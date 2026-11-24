import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr, ValidationError
from typer.testing import CliRunner

from llm_lab.cli import app
from llm_lab.cli.jobs import get_instance_provider, get_job_record_store
from llm_lab.jobs.orchestrator import FileJobRecordStore, JobOrchestrator
from llm_lab.jobs.protocol import (
    Aborted,
    ExplicitOffer,
    Failed,
    JobRecord,
    JobSpec,
    ResourceRequest,
    Running,
    Succeeded,
)
from llm_lab.jobs.vast import VastInstanceProvider, VastSettings
from llm_lab.settings import Settings


@dataclass
class FakeVast:
    offers: list[dict[str, float]] = field(
        default_factory=lambda: [{"id": 12, "dph_total": 0.9}, {"id": 11, "dph_total": 0.4}]
    )
    unavailable_offers: set[int] = field(default_factory=set)
    actual_statuses: Iterator[str] = field(default_factory=lambda: iter(["loading", "running"]))
    log_text: str = "step 1\nllm-lab-job-exit-code=0\n"
    missing_instance_response: httpx.Response = field(
        default_factory=lambda: httpx.Response(404, json={"success": False, "error": "not_found"})
    )
    fail_before: dict[tuple[str, str], Callable[[], httpx.Response]] = field(default_factory=dict)
    live_instances: dict[int, str] = field(default_factory=dict)
    rented_offers: list[int] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        parts = request.url.path.strip("/").split("/")
        if request.url.host == "logs.test":
            if "Authorization" in request.headers:
                return httpx.Response(400)
            return httpx.Response(200, text=self.log_text)
        if request.headers["Authorization"] != "Bearer secret-key":
            return httpx.Response(401)
        match request.method, parts:
            case "POST", ["api", "v0", "bundles"]:
                return httpx.Response(200, json={"offers": self.offers})
            case "PUT", ["api", "v0", "asks", offer_id]:
                if int(offer_id) in self.unavailable_offers:
                    return httpx.Response(404, json={"success": False, "error": "invalid_args"})
                instance_id = 9000 + len(self.rented_offers)
                self.rented_offers.append(int(offer_id))
                self.live_instances[instance_id] = json.loads(request.content)["label"]
                if failure := self.fail_before.get(("PUT", "asks")):
                    return failure()
                return httpx.Response(200, json={"success": True, "new_contract": instance_id})
            case "PUT", ["api", "v0", "instances", "request_logs", instance_id]:
                if failure := self.fail_before.get(("PUT", "request_logs")):
                    return failure()
                return httpx.Response(200, json={"success": True, "result_url": f"https://logs.test/{instance_id}.log"})
            case "GET", ["api", "v0", "instances", instance_id]:
                if failure := self.fail_before.get(("GET", "instances")):
                    return failure()
                if int(instance_id) not in self.live_instances:
                    return httpx.Response(404)
                return httpx.Response(200, json={"instances": {"actual_status": next(self.actual_statuses)}})
            case "GET", ["api", "v1", "instances"]:
                label = json.loads(request.url.params["select_filters"])["label"]["eq"]
                matching = [{"id": instance_id} for instance_id, live in self.live_instances.items() if live == label]
                return httpx.Response(200, json={"success": True, "instances": matching})
            case "DELETE", ["api", "v0", "instances", instance_id]:
                if int(instance_id) not in self.live_instances:
                    return self.missing_instance_response
                del self.live_instances[int(instance_id)]
                return httpx.Response(200, json={"success": True})
            case _:
                return httpx.Response(400)


@dataclass
class SteppingClock:
    now: datetime = datetime(2026, 1, 1, tzinfo=UTC)

    def read(self) -> datetime:
        current = self.now
        self.now += timedelta(seconds=5)
        return current


@dataclass
class Workbench:
    vast: FakeVast
    store: FileJobRecordStore
    provider: VastInstanceProvider
    orchestrator: JobOrchestrator
    sleeps: list[float]


@pytest.fixture
def workbench(tmp_path: Path) -> Workbench:
    vast = FakeVast()
    sleeps: list[float] = []
    settings = Settings(job_records_directory=tmp_path / "records")
    store = FileJobRecordStore(settings)
    provider = VastInstanceProvider(
        VastSettings(api_key=SecretStr("secret-key")),
        httpx.Client(transport=httpx.MockTransport(vast.handle)),
        sleeps.append,
    )
    return Workbench(
        vast, store, provider, JobOrchestrator(settings, provider, store, SteppingClock().read, sleeps.append), sleeps
    )


def build_spec(placement: ResourceRequest | ExplicitOffer | None = None) -> JobSpec:
    return JobSpec(
        name="train",
        image="ghcr.io/example/train:1",
        command=("python", "train.py", "--note", "two words"),
        environment={"SEED": "1"},
        disk_gb=40,
        placement=placement
        or ResourceRequest(kind="resources", gpu_name="RTX 4090", gpu_count=1, max_dollars_per_hour=1.0),
    )


def test_job_rents_the_offer_with_the_lowest_price_and_destroys_the_instance_after_success(
    workbench: Workbench,
) -> None:
    record = workbench.orchestrator.execute(build_spec())

    submitted_at = datetime(2026, 1, 1, tzinfo=UTC)
    assert record.status == Succeeded(
        submitted_at=submitted_at, finished_at=submitted_at + timedelta(seconds=5), instance_id=9000
    )
    assert workbench.store.read("train") == record
    assert workbench.vast.rented_offers == [11]
    assert workbench.vast.live_instances == {}
    assert workbench.sleeps == [15]
    search, rent = (json.loads(request.content) for request in workbench.vast.requests[:2])
    assert search["gpu_name"] == {"eq": "RTX 4090"}
    assert search["dph_total"] == {"lte": 1.0}
    assert rent == {
        "client_id": "me",
        "image": "ghcr.io/example/train:1",
        "disk": 40.0,
        "label": "llm-lab-job-train",
        "runtype": "args",
        "args": [
            "sh",
            "-c",
            "if [ -e /llm-lab-job-done ]; then exit 0; fi\n"
            "python train.py --note 'two words'\n"
            "exit_code=$?\n"
            "touch /llm-lab-job-done\n"
            'echo "llm-lab-job-exit-code=$exit_code"\n'
            "sleep 300",
        ],
        "env": {"SEED": "1"},
    }


def test_job_with_nonzero_exit_code_is_recorded_as_failed_and_its_instance_is_destroyed(workbench: Workbench) -> None:
    workbench.vast.log_text = "Traceback\nllm-lab-job-exit-code=3\n"

    record = workbench.orchestrator.execute(build_spec())

    assert isinstance(record.status, Failed)
    assert record.status.exit_code == 3
    assert workbench.vast.live_instances == {}


def test_job_rents_the_next_offer_when_an_offer_is_not_available(workbench: Workbench) -> None:
    workbench.vast.unavailable_offers = {11}

    record = workbench.orchestrator.execute(build_spec())

    assert isinstance(record.status, Succeeded)
    assert workbench.vast.rented_offers == [12]


def test_job_with_an_explicit_offer_rents_that_offer_without_a_search(workbench: Workbench) -> None:
    record = workbench.orchestrator.execute(build_spec(ExplicitOffer(kind="offer", offer_id=77)))

    assert isinstance(record.status, Succeeded)
    assert workbench.vast.rented_offers == [77]
    assert workbench.vast.requests[0].method == "PUT"


def test_job_is_aborted_when_no_offer_is_available(workbench: Workbench) -> None:
    workbench.vast.offers = []

    record = workbench.orchestrator.execute(build_spec())

    assert isinstance(record.status, Aborted)
    assert workbench.vast.rented_offers == []


def test_job_is_aborted_and_its_instance_destroyed_when_the_container_exits_without_an_exit_code(
    workbench: Workbench,
) -> None:
    workbench.vast.actual_statuses = iter(["loading", "exited"])
    workbench.vast.log_text = "killed\n"

    record = workbench.orchestrator.execute(build_spec())

    assert isinstance(record.status, Aborted)
    assert "exited" in record.status.reason
    assert workbench.vast.live_instances == {}


def test_job_is_aborted_when_the_log_of_an_exited_instance_is_refused(workbench: Workbench) -> None:
    workbench.vast.actual_statuses = iter(["exited"])
    workbench.vast.fail_before[("PUT", "request_logs")] = lambda: httpx.Response(404)

    record = workbench.orchestrator.execute(build_spec())

    assert isinstance(record.status, Aborted)
    assert workbench.vast.live_instances == {}


def test_abandon_accepts_an_error_from_the_destroy_request_when_the_instance_does_not_exist(
    workbench: Workbench,
) -> None:
    workbench.vast.missing_instance_response = httpx.Response(400, json={"success": False, "error": "invalid_args"})
    workbench.store.write(
        JobRecord(spec=build_spec(), status=Running(submitted_at=datetime(2026, 1, 1, tzinfo=UTC), instance_id=501))
    )

    record = workbench.orchestrator.abandon("train")

    assert isinstance(record.status, Aborted)


def test_instance_is_destroyed_when_observation_raises_an_error(workbench: Workbench) -> None:
    workbench.vast.fail_before[("GET", "instances")] = lambda: httpx.Response(500)

    with pytest.raises(httpx.HTTPStatusError):
        workbench.orchestrator.execute(build_spec())

    assert workbench.vast.rented_offers == [11]
    assert workbench.vast.live_instances == {}
    assert isinstance(workbench.store.read("train").status, Running)


def raise_read_timeout() -> httpx.Response:
    raise httpx.ReadTimeout("the response of the rent request was lost")


def test_instance_is_destroyed_when_the_response_of_the_rent_request_is_lost(workbench: Workbench) -> None:
    workbench.vast.fail_before[("PUT", "asks")] = raise_read_timeout

    with pytest.raises(httpx.ReadTimeout):
        workbench.orchestrator.execute(build_spec())

    assert workbench.vast.rented_offers == [11]
    assert workbench.vast.live_instances == {}


def test_rate_limited_request_is_sent_again_after_a_delay(workbench: Workbench) -> None:
    responses = iter([httpx.Response(429), httpx.Response(429)])
    workbench.vast.fail_before[("GET", "instances")] = lambda: (
        next(responses, None) or httpx.Response(200, json={"instances": {"actual_status": "running"}})
    )

    record = workbench.orchestrator.execute(build_spec())

    assert isinstance(record.status, Succeeded)
    assert workbench.sleeps == [2, 4]


def test_abandon_destroys_the_instance_of_a_running_job_and_records_the_job_as_aborted(workbench: Workbench) -> None:
    spec = build_spec()
    workbench.vast.live_instances = {501: "llm-lab-job-train", 502: "llm-lab-job-train", 503: "llm-lab-job-other"}
    workbench.store.write(
        JobRecord(spec=spec, status=Running(submitted_at=datetime(2026, 1, 1, tzinfo=UTC), instance_id=501))
    )

    record = workbench.orchestrator.abandon("train")

    assert isinstance(record.status, Aborted)
    assert workbench.store.read("train") == record
    assert workbench.vast.live_instances == {503: "llm-lab-job-other"}


@pytest.mark.parametrize(
    "change",
    [
        {"name": "Not/A/Name"},
        {"command": []},
        {"retries": 2},
        {"placement": {"kind": "resources", "gpu_name": "RTX 4090"}},
    ],
)
def test_invalid_manifest_is_rejected(change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        JobSpec.model_validate(build_spec().model_dump() | change)


@pytest.mark.parametrize(
    ("log_text", "cli_exit_code"), [("llm-lab-job-exit-code=0\n", 0), ("llm-lab-job-exit-code=7\n", 1)]
)
def test_cli_runs_a_manifest_file_and_reports_the_record(
    tmp_path: Path, workbench: Workbench, log_text: str, cli_exit_code: int
) -> None:
    workbench.vast.log_text = log_text
    workbench.vast.actual_statuses = iter(["running"])
    manifest = tmp_path / "job.yaml"
    manifest.write_text(
        "name: train\nimage: ghcr.io/example/train:1\ncommand: [python, train.py]\ndisk_gb: 40\n"
        "placement:\n  kind: resources\n  gpu_name: RTX 4090\n  gpu_count: 1\n  max_dollars_per_hour: 1.0\n"
    )
    overridden = app.with_dependency_overrides(
        {get_job_record_store: lambda: workbench.store, get_instance_provider: lambda: workbench.provider}
    )

    run_result = CliRunner().invoke(overridden, ["jobs", "run", str(manifest)], catch_exceptions=False)
    status_result = CliRunner().invoke(overridden, ["jobs", "status", "train"], catch_exceptions=False)

    assert run_result.exit_code == cli_exit_code
    assert JobRecord.model_validate_json(run_result.output) == workbench.store.read("train")
    assert status_result.output == run_result.output
    assert workbench.vast.live_instances == {}
