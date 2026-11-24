import json
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from keyring.backend import KeyringBackend
from pydantic import SecretStr, ValidationError
from typer.testing import CliRunner

from llm_lab.cli import app
from llm_lab.cli.jobs import get_http_client, get_job_record_store
from llm_lab.cli.secrets import get_secret_source
from llm_lab.jobs.orchestrator import JobOrchestrator
from llm_lab.jobs.protocol import (
    Aborted,
    ExplicitOffer,
    Failed,
    JobSpec,
    Pending,
    ResourceRequest,
    Running,
    Succeeded,
)
from llm_lab.jobs.secrets import KeyringSecretSource
from llm_lab.jobs.store import SqliteJobRecordStore
from llm_lab.jobs.vast import VastInstanceProvider, VastSettings


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


class MemoryKeyring(KeyringBackend):
    def __init__(self) -> None:
        super().__init__()
        self.passwords: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.passwords.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.passwords[service, username] = password

    def delete_password(self, service: str, username: str) -> None:
        del self.passwords[service, username]


@dataclass
class Workbench:
    vast: FakeVast
    secret_source: KeyringSecretSource
    store: SqliteJobRecordStore
    provider: VastInstanceProvider
    orchestrator: JobOrchestrator
    sleeps: list[float]


@pytest.fixture
def workbench(tmp_path: Path) -> Iterator[Workbench]:
    vast = FakeVast()
    sleeps: list[float] = []
    secret_source = KeyringSecretSource(MemoryKeyring(), "llm-lab-test")
    secret_source.write("VAST_API_KEY", SecretStr("secret-key"))
    secret_source.write("HF_TOKEN", SecretStr("hf-secret-value"))
    provider = VastInstanceProvider(
        VastSettings(api_key=secret_source.read("VAST_API_KEY")),
        httpx.Client(transport=httpx.MockTransport(vast.handle)),
        sleeps.append,
    )
    connection = sqlite3.connect(tmp_path / "jobs.sqlite3")
    store = SqliteJobRecordStore(connection)
    yield Workbench(
        vast,
        secret_source,
        store,
        provider,
        JobOrchestrator(provider, store, secret_source, SteppingClock().read),
        sleeps,
    )
    connection.close()


def build_spec(placement: ResourceRequest | ExplicitOffer | None = None) -> JobSpec:
    return JobSpec(
        name="train",
        image="ghcr.io/example/train:1",
        command=("python", "train.py", "--note", "two words"),
        environment={"SEED": "1"},
        secret_environment=("HF_TOKEN",),
        disk_gb=40,
        ttl_seconds_after_finished=600,
        placement=placement
        or ResourceRequest(kind="resources", gpu_name="RTX 4090", gpu_count=1, max_dollars_per_hour=1.0),
    )


def test_submit_rents_the_offer_with_the_lowest_price_and_returns_while_the_instance_runs(workbench: Workbench) -> None:
    record = workbench.orchestrator.submit(build_spec())

    assert record.status == Running(submitted_at=datetime(2026, 1, 1, tzinfo=UTC), instance_id=9000)
    assert workbench.store.read_latest("train") == record
    assert workbench.vast.rented_offers == [11]
    assert workbench.vast.live_instances == {9000: "llm-lab-job-train"}
    assert workbench.sleeps == []
    search, rent = (json.loads(request.content) for request in workbench.vast.requests)
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
            "if [ ! -e /llm-lab-job-done ]; then\n"
            "python train.py --note 'two words'\n"
            "exit_code=$?\n"
            "touch /llm-lab-job-done\n"
            'echo "llm-lab-job-exit-code=$exit_code"\n'
            "fi\n"
            "sleep 600\n"
            'curl -fsS -X DELETE -H "Authorization: Bearer $CONTAINER_API_KEY" '
            '"https://console.vast.ai/api/v0/instances/$CONTAINER_ID/"',
        ],
        "env": {"SEED": "1", "HF_TOKEN": "hf-secret-value"},
    }
    assert "hf-secret-value" not in record.model_dump_json()
    stored = workbench.store.connection.execute("SELECT group_concat(record) FROM job_transitions").fetchone()
    assert '"secret_environment":["HF_TOKEN"]' in stored[0]
    assert "hf-secret-value" not in stored[0]


def test_submit_rents_no_instance_and_records_no_job_when_a_secret_is_not_in_the_keyring(workbench: Workbench) -> None:
    spec = build_spec().model_copy(update={"secret_environment": ("HF_TOKEN", "WANDB_API_KEY")})

    with pytest.raises(LookupError, match="WANDB_API_KEY"):
        workbench.orchestrator.submit(spec)

    assert workbench.vast.requests == []
    assert workbench.store.read_latest("train") is None


def test_sync_leaves_a_job_running_while_its_command_has_not_reported_an_exit_code(workbench: Workbench) -> None:
    workbench.vast.log_text = "step 1\n"
    submitted = workbench.orchestrator.submit(build_spec())

    assert workbench.orchestrator.sync("train") == submitted
    assert workbench.orchestrator.sync("train") == submitted
    assert workbench.vast.live_instances == {9000: "llm-lab-job-train"}
    assert len(workbench.store.read_history("train")) == 2


@pytest.mark.parametrize(
    ("log_text", "status_type"), [("llm-lab-job-exit-code=0\n", Succeeded), ("llm-lab-job-exit-code=3\n", Failed)]
)
def test_sync_records_the_exit_code_and_destroys_the_instance(
    workbench: Workbench, log_text: str, status_type: type[Succeeded | Failed]
) -> None:
    workbench.vast.log_text = log_text
    workbench.vast.actual_statuses = iter(["running"])
    workbench.orchestrator.submit(build_spec())

    record = workbench.orchestrator.sync("train")

    assert isinstance(record.status, status_type)
    assert workbench.store.read_latest("train") == record
    assert workbench.vast.live_instances == {}
    assert [type(transition.record.status) for transition in workbench.store.read_history("train")] == [
        Pending,
        Running,
        status_type,
    ]
    assert workbench.orchestrator.sync("train") == record


def test_sync_all_finishes_each_job_that_is_not_finished(workbench: Workbench) -> None:
    workbench.vast.actual_statuses = iter(["running", "running"])
    workbench.orchestrator.submit(build_spec())
    workbench.orchestrator.submit(build_spec().model_copy(update={"name": "evaluate"}))

    records = workbench.orchestrator.sync_all()

    assert [(record.spec.name, record.status.condition) for record in records] == [
        ("evaluate", "succeeded"),
        ("train", "succeeded"),
    ]
    assert workbench.vast.live_instances == {}


def test_sync_records_a_job_as_aborted_when_its_instance_destroyed_itself(workbench: Workbench) -> None:
    workbench.orchestrator.submit(build_spec())
    workbench.vast.live_instances.clear()

    record = workbench.orchestrator.sync("train")

    assert isinstance(record.status, Aborted)
    assert "missing" in record.status.reason


@pytest.mark.parametrize("request_logs_response", [None, httpx.Response(404)])
def test_sync_records_a_job_as_aborted_and_destroys_the_instance_when_the_container_exited_without_an_exit_code(
    workbench: Workbench, request_logs_response: httpx.Response | None
) -> None:
    workbench.vast.actual_statuses = iter(["exited"])
    workbench.vast.log_text = "killed\n"
    if request_logs_response is not None:
        workbench.vast.fail_before[("PUT", "request_logs")] = lambda: request_logs_response
    workbench.orchestrator.submit(build_spec())

    record = workbench.orchestrator.sync("train")

    assert isinstance(record.status, Aborted)
    assert "exited" in record.status.reason
    assert workbench.vast.live_instances == {}


def test_submit_rents_the_next_offer_when_an_offer_is_not_available(workbench: Workbench) -> None:
    workbench.vast.unavailable_offers = {11}

    record = workbench.orchestrator.submit(build_spec())

    assert isinstance(record.status, Running)
    assert workbench.vast.rented_offers == [12]


def test_submit_with_an_explicit_offer_rents_that_offer_without_a_search(workbench: Workbench) -> None:
    record = workbench.orchestrator.submit(build_spec(ExplicitOffer(kind="offer", offer_id=77)))

    assert isinstance(record.status, Running)
    assert workbench.vast.rented_offers == [77]
    assert workbench.vast.requests[0].method == "PUT"


def test_submit_records_a_job_as_aborted_when_no_offer_is_available(workbench: Workbench) -> None:
    workbench.vast.offers = []

    record = workbench.orchestrator.submit(build_spec())

    assert isinstance(record.status, Aborted)
    assert workbench.vast.rented_offers == []


def test_submit_rejects_a_job_name_that_is_not_finished_and_accepts_it_after_the_job_finished(
    workbench: Workbench,
) -> None:
    workbench.vast.actual_statuses = iter(["running"])
    workbench.orchestrator.submit(build_spec())

    with pytest.raises(ValueError, match="not finished"):
        workbench.orchestrator.submit(build_spec())

    workbench.orchestrator.sync("train")
    assert isinstance(workbench.orchestrator.submit(build_spec()).status, Running)
    assert workbench.vast.rented_offers == [11, 11]


def raise_read_timeout() -> httpx.Response:
    raise httpx.ReadTimeout("the response of the rent request was lost")


def test_instance_is_destroyed_when_the_response_of_the_rent_request_is_lost(workbench: Workbench) -> None:
    workbench.vast.fail_before[("PUT", "asks")] = raise_read_timeout

    with pytest.raises(httpx.ReadTimeout):
        workbench.orchestrator.submit(build_spec())

    assert workbench.vast.rented_offers == [11]
    assert workbench.vast.live_instances == {}
    record = workbench.orchestrator.sync("train")
    assert isinstance(record.status, Aborted)


def test_sync_keeps_the_job_running_when_observation_raises_an_error(workbench: Workbench) -> None:
    submitted = workbench.orchestrator.submit(build_spec())
    workbench.vast.fail_before[("GET", "instances")] = lambda: httpx.Response(500)

    with pytest.raises(httpx.HTTPStatusError):
        workbench.orchestrator.sync("train")

    assert workbench.store.read_latest("train") == submitted
    assert workbench.vast.live_instances == {9000: "llm-lab-job-train"}


def test_rate_limited_request_is_sent_again_after_a_delay(workbench: Workbench) -> None:
    workbench.orchestrator.submit(build_spec())
    responses = iter([httpx.Response(429), httpx.Response(429)])
    workbench.vast.fail_before[("GET", "instances")] = lambda: (
        next(responses, None) or httpx.Response(200, json={"instances": {"actual_status": "running"}})
    )

    record = workbench.orchestrator.sync("train")

    assert isinstance(record.status, Succeeded)
    assert workbench.sleeps == [2, 4]


@pytest.mark.parametrize(
    "missing_instance_response",
    [
        httpx.Response(404, json={"success": False, "error": "not_found"}),
        httpx.Response(400, json={"success": False, "error": "invalid_args"}),
    ],
)
def test_abandon_destroys_each_instance_of_the_job_and_records_the_job_as_aborted(
    workbench: Workbench, missing_instance_response: httpx.Response
) -> None:
    workbench.vast.missing_instance_response = missing_instance_response
    workbench.orchestrator.submit(build_spec())
    workbench.vast.live_instances |= {502: "llm-lab-job-train", 503: "llm-lab-job-other"}

    record = workbench.orchestrator.abandon("train")

    assert isinstance(record.status, Aborted)
    assert workbench.store.read_latest("train") == record
    assert workbench.vast.live_instances == {503: "llm-lab-job-other"}
    assert workbench.orchestrator.abandon("train") == record


def test_unknown_job_name_is_rejected(workbench: Workbench) -> None:
    with pytest.raises(LookupError):
        workbench.orchestrator.sync("absent")


@pytest.mark.parametrize(
    "change",
    [
        {"name": "Not/A/Name"},
        {"command": []},
        {"retries": 2},
        {"ttl_seconds_after_finished": 0},
        {"secret_environment": ["SEED"]},
        {"secret_environment": ["not a name"]},
        {"placement": {"kind": "resources", "gpu_name": "RTX 4090"}},
    ],
)
def test_invalid_manifest_is_rejected(change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        JobSpec.model_validate(build_spec().model_dump() | change)


@pytest.mark.parametrize(
    ("log_text", "status_exit_code"), [("llm-lab-job-exit-code=0\n", 0), ("llm-lab-job-exit-code=7\n", 1)]
)
def test_cli_submits_a_manifest_file_and_reports_the_job_from_the_database(
    tmp_path: Path, workbench: Workbench, log_text: str, status_exit_code: int
) -> None:
    workbench.vast.log_text = log_text
    workbench.vast.actual_statuses = iter(["running"])
    manifest = tmp_path / "job.yaml"
    manifest.write_text(
        "name: train\nimage: ghcr.io/example/train:1\ncommand: [python, train.py]\ndisk_gb: 40\n"
        "ttl_seconds_after_finished: 600\nsecret_environment: [WANDB_API_KEY]\n"
        "placement:\n  kind: resources\n  gpu_name: RTX 4090\n  gpu_count: 1\n  max_dollars_per_hour: 1.0\n"
    )
    overridden = app.with_dependency_overrides(
        {
            get_job_record_store: lambda: workbench.store,
            get_secret_source: lambda: workbench.secret_source,
            get_http_client: lambda: httpx.Client(transport=httpx.MockTransport(workbench.vast.handle)),
        }
    )

    def invoke(*arguments: str) -> tuple[int, str]:
        result = CliRunner().invoke(overridden, ["jobs", *arguments], catch_exceptions=False)
        return result.exit_code, result.output

    result = CliRunner().invoke(overridden, ["secrets", "set", "WANDB_API_KEY"], input="wandb-secret-value\n")
    assert "wandb-secret-value" not in result.output
    submit_code, submit_output = invoke("submit", str(manifest))
    assert submit_code == 0
    assert "wandb-secret-value" not in submit_output
    assert json.loads(workbench.vast.requests[-1].content)["env"] == {"WANDB_API_KEY": "wandb-secret-value"}
    assert invoke("list") == (0, "train\trunning\n")
    status_code, status_output = invoke("status", "train")
    condition = "succeeded" if status_exit_code == 0 else "failed"
    assert status_code == status_exit_code
    assert f'"condition": "{condition}"' in status_output
    assert invoke("sync") == (0, f"train\t{condition}\n")
    assert [line.split('"condition":"')[1].split('"')[0] for line in invoke("history", "train")[1].splitlines()] == [
        "pending",
        "running",
        condition,
    ]
    assert workbench.vast.live_instances == {}
