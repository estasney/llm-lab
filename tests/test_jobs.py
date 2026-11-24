import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import keyring
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from keyring.backend import KeyringBackend
from pydantic import SecretStr, ValidationError
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from llm_lab.cli import app
from llm_lab.cli.jobs.deps import get_http_client
from llm_lab.database import build_migration_config
from llm_lab.jobs.orchestrator import JobOrchestrator
from llm_lab.jobs.protocol import (
    Aborted,
    ExplicitOffer,
    Failed,
    JobSpec,
    PlacementKind,
    ResourceRequest,
    Running,
    Succeeded,
)
from llm_lab.jobs.secrets import SettingsSecretSource
from llm_lab.jobs.tables import Base, JobTransitionRow
from llm_lab.jobs.vast import VastInstanceProvider
from llm_lab.keyring_settings import SecretName
from llm_lab.settings import Settings, get_settings


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
    engine: Engine
    provider: VastInstanceProvider
    orchestrator: JobOrchestrator
    sleeps: list[float]


@pytest.fixture
def workbench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Workbench]:
    vast = FakeVast()
    sleeps: list[float] = []
    previous_keyring = keyring.get_keyring()
    keyring.set_keyring(MemoryKeyring())
    keyring.set_password("llm-lab", "VAST_API_KEY", "secret-key")
    keyring.set_password("llm-lab", "HF_TOKEN", "hf-secret-value")
    monkeypatch.setenv("LLM_LAB_JOB_DATABASE_PATH", str(tmp_path / "jobs.sqlite3"))
    settings = Settings()
    engine = settings.database_engine
    command.upgrade(build_migration_config(engine), "head")
    session = Session(engine)
    provider = VastInstanceProvider(
        settings,
        httpx.Client(transport=httpx.MockTransport(vast.handle)),
        sleeps.append,
    )
    yield Workbench(
        vast,
        engine,
        provider,
        JobOrchestrator(provider, session, SettingsSecretSource(settings), SteppingClock().read),
        sleeps,
    )
    session.close()
    engine.dispose()
    discard_cached_settings()
    keyring.set_keyring(previous_keyring)


def discard_cached_settings() -> None:
    get_settings().database_engine.dispose()
    get_settings.cache_clear()


def read_conditions(engine: Engine, name: str) -> list[str]:
    with Session(engine) as session:
        return list(
            session.scalars(
                select(JobTransitionRow.condition)
                .where(JobTransitionRow.name == name)
                .order_by(JobTransitionRow.sequence)
            )
        )


def build_spec(placement: ResourceRequest | ExplicitOffer | None = None) -> JobSpec:
    return JobSpec(
        name="train",
        image="ghcr.io/example/train:1",
        command=("python", "train.py", "--note", "two words"),
        environment={"SEED": "1"},
        secret_environment=(SecretName.hf_token,),
        disk_gb=40,
        ttl_seconds_after_finished=600,
        placement=placement
        or ResourceRequest(kind=PlacementKind.resources, gpu_name="RTX 4090", gpu_count=1, max_dollars_per_hour=1.0),
    )


def test_submit_rents_the_offer_with_the_lowest_price_and_returns_while_the_instance_runs(workbench: Workbench) -> None:
    record = workbench.orchestrator.submit(build_spec())

    assert record.status == Running(submitted_at=datetime(2026, 1, 1, tzinfo=UTC), instance_id=9000)
    assert workbench.orchestrator.read("train") == record
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
    with workbench.engine.connect() as connection:
        stored = "".join(connection.execute(text("SELECT record FROM job_transitions")).scalars())
    assert '"secret_environment":["HF_TOKEN"]' in stored
    assert "hf-secret-value" not in stored


def test_each_database_connection_has_the_sqlite_pragmas(workbench: Workbench) -> None:
    with workbench.engine.connect() as first, workbench.engine.connect() as second:
        for connection in (first, second):
            pragmas = {
                name: connection.execute(text(f"PRAGMA {name}")).scalar_one()
                for name in ("foreign_keys", "journal_mode", "synchronous", "busy_timeout")
            }
            assert pragmas == {"foreign_keys": 1, "journal_mode": "wal", "synchronous": 1, "busy_timeout": 5000}


def test_migrations_create_the_schema_of_the_models(workbench: Workbench) -> None:
    with workbench.engine.connect() as connection:
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []


def test_sync_leaves_a_job_running_while_its_command_has_not_reported_an_exit_code(workbench: Workbench) -> None:
    workbench.vast.log_text = "step 1\n"
    submitted = workbench.orchestrator.submit(build_spec())

    assert workbench.orchestrator.sync("train") == submitted
    assert workbench.orchestrator.sync("train") == submitted
    assert workbench.vast.live_instances == {9000: "llm-lab-job-train"}
    assert read_conditions(workbench.engine, "train") == ["pending", "running"]


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
    assert workbench.orchestrator.read("train") == record
    assert workbench.vast.live_instances == {}
    assert read_conditions(workbench.engine, "train") == ["pending", "running", record.status.condition]
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
    record = workbench.orchestrator.submit(build_spec(ExplicitOffer(kind=PlacementKind.offer, offer_id=77)))

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

    assert workbench.orchestrator.read("train") == submitted
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
    assert workbench.orchestrator.read("train") == record
    assert workbench.vast.live_instances == {503: "llm-lab-job-other"}
    assert workbench.orchestrator.abandon("train") == record


def test_settings_read_a_keyring_entry_only_for_a_field_with_the_annotation(workbench: Workbench) -> None:
    keyring.set_password("llm-lab", "vast_api_url", "https://wrong.test")
    keyring.set_password("llm-lab", "VAST_API_URL", "https://wrong.test")

    settings = Settings()

    assert settings.vast_api_key == SecretStr("secret-key")
    assert settings.vast_api_url == "https://console.vast.ai/api"


def test_settings_prefer_an_environment_variable_to_a_keyring_entry(
    workbench: Workbench, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_LAB_VAST_API_KEY", "key-from-environment")

    assert Settings().vast_api_key == SecretStr("key-from-environment")


def test_cli_rejects_a_vast_command_when_the_api_key_has_no_keyring_entry(workbench: Workbench) -> None:
    keyring.delete_password("llm-lab", "VAST_API_KEY")
    overridden = app

    assert CliRunner().invoke(overridden, ["jobs", "list"], catch_exceptions=False).exit_code == 0
    with pytest.raises(LookupError, match="secrets set VAST_API_KEY"):
        CliRunner().invoke(overridden, ["jobs", "sync"], catch_exceptions=False)


def test_cli_prompts_for_the_job_name_and_the_manifest_when_they_are_omitted(
    tmp_path: Path, workbench: Workbench
) -> None:
    workbench.orchestrator.submit(build_spec())

    history = CliRunner().invoke(app, ["jobs", "history"], input="train\n", catch_exceptions=False)
    submit = CliRunner().invoke(app, ["jobs", "submit"], input=f"{tmp_path / 'absent.yaml'}\n")

    assert "Job name: " in history.output
    assert history.output.count('"condition":') == 2
    assert "Manifest file: " in submit.output
    assert submit.exit_code == 2
    assert "does not exist" in submit.output


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
        {"environment": {"HF_TOKEN": "plain"}},
        {"secret_environment": ["WANDB_API_KEY"]},
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
        "ttl_seconds_after_finished: 600\nsecret_environment: [HF_TOKEN]\n"
        "placement:\n  kind: resources\n  gpu_name: RTX 4090\n  gpu_count: 1\n  max_dollars_per_hour: 1.0\n"
    )
    overridden = app.with_dependency_overrides(
        {
            get_http_client: lambda: httpx.Client(transport=httpx.MockTransport(workbench.vast.handle)),
        }
    )

    def invoke(*arguments: str) -> tuple[int, str]:
        result = CliRunner().invoke(overridden, ["jobs", *arguments], catch_exceptions=False)
        return result.exit_code, result.output

    keyring.delete_password("llm-lab", "HF_TOKEN")
    with pytest.raises(LookupError, match="secrets set HF_TOKEN"):
        invoke("submit", str(manifest))
    assert workbench.vast.requests == []
    assert read_conditions(workbench.engine, "train") == []
    assert CliRunner().invoke(overridden, ["secrets", "set", "WANDB_API_KEY"]).exit_code == 2
    discard_cached_settings()
    result = CliRunner().invoke(
        overridden, ["secrets", "set"], input="NOT_A_SECRET\nHF_TOKEN\nwandb-secret-value\nwandb-secret-value\n"
    )
    assert result.exit_code == 0
    assert "wandb-secret-value" not in result.output
    submit_code, submit_output = invoke("submit", str(manifest))
    assert submit_code == 0
    assert "wandb-secret-value" not in submit_output
    assert json.loads(workbench.vast.requests[-1].content)["env"] == {"HF_TOKEN": "wandb-secret-value"}
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
