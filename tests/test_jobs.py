import base64
import json
import os
import shlex
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import assert_never

import httpx
import keyring
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat, load_ssh_private_key
from keyring.backend import KeyringBackend
from pydantic import SecretStr, ValidationError
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from llm_lab.cli import app
from llm_lab.cli.jobs.deps import get_artifact_downloader, get_http_client
from llm_lab.database import build_migration_config
from llm_lab.errors import CommandError
from llm_lab.jobs.access import CodeDelivery, JobAccessStore, RsyncDownloader, RsyncUploader, SshEndpoint
from llm_lab.jobs.code import WorkspaceCodeSource
from llm_lab.jobs.orchestrator import JobOrchestrator
from llm_lab.jobs.protocol import (
    Aborted,
    ExplicitOffer,
    Failed,
    JobCode,
    JobSpec,
    PlacementKind,
    ResourceRequest,
    Running,
    Succeeded,
)
from llm_lab.jobs.tables import Base, JobTransitionRow
from llm_lab.jobs.vast import VastInstanceProvider, build_offer_query
from llm_lab.keyring_settings import SecretName
from llm_lab.secrets import SettingsSecretSource
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
    ports: dict[str, list[dict[str, str]]] = field(
        default_factory=lambda: {"2222/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41022"}]}
    )

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
                return httpx.Response(
                    200,
                    json={
                        "instances": {
                            "actual_status": next(self.actual_statuses, "running"),
                            "public_ipaddr": "203.0.113.5",
                            "ports": self.ports,
                        }
                    },
                )
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
class FakeDownloader:
    directory: Path
    downloads: list[tuple[str, SshEndpoint, str]] = field(default_factory=list)
    failure: CommandError | None = None

    def download(self, job_name: str, endpoint: SshEndpoint, remote_directory: str) -> Path:
        if self.failure is not None:
            raise self.failure
        self.downloads.append((job_name, endpoint, remote_directory))
        return self.directory / job_name


@dataclass
class FakeCodeSource:
    code: JobCode = field(
        default_factory=lambda: JobCode(root=Path("/work"), files=("jobs/train/train.py", "uv.lock"), digest="0a1b")
    )

    def read(self, package: str) -> JobCode:
        return self.code


@dataclass
class FakeUploader:
    deliveries: Iterator[CodeDelivery] = field(default_factory=lambda: iter([CodeDelivery.delivered]))
    uploads: list[tuple[str, SshEndpoint, JobCode]] = field(default_factory=list)
    failure: CommandError | None = None

    def upload(self, job_name: str, endpoint: SshEndpoint, code: JobCode) -> CodeDelivery:
        if self.failure is not None:
            raise self.failure
        self.uploads.append((job_name, endpoint, code))
        return next(self.deliveries, CodeDelivery.unreachable)


@dataclass
class Workbench:
    vast: FakeVast
    engine: Engine
    provider: VastInstanceProvider
    orchestrator: JobOrchestrator
    sleeps: list[float]
    access: JobAccessStore
    uploader: FakeUploader
    downloader: FakeDownloader


@pytest.fixture
def workbench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Workbench]:
    vast = FakeVast()
    sleeps: list[float] = []
    previous_keyring = keyring.get_keyring()
    keyring.set_keyring(MemoryKeyring())
    keyring.set_password("llm-lab", "VAST_API_KEY", "secret-key")
    keyring.set_password("llm-lab", "HF_TOKEN", "hf-secret-value")
    monkeypatch.setenv("LLM_LAB_JOB_DATABASE_PATH", str(tmp_path / "jobs.sqlite3"))
    monkeypatch.setenv("LLM_LAB_JOB_ACCESS_DIR", str(tmp_path / "access"))
    monkeypatch.setenv("LLM_LAB_JOB_DOWNLOADS_DIR", str(tmp_path / "downloads"))
    settings = Settings()
    access = JobAccessStore(settings)
    uploader = FakeUploader()
    downloader = FakeDownloader(settings.job_downloads_dir)
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
        JobOrchestrator(
            settings,
            provider,
            session,
            SettingsSecretSource(settings),
            access,
            FakeCodeSource(),
            uploader,
            downloader,
            SteppingClock().read,
            sleeps.append,
        ),
        sleeps,
        access,
        uploader,
        downloader,
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
    """
    Arrange: a fake Vast.ai with two offers at different prices, and a job spec that requests a GPU, sets an
        environment variable, a secret and a command with an argument that contains a space.
    Act: submit the job.
    Assert: the job is running on the instance of the offer with the lowest price, no request waited for a retry, the
        search filters on the GPU, the price ceiling and a direct port, the rent request has the image, the disk, the
        label, the environment with the secret value and the SSH keys, and a script that starts sshd, runs the
        quoted command one time, prints the exit code and then destroys the instance after the time to live, and
        the record and the database have the secret name but not the secret value.
    """
    record = workbench.orchestrator.submit(build_spec())

    assert record.status == Running(submitted_at=datetime(2026, 1, 1, tzinfo=UTC), instance_id=9000)
    assert workbench.orchestrator.read("train") == record
    assert workbench.vast.rented_offers == [11]
    assert workbench.vast.live_instances == {9000: "llm-lab-job-train"}
    assert workbench.sleeps == []
    search, rent = (json.loads(request.content) for request in workbench.vast.requests)
    assert search["gpu_name"] == {"eq": "RTX 4090"}
    assert search["dph_total"] == {"lte": 1.0}
    assert search["direct_port_count"] == {"gte": 1}
    environment = rent.pop("env")
    assert environment.keys() == {
        "SEED",
        "HF_TOKEN",
        "LLM_LAB_SSH_AUTHORIZED_KEY",
        "LLM_LAB_SSH_HOST_KEY",
        "-p 2222:2222",
    }
    assert (environment["SEED"], environment["HF_TOKEN"], environment["-p 2222:2222"]) == ("1", "hf-secret-value", "1")
    assert rent == {
        "client_id": "me",
        "image": "ghcr.io/example/train:1",
        "disk": 40.0,
        "label": "llm-lab-job-train",
        "runtype": "args",
        "args": [
            "sh",
            "-c",
            "if command -v sshd >/dev/null 2>&1; then\n"
            "mkdir -p -m 700 /run/sshd /root/.ssh\n"
            "mkdir -p /etc/profile.d\n"
            "printf '%s\\n' \"$LLM_LAB_SSH_AUTHORIZED_KEY\" > /run/sshd/llm-lab-authorized-keys\n"
            "(umask 077 && printf '%s' \"$LLM_LAB_SSH_HOST_KEY\" | base64 -d > /run/sshd/llm-lab-host-key)\n"
            'printf \'export PATH="%s"\\n\' "$PATH" > /etc/profile.d/llm-lab-path.sh\n'
            '"$(command -v sshd)" -p 2222 -h /run/sshd/llm-lab-host-key '
            "-o 'AuthorizedKeysFile=/run/sshd/llm-lab-authorized-keys .ssh/authorized_keys' "
            "-o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o PermitRootLogin=prohibit-password\n"
            "fi\n"
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
    }
    assert "hf-secret-value" not in record.model_dump_json()
    with workbench.engine.connect() as connection:
        stored = "".join(connection.execute(text("SELECT record FROM job_transitions")).scalars())
    assert '"secret_environment":["HF_TOKEN"]' in stored
    assert "hf-secret-value" not in stored


def test_each_database_connection_has_the_sqlite_pragmas(workbench: Workbench) -> None:
    """
    Arrange: the migrated job database of the workbench.
    Act: open two connections and read the foreign_keys, journal_mode, synchronous and busy_timeout pragmas on each.
    Assert: each connection has foreign keys on, the WAL journal mode, the NORMAL synchronous level and a busy timeout
        of 5000 milliseconds.
    """
    with workbench.engine.connect() as first, workbench.engine.connect() as second:
        for connection in (first, second):
            pragmas = {
                name: connection.execute(text(f"PRAGMA {name}")).scalar_one()
                for name in ("foreign_keys", "journal_mode", "synchronous", "busy_timeout")
            }
            assert pragmas == {"foreign_keys": 1, "journal_mode": "wal", "synchronous": 1, "busy_timeout": 5000}


def test_cli_reports_expected_errors_as_one_message(
    tmp_path: Path, workbench: Workbench, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Arrange: a manifest file with a job name that is not valid and no command, and a second database file that has no
        migrations.
    Act: submit the manifest with the CLI, then list the jobs with the CLI on the second database file.
    Assert: the submit command exits with code 1 and prints one error that names the manifest file and each field that
        is not valid, and the list command exits with code 1 and prints one error that tells the user to upgrade
        the database.
    """
    manifest = tmp_path / "job.yaml"
    manifest.write_text("name: Not/A/Name\nimage: busybox\n")
    invalid_manifest = CliRunner().invoke(app, ["jobs", "submit", str(manifest)], catch_exceptions=False)
    monkeypatch.setenv("LLM_LAB_JOB_DATABASE_PATH", str(tmp_path / "new.sqlite3"))
    discard_cached_settings()
    old_schema = CliRunner().invoke(app, ["jobs", "list"], catch_exceptions=False)

    assert invalid_manifest.exit_code == 1
    assert invalid_manifest.output.startswith(f"Error: {manifest} is not a valid job manifest:\n  name: ")
    assert "  command: Field required\n" in invalid_manifest.output
    assert (old_schema.exit_code, old_schema.output) == (
        1,
        "Error: The job database is not current. Run `llm-lab db upgrade`.\n",
    )


def test_cli_upgrade_reports_whether_it_applied_a_migration(
    tmp_path: Path, workbench: Workbench, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Arrange: a database file that has no migrations.
    Act: run the database upgrade command of the CLI two times.
    Assert: the first run prints the revision that it migrated to, and the second run prints that the database is
        already current.
    """
    monkeypatch.setenv("LLM_LAB_JOB_DATABASE_PATH", str(tmp_path / "new.sqlite3"))
    discard_cached_settings()

    first = CliRunner().invoke(app, ["db", "upgrade"], catch_exceptions=False)
    second = CliRunner().invoke(app, ["db", "upgrade"], catch_exceptions=False)

    assert first.output == "Migrated to 9395d98d9cb4.\n"
    assert second.output == "Already current.\n"


def test_migrations_create_the_schema_of_the_models(workbench: Workbench) -> None:
    """
    Arrange: a database that the migrations upgraded to the head revision.
    Act: compare the schema of the database with the metadata of the table models.
    Assert: the comparison finds no differences.
    """
    with workbench.engine.connect() as connection:
        assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []


def test_sync_leaves_a_job_running_while_its_command_has_not_reported_an_exit_code(workbench: Workbench) -> None:
    """
    Arrange: a submitted job whose instance first has the status loading and then the status running, and whose log
        has no exit code.
    Act: sync the job two times.
    Assert: each sync returns the record of the submission, the instance is not destroyed, and the database has only
        the pending and the running transitions.
    """
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
    """
    Arrange: a submitted job whose instance is running and whose log has an exit code.
    Act: sync the job.
    Assert: the status of the job is succeeded for exit code 0 and failed for a different exit code, the stored record
        is the same, the instance is destroyed, the database has the pending, the running and the final
        transitions, and a second sync returns the same record.
    """
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
    """
    Arrange: two submitted jobs whose instances are running and whose logs have the exit code 0.
    Act: sync all jobs.
    Assert: the result has the two jobs in the order of their names, each job succeeded, and no instance is live.
    """
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
    """
    Arrange: a submitted job whose instance is not on Vast.ai.
    Act: sync the job.
    Assert: the job is aborted, and the reason says that the instance is missing.
    """
    workbench.orchestrator.submit(build_spec())
    workbench.vast.live_instances.clear()

    record = workbench.orchestrator.sync("train")

    assert isinstance(record.status, Aborted)
    assert "missing" in record.status.reason


@pytest.mark.parametrize("request_logs_response", [None, httpx.Response(404)])
def test_sync_records_a_job_as_aborted_and_destroys_the_instance_when_the_container_exited_without_an_exit_code(
    workbench: Workbench, request_logs_response: httpx.Response | None
) -> None:
    """
    Arrange: a submitted job whose instance has the status exited, and whose log has no exit code or whose log request
        returns status 404.
    Act: sync the job.
    Assert: the job is aborted, the reason says that the instance exited, and the instance is destroyed.
    """
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
    """
    Arrange: a fake Vast.ai whose offer with the lowest price is not available to rent.
    Act: submit a job.
    Assert: the job is running, and the instance is from the offer with the next price.
    """
    workbench.vast.unavailable_offers = {11}

    record = workbench.orchestrator.submit(build_spec())

    assert isinstance(record.status, Running)
    assert workbench.vast.rented_offers == [12]


def test_submit_with_an_explicit_offer_rents_that_offer_without_a_search(workbench: Workbench) -> None:
    """
    Arrange: a job spec whose placement is an explicit offer.
    Act: submit the job.
    Assert: the job is running on that offer, and the first request to Vast.ai is the rent request, not a search.
    """
    record = workbench.orchestrator.submit(build_spec(ExplicitOffer(kind=PlacementKind.offer, offer_id=77)))

    assert isinstance(record.status, Running)
    assert workbench.vast.rented_offers == [77]
    assert workbench.vast.requests[0].method == "PUT"


def test_submit_records_a_job_as_aborted_when_no_offer_is_available(workbench: Workbench) -> None:
    """
    Arrange: a fake Vast.ai that has no offers.
    Act: submit a job.
    Assert: the job is aborted, and no offer is rented.
    """
    workbench.vast.offers = []

    record = workbench.orchestrator.submit(build_spec())

    assert isinstance(record.status, Aborted)
    assert workbench.vast.rented_offers == []


def test_submit_rejects_a_job_name_that_is_not_finished_and_accepts_it_after_the_job_finished(
    workbench: Workbench,
) -> None:
    """
    Arrange: a submitted job whose instance is running and whose log has the exit code 0.
    Act: submit a job with the same name, then sync the first job and submit the job with the same name again.
    Assert: the submission before the sync raises an error that says the job is not finished, and the submission after
        the sync gives a running job on a second rented instance.
    """
    workbench.vast.actual_statuses = iter(["running"])
    workbench.orchestrator.submit(build_spec())

    with pytest.raises(CommandError, match="not finished"):
        workbench.orchestrator.submit(build_spec())

    workbench.orchestrator.sync("train")
    assert isinstance(workbench.orchestrator.submit(build_spec()).status, Running)
    assert workbench.vast.rented_offers == [11, 11]


def raise_read_timeout() -> httpx.Response:
    raise httpx.ReadTimeout("the response of the rent request was lost")


def test_instance_is_destroyed_when_the_response_of_the_rent_request_is_lost(workbench: Workbench) -> None:
    """
    Arrange: a fake Vast.ai that creates the instance for a rent request and then raises a read timeout.
    Act: submit a job, then sync the job.
    Assert: the submission raises an error that says the request did not complete, the rented instance is destroyed,
        and the sync records the job as aborted.
    """
    workbench.vast.fail_before[("PUT", "asks")] = raise_read_timeout

    with pytest.raises(CommandError, match="did not complete"):
        workbench.orchestrator.submit(build_spec())

    assert workbench.vast.rented_offers == [11]
    assert workbench.vast.live_instances == {}
    record = workbench.orchestrator.sync("train")
    assert isinstance(record.status, Aborted)


def test_sync_keeps_the_job_running_when_observation_raises_an_error(workbench: Workbench) -> None:
    """
    Arrange: a submitted job, and a fake Vast.ai that returns status 500 for each request for the instance.
    Act: sync the job.
    Assert: the sync raises an error that names the status 500, the stored record is the record of the submission, and
        the instance is not destroyed.
    """
    submitted = workbench.orchestrator.submit(build_spec())
    workbench.vast.fail_before[("GET", "instances")] = lambda: httpx.Response(500)

    with pytest.raises(CommandError, match="status 500"):
        workbench.orchestrator.sync("train")

    assert workbench.orchestrator.read("train") == submitted
    assert workbench.vast.live_instances == {9000: "llm-lab-job-train"}


def test_rate_limited_request_is_sent_again_after_a_delay(workbench: Workbench) -> None:
    """
    Arrange: a submitted job, and a fake Vast.ai that returns status 429 two times for the request for the instance and
        then returns the status running.
    Act: sync the job.
    Assert: the job succeeded, and the provider waited 2 seconds and then 4 seconds before it sent the request again.
    """
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
    """
    Arrange: a fake Vast.ai that returns an error for a delete request of an instance that is not live, a submitted
        job without an artifacts directory, a second instance with the label of the job, and an instance of a
        different job.
    Act: abandon the job with download, then abandon the job a second time.
    Assert: the job is aborted, the stored record is the same, only the instance of the different job is live, no
        download occurred, and the second abandon returns the same record.
    """
    workbench.vast.missing_instance_response = missing_instance_response
    workbench.orchestrator.submit(build_spec())
    workbench.vast.live_instances |= {502: "llm-lab-job-train", 503: "llm-lab-job-other"}

    abandonment = workbench.orchestrator.abandon("train", download=True)

    assert isinstance(abandonment.record.status, Aborted)
    assert workbench.orchestrator.read("train") == abandonment.record
    assert workbench.vast.live_instances == {503: "llm-lab-job-other"}
    assert abandonment.downloaded_to is None, "Expected no download for a job without an artifacts directory."
    assert workbench.orchestrator.abandon("train", download=True).record == abandonment.record


def test_settings_read_a_keyring_entry_only_for_a_field_with_the_annotation(workbench: Workbench) -> None:
    """
    Arrange: keyring entries for the API key, which has the keyring annotation, and for the API URL, which does not.
    Act: create the settings.
    Assert: the API key has the value from the keyring, and the API URL has its default value.
    """
    keyring.set_password("llm-lab", "vast_api_url", "https://wrong.test")
    keyring.set_password("llm-lab", "VAST_API_URL", "https://wrong.test")

    settings = Settings()

    assert settings.vast_api_key == SecretStr("secret-key")
    assert settings.vast_api_url == "https://console.vast.ai/api"


def test_settings_prefer_an_environment_variable_to_a_keyring_entry(
    workbench: Workbench, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Arrange: a keyring entry and an environment variable with different values for the API key.
    Act: create the settings.
    Assert: the API key has the value from the environment variable.
    """
    monkeypatch.setenv("LLM_LAB_VAST_API_KEY", "key-from-environment")

    assert Settings().vast_api_key == SecretStr("key-from-environment")


def test_cli_rejects_a_vast_command_when_the_api_key_has_no_keyring_entry(workbench: Workbench) -> None:
    """
    Arrange: a keyring without the entry for the API key.
    Act: list the jobs with the CLI, then sync the jobs with the CLI.
    Assert: the list command exits with code 0, and the sync command exits with code 1 and prints an error that tells
        the user to set the API key.
    """
    keyring.delete_password("llm-lab", "VAST_API_KEY")
    overridden = app

    assert CliRunner().invoke(overridden, ["jobs", "list"], catch_exceptions=False).exit_code == 0
    sync = CliRunner().invoke(overridden, ["jobs", "sync"], catch_exceptions=False)
    assert sync.exit_code == 1
    assert sync.output == "Error: The Vast.ai API key is not set. Run `llm-lab secrets set VAST_API_KEY`.\n"


def test_cli_prompts_for_the_job_name_and_the_manifest_when_they_are_omitted(
    tmp_path: Path, workbench: Workbench
) -> None:
    """
    Arrange: a submitted job.
    Act: run the history command of the CLI without a job name and give the name at the prompt, then run the submit
        command without a manifest and give the path of a file that does not exist at the prompt.
    Assert: the history command prompts for the job name and prints the two transitions of the job, and the submit
        command prompts for the manifest file and exits with code 2 and an error that says the file does not exist.
    """
    workbench.orchestrator.submit(build_spec())

    history = CliRunner().invoke(app, ["jobs", "history"], input="train\n", catch_exceptions=False)
    submit = CliRunner().invoke(app, ["jobs", "submit"], input=f"{tmp_path / 'absent.yaml'}\n")

    assert "Job name: " in history.output
    assert history.output.count('"condition":') == 2
    assert "Manifest file: " in submit.output
    assert submit.exit_code == 2
    assert "does not exist" in submit.output


def test_unknown_job_name_is_rejected(workbench: Workbench) -> None:
    """
    Arrange: a job database that has no job with the name absent.
    Act: sync the job with the name absent.
    Assert: the sync raises an error that names the job.
    """
    with pytest.raises(CommandError, match="absent"):
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
    """
    Arrange: the fields of a valid job spec with one change that makes the spec not valid.
    Act: validate the changed fields as a job spec.
    Assert: the validation raises a validation error.
    """
    with pytest.raises(ValidationError):
        JobSpec.model_validate(build_spec().model_dump() | change)


@pytest.mark.parametrize(
    ("log_text", "status_exit_code"), [("llm-lab-job-exit-code=0\n", 0), ("llm-lab-job-exit-code=7\n", 1)]
)
def test_cli_submits_a_manifest_file_and_reports_the_job_from_the_database(
    tmp_path: Path, workbench: Workbench, log_text: str, status_exit_code: int
) -> None:
    """
    Arrange: a manifest file of a job that uses the secret HF_TOKEN, a keyring without that secret, and a fake Vast.ai
        whose instance is running and whose log has an exit code.
    Act: submit the manifest with the CLI, set the secret with the CLI, submit the manifest again, and then run the
        list, status, sync and history commands of the CLI.
    Assert: the first submission fails with an error that tells the user to set the secret and sends no request, the
        secrets command rejects a secret name that is not known, the secret value is not in the output and goes to
        the rent request, the status command exits with code 0 for a succeeded job and code 1 for a failed job,
        the history has the pending, the running and the final transitions, and no instance is live.
    """
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
    assert invoke("submit", str(manifest)) == (
        1,
        "Error: The secret HF_TOKEN is not set. Run `llm-lab secrets set HF_TOKEN`.\n",
    )
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
    assert json.loads(workbench.vast.requests[-1].content)["env"]["HF_TOKEN"] == "wandb-secret-value"
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


def test_offer_query_filters_on_cuda_13_unless_the_request_sets_another_floor() -> None:
    """
    Arrange: a resource request without a minimum CUDA version, and a copy of it with a minimum CUDA version of 13.2.
    Act: build the offer query of each request.
    Assert: the query of the first request filters on CUDA 13.0 or higher, and the query of the copy filters on CUDA
        13.2 or higher.
    """
    request = ResourceRequest(kind=PlacementKind.resources, gpu_name="RTX 4090", gpu_count=1, max_dollars_per_hour=1.0)

    actual = [
        build_offer_query(placement, 40, 10)["cuda_max_good"]
        for placement in (request, request.model_copy(update={"min_cuda": 13.2}))
    ]

    expected = [{"gte": 13.0}, {"gte": 13.2}]
    assert actual == expected, f"Expected the CUDA filters {expected}, got {actual}."


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        ({}, {}),
        (
            {
                "max_compute_dollars_per_hour": 0.4,
                "max_storage_dollars_per_gb_month": 0.2,
                "max_download_dollars_per_gb": 0.005,
                "max_upload_dollars_per_gb": 0.01,
            },
            {
                "dph_base": {"lte": 0.4},
                "storage_cost": {"lte": 0.2},
                "inet_down_cost": {"lte": 0.005},
                "inet_up_cost": {"lte": 0.01},
            },
        ),
        ({"countries": {"include": ["US", "CA"]}}, {"geolocation": {"in": ["US", "CA"]}}),
        ({"countries": {"exclude": ["CN"]}}, {"geolocation": {"notin": ["CN"]}}),
        ({"datacenter_only": True}, {"datacenter": {"eq": True}}),
        ({"min_duration_hours": 48}, {"duration": {"gte": 172800}}),
    ],
    ids=["no-filters", "price-parts", "include-countries", "exclude-countries", "datacenter-only", "min-duration"],
)
def test_offer_query_adds_a_clause_for_each_optional_filter_of_the_request(
    update: dict[str, object], expected: dict[str, object]
) -> None:
    """
    Arrange: a resource request without optional filters, and the same request with zero or more optional filters.
    Act: build the offer query of each request.
    Assert: the clauses that are different between the two queries are only the clauses for the optional filters that
        the second request sets.
    """
    fields = {"kind": "resources", "gpu_name": "RTX 4090", "gpu_count": 1, "max_dollars_per_hour": 1.0}
    unfiltered = build_offer_query(ResourceRequest.model_validate(fields), 40, 10)

    query = build_offer_query(ResourceRequest.model_validate(fields | update), 40, 10)

    actual = {key: value for key, value in query.items() if unfiltered.get(key) != value}
    assert actual == expected, f"Expected the added clauses {expected}, got {actual}."


@pytest.mark.parametrize(
    "countries",
    [{"include": ["US"], "exclude": ["CN"]}, {"include": ["ZZ"]}, {"include": ["USA"]}, {"exclude": []}],
    ids=["include-and-exclude", "unassigned-code", "three-letter-code", "empty-set"],
)
def test_resource_request_rejects_an_invalid_country_filter(countries: dict[str, list[str]]) -> None:
    """
    Arrange: the fields of a valid resource request and a country filter that is not valid.
    Act: validate the fields with the country filter as a resource request.
    Assert: the validation raises a validation error.
    """
    fields = {"kind": "resources", "gpu_name": "RTX 4090", "gpu_count": 1, "max_dollars_per_hour": 1.0}

    with pytest.raises(ValidationError):
        ResourceRequest.model_validate(fields | {"countries": countries})


def build_spec_with_artifacts() -> JobSpec:
    return build_spec().model_copy(update={"artifacts_directory": "/workspace/artifacts"})


def test_instance_receives_the_public_half_of_the_local_identity_and_the_private_half_of_the_pinned_host_key(
    workbench: Workbench,
) -> None:
    """
    Arrange: a job spec.
    Act: submit the job.
    Assert: the rent request has the public key of the local identity file as the authorized key, the known hosts file
        has the alias of the job and the public key of the host key that the rent request has, and the identity
        file has the mode 0o600.
    """
    workbench.orchestrator.submit(build_spec())

    environment = json.loads(workbench.vast.requests[-1].content)["env"]
    directory = workbench.access.directory / "train"
    identity = load_ssh_private_key((directory / "id_ed25519").read_bytes(), password=None)
    host_key = load_ssh_private_key(base64.b64decode(environment["LLM_LAB_SSH_HOST_KEY"]), password=None)
    actual = (
        environment["LLM_LAB_SSH_AUTHORIZED_KEY"],
        (directory / "known_hosts").read_text(),
        (directory / "id_ed25519").stat().st_mode & 0o777,
    )
    expected = (
        identity.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH).decode(),
        f"llm-lab-train {host_key.public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH).decode()}\n",
        0o600,
    )
    assert actual == expected, f"Expected the identity, the pinned host key and the file mode {expected}, got {actual}."


def test_abandon_downloads_the_artifacts_directory_and_then_destroys_the_instance(workbench: Workbench) -> None:
    """
    Arrange: a submitted job with an artifacts directory.
    Act: abandon the job with download.
    Assert: the downloader received the job name, the public SSH endpoint of the instance and the artifacts
        directory, the result has the local download directory, the job is aborted, the instance is destroyed, and
        the SSH keys of the job are removed.
    """
    workbench.orchestrator.submit(build_spec_with_artifacts())

    abandonment = workbench.orchestrator.abandon("train", download=True)

    assert workbench.downloader.downloads == [("train", SshEndpoint("203.0.113.5", 41022), "/workspace/artifacts")]
    assert abandonment.downloaded_to == workbench.downloader.directory / "train"
    assert isinstance(abandonment.record.status, Aborted)
    assert workbench.vast.live_instances == {}
    assert not (workbench.access.directory / "train").exists(), "Expected the throwaway keys to be removed."


def test_abandon_keeps_the_job_running_when_the_download_fails(workbench: Workbench) -> None:
    """
    Arrange: a submitted job with an artifacts directory, and a downloader that raises an error.
    Act: abandon the job with download.
    Assert: the abandon raises the error of the downloader, the job is running, the instance is live, and the SSH keys
        of the job are kept.
    """
    workbench.orchestrator.submit(build_spec_with_artifacts())
    workbench.downloader.failure = CommandError("rsync exited with status 12")

    with pytest.raises(CommandError, match="status 12"):
        workbench.orchestrator.abandon("train", download=True)

    assert isinstance(workbench.orchestrator.read("train").status, Running)
    assert workbench.vast.live_instances == {9000: "llm-lab-job-train"}
    assert (workbench.access.directory / "train").exists(), "Expected the keys to stay for another attempt."


def test_abandon_without_download_destroys_the_instance_at_once(workbench: Workbench) -> None:
    """
    Arrange: a submitted job with an artifacts directory.
    Act: abandon the job without download.
    Assert: the downloader received no request, the result has no download directory, and the instance is destroyed.
    """
    workbench.orchestrator.submit(build_spec_with_artifacts())

    abandonment = workbench.orchestrator.abandon("train", download=False)

    assert (workbench.downloader.downloads, abandonment.downloaded_to, workbench.vast.live_instances) == ([], None, {})


def test_throwaway_keys_are_removed_when_the_job_finishes_or_no_offer_is_available(workbench: Workbench) -> None:
    """
    Arrange: a submitted job whose instance is running and whose log has the exit code 0.
    Act: sync the job, then remove all offers and submit the job again.
    Assert: the SSH keys of the job are removed after the sync, and they are removed after the submission that has no
        offer.
    """
    workbench.orchestrator.submit(build_spec())
    workbench.vast.actual_statuses = iter(["running"])
    workbench.orchestrator.sync("train")
    finished_keys = (workbench.access.directory / "train").exists()
    workbench.vast.offers = []
    workbench.orchestrator.submit(build_spec())

    assert (finished_keys, (workbench.access.directory / "train").exists()) == (False, False)


def test_ssh_endpoint_is_rejected_when_the_instance_does_not_map_the_ssh_port(workbench: Workbench) -> None:
    """
    Arrange: a submitted job whose instance maps only a port that is not the SSH port.
    Act: find the SSH endpoint of the job.
    Assert: the call raises an error that names the SSH port 2222.
    """
    workbench.orchestrator.submit(build_spec())
    workbench.vast.ports = {"8080/tcp": [{"HostIp": "0.0.0.0", "HostPort": "40080"}]}

    with pytest.raises(CommandError, match="SSH port 2222"):
        workbench.orchestrator.find_ssh_endpoint("train")


def test_cli_opens_a_shell_with_the_throwaway_key_and_destroy_reports_the_download(
    workbench: Workbench, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Arrange: a submitted job with an artifacts directory, a replacement for os.execvp that records its arguments, and a
        CLI that uses the fake Vast.ai and the fake downloader.
    Act: run the ssh command of the CLI, then run the destroy command of the CLI.
    Assert: the ssh command exits with code 0 and runs ssh with the mapped port, the identity file, the known hosts
        file, strict host key checking, the host key alias, the root user and the public address, and the destroy
        command exits with code 1, prints the local download directory and destroys the instance.
    """
    workbench.orchestrator.submit(build_spec_with_artifacts())
    executed: list[list[str]] = []

    def record_execution(file: str, arguments: list[str]) -> None:
        executed.append([file, *arguments])

    monkeypatch.setattr(os, "execvp", record_execution)
    overridden = app.with_dependency_overrides(
        {
            get_http_client: lambda: httpx.Client(transport=httpx.MockTransport(workbench.vast.handle)),
            get_artifact_downloader: lambda: workbench.downloader,
        }
    )

    ssh = CliRunner().invoke(overridden, ["jobs", "ssh", "train"], catch_exceptions=False)
    destroy = CliRunner().invoke(overridden, ["jobs", "destroy", "train"], catch_exceptions=False)

    directory = workbench.access.directory / "train"
    assert ssh.exit_code == 0
    assert executed == [
        [
            "ssh",
            "ssh",
            "-p",
            "41022",
            "-i",
            str(directory / "id_ed25519"),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            f"UserKnownHostsFile={directory / 'known_hosts'}",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "HostKeyAlias=llm-lab-train",
            "-l",
            "root",
            "203.0.113.5",
        ]
    ]
    assert destroy.exit_code == 1, "Expected the exit code of an aborted job."
    assert f"Downloaded the artifacts of job train to {workbench.downloader.directory / 'train'}." in destroy.output
    assert workbench.vast.live_instances == {}


@pytest.mark.parametrize(("status", "expected_error"), [(0, None), (12, "rsync exited with status 12")])
def test_rsync_downloader_copies_the_remote_directory_through_the_job_ssh_command(
    tmp_path: Path, workbench: Workbench, monkeypatch: pytest.MonkeyPatch, status: int, expected_error: str | None
) -> None:
    """
    Arrange: a fake rsync program on the PATH that records its arguments and exits with status 0 or with a
        different status, and the SSH keys of a job.
    Act: download the artifacts directory of the job with the rsync downloader.
    Assert: the download returns the local download directory for status 0 and raises an error that names the status
        for a different status, and rsync received the archive and partial options, the SSH command of the job and
        the remote and local directories.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    recorded = tmp_path / "rsync-arguments"
    (tools / "rsync").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {recorded}\nexit {status}\n')
    (tools / "rsync").chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))
    workbench.access.create("train")
    downloader = RsyncDownloader(Settings(), workbench.access)
    endpoint = SshEndpoint("203.0.113.5", 41022)

    match expected_error:
        case None:
            assert downloader.download("train", endpoint, "/workspace/artifacts") == tmp_path / "downloads" / "train"
        case str():
            with pytest.raises(CommandError, match=expected_error):
                downloader.download("train", endpoint, "/workspace/artifacts")
        case _:
            assert_never(expected_error)

    *ssh, host = workbench.access.build_ssh_command("train", endpoint)
    expected = [
        "--archive",
        "--partial",
        "--rsh",
        shlex.join(ssh),
        f"{host}:/workspace/artifacts/",
        f"{tmp_path / 'downloads' / 'train'}/",
    ]
    assert recorded.read_text().splitlines() == expected


def build_spec_with_package() -> JobSpec:
    return build_spec().model_copy(update={"package": "train-job"})


def test_submit_uploads_the_code_of_the_package_and_the_instance_runs_the_command_after_the_code_arrives(
    workbench: Workbench,
) -> None:
    """
    Arrange: a job spec that names a workspace package, and an uploader that reports the instance as not reachable
        one time and then delivers the code.
    Act: submit the job.
    Assert: the job is running, the record and each stored transition have the digest of the code, the uploader
        received the code two times at the public SSH endpoint of the instance with one delay between the attempts,
        and the script of the instance waits for the marker file of the code with a time limit, changes to the code
        directory and runs the command with uv for the package.
    """
    workbench.uploader.deliveries = iter([CodeDelivery.unreachable, CodeDelivery.delivered])

    record = workbench.orchestrator.submit(build_spec_with_package())

    code = FakeCodeSource().code
    endpoint = SshEndpoint("203.0.113.5", 41022)
    assert isinstance(record.status, Running)
    assert record.code_digest == "0a1b"
    assert workbench.orchestrator.read("train") == record
    assert workbench.uploader.uploads == [("train", endpoint, code), ("train", endpoint, code)]
    assert workbench.sleeps == [10]
    rent = next(request for request in workbench.vast.requests if request.url.path.startswith("/api/v0/asks/"))
    script = json.loads(rent.content)["args"][2]
    expected = (
        "timeout 900 sh -c 'until [ -e /workspace/code/.llm-lab-code-complete ]; do sleep 1; done' "
        "&& cd /workspace/code "
        "&& uv run --locked --package train-job -- python train.py --note 'two words'\n"
    )
    assert expected in script, f"Expected the script to contain {expected!r}, got {script!r}."


def test_submit_destroys_the_instance_and_records_the_job_as_aborted_when_the_code_does_not_arrive_in_time(
    workbench: Workbench,
) -> None:
    """
    Arrange: a job spec that names a workspace package, and an uploader that always reports the instance as not
        reachable.
    Act: submit the job.
    Assert: the job is aborted with a reason that names the code, the stored record is the same, the instance is
        destroyed, the SSH keys of the job are removed, and the database has the pending, the running and the
        aborted transitions.
    """
    workbench.uploader.deliveries = iter([])

    record = workbench.orchestrator.submit(build_spec_with_package())

    assert isinstance(record.status, Aborted)
    assert "code" in record.status.reason
    assert workbench.orchestrator.read("train") == record
    assert workbench.vast.live_instances == {}
    assert not (workbench.access.directory / "train").exists(), "Expected the throwaway keys to be removed."
    assert read_conditions(workbench.engine, "train") == ["pending", "running", "aborted"]


def test_submit_destroys_the_instance_when_the_upload_of_the_code_raises_an_error(workbench: Workbench) -> None:
    """
    Arrange: a job spec that names a workspace package, and an uploader that raises an error.
    Act: submit the job.
    Assert: the submission raises the error of the uploader, the instance is destroyed, and the stored job is aborted.
    """
    workbench.uploader.failure = CommandError("rsync exited with status 23")

    with pytest.raises(CommandError, match="status 23"):
        workbench.orchestrator.submit(build_spec_with_package())

    assert workbench.vast.live_instances == {}
    assert isinstance(workbench.orchestrator.read("train").status, Aborted)


def test_submit_waits_for_the_code_upload_while_the_instance_has_no_public_ssh_port(workbench: Workbench) -> None:
    """
    Arrange: a job spec that names a workspace package, and an instance that reports no ports.
    Act: submit the job.
    Assert: the uploader received no request, and the job is aborted after the time limit for the code.
    """
    workbench.vast.ports = {}

    record = workbench.orchestrator.submit(build_spec_with_package())

    assert workbench.uploader.uploads == []
    assert isinstance(record.status, Aborted)


def write_workspace(root: Path) -> None:
    files = {
        "pyproject.toml": '[tool.uv.workspace]\nmembers = ["packages/*", "jobs/*"]\n',
        ".gitignore": "__pycache__/\n",
        "uv.lock": (
            'version = 1\n\n[manifest]\nmembers = ["cli", "shared", "train-job"]\n\n'
            '[[package]]\nname = "cli"\nsource = { editable = "packages/cli" }\n\n'
            '[[package]]\nname = "httpx"\nsource = { registry = "https://pypi.org/simple" }\n\n'
            '[[package]]\nname = "shared"\nsource = { editable = "packages/shared" }\n'
            'dependencies = [{ name = "httpx" }]\n\n'
            '[[package]]\nname = "train-job"\nsource = { editable = "jobs/train" }\n'
            'dependencies = [{ name = "shared" }]\n'
        ),
        "packages/cli/pyproject.toml": '[project]\nname = "cli"\n',
        "packages/cli/src/cli/__init__.py": "",
        "packages/shared/pyproject.toml": '[project]\nname = "shared"\n',
        "packages/shared/src/shared/__init__.py": "",
        "packages/shared/src/shared/__pycache__/cached.pyc": "",
        "jobs/train/pyproject.toml": '[project]\nname = "train-job"\n',
        "jobs/train/train.py": "print('train')\n",
        "notes.txt": "",
    }
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(content)
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "pyproject.toml", "uv.lock", "packages"], check=True)


def test_code_of_a_package_is_its_member_the_members_that_it_needs_and_the_project_files_of_the_workspace(
    tmp_path: Path,
) -> None:
    """
    Arrange: a git repository with a uv workspace of three members, where the job member depends on the shared member
        and is not yet tracked by git, the shared member has a file that git ignores, and the root has a file that
        is not part of a member.
    Act: read the code of the job member two times, with a change to a file of the job member between the reads.
    Assert: the code has the root project file, the lock file, the project file of each member and all other files of
        the job member and the shared member, it does not have the ignored file, the root file or the source of
        the member that the job does not need, and the digest is different after the change.
    """
    write_workspace(tmp_path)
    source = WorkspaceCodeSource(Settings(workspace_dir=tmp_path))

    code = source.read("train-job")
    (tmp_path / "jobs/train/train.py").write_text("print('changed')\n")
    changed = source.read("train-job")

    expected = (
        "jobs/train/pyproject.toml",
        "jobs/train/train.py",
        "packages/cli/pyproject.toml",
        "packages/shared/pyproject.toml",
        "packages/shared/src/shared/__init__.py",
        "pyproject.toml",
        "uv.lock",
    )
    assert (code.root, code.files) == (tmp_path, expected), f"Expected the files {expected}, got {code.files}."
    assert changed.files == code.files
    assert changed.digest != code.digest, "Expected the digest to follow the contents of the files."


def test_code_of_a_package_that_is_not_a_workspace_member_is_rejected(tmp_path: Path) -> None:
    """
    Arrange: a git repository with a uv workspace, and a directory that has no lock file.
    Act: read the code of a registry package of the lock file, then read the code of a member in the directory
        without a lock file.
    Assert: the first read raises an error that names the package and the members, and the second read raises an
        error that names the missing lock file.
    """
    write_workspace(tmp_path / "workspace")

    with pytest.raises(CommandError, match=r"no member with the name httpx.*'cli', 'shared', 'train-job'"):
        WorkspaceCodeSource(Settings(workspace_dir=tmp_path / "workspace")).read("httpx")
    with pytest.raises(CommandError, match=r"uv\.lock does not exist"):
        WorkspaceCodeSource(Settings(workspace_dir=tmp_path / "empty")).read("train-job")


@pytest.mark.parametrize(
    ("rsync_status", "ssh_status", "expected"),
    [
        (0, 0, CodeDelivery.delivered),
        (255, 0, CodeDelivery.unreachable),
        (0, 255, CodeDelivery.unreachable),
        (23, 0, "rsync exited with status 23"),
    ],
)
def test_rsync_uploader_copies_the_listed_files_and_then_creates_the_marker_file(
    tmp_path: Path,
    workbench: Workbench,
    monkeypatch: pytest.MonkeyPatch,
    rsync_status: int,
    ssh_status: int,
    expected: CodeDelivery | str,
) -> None:
    """
    Arrange: fake rsync and ssh programs on the PATH that record their arguments and their input and exit with the
        given statuses, and the SSH keys of a job.
    Act: upload the code of the job with the rsync uploader.
    Assert: the upload reports delivered when both programs exit with status 0, reports unreachable when one of them
        exits with status 255, and raises an error that names any other status. rsync received the names of the
        code files on its input and the SSH command of the job with batch mode and a connection time limit, and
        ssh ran only after a successful copy and created the marker file in the code directory.
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    for program, status in (("rsync", rsync_status), ("ssh", ssh_status)):
        (tools / program).write_text(
            f'#!/bin/sh\nprintf "%s\\n" "$@" > {tmp_path}/{program}-arguments\n'
            f"/bin/cat > {tmp_path}/{program}-input\nexit {status}\n"
        )
        (tools / program).chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))
    workbench.access.create("train")
    uploader = RsyncUploader(Settings(), workbench.access)
    endpoint = SshEndpoint("203.0.113.5", 41022)
    code = JobCode(root=tmp_path / "workspace", files=("jobs/train/train.py", "uv.lock"), digest="0a1b")

    match expected:
        case CodeDelivery():
            assert uploader.upload("train", endpoint, code) == expected
        case str():
            with pytest.raises(CommandError, match=expected):
                uploader.upload("train", endpoint, code)
        case _:
            assert_never(expected)

    *ssh, host = workbench.access.build_ssh_command("train", endpoint)
    ssh += ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    assert (tmp_path / "rsync-arguments").read_text().splitlines() == [
        "--archive",
        "--compress",
        "--from0",
        "--files-from=-",
        "--rsh",
        shlex.join(ssh),
        f"{tmp_path / 'workspace'}/",
        f"{host}:/workspace/code/",
    ]
    assert (tmp_path / "rsync-input").read_text() == "jobs/train/train.py\0uv.lock"
    marker_commands = (
        [(tmp_path / "ssh-arguments").read_text().splitlines()[-1]] if (tmp_path / "ssh-arguments").exists() else []
    )
    expected_commands = ["touch /workspace/code/.llm-lab-code-complete"] if rsync_status == 0 else []
    assert marker_commands == expected_commands
