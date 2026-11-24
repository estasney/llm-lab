from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from llm_lab.cli import app
from llm_lab.cli.jobs import get_job_orchestrator, get_job_record_store
from llm_lab.jobs.orchestrator import FileJobRecordStore, JobOrchestrator
from llm_lab.jobs.protocol import Failed, JobRecord, JobSpec, Running, Succeeded
from llm_lab.settings import Settings


@dataclass
class ScriptedRuntime:
    exit_code: int
    store: FileJobRecordStore
    records_seen_during_run: list[JobRecord] = field(default_factory=list)

    def run_to_exit(self, spec: JobSpec) -> int:
        self.records_seen_during_run.append(self.store.read(spec.name))
        return self.exit_code


@dataclass
class SteppingClock:
    now: datetime = datetime(2026, 1, 1, tzinfo=UTC)

    def read(self) -> datetime:
        current = self.now
        self.now += timedelta(seconds=5)
        return current


@pytest.fixture
def store(tmp_path: Path) -> FileJobRecordStore:
    return FileJobRecordStore(Settings(job_records_directory=tmp_path / "records"))


def test_job_that_exits_with_zero_is_recorded_as_succeeded(store: FileJobRecordStore) -> None:
    spec = JobSpec(name="train", image="busybox", command=("true",))
    runtime = ScriptedRuntime(0, store)

    record = JobOrchestrator(runtime, store, SteppingClock().read).execute(spec)

    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    assert runtime.records_seen_during_run == [JobRecord(spec=spec, status=Running(started_at=started_at))]
    assert record.status == Succeeded(started_at=started_at, finished_at=started_at + timedelta(seconds=5))
    assert store.read("train") == record


def test_job_that_exits_with_nonzero_is_recorded_as_failed(store: FileJobRecordStore) -> None:
    spec = JobSpec(name="train", image="busybox")

    record = JobOrchestrator(ScriptedRuntime(3, store), store, SteppingClock().read).execute(spec)

    assert isinstance(record.status, Failed)
    assert record.status.exit_code == 3
    assert store.read("train") == record


@pytest.mark.parametrize(
    "manifest",
    [
        {"name": "Not/A/Name", "image": "busybox"},
        {"name": "train"},
        {"name": "train", "image": "busybox", "retries": 2},
    ],
)
def test_invalid_manifest_is_rejected(manifest: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        JobSpec.model_validate(manifest)


@pytest.mark.parametrize(("container_exit_code", "cli_exit_code"), [(0, 0), (7, 1)])
def test_cli_runs_a_manifest_file_and_reports_the_record(
    tmp_path: Path, store: FileJobRecordStore, container_exit_code: int, cli_exit_code: int
) -> None:
    manifest = tmp_path / "job.yaml"
    manifest.write_text("name: train\nimage: busybox\ncommand: [echo, hello]\nenvironment:\n  SEED: '1'\n")
    overridden = app.with_dependency_overrides(
        {
            get_job_record_store: lambda: store,
            get_job_orchestrator: lambda: JobOrchestrator(
                ScriptedRuntime(container_exit_code, store), store, SteppingClock().read
            ),
        }
    )

    run_result = CliRunner().invoke(overridden, ["jobs", "run", str(manifest)], catch_exceptions=False)
    status_result = CliRunner().invoke(overridden, ["jobs", "status", "train"], catch_exceptions=False)

    assert run_result.exit_code == cli_exit_code
    assert JobRecord.model_validate_json(run_result.output) == store.read("train")
    assert status_result.output == run_result.output
