import subprocess
from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from llm_lab.jobs.protocol import JobRecord, JobSpec, Running, conclude
from llm_lab.settings import Settings


class ContainerRuntime(Protocol):
    def run_to_exit(self, spec: JobSpec) -> int: ...


class JobRecordStore(Protocol):
    def write(self, record: JobRecord) -> None: ...

    def read(self, name: str) -> JobRecord: ...


class DockerRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run_to_exit(self, spec: JobSpec) -> int:
        environment_flags = [flag for key, value in spec.environment.items() for flag in ("--env", f"{key}={value}")]
        completed = subprocess.run(
            [self.settings.container_executable, "run", "--rm", *environment_flags, spec.image, *spec.command],
            check=False,
        )
        return completed.returncode


class FileJobRecordStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def write(self, record: JobRecord) -> None:
        self.settings.job_records_directory.mkdir(parents=True, exist_ok=True)
        (self.settings.job_records_directory / f"{record.spec.name}.json").write_text(record.model_dump_json(indent=2))

    def read(self, name: str) -> JobRecord:
        return JobRecord.model_validate_json((self.settings.job_records_directory / f"{name}.json").read_text())


class JobOrchestrator:
    def __init__(self, runtime: ContainerRuntime, store: JobRecordStore, read_clock: Callable[[], datetime]) -> None:
        self.runtime = runtime
        self.store = store
        self.read_clock = read_clock

    def execute(self, spec: JobSpec) -> JobRecord:
        running = Running(started_at=self.read_clock())
        self.store.write(JobRecord(spec=spec, status=running))
        exit_code = self.runtime.run_to_exit(spec)
        record = JobRecord(spec=spec, status=conclude(running, self.read_clock(), exit_code))
        self.store.write(record)
        return record
