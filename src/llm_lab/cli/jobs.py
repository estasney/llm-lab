import os
import signal
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Annotated, assert_never

import httpx
import typer
import yaml
from pydantic import SecretStr

from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.dependencies import Depends
from llm_lab.jobs.orchestrator import FileJobRecordStore, InstanceProvider, JobOrchestrator, JobRecordStore
from llm_lab.jobs.protocol import Aborted, Failed, JobRecord, JobSpec, Pending, Running, Succeeded
from llm_lab.jobs.vast import VastInstanceProvider, VastSettings
from llm_lab.settings import SettingsDep

app = DependentTyper(name="jobs", no_args_is_help=True)


def get_job_record_store(settings: SettingsDep) -> JobRecordStore:
    return FileJobRecordStore(settings)


type JobRecordStoreDep = Annotated[JobRecordStore, Depends(get_job_record_store)]


def get_instance_provider() -> Iterator[InstanceProvider]:
    with httpx.Client(timeout=60) as http:
        yield VastInstanceProvider(VastSettings(api_key=SecretStr(os.environ["VAST_API_KEY"])), http, time.sleep)


type InstanceProviderDep = Annotated[InstanceProvider, Depends(get_instance_provider)]


def exit_on_termination(signal_number: int, frame: FrameType | None) -> None:
    raise SystemExit(128 + signal_number)


def get_job_orchestrator(
    settings: SettingsDep, provider: InstanceProviderDep, store: JobRecordStoreDep
) -> Iterator[JobOrchestrator]:
    previous_handler = signal.signal(signal.SIGTERM, exit_on_termination)
    try:
        yield JobOrchestrator(settings, provider, store, lambda: datetime.now(UTC), time.sleep)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


type JobOrchestratorDep = Annotated[JobOrchestrator, Depends(get_job_orchestrator)]


def report(record: JobRecord) -> None:
    typer.echo(record.model_dump_json(indent=2))
    match record.status:
        case Pending() | Running() | Succeeded():
            pass
        case Failed() | Aborted():
            raise typer.Exit(1)
        case _:
            assert_never(record.status)


@app.command("run")
def run_job(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)], orchestrator: JobOrchestratorDep
) -> None:
    report(orchestrator.execute(JobSpec.model_validate(yaml.safe_load(manifest.read_text()))))


@app.command("status")
def show_job_status(name: str, store: JobRecordStoreDep) -> None:
    report(store.read(name))


@app.command("destroy")
def destroy_job(name: str, orchestrator: JobOrchestratorDep) -> None:
    report(orchestrator.abandon(name))
