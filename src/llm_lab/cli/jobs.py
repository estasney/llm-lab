import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, assert_never

import httpx
import typer
import yaml

from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.cli.secrets import SecretSourceDep
from llm_lab.dependencies import Depends
from llm_lab.jobs.orchestrator import InstanceProvider, JobOrchestrator
from llm_lab.jobs.protocol import Aborted, Failed, JobRecord, JobSpec, Pending, Running, Succeeded
from llm_lab.jobs.store import SqliteJobRecordStore
from llm_lab.jobs.vast import VastInstanceProvider, VastSettings
from llm_lab.settings import SettingsDep

app = DependentTyper(name="jobs", no_args_is_help=True)


def get_job_record_store(settings: SettingsDep) -> Iterator[SqliteJobRecordStore]:
    settings.job_database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(settings.job_database_path)) as connection:
        yield SqliteJobRecordStore(connection)


type JobRecordStoreDep = Annotated[SqliteJobRecordStore, Depends(get_job_record_store)]


def get_http_client() -> Iterator[httpx.Client]:
    with httpx.Client(timeout=60) as http:
        yield http


type HttpClientDep = Annotated[httpx.Client, Depends(get_http_client)]


def get_instance_provider(secret_source: SecretSourceDep, http: HttpClientDep) -> InstanceProvider:
    return VastInstanceProvider(VastSettings(api_key=secret_source.read("VAST_API_KEY")), http, time.sleep)


type InstanceProviderDep = Annotated[InstanceProvider, Depends(get_instance_provider)]


def get_job_orchestrator(
    provider: InstanceProviderDep, store: JobRecordStoreDep, secret_source: SecretSourceDep
) -> JobOrchestrator:
    return JobOrchestrator(provider, store, secret_source, lambda: datetime.now(UTC))


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


@app.command("submit")
def submit_job(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)], orchestrator: JobOrchestratorDep
) -> None:
    report(orchestrator.submit(JobSpec.model_validate(yaml.safe_load(manifest.read_text()))))


@app.command("status")
def show_job_status(name: str, orchestrator: JobOrchestratorDep) -> None:
    report(orchestrator.sync(name))


@app.command("sync")
def sync_jobs(orchestrator: JobOrchestratorDep) -> None:
    for record in orchestrator.sync_all():
        typer.echo(f"{record.spec.name}\t{record.status.condition}")


@app.command("list")
def list_jobs(store: JobRecordStoreDep) -> None:
    for record in store.list_latest():
        typer.echo(f"{record.spec.name}\t{record.status.condition}")


@app.command("history")
def show_job_history(name: str, store: JobRecordStoreDep) -> None:
    for transition in store.read_history(name):
        typer.echo(f"{transition.recorded_at.isoformat()}\t{transition.record.status.model_dump_json()}")


@app.command("destroy")
def destroy_job(name: str, orchestrator: JobOrchestratorDep) -> None:
    report(orchestrator.abandon(name))
