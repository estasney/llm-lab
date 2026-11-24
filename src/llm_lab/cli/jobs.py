from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, assert_never

import typer
import yaml

from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.dependencies import Depends
from llm_lab.jobs.orchestrator import DockerRuntime, FileJobRecordStore, JobOrchestrator, JobRecordStore
from llm_lab.jobs.protocol import Failed, JobRecord, JobSpec, Running, Succeeded
from llm_lab.settings import SettingsDep

app = DependentTyper(name="jobs", no_args_is_help=True)


def get_job_record_store(settings: SettingsDep) -> JobRecordStore:
    return FileJobRecordStore(settings)


type JobRecordStoreDep = Annotated[JobRecordStore, Depends(get_job_record_store)]


def get_job_orchestrator(settings: SettingsDep, store: JobRecordStoreDep) -> JobOrchestrator:
    return JobOrchestrator(DockerRuntime(settings), store, lambda: datetime.now(UTC))


type JobOrchestratorDep = Annotated[JobOrchestrator, Depends(get_job_orchestrator)]


def report(record: JobRecord) -> None:
    typer.echo(record.model_dump_json(indent=2))
    match record.status:
        case Running() | Succeeded():
            pass
        case Failed():
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
