from datetime import UTC
from pathlib import Path
from typing import Annotated, assert_never

import typer
import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.cli.jobs.deps import DatabaseEngineDep, JobOrchestratorDep
from llm_lab.jobs.protocol import Aborted, Failed, JobRecord, JobSpec, Pending, Running, Succeeded
from llm_lab.jobs.tables import JobTransitionRow, select_latest_transitions

app = DependentTyper(name="jobs", no_args_is_help=True)


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
def list_jobs(engine: DatabaseEngineDep) -> None:
    with Session(engine) as session:
        for row in session.scalars(select_latest_transitions()):
            typer.echo(f"{row.name}\t{row.condition}")


@app.command("history")
def show_job_history(name: str, engine: DatabaseEngineDep) -> None:
    with Session(engine) as session:
        rows = session.scalars(
            select(JobTransitionRow).where(JobTransitionRow.name == name).order_by(JobTransitionRow.sequence)
        )
        for row in rows:
            status = JobRecord.model_validate_json(row.record).status
            typer.echo(f"{row.recorded_at.replace(tzinfo=UTC).isoformat()}\t{status.model_dump_json()}")


@app.command("destroy")
def destroy_job(name: str, orchestrator: JobOrchestratorDep) -> None:
    report(orchestrator.abandon(name))
