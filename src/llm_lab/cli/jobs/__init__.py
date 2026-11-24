from datetime import UTC
from pathlib import Path
from typing import Annotated, assert_never

import typer
import yaml
from sqlalchemy import select

from llm_lab.cli.db.deps import DatabaseSessionDep
from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.cli.jobs.deps import JobOrchestratorDep
from llm_lab.jobs.protocol import Aborted, Failed, JobRecord, JobSpec, Pending, Running, Succeeded
from llm_lab.jobs.tables import JobTransitionRow, select_latest_transitions

app = DependentTyper(name="jobs", no_args_is_help=True, help="Run container jobs on Vast.ai instances.")


def prompt_for_manifest() -> Path:
    return typer.prompt("Manifest file", type=Path)


def prompt_for_job_name() -> str:
    return typer.prompt("Job name", type=str)


type ManifestArgument = Annotated[
    Path,
    typer.Argument(
        exists=True,
        dir_okay=False,
        help="YAML file that defines the job. A prompt asks for it when it is omitted.",
        default_factory=prompt_for_manifest,
    ),
]
type JobNameArgument = Annotated[
    str,
    typer.Argument(
        help="Name of the job, as its manifest gives it. A prompt asks for it when it is omitted.",
        default_factory=prompt_for_job_name,
    ),
]


def report(record: JobRecord) -> None:
    typer.echo(record.model_dump_json(indent=2))
    match record.status:
        case Pending() | Running() | Succeeded():
            pass
        case Failed() | Aborted():
            raise typer.Exit(1)
        case _:
            assert_never(record.status)


@app.command("submit", help="Rent a Vast.ai instance for the job in a manifest, record it, and exit.")
def submit_job(manifest: ManifestArgument, orchestrator: JobOrchestratorDep) -> None:
    report(orchestrator.submit(JobSpec.model_validate(yaml.safe_load(manifest.read_text()))))


@app.command("status", help="Read the instance of one job, and finish the job when its command has ended.")
def show_job_status(name: JobNameArgument, orchestrator: JobOrchestratorDep) -> None:
    report(orchestrator.sync(name))


@app.command("sync", help="Read the instance of each job that is not finished, and finish those that have ended.")
def sync_jobs(orchestrator: JobOrchestratorDep) -> None:
    for record in orchestrator.sync_all():
        typer.echo(f"{record.spec.name}\t{record.status.condition}")


@app.command("list", help="Show the recorded state of each job, without a request to Vast.ai.")
def list_jobs(session: DatabaseSessionDep) -> None:
    for row in session.scalars(select_latest_transitions()):
        typer.echo(f"{row.name}\t{row.condition}")


@app.command("history", help="Show each recorded state transition of one job, without a request to Vast.ai.")
def show_job_history(name: JobNameArgument, session: DatabaseSessionDep) -> None:
    rows = session.scalars(
        select(JobTransitionRow).where(JobTransitionRow.name == name).order_by(JobTransitionRow.sequence)
    )
    for row in rows:
        status = JobRecord.model_validate_json(row.record).status
        typer.echo(f"{row.recorded_at.replace(tzinfo=UTC).isoformat()}\t{status.model_dump_json()}")


@app.command("destroy", help="Destroy each Vast.ai instance of one job and record the job as aborted.")
def destroy_job(name: JobNameArgument, orchestrator: JobOrchestratorDep) -> None:
    report(orchestrator.abandon(name))
