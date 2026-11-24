from datetime import UTC
from pathlib import Path
from typing import Annotated, assert_never

import typer
import yaml
from sqlalchemy import select

from llm_lab.cli.db.deps import DatabaseSessionDep
from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.cli.jobs.deps import JobOrchestratorDep
from llm_lab.cli.yaml_files import reading_yaml_file
from llm_lab.jobs.protocol import Aborted, Failed, JobRecord, JobSpec, Pending, Running, Succeeded
from llm_lab.jobs.tables import JobTransitionRow, select_latest_transitions

app = DependentTyper(name="jobs", no_args_is_help=True, help="Run jobs on Vast.ai.")


def prompt_for_manifest() -> Path:
    return typer.prompt("Manifest file", type=Path)


def prompt_for_job_name() -> str:
    return typer.prompt("Job name", type=str)


type ManifestArgument = Annotated[
    Path,
    typer.Argument(
        exists=True,
        dir_okay=False,
        help="Job manifest file.",
        default_factory=prompt_for_manifest,
    ),
]
type JobNameArgument = Annotated[
    str,
    typer.Argument(
        help="Job name.",
        default_factory=prompt_for_job_name,
    ),
]


def load_job_spec(manifest: Path) -> JobSpec:
    with reading_yaml_file(manifest, "job manifest"):
        return JobSpec.model_validate(yaml.safe_load(manifest.read_text()))


def report(record: JobRecord) -> None:
    typer.echo(record.model_dump_json(indent=2))
    match record.status:
        case Pending() | Running() | Succeeded():
            pass
        case Failed() | Aborted():
            raise typer.Exit(1)
        case _:
            assert_never(record.status)


@app.command("submit", help="Start a job.")
def submit_job(manifest: ManifestArgument, orchestrator: JobOrchestratorDep) -> None:
    report(orchestrator.submit(load_job_spec(manifest)))


@app.command("status", help="Refresh and show a job.")
def show_job_status(name: JobNameArgument, orchestrator: JobOrchestratorDep) -> None:
    report(orchestrator.sync(name))


@app.command("sync", help="Refresh all unfinished jobs.")
def sync_jobs(orchestrator: JobOrchestratorDep) -> None:
    for record in orchestrator.sync_all():
        typer.echo(f"{record.spec.name}\t{record.status.condition}")


@app.command("list", help="List jobs.")
def list_jobs(session: DatabaseSessionDep) -> None:
    for row in session.scalars(select_latest_transitions()):
        typer.echo(f"{row.name}\t{row.condition}")


@app.command("history", help="Show the state history of a job.")
def show_job_history(name: JobNameArgument, session: DatabaseSessionDep) -> None:
    rows = session.scalars(
        select(JobTransitionRow).where(JobTransitionRow.name == name).order_by(JobTransitionRow.sequence)
    )
    for row in rows:
        status = JobRecord.model_validate_json(row.record).status
        typer.echo(f"{row.recorded_at.replace(tzinfo=UTC).isoformat()}\t{status.model_dump_json()}")


@app.command("destroy", help="Destroy the instances of a job.")
def destroy_job(name: JobNameArgument, orchestrator: JobOrchestratorDep) -> None:
    report(orchestrator.abandon(name))
