from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import typer

from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.settings import SettingsDep
from llm_lab.studio.client import Progress, RunningJob, StudioClientDep
from llm_lab.studio.recipe import JobRequest, RecipeFile, RunRequest
from llm_lab.studio.ssh import RunDirectorySourceDep

app = DependentTyper(name="datasets", no_args_is_help=True)


def describe_running_job(job: RunningJob) -> str:
    match job.progress:
        case Progress(done=int() as done, total=int() as total):
            counts = f"{done}/{total}"
        case _:
            counts = None
    return " ".join(part for part in (job.status, job.stage, job.current_column, counts) if part is not None)


@app.command("create")
def create_dataset(
    recipe_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    name: Annotated[str, typer.Option()],
    rows: Annotated[int, typer.Option(min=1)],
    settings: SettingsDep,
    studio: StudioClientDep,
    run_directories: RunDirectorySourceDep,
) -> None:
    recipe_file = RecipeFile.model_validate_json(recipe_path.read_bytes())
    request = JobRequest(recipe=recipe_file.recipe, run=RunRequest(rows=rows, run_name=name))
    dataset_dir = settings.datasets_dir / request.run.run_name
    if dataset_dir.exists():
        raise FileExistsError(f"dataset {request.run.run_name!r} already exists at {dataset_dir}")
    completed = studio.run_job(request, lambda job: typer.echo(describe_running_job(job)))
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=settings.datasets_dir, prefix=f".{request.run.run_name}-") as partial_dir:
        fetched_dir = Path(partial_dir) / request.run.run_name
        run_directories.fetch_directory(completed.artifact_path, fetched_dir)
        fetched_dir.rename(dataset_dir)
    typer.echo(dataset_dir)


@app.command("list")
def list_datasets(settings: SettingsDep) -> None:
    pass


@app.command("delete")
def delete_dataset(settings: SettingsDep) -> None:
    pass
