from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import typer

from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.settings import SettingsDep
from llm_lab.studio.client import StudioClientDep
from llm_lab.studio.recipe import JobRequest, RecipeFile, RunRequest
from llm_lab.studio.ssh import RunDirectorySourceDep

app = DependentTyper(name="datasets", no_args_is_help=True)


@app.command("create")
def create_dataset(
    recipe_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="JSON file from Studio's Copy recipe button."),
    ],
    name: Annotated[str, typer.Option(help="Name of the dataset, in lowercase letters, digits and hyphens.")],
    rows: Annotated[int, typer.Option(min=1, help="Number of rows to generate.")],
    settings: SettingsDep,
    studio: StudioClientDep,
    run_directories: RunDirectorySourceDep,
) -> None:
    """Run a recipe on Studio and store its run directory as a dataset."""
    recipe_file = RecipeFile.model_validate_json(recipe_path.read_bytes())
    request = JobRequest(recipe=recipe_file.recipe, run=RunRequest(rows=rows, run_name=name))
    dataset_dir = settings.datasets_dir / request.run.run_name
    if dataset_dir.exists():
        raise FileExistsError(f"dataset {request.run.run_name!r} already exists at {dataset_dir}")
    completed = studio.run_job(
        request, lambda job: typer.echo(" ".join(filter(None, (job.status, job.stage, job.current_column))))
    )
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
