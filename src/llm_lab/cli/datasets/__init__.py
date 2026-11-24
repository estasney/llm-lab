import json
from pathlib import Path
from typing import Annotated

import typer

from llm_lab.cli.datasets.deps import DatasetDesignerDep, DatasetHubDep, ModelServerDep
from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.cli.jobs import report
from llm_lab.cli.jobs.deps import JobOrchestratorDep
from llm_lab.cli.yaml_files import reading_yaml_file
from llm_lab.datasets.designer import DatasetDesigner
from llm_lab.datasets.hub import DatasetHub, HubDataset, parse_hub_dataset
from llm_lab.datasets.job import build_dataset_job
from llm_lab.datasets.recipe import Recipe, RecipeFile, parse_recipe
from llm_lab.errors import CommandError
from llm_lab.settings import SettingsDep

app = DependentTyper(name="datasets", no_args_is_help=True, help="Create datasets from recipes.")

type RecipeArgument = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Recipe file.")]
type DatasetOption = Annotated[
    HubDataset,
    typer.Option(parser=parse_hub_dataset, metavar="NAMESPACE/NAME", help="Hub repository of the dataset."),
]
type RowsOption = Annotated[int, typer.Option(min=1, help="Number of rows to generate.")]


def load_recipe(path: Path) -> Recipe:
    with reading_yaml_file(path, "recipe"):
        return parse_recipe(path.read_text())


def check_new_dataset(recipe: Recipe, dataset: HubDataset, designer: DatasetDesigner, hub: DatasetHub) -> None:
    designer.validate(recipe)
    if hub.exists(dataset):
        raise CommandError(f"The dataset {dataset.repo_id} already exists on the Hub.")


@app.command("submit", help="Start a Vast.ai job that creates the dataset of a recipe.")
def submit_dataset(
    recipe_path: RecipeArgument,
    dataset: DatasetOption,
    rows: RowsOption,
    settings: SettingsDep,
    designer: DatasetDesignerDep,
    hub: DatasetHubDep,
    orchestrator: JobOrchestratorDep,
) -> None:
    if settings.datasets_image is None:
        raise CommandError("The datasets image is not set. Set the environment variable LLM_LAB_DATASETS_IMAGE.")
    recipe = load_recipe(recipe_path)
    check_new_dataset(recipe, dataset, designer, hub)
    report(orchestrator.submit(build_dataset_job(recipe, dataset, rows, settings.datasets_image)))


@app.command("create", help="Serve the model of a recipe with vLLM, generate the dataset and upload it to the Hub.")
def create_dataset(
    recipe_path: RecipeArgument,
    dataset: DatasetOption,
    rows: RowsOption,
    server: ModelServerDep,
    designer: DatasetDesignerDep,
    hub: DatasetHubDep,
) -> None:
    recipe = load_recipe(recipe_path)
    check_new_dataset(recipe, dataset, designer, hub)
    with server.serve(recipe):
        directory = designer.create(recipe, dataset.name, rows)
    typer.echo(hub.publish(dataset, directory, recipe))


@app.command("schema", help="Print the JSON schema of recipe files.")
def print_recipe_schema() -> None:
    typer.echo(json.dumps(RecipeFile.model_json_schema(), indent=2))


@app.command("list")
def list_datasets(settings: SettingsDep) -> None:
    pass


@app.command("delete")
def delete_dataset(settings: SettingsDep) -> None:
    pass
