import json
from pathlib import Path
from typing import Annotated, assert_never

import typer
from pydantic.json_schema import GenerateJsonSchema

from llm_lab.cli.datasets.deps import DatasetDesignerDep, DatasetHubDep, ModelServerDep, SeedStoreDep
from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.cli.jobs import report
from llm_lab.cli.jobs.deps import JobOrchestratorDep
from llm_lab.cli.yaml_files import reading_yaml_file
from llm_lab.datasets.designer import DatasetDesigner
from llm_lab.datasets.hub import (
    DatasetHub,
    EmptyRepository,
    HubDataset,
    HubRepository,
    StoredDataset,
    parse_hub_dataset,
)
from llm_lab.datasets.job import build_dataset_job
from llm_lab.datasets.recipe import Recipe, RecipeFile, parse_recipe
from llm_lab.datasets.seeds import SeedStore, resolve_recipe
from llm_lab.errors import CommandError
from llm_lab.settings import SettingsDep

app = DependentTyper(name="datasets", no_args_is_help=True, help="Create datasets from recipes.")

type RecipeArgument = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Recipe file.")]
type DatasetOption = Annotated[
    HubDataset,
    typer.Option(parser=parse_hub_dataset, metavar="NAMESPACE/NAME", help="Hub repository of the dataset."),
]
type RowsOption = Annotated[int, typer.Option(min=1, help="Number of rows to generate.")]


def load_recipe(path: Path, dataset: HubDataset, seeds: SeedStore) -> Recipe:
    with reading_yaml_file(path, "recipe"):
        return resolve_recipe(path.read_text(), path.parent, dataset, seeds)


def check_dataset(recipe: Recipe, dataset: HubDataset, designer: DatasetDesigner, hub: DatasetHub) -> HubRepository:
    designer.validate(recipe)
    repository = hub.read(dataset)
    match repository:
        case StoredDataset(recipe_source=source):
            with reading_yaml_file(f"{dataset.repo_id}/recipe.yaml", "recipe"):
                stored = parse_recipe(source)
            if not recipe.generates_rows_like(stored):
                raise CommandError(
                    f"The recipe differs from the recipe of the dataset {dataset.repo_id}. "
                    "Rows of one dataset come from one recipe."
                )
        case EmptyRepository() | None:
            pass
        case _:
            assert_never(repository)
    return repository


@app.command("submit", help="Start a Vast.ai job that creates the dataset of a recipe.")
def submit_dataset(
    recipe_path: RecipeArgument,
    dataset: DatasetOption,
    rows: RowsOption,
    settings: SettingsDep,
    designer: DatasetDesignerDep,
    hub: DatasetHubDep,
    seeds: SeedStoreDep,
    orchestrator: JobOrchestratorDep,
) -> None:
    if settings.datasets_image is None:
        raise CommandError("The datasets image is not set. Set the environment variable LLM_LAB_DATASETS_IMAGE.")
    recipe = load_recipe(recipe_path, dataset, seeds)
    check_dataset(recipe, dataset, designer, hub)
    report(orchestrator.submit(build_dataset_job(recipe, dataset, rows, settings.datasets_image)))


@app.command("create", help="Serve the model of a recipe with vLLM, generate the dataset and upload it to the Hub.")
def create_dataset(
    recipe_path: RecipeArgument,
    dataset: DatasetOption,
    rows: RowsOption,
    server: ModelServerDep,
    designer: DatasetDesignerDep,
    hub: DatasetHubDep,
    seeds: SeedStoreDep,
) -> None:
    recipe = load_recipe(recipe_path, dataset, seeds)
    repository = check_dataset(recipe, dataset, designer, hub)
    with server.serve(recipe):
        directory = designer.create(recipe, dataset.name, rows)
    typer.echo(hub.publish(dataset, repository, directory, recipe))


@app.command("schema", help="Print the JSON schema of recipe files.")
def print_recipe_schema() -> None:
    schema = {"$schema": GenerateJsonSchema.schema_dialect, **RecipeFile.model_json_schema()}
    typer.echo(json.dumps(schema, indent=2))


@app.command("list")
def list_datasets(settings: SettingsDep) -> None:
    pass


@app.command("delete")
def delete_dataset(settings: SettingsDep) -> None:
    pass
