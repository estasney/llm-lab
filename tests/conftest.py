from dataclasses import dataclass, field
from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

from llm_lab.datasets.hub import HubDataset, HubRepository
from llm_lab.datasets.job import JobRecipe, prepare_job_recipe
from llm_lab.datasets.recipe import Recipe, parse_recipe


def build_model_config(alias: str, model: str = "Qwen/Qwen3-8B", provider: str = "vllm") -> dict[str, str]:
    return {"alias": alias, "model": model, "provider": provider}


def build_recipe_source(
    model_configs: list[dict[str, str]], *, prompt: str = "Write one sentence about {{ topic }}.", **sections: object
) -> str:
    return yaml.safe_dump(
        {
            "job": {
                "disk_gb": 120,
                "ttl_seconds_after_finished": 600,
                "placement": {
                    "kind": "resources",
                    "gpu_name": "RTX 6000Ada",
                    "gpu_count": 2,
                    "max_dollars_per_hour": 2.5,
                },
            },
            "vllm_args": ["--tensor-parallel-size", "2"],
            "data_designer": {
                "model_configs": model_configs,
                "columns": [
                    {
                        "name": "topic",
                        "column_type": "sampler",
                        "sampler_type": "category",
                        "params": {"values": ["tides", "volcanoes"]},
                    },
                    {"name": "sentence", "column_type": "llm-text", "model_alias": "writer", "prompt": prompt},
                ],
            },
            **sections,
        }
    )


@dataclass
class Journal:
    events: list[str] = field(default_factory=list)
    opened: int = 0

    def open_resource(self, name: str) -> int:
        self.opened += 1
        self.events.append(f"open {name} {self.opened}")
        return self.opened


@dataclass
class FakeDesigner:
    journal: Journal
    directory: Path

    def validate(self, recipe: Recipe) -> None:
        self.journal.events.append(f"validate {recipe.served_model}")

    def create(self, recipe: Recipe, name: str, rows: int) -> Path:
        self.journal.events.append(f"create {name} {rows}")
        return self.directory


@dataclass
class FakeHub:
    journal: Journal
    repositories: dict[str, HubRepository] = field(default_factory=dict[str, HubRepository])
    published: list[HubRepository] = field(default_factory=list[HubRepository])

    def read(self, dataset: HubDataset) -> HubRepository:
        self.journal.events.append(f"read {dataset.repo_id}")
        return self.repositories.get(dataset.repo_id)

    def publish(self, dataset: HubDataset, repository: HubRepository, directory: Path, recipe: Recipe) -> str:
        self.journal.events.append(f"publish {dataset.repo_id} {directory.name}")
        self.published.append(repository)
        return f"https://huggingface.co/datasets/{dataset.repo_id}"


@dataclass
class FakeSeedStore:
    journal: Journal
    patterns: list[Path] = field(default_factory=list[Path])

    def store(self, pattern: Path, dataset: HubDataset) -> str:
        self.journal.events.append(f"store seeds of {dataset.repo_id}")
        self.patterns.append(pattern)
        return f"buckets/{dataset.namespace}/seeds/0a1b/{pattern.name}"


def run_cli(app: typer.Typer, arguments: list[str]) -> str:
    return CliRunner().invoke(app, arguments, catch_exceptions=False).output


def invoke_cli(app: typer.Typer, arguments: list[str]) -> tuple[int, str]:
    result = CliRunner().invoke(app, arguments, catch_exceptions=False)
    return result.exit_code, result.output


def follow_context_chain(error: BaseException) -> list[BaseException]:
    chain = [error]
    while (context := chain[-1].__context__) is not None and context not in chain:
        chain.append(context)
    return chain


@pytest.fixture
def journal() -> Journal:
    return Journal()


@pytest.fixture
def recipe_source() -> str:
    return build_recipe_source([build_model_config("writer")])


@pytest.fixture
def recipe(recipe_source: str) -> Recipe:
    return parse_recipe(recipe_source)


@pytest.fixture
def job_recipe(recipe: Recipe) -> JobRecipe:
    return prepare_job_recipe(recipe, HubDataset(namespace="lab", name="sentences"), FakeSeedStore(Journal()))


@pytest.fixture
def recipe_path(tmp_path: Path, recipe_source: str) -> Path:
    path = tmp_path / "sentences.yaml"
    path.write_text(recipe_source)
    return path
