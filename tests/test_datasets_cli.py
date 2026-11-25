import base64
import gzip
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from llm_lab.cli import app
from llm_lab.cli.datasets.deps import get_dataset_designer, get_dataset_hub, get_model_server
from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.cli.jobs.deps import get_job_orchestrator
from llm_lab.datasets.hub import HubDataset, StoredDataset
from llm_lab.datasets.job import build_dataset_job
from llm_lab.datasets.recipe import PortableRecipe, Recipe, parse_recipe
from llm_lab.datasets.repository import RepositoryFilePaths, RepositoryMetadata
from llm_lab.datasets.seeds import plan_seed_upload
from llm_lab.jobs.protocol import JobRecord, JobSpec, Running
from llm_lab.keyring_settings import SecretName
from llm_lab.settings import Settings, get_settings
from tests.conftest import FakeDesigner, FakeHub, Journal, build_model_config, build_recipe_source, invoke_cli, run_cli


@dataclass
class FakeServer:
    journal: Journal

    @contextmanager
    def serve(self, recipe: Recipe) -> Generator[None]:
        self.journal.events.append(f"start {recipe.served_model}")
        yield
        self.journal.events.append("stop")


@dataclass
class FakeOrchestrator:
    submitted: list[JobSpec] = field(default_factory=list[JobSpec])

    def submit(self, spec: JobSpec) -> JobRecord:
        self.submitted.append(spec)
        return JobRecord(spec=spec, status=Running(submitted_at=datetime(2026, 1, 1, tzinfo=UTC), instance_id=9000))


def build_app(
    journal: Journal, hub: FakeHub, orchestrator: FakeOrchestrator, datasets_image: str | None
) -> DependentTyper:
    return app.with_dependency_overrides(
        {
            get_settings: lambda: Settings(datasets_image=datasets_image),
            get_model_server: lambda: FakeServer(journal),
            get_dataset_designer: lambda: FakeDesigner(journal, Path("artifacts") / "sentences"),
            get_dataset_hub: lambda: hub,
            get_job_orchestrator: lambda: orchestrator,
        }
    )


def test_create_generates_while_the_model_is_served_and_publishes_after(journal: Journal, recipe_path: Path) -> None:
    cli = build_app(journal, FakeHub(journal), FakeOrchestrator(), None)

    output = run_cli(cli, ["datasets", "create", str(recipe_path), "--dataset", "lab/sentences", "--rows", "4"])

    expected = [
        "validate Qwen/Qwen3-8B",
        "read lab/sentences",
        "start Qwen/Qwen3-8B",
        "create sentences 4",
        "stop",
        "publish lab/sentences sentences",
    ]
    assert journal.events == expected, f"Expected {expected}, got {journal.events}."
    assert output == "https://huggingface.co/datasets/lab/sentences\n", f"Expected the dataset URL, got {output!r}."


def build_stored_dataset(recipe_source: str) -> StoredDataset:
    return StoredDataset(
        revision="abc123",
        recipe_source=recipe_source,
        metadata=RepositoryMetadata(
            actual_num_records=4,
            target_num_records=4,
            file_paths=RepositoryFilePaths(data=("data/batch_00000.parquet",)),
        ),
    )


def test_create_adds_rows_to_a_dataset_of_a_recipe_with_other_job_settings(journal: Journal, recipe_path: Path) -> None:
    stored = build_stored_dataset(build_recipe_source([build_model_config("writer")]).replace("600", "900"))
    hub = FakeHub(journal, repositories={"lab/sentences": stored})
    cli = build_app(journal, hub, FakeOrchestrator(), None)

    actual = invoke_cli(cli, ["datasets", "create", str(recipe_path), "--dataset", "lab/sentences", "--rows", "4"])

    expected = (0, "https://huggingface.co/datasets/lab/sentences\n")
    assert actual == expected, f"Expected {expected}, got {actual}."
    assert hub.published == [stored], f"Expected the rows to go onto the stored dataset, got {hub.published}."


@pytest.mark.parametrize("command", ["create", "submit"])
def test_command_starts_nothing_when_the_dataset_has_another_recipe(
    journal: Journal, recipe_path: Path, command: str
) -> None:
    orchestrator = FakeOrchestrator()
    stored = build_stored_dataset(build_recipe_source([build_model_config("writer")], prompt="Name a {{ topic }}."))
    hub = FakeHub(journal, repositories={"lab/sentences": stored})
    cli = build_app(journal, hub, orchestrator, "ghcr.io/lab/datasets:1")

    actual = invoke_cli(cli, ["datasets", command, str(recipe_path), "--dataset", "lab/sentences", "--rows", "4"])

    expected = (
        1,
        "Error: The recipe differs from the recipe of the dataset lab/sentences. "
        "Rows of one dataset come from one recipe.\n",
    )
    assert actual == expected, f"Expected {expected}, got {actual}."
    assert journal.events == ["validate Qwen/Qwen3-8B", "read lab/sentences"], f"Got {journal.events}."
    assert orchestrator.submitted == [], f"Expected no job, got {orchestrator.submitted}."


def test_command_reports_a_stored_recipe_that_is_not_valid(journal: Journal, recipe_path: Path) -> None:
    hub = FakeHub(journal, repositories={"lab/sentences": build_stored_dataset("job: {}\n")})
    cli = build_app(journal, hub, FakeOrchestrator(), None)

    exit_code, output = invoke_cli(
        cli, ["datasets", "create", str(recipe_path), "--dataset", "lab/sentences", "--rows", "4"]
    )

    assert (exit_code, output.splitlines()[0]) == (1, "Error: lab/sentences/recipe.yaml is not a valid recipe:"), (
        f"Got {output!r}."
    )
    assert hub.published == [], f"Expected no upload, got {hub.published}."


@pytest.mark.parametrize("dataset", ["sentences", "lab/Sentences", "lab/sentences/extra"])
def test_command_rejects_a_dataset_that_is_not_a_repository_id(
    journal: Journal, recipe_path: Path, dataset: str
) -> None:
    cli = build_app(journal, FakeHub(journal), FakeOrchestrator(), None)

    exit_code, _ = invoke_cli(cli, ["datasets", "create", str(recipe_path), "--dataset", dataset, "--rows", "4"])

    assert exit_code == 2, f"Expected a usage error for {dataset!r}, got exit code {exit_code}."
    assert journal.events == [], f"Expected no work for a rejected dataset, got {journal.events}."


def test_command_reports_each_invalid_section_of_a_recipe(journal: Journal, tmp_path: Path) -> None:
    recipe_path = tmp_path / "sentences.yaml"
    recipe_path.write_text(build_recipe_source([build_model_config("writer")], studio={"ui": {}}))
    cli = build_app(journal, FakeHub(journal), FakeOrchestrator(), None)

    actual = invoke_cli(cli, ["datasets", "create", str(recipe_path), "--dataset", "lab/sentences", "--rows", "4"])

    expected = (1, f"Error: {recipe_path} is not a valid recipe:\n  studio: Extra inputs are not permitted\n")
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_submit_starts_a_job_that_runs_the_recipe_in_the_datasets_image(
    journal: Journal, recipe_path: Path, portable_recipe: PortableRecipe
) -> None:
    orchestrator = FakeOrchestrator()
    cli = build_app(journal, FakeHub(journal), orchestrator, "ghcr.io/lab/datasets:1")

    exit_code, output = invoke_cli(
        cli, ["datasets", "submit", str(recipe_path), "--dataset", "lab/sentences", "--rows", "4"]
    )

    expected = [
        build_dataset_job(portable_recipe, HubDataset(namespace="lab", name="sentences"), 4, "ghcr.io/lab/datasets:1")
    ]
    assert orchestrator.submitted == expected, f"Expected {expected}, got {orchestrator.submitted}."
    assert (exit_code, JobRecord.model_validate_json(output).status.condition) == (0, "running"), f"Got {output!r}."
    assert journal.events == ["validate Qwen/Qwen3-8B", "read lab/sentences"], f"Got {journal.events}."


def test_submit_uploads_local_seeds_and_starts_a_job_that_reads_them_from_the_bucket(
    journal: Journal, tmp_path: Path
) -> None:
    seeds = tmp_path / "questions.parquet"
    pd.DataFrame({"question": ["Why is the sky blue?"]}).to_parquet(seeds)
    recipe_path = tmp_path / "questions.yaml"
    recipe_path.write_text(
        build_recipe_source([build_model_config("writer")], seed_source={"seed_type": "local", "path": str(seeds)})
    )
    orchestrator = FakeOrchestrator()
    hub = FakeHub(journal)
    cli = build_app(journal, hub, orchestrator, "ghcr.io/lab/datasets:1")

    exit_code, _ = invoke_cli(
        cli, ["datasets", "submit", str(recipe_path), "--dataset", "lab/questions", "--rows", "4"]
    )

    plan = plan_seed_upload(parse_recipe(recipe_path.read_text()), "lab/llm-lab-seeds")
    dataset = HubDataset(namespace="lab", name="questions")
    actual = (exit_code, journal.events, hub.uploads, orchestrator.submitted)
    expected = (
        0,
        ["validate Qwen/Qwen3-8B", "read lab/questions", "upload seeds to lab/llm-lab-seeds"],
        [plan.upload],
        [build_dataset_job(plan.recipe, dataset, 4, "ghcr.io/lab/datasets:1")],
    )
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_submit_requires_the_datasets_image(journal: Journal, recipe_path: Path) -> None:
    orchestrator = FakeOrchestrator()
    cli = build_app(journal, FakeHub(journal), orchestrator, None)

    actual = invoke_cli(cli, ["datasets", "submit", str(recipe_path), "--dataset", "lab/sentences", "--rows", "4"])

    expected = (1, "Error: The datasets image is not set. Set the environment variable LLM_LAB_DATASETS_IMAGE.\n")
    assert actual == expected, f"Expected {expected}, got {actual}."
    assert orchestrator.submitted == [], f"Expected no job, got {orchestrator.submitted}."


def test_dataset_job_carries_the_recipe_and_runs_it_with_the_hub_token(portable_recipe: PortableRecipe) -> None:
    spec = build_dataset_job(
        portable_recipe, HubDataset(namespace="lab", name="sentences"), 4, "ghcr.io/lab/datasets:1"
    )

    decoded = gzip.decompress(base64.b64decode(spec.environment["LLM_LAB_RECIPE"])).decode()
    actual = (spec.name, spec.image, spec.command, set(spec.environment), spec.secret_environment, spec.disk_gb)
    expected = (
        "sentences",
        "ghcr.io/lab/datasets:1",
        ("llm-lab-dataset-job", "--dataset", "lab/sentences", "--rows", "4"),
        {"LLM_LAB_RECIPE", "LLM_LAB_ARTIFACTS_DIR"},
        (SecretName.hf_token,),
        120,
    )
    assert actual == expected, f"Expected {expected}, got {actual}."
    assert (spec.ttl_seconds_after_finished, spec.placement) == (600, portable_recipe.recipe.job.placement), (
        "Expected the recipe job."
    )
    assert decoded == portable_recipe.recipe.source, "Expected the job to carry the recipe file unchanged."
    assert spec.artifacts_directory == spec.environment["LLM_LAB_ARTIFACTS_DIR"], (
        "Expected destroy to download the directory where Data Designer writes."
    )


def test_committed_recipe_schema_matches_the_schema_command() -> None:
    committed = (Path(__file__).parent.parent / "recipes" / "recipe.schema.json").read_text()
    assert run_cli(app, ["datasets", "schema"]) == committed, (
        "Expected recipes/recipe.schema.json to match the output of llm-lab datasets schema."
    )
