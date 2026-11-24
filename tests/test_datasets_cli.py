import base64
import gzip
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from llm_lab.cli import app
from llm_lab.cli.datasets.deps import get_dataset_designer, get_dataset_hub, get_model_server
from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.cli.jobs.deps import get_job_orchestrator
from llm_lab.datasets.hub import HubDataset
from llm_lab.datasets.job import build_dataset_job, require_image_cuda
from llm_lab.datasets.recipe import Recipe
from llm_lab.jobs.protocol import ExplicitOffer, JobRecord, JobSpec, PlacementKind, ResourceRequest, Running
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
        "exists lab/sentences",
        "start Qwen/Qwen3-8B",
        "create sentences 4",
        "stop",
        "publish lab/sentences sentences",
    ]
    assert journal.events == expected, f"Expected {expected}, got {journal.events}."
    assert output == "https://huggingface.co/datasets/lab/sentences\n", f"Expected the dataset URL, got {output!r}."


@pytest.mark.parametrize("command", ["create", "submit"])
def test_command_starts_nothing_when_the_dataset_exists(journal: Journal, recipe_path: Path, command: str) -> None:
    orchestrator = FakeOrchestrator()
    cli = build_app(journal, FakeHub(journal, existing={"lab/sentences"}), orchestrator, "ghcr.io/lab/datasets:1")

    actual = invoke_cli(cli, ["datasets", command, str(recipe_path), "--dataset", "lab/sentences", "--rows", "4"])

    expected = (1, "Error: The dataset lab/sentences already exists on the Hub.\n")
    assert actual == expected, f"Expected {expected}, got {actual}."
    assert journal.events == ["validate Qwen/Qwen3-8B", "exists lab/sentences"], f"Got {journal.events}."
    assert orchestrator.submitted == [], f"Expected no job, got {orchestrator.submitted}."


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
    journal: Journal, recipe_path: Path, recipe: Recipe
) -> None:
    orchestrator = FakeOrchestrator()
    cli = build_app(journal, FakeHub(journal), orchestrator, "ghcr.io/lab/datasets:1")

    exit_code, output = invoke_cli(
        cli, ["datasets", "submit", str(recipe_path), "--dataset", "lab/sentences", "--rows", "4"]
    )

    expected = [
        build_dataset_job(recipe, HubDataset(namespace="lab", name="sentences"), 4, "ghcr.io/lab/datasets:1", 13.0)
    ]
    assert orchestrator.submitted == expected, f"Expected {expected}, got {orchestrator.submitted}."
    assert (exit_code, JobRecord.model_validate_json(output).status.condition) == (0, "running"), f"Got {output!r}."
    assert journal.events == ["validate Qwen/Qwen3-8B", "exists lab/sentences"], f"Got {journal.events}."


def test_submit_requires_the_datasets_image(journal: Journal, recipe_path: Path) -> None:
    orchestrator = FakeOrchestrator()
    cli = build_app(journal, FakeHub(journal), orchestrator, None)

    actual = invoke_cli(cli, ["datasets", "submit", str(recipe_path), "--dataset", "lab/sentences", "--rows", "4"])

    expected = (1, "Error: The datasets image is not set. Set the environment variable LLM_LAB_DATASETS_IMAGE.\n")
    assert actual == expected, f"Expected {expected}, got {actual}."
    assert orchestrator.submitted == [], f"Expected no job, got {orchestrator.submitted}."


def test_dataset_job_carries_the_recipe_and_runs_it_with_the_hub_token(recipe: Recipe) -> None:
    spec = build_dataset_job(recipe, HubDataset(namespace="lab", name="sentences"), 4, "ghcr.io/lab/datasets:1", 13.0)

    decoded = gzip.decompress(base64.b64decode(spec.environment["LLM_LAB_RECIPE"])).decode()
    actual = (spec.name, spec.image, spec.command, set(spec.environment), spec.secret_environment, spec.disk_gb)
    expected = (
        "sentences",
        "ghcr.io/lab/datasets:1",
        ("llm-lab-dataset-job", "--dataset", "lab/sentences", "--rows", "4"),
        {"LLM_LAB_RECIPE"},
        (SecretName.hf_token,),
        120,
    )
    assert actual == expected, f"Expected {expected}, got {actual}."
    expected_placement = ResourceRequest(
        kind=PlacementKind.resources, gpu_name="RTX 6000Ada", gpu_count=2, max_dollars_per_hour=2.5, min_cuda=13.0
    )
    assert (spec.ttl_seconds_after_finished, spec.placement) == (600, expected_placement), f"Got {spec}."
    assert decoded == recipe.source, "Expected the job to carry the recipe file unchanged."


@pytest.mark.parametrize(
    ("placement", "expected"),
    [
        (
            ResourceRequest(kind=PlacementKind.resources, gpu_name="H100", gpu_count=1, max_dollars_per_hour=3.0),
            13.0,
        ),
        (
            ResourceRequest(
                kind=PlacementKind.resources, gpu_name="H100", gpu_count=1, max_dollars_per_hour=3.0, min_cuda=12.8
            ),
            13.0,
        ),
        (
            ResourceRequest(
                kind=PlacementKind.resources, gpu_name="H100", gpu_count=1, max_dollars_per_hour=3.0, min_cuda=13.1
            ),
            13.1,
        ),
    ],
    ids=["recipe-without-floor", "recipe-floor-below-image", "recipe-floor-above-image"],
)
def test_dataset_job_requires_the_higher_cuda_floor_of_the_recipe_and_the_image(
    placement: ResourceRequest, expected: float
) -> None:
    actual = require_image_cuda(placement, 13.0)

    assert actual == placement.model_copy(update={"min_cuda": expected}), f"Expected min_cuda {expected}, got {actual}."


def test_dataset_job_keeps_an_explicit_offer() -> None:
    offer = ExplicitOffer(kind=PlacementKind.offer, offer_id=42)

    assert require_image_cuda(offer, 13.0) == offer, "Expected the explicit offer unchanged."
