from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from llm_lab.datasets.recipe import PortableRecipe, RecipeJob, parse_portable_recipe, parse_recipe
from llm_lab.errors import CommandError
from llm_lab.jobs.protocol import PlacementKind, ResourceRequest
from tests.conftest import build_model_config, build_recipe_source


def test_recipe_serves_the_one_model_of_its_model_configs() -> None:
    source = build_recipe_source([build_model_config("writer"), build_model_config("judge")])

    recipe = parse_recipe(source)

    actual = (recipe.served_model, recipe.provider_name, recipe.job, recipe.vllm_args, recipe.source)
    expected = (
        "Qwen/Qwen3-8B",
        "vllm",
        RecipeJob(
            disk_gb=120,
            ttl_seconds_after_finished=600,
            placement=ResourceRequest(
                kind=PlacementKind.resources, gpu_name="RTX 6000Ada", gpu_count=2, max_dollars_per_hour=2.5
            ),
        ),
        ("--tensor-parallel-size", "2"),
        source,
    )
    assert actual == expected, f"Expected {expected}, got {actual}."


@pytest.mark.parametrize(
    "model_configs",
    [
        [build_model_config("writer"), build_model_config("judge", model="Qwen/Qwen3-32B")],
        [build_model_config("writer"), build_model_config("judge", provider="nvidia")],
        [],
    ],
    ids=["two-models", "two-providers", "no-models"],
)
def test_recipe_without_exactly_one_model_on_one_provider_is_rejected(model_configs: list[dict[str, str]]) -> None:
    with pytest.raises(CommandError, match="one model on one provider"):
        parse_recipe(build_recipe_source(model_configs))


@pytest.mark.parametrize(
    "source",
    [
        build_recipe_source([build_model_config("writer")], studio={"ui": {}}),
        build_recipe_source([build_model_config("writer")], job={"disk_gb": 120, "ttl_seconds_after_finished": 600}),
        build_recipe_source([build_model_config("writer")]).replace("llm-text", "llm-poem"),
    ],
    ids=["unknown-section", "no-placement", "unknown-column-type"],
)
def test_recipe_with_an_invalid_section_is_rejected(source: str) -> None:
    with pytest.raises(ValidationError):
        parse_recipe(source)


@pytest.mark.parametrize(
    ("other", "expected"),
    [
        (build_recipe_source([build_model_config("writer")]).replace("2.5", "4.0"), True),
        (build_recipe_source([build_model_config("writer")], prompt="Name a {{ topic }}."), False),
        (build_recipe_source([build_model_config("writer")]).replace("'2'", "'4'"), False),
    ],
    ids=["other-job", "other-column", "other-vllm-args"],
)
def test_recipes_generate_like_rows_when_only_the_job_differs(
    portable_recipe: PortableRecipe, other: str, *, expected: bool
) -> None:
    actual = portable_recipe.generates_rows_like(parse_portable_recipe(other))
    assert actual == expected, f"Expected {expected} for {other!r}, got {actual}."


@pytest.mark.parametrize(
    "seed_source",
    [
        {"seed_type": "hf", "path": "buckets/lab/llm-lab-seeds/seeds/0f/*.parquet"},
        {"seed_type": "hf", "path": "datasets/lab/questions/data/*.parquet"},
    ],
    ids=["bucket", "dataset"],
)
def test_recipe_with_hub_seeds_runs_on_another_machine(seed_source: dict[str, str]) -> None:
    source = build_recipe_source([build_model_config("writer")], seed_source=seed_source)

    portable = parse_portable_recipe(source)

    assert portable.recipe.source == source, "Expected the portable recipe to keep the recipe file unchanged."


def test_recipe_with_local_seeds_does_not_run_on_another_machine(tmp_path: Path) -> None:
    seeds = tmp_path / "questions.parquet"
    pd.DataFrame({"question": ["Why is the sky blue?"]}).to_parquet(seeds)
    source = build_recipe_source([build_model_config("writer")], seed_source={"seed_type": "local", "path": str(seeds)})

    with pytest.raises(CommandError, match="only from an hf seed source"):
        parse_portable_recipe(source)
