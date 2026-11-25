import pytest
from pydantic import ValidationError

from llm_lab.datasets.recipe import HubSeed, Recipe, RecipeJob, SeedLocation, parse_recipe
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


def build_hub_seed(path: str) -> dict[str, str]:
    return {"kind": "hub", "path": f"buckets/lab/llm-lab-seeds/{path}"}


def test_recipe_runs_with_the_hub_seed_that_it_names() -> None:
    recipe = parse_recipe(build_recipe_source([build_model_config("writer")], seed=build_hub_seed("0a1b/*.parquet")))

    expected = HubSeed(kind=SeedLocation.hub, path="buckets/lab/llm-lab-seeds/0a1b/*.parquet")
    assert recipe.seed == expected, f"Expected {expected}, got {recipe.seed}."


def test_recipe_with_local_seeds_does_not_run_before_they_are_uploaded() -> None:
    source = build_recipe_source([build_model_config("writer")], seed={"kind": "local", "path": "a.parquet"})

    with pytest.raises(CommandError, match="Upload them before the recipe runs"):
        parse_recipe(source)


@pytest.mark.parametrize(
    "source",
    [
        build_recipe_source([build_model_config("writer")], studio={"ui": {}}),
        build_recipe_source([build_model_config("writer")], job={"disk_gb": 120, "ttl_seconds_after_finished": 600}),
        build_recipe_source([build_model_config("writer")]).replace("llm-text", "llm-poem"),
        build_recipe_source([build_model_config("writer")]).replace(
            "data_designer:\n",
            "data_designer:\n  seed_config:\n    source:\n      seed_type: hf\n      path: x.parquet\n",
        ),
    ],
    ids=["unknown-section", "no-placement", "unknown-column-type", "data-designer-seed"],
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
        (build_recipe_source([build_model_config("writer")], seed=build_hub_seed("seeds/0a1b/*.parquet")), False),
    ],
    ids=["other-job", "other-column", "other-vllm-args", "other-seed"],
)
def test_recipes_generate_like_rows_when_only_the_job_differs(recipe: Recipe, other: str, *, expected: bool) -> None:
    actual = recipe.generates_rows_like(parse_recipe(other))
    assert actual == expected, f"Expected {expected} for {other!r}, got {actual}."
