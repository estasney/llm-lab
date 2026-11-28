import pytest
from pydantic import ValidationError

from llm_lab.datasets.recipe import HubSeed, LocalSeed, Recipe, RecipeJob, SeedLocation, parse_recipe
from llm_lab.errors import CommandError
from llm_lab.jobs.protocol import PlacementKind, ResourceRequest
from tests.conftest import build_model_config, build_recipe_source


def test_recipe_serves_the_one_model_of_its_model_configs() -> None:
    """
    Arrange: a recipe source with two model configs that name the same model on the same provider.
    Act: parse the recipe.
    Assert: the recipe has the model and the provider of the model configs, the job and the vLLM arguments of the
        source, and the source text.
    """
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
    """
    Arrange: a recipe source whose model configs name two models, two providers or no model.
    Act: parse the recipe.
    Assert: the parse raises a CommandError which tells that a recipe must name one model on one provider.
    """
    with pytest.raises(CommandError, match="one model on one provider"):
        parse_recipe(build_recipe_source(model_configs))


def build_hub_seed(path: str) -> dict[str, str]:
    return {"kind": "hub", "path": f"buckets/lab/llm-lab-seeds/{path}"}


def test_recipe_runs_with_the_hub_seed_that_it_names() -> None:
    """
    Arrange: a recipe source with a seed of the kind hub, whose path is in a Hub bucket.
    Act: parse the recipe.
    Assert: the seed of the recipe is a Hub seed with the same path.
    """
    recipe = parse_recipe(build_recipe_source([build_model_config("writer")], seed=build_hub_seed("0a1b/*.parquet")))

    expected = HubSeed(kind=SeedLocation.hub, path="buckets/lab/llm-lab-seeds/0a1b/*.parquet")
    assert recipe.seed == expected, f"Expected {expected}, got {recipe.seed}."


def test_recipe_reads_local_seeds_from_the_path_that_it_names() -> None:
    """
    Arrange: a recipe source with a seed of the kind local, whose path is a local file.
    Act: parse the recipe.
    Assert: the seed of the recipe is a local seed with the same path.
    """
    recipe = parse_recipe(
        build_recipe_source([build_model_config("writer")], seed={"kind": "local", "path": "a.parquet"})
    )

    expected = LocalSeed(kind=SeedLocation.local, path="a.parquet")
    assert recipe.seed == expected, f"Expected {expected}, got {recipe.seed}."


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
    """
    Arrange: a recipe source with an unknown section, a job without a placement, a column of an unknown type, or a
        seed in the data_designer section.
    Act: parse the recipe.
    Assert: the parse raises a ValidationError.
    """
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
    """
    Arrange: a recipe, and a second recipe source that is different from it only in the job, in the prompt of a
        column or in the vLLM arguments, or that adds a seed.
    Act: parse the second recipe source, and compare the two recipes with generates_rows_like.
    Assert: the result is true only when the difference is in the job.
    """
    actual = recipe.generates_rows_like(parse_recipe(other))
    assert actual == expected, f"Expected {expected} for {other!r}, got {actual}."
