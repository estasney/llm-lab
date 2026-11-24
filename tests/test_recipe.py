import pytest
from pydantic import ValidationError

from llm_lab.datasets.recipe import RecipeJob, parse_recipe
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
