from pathlib import Path

import pytest
from data_designer.config.seed import SamplingStrategy

from llm_lab.datasets.hub import HubDataset
from llm_lab.datasets.recipe import HubSeed, HubSeedFiles, SeedLocation, parse_recipe
from llm_lab.datasets.seeds import name_seed_folder, resolve_recipe
from tests.conftest import FakeSeedStore, Journal, build_model_config, build_recipe_source


def test_recipe_with_local_seeds_runs_from_the_files_that_the_store_uploads(journal: Journal, tmp_path: Path) -> None:
    store = FakeSeedStore(journal)
    seed = {"files": {"kind": "local", "path": "seeds/*.parquet"}, "sampling_strategy": "shuffle"}
    source = build_recipe_source([build_model_config("writer")], seed=seed)

    recipe = resolve_recipe(source, tmp_path, HubDataset(namespace="lab", name="questions"), store)

    expected_seed = HubSeed(
        files=HubSeedFiles(kind=SeedLocation.hub, path="buckets/lab/seeds/0a1b/*.parquet"),
        sampling_strategy=SamplingStrategy.SHUFFLE,
    )
    actual = (store.patterns, recipe.seed, parse_recipe(recipe.source) == recipe)
    expected = ([tmp_path / "seeds" / "*.parquet"], expected_seed, True)
    assert actual == expected, f"Expected {expected}, got {actual}."


@pytest.mark.parametrize(
    "seed",
    [None, {"files": {"kind": "hub", "path": "datasets/lab/questions/data/*.parquet"}}],
    ids=["no-seed", "hub-seed"],
)
def test_recipe_without_local_seeds_runs_as_written(
    journal: Journal, tmp_path: Path, seed: dict[str, object] | None
) -> None:
    source = build_recipe_source([build_model_config("writer")], seed=seed)

    recipe = resolve_recipe(source, tmp_path, HubDataset(namespace="lab", name="questions"), FakeSeedStore(journal))

    assert (recipe.source, journal.events) == (source, []), f"Got {recipe.source!r} and {journal.events}."


@pytest.mark.parametrize(
    ("other", "same_folder"),
    [
        ({"part-1.parquet": "bb", "part-0.parquet": "aa"}, True),
        ({"part-0.parquet": "aa", "part-1.parquet": "cc"}, False),
        ({"part-0.parquet": "aa", "part-2.parquet": "bb"}, False),
    ],
    ids=["same-files", "other-contents", "other-name"],
)
def test_seed_folder_follows_the_names_and_contents_of_the_files(other: dict[str, str], *, same_folder: bool) -> None:
    folders = name_seed_folder({"part-0.parquet": "aa", "part-1.parquet": "bb"}), name_seed_folder(other)

    assert (folders[0] == folders[1]) == same_folder, f"Expected same folder {same_folder}, got {folders}."


def test_stored_recipe_keeps_the_data_designer_section_of_the_written_recipe(journal: Journal, tmp_path: Path) -> None:
    written = (Path(__file__).parent.parent / "recipes" / "concise-dpo.yaml").read_text()
    source = f"{written}\nseed:\n  files:\n    kind: local\n    path: questions.parquet\n"

    recipe = resolve_recipe(source, tmp_path, HubDataset(namespace="lab", name="answers"), FakeSeedStore(journal))

    assert recipe.data_designer == parse_recipe(written).data_designer, "Expected data_designer to stay as written."
