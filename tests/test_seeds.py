from pathlib import Path

import pandas as pd
import pytest
from data_designer.config.seed import SamplingStrategy

from llm_lab.datasets.hub import HubDataset
from llm_lab.datasets.recipe import HubSeed, SeedLocation, parse_recipe
from llm_lab.datasets.seeds import read_seed_folder, resolve_recipe
from llm_lab.errors import CommandError
from tests.conftest import FakeSeedStore, Journal, build_model_config, build_recipe_source


def write_seed_files(directory: Path, questions: dict[str, str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, question in questions.items():
        pd.DataFrame({"question": [question]}).to_parquet(directory / name)
    return directory


def test_recipe_with_local_seeds_runs_from_the_files_that_the_store_uploads(journal: Journal, tmp_path: Path) -> None:
    store = FakeSeedStore(journal)
    seed = {"kind": "local", "path": "seeds/*.parquet", "sampling_strategy": "shuffle"}
    source = build_recipe_source([build_model_config("writer")], seed=seed)

    recipe = resolve_recipe(source, tmp_path, HubDataset(namespace="lab", name="questions"), store)

    expected_seed = HubSeed(
        kind=SeedLocation.hub, path="buckets/lab/seeds/0a1b/*.parquet", sampling_strategy=SamplingStrategy.SHUFFLE
    )
    actual = (store.patterns, recipe.seed, parse_recipe(recipe.source) == recipe)
    expected = ([tmp_path / "seeds" / "*.parquet"], expected_seed, True)
    assert actual == expected, f"Expected {expected}, got {actual}."


@pytest.mark.parametrize(
    "seed",
    [None, {"kind": "hub", "path": "datasets/lab/questions/data/*.parquet"}],
    ids=["no-seed", "hub-seed"],
)
def test_recipe_without_local_seeds_runs_as_written(
    journal: Journal, tmp_path: Path, seed: dict[str, str] | None
) -> None:
    source = build_recipe_source([build_model_config("writer")], seed=seed)

    recipe = resolve_recipe(source, tmp_path, HubDataset(namespace="lab", name="questions"), FakeSeedStore(journal))

    assert (recipe.source, journal.events) == (source, []), f"Got {recipe.source!r} and {journal.events}."


def test_stored_recipe_keeps_the_data_designer_section_of_the_written_recipe(journal: Journal, tmp_path: Path) -> None:
    written = (Path(__file__).parent.parent / "recipes" / "concise-dpo.yaml").read_text()
    source = f"{written}\nseed:\n  kind: local\n  path: questions.parquet\n"

    recipe = resolve_recipe(source, tmp_path, HubDataset(namespace="lab", name="answers"), FakeSeedStore(journal))

    assert recipe.data_designer == parse_recipe(written).data_designer, "Expected data_designer to stay as written."


def test_seed_folder_holds_every_file_that_the_pattern_matches(tmp_path: Path) -> None:
    seeds = write_seed_files(tmp_path, {"part-0.parquet": "What is a tide?", "part-1.parquet": "Why is the sky blue?"})

    folder = read_seed_folder(seeds / "*.parquet")

    expected = {"part-0.parquet": seeds / "part-0.parquet", "part-1.parquet": seeds / "part-1.parquet"}
    assert folder.files == expected, f"Expected {expected}, got {folder.files}."


@pytest.mark.parametrize(
    ("other", "same_folder"),
    [
        ({"part-0.parquet": "What is a tide?", "part-1.parquet": "Why is the sky blue?"}, True),
        ({"part-0.parquet": "What is a tide?", "part-1.parquet": "Why is the sea salty?"}, False),
        ({"part-0.parquet": "What is a tide?", "part-2.parquet": "Why is the sky blue?"}, False),
    ],
    ids=["same-files", "other-contents", "other-name"],
)
def test_seed_folder_name_follows_the_names_and_contents_of_the_files(
    tmp_path: Path, other: dict[str, str], *, same_folder: bool
) -> None:
    seeds = write_seed_files(
        tmp_path / "seeds", {"part-0.parquet": "What is a tide?", "part-1.parquet": "Why is the sky blue?"}
    )
    others = write_seed_files(tmp_path / "others", other)

    names = read_seed_folder(seeds / "*.parquet").name, read_seed_folder(others / "*.parquet").name

    assert (names[0] == names[1]) == same_folder, f"Expected same folder {same_folder}, got {names}."


def test_seed_folder_needs_at_least_one_file(tmp_path: Path) -> None:
    with pytest.raises(CommandError, match="No seed files match"):
        read_seed_folder(tmp_path / "*.parquet")
