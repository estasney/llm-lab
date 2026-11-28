from pathlib import Path

import pandas as pd
import pytest
from data_designer.config.seed import SamplingStrategy

from llm_lab.datasets.hub import HubDataset
from llm_lab.datasets.job import prepare_job_recipe
from llm_lab.datasets.recipe import HubSeed, SeedLocation, parse_recipe
from llm_lab.datasets.seeds import read_seed_folder
from llm_lab.errors import CommandError
from tests.conftest import FakeSeedStore, Journal, build_model_config, build_recipe_source


def write_seed_files(directory: Path, questions: dict[str, str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name, question in questions.items():
        pd.DataFrame({"question": [question]}).to_parquet(directory / name)
    return directory


def test_job_recipe_reads_the_local_seeds_that_the_store_uploads(journal: Journal) -> None:
    """
    Arrange: a recipe with the local seed "seeds/*.parquet" and the shuffle sampling strategy, and a fake seed
        store.
    Act: prepare the job recipe.
    Assert: the store receives the seed pattern, and the seed of the job recipe and the seed in its source are
        a Hub seed at the bucket path that the store returns, with the shuffle sampling strategy.
    """
    store = FakeSeedStore(journal)
    seed = {"kind": "local", "path": "seeds/*.parquet", "sampling_strategy": "shuffle"}
    recipe = parse_recipe(build_recipe_source([build_model_config("writer")], seed=seed))

    job_recipe = prepare_job_recipe(recipe, HubDataset(namespace="lab", name="questions"), store)

    expected_seed = HubSeed(
        kind=SeedLocation.hub, path="buckets/lab/seeds/0a1b/*.parquet", sampling_strategy=SamplingStrategy.SHUFFLE
    )
    actual = (store.patterns, job_recipe.seed, parse_recipe(job_recipe.source).seed)
    expected = ([Path("seeds/*.parquet")], expected_seed, expected_seed)
    assert actual == expected, f"Expected {expected}, got {actual}."


@pytest.mark.parametrize(
    "seed",
    [None, {"kind": "hub", "path": "datasets/lab/questions/data/*.parquet"}],
    ids=["no-seed", "hub-seed"],
)
def test_job_recipe_without_local_seeds_is_the_recipe_as_written(journal: Journal, seed: dict[str, str] | None) -> None:
    """
    Arrange: a recipe without a local seed, and a fake seed store.
    Act: prepare the job recipe.
    Assert: the job recipe has the source and the seed of the recipe, and the seed store records no events.
    """
    recipe = parse_recipe(build_recipe_source([build_model_config("writer")], seed=seed))

    job_recipe = prepare_job_recipe(recipe, HubDataset(namespace="lab", name="questions"), FakeSeedStore(journal))

    assert (job_recipe.source, job_recipe.seed, journal.events) == (recipe.source, recipe.seed, []), (
        f"Got {job_recipe}."
    )


def test_job_recipe_keeps_the_data_designer_section_of_the_written_recipe(journal: Journal) -> None:
    """
    Arrange: the recipe recipes/concise-dpo.yaml with a local seed added at the end.
    Act: prepare the job recipe, which writes the source of the recipe again with a Hub seed.
    Assert: the data_designer section of the job recipe source is equal to the data_designer section of the
        written recipe.
    """
    written = (Path(__file__).parent.parent / "recipes" / "concise-dpo.yaml").read_text()
    recipe = parse_recipe(f"{written}\nseed:\n  kind: local\n  path: questions.parquet\n")

    job_recipe = prepare_job_recipe(recipe, HubDataset(namespace="lab", name="answers"), FakeSeedStore(journal))

    assert parse_recipe(job_recipe.source).data_designer == parse_recipe(written).data_designer, (
        "Expected data_designer to stay as written."
    )


def test_seed_folder_holds_every_file_that_the_pattern_matches(tmp_path: Path) -> None:
    """
    Arrange: a folder with two parquet seed files.
    Act: read the seed folder of the pattern "*.parquet" in that folder.
    Assert: the files of the seed folder are the names of the two files with their paths.
    """
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
    """
    Arrange: two folders of two seed files each, where the second folder has the same files as the first, or
        one file with a different content or name.
    Act: read the seed folder of each folder.
    Assert: the two seed folders have the same name only when the names and the contents of the files are the
        same.
    """
    seeds = write_seed_files(
        tmp_path / "seeds", {"part-0.parquet": "What is a tide?", "part-1.parquet": "Why is the sky blue?"}
    )
    others = write_seed_files(tmp_path / "others", other)

    names = read_seed_folder(seeds / "*.parquet").name, read_seed_folder(others / "*.parquet").name

    assert (names[0] == names[1]) == same_folder, f"Expected same folder {same_folder}, got {names}."


def test_seed_folder_needs_at_least_one_file(tmp_path: Path) -> None:
    """
    Arrange: an empty folder.
    Act: read the seed folder of the pattern "*.parquet" in that folder.
    Assert: the read raises a CommandError with the message that no seed files match.
    """
    with pytest.raises(CommandError, match="No seed files match"):
        read_seed_folder(tmp_path / "*.parquet")
