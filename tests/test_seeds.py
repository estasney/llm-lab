from pathlib import Path, PurePosixPath

import pandas as pd
import pytest
import yaml

from llm_lab.datasets.recipe import parse_recipe
from llm_lab.datasets.seeds import SeedPlan, plan_seed_upload
from tests.conftest import build_model_config, build_recipe_source


def write_seed_files(directory: Path, questions: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index, question in enumerate(questions):
        pd.DataFrame({"question": [question]}).to_parquet(directory / f"part-{index}.parquet")
    return directory


def plan_local_seeds(path: str) -> SeedPlan:
    source = build_recipe_source([build_model_config("writer")], seed_source={"seed_type": "local", "path": path})
    return plan_seed_upload(parse_recipe(source), "lab/llm-lab-seeds")


def read_seed_path(plan: SeedPlan) -> str:
    return yaml.safe_load(plan.recipe.recipe.source)["data_designer"]["seed_config"]["source"]["path"]


def find_upload_folder(plan: SeedPlan) -> PurePosixPath:
    if plan.upload is None:
        pytest.fail("Expected local seed files to need an upload.")
    folders = {PurePosixPath(path).parent for path in plan.upload.files}
    if len(folders) != 1:
        pytest.fail(f"Expected the seed files in one bucket folder, got {folders}.")
    return folders.pop()


def test_local_seed_files_go_to_the_bucket_folder_that_the_recipe_reads(tmp_path: Path) -> None:
    seeds = write_seed_files(tmp_path / "seeds", ["Why is the sky blue?", "What is a tide?"])

    plan = plan_local_seeds(f"{seeds}/*.parquet")

    folder = find_upload_folder(plan)
    actual = (plan.upload and plan.upload.bucket_id, plan.upload and dict(plan.upload.files), read_seed_path(plan))
    expected = (
        "lab/llm-lab-seeds",
        {f"{folder}/part-0.parquet": seeds / "part-0.parquet", f"{folder}/part-1.parquet": seeds / "part-1.parquet"},
        f"buckets/lab/llm-lab-seeds/{folder}/*.parquet",
    )
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_one_local_seed_file_keeps_its_name_in_the_bucket(tmp_path: Path) -> None:
    seeds = write_seed_files(tmp_path / "seeds", ["Why is the sky blue?"])

    plan = plan_local_seeds(str(seeds / "part-0.parquet"))

    folder = find_upload_folder(plan)
    assert read_seed_path(plan) == f"buckets/lab/llm-lab-seeds/{folder}/part-0.parquet", f"Got {read_seed_path(plan)}."


@pytest.mark.parametrize(
    ("other_questions", "same_folder"),
    [(["Why is the sky blue?", "What is a tide?"], True), (["Why is the sky blue?", "What is a wave?"], False)],
    ids=["same-contents", "other-contents"],
)
def test_bucket_folder_follows_the_contents_of_the_seed_files(
    tmp_path: Path, other_questions: list[str], *, same_folder: bool
) -> None:
    seeds = write_seed_files(tmp_path / "seeds", ["Why is the sky blue?", "What is a tide?"])
    other = write_seed_files(tmp_path / "other", other_questions)

    folders = (
        find_upload_folder(plan_local_seeds(f"{seeds}/*.parquet")),
        find_upload_folder(plan_local_seeds(f"{other}/*.parquet")),
    )

    assert (folders[0] == folders[1]) == same_folder, f"Expected same folder {same_folder}, got {folders}."


@pytest.mark.parametrize(
    "seed_source",
    [None, {"seed_type": "hf", "path": "datasets/lab/questions/data/*.parquet"}],
    ids=["no-seeds", "hub-seeds"],
)
def test_recipe_without_local_seeds_runs_unchanged_with_no_upload(seed_source: dict[str, str] | None) -> None:
    source = build_recipe_source([build_model_config("writer")], seed_source=seed_source)

    plan = plan_seed_upload(parse_recipe(source), "lab/llm-lab-seeds")

    assert (plan.recipe.recipe.source, plan.upload) == (source, None), f"Got {plan}."
