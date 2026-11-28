import json
from collections.abc import Mapping
from pathlib import Path
from typing import assert_never

import pytest
import yaml

from llm_lab.datasets.recipe import Recipe
from llm_lab.datasets.repository import (
    GeneratedRun,
    RepositoryFilePaths,
    RepositoryMetadata,
    plan_repository_files,
    read_generated_run,
)


def read_document(files: Mapping[str, Path | bytes], path: str) -> str:
    content = files[path]
    match content:
        case bytes():
            return content.decode()
        case Path():
            raise TypeError(f"Expected {path} to be written by llm-lab, got the run file {content}.")
        case _:
            assert_never(content)


@pytest.fixture
def run_directory(tmp_path: Path) -> Path:
    for path in [
        "parquet-files/batch_00000.parquet",
        "parquet-files/batch_00001.parquet",
        "processors-files/chat/batch_00000.parquet",
        "images/a/one.png",
        "builder_config.json",
    ]:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(path)
    metadata = {
        "actual_num_records": 7,
        "target_num_records": 8,
        "dataset_name": "sentences",
        "file_paths": {
            "parquet-files": ["parquet-files/batch_00000.parquet", "parquet-files/batch_00001.parquet"],
            "processor-files": {"chat": ["processors-files/chat/batch_00000.parquet"]},
        },
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return tmp_path


@pytest.fixture
def run(run_directory: Path) -> GeneratedRun:
    return read_generated_run(run_directory)


@pytest.fixture
def stored() -> RepositoryMetadata:
    return RepositoryMetadata.model_validate_json(
        json.dumps(
            {
                "actual_num_records": 4,
                "target_num_records": 4,
                "file_paths": {"data": ["data/batch_00000.parquet"]},
                "column_statistics": [],
            }
        )
    )


def test_run_files_get_names_of_the_run_beside_the_stored_files(
    run: GeneratedRun, stored: RepositoryMetadata, recipe: Recipe
) -> None:
    files = plan_repository_files("sentences", "run1", run, stored, recipe)

    expected = {
        "data/run1-batch_00000.parquet": run.directory / "parquet-files/batch_00000.parquet",
        "data/run1-batch_00001.parquet": run.directory / "parquet-files/batch_00001.parquet",
        "chat/run1-batch_00000.parquet": run.directory / "processors-files/chat/batch_00000.parquet",
        "images/a/one.png": run.directory / "images/a/one.png",
        "builder_config.json": run.directory / "builder_config.json",
    }
    actual = {path: content for path, content in files.items() if isinstance(content, Path)}
    assert actual == expected, f"Expected {expected}, got {actual}."
    assert files["recipe.yaml"] == recipe.source.encode(), "Expected the recipe to be stored."


def test_metadata_describes_the_stored_rows_and_the_rows_of_the_run(
    run: GeneratedRun, stored: RepositoryMetadata, recipe: Recipe
) -> None:
    files = plan_repository_files("sentences", "run1", run, stored, recipe)

    expected = RepositoryMetadata(
        actual_num_records=11,
        target_num_records=12,
        file_paths=RepositoryFilePaths(
            data=("data/batch_00000.parquet", "data/run1-batch_00000.parquet", "data/run1-batch_00001.parquet"),
            processor_files={"chat": ("chat/run1-batch_00000.parquet",)},
        ),
    )
    actual = RepositoryMetadata.model_validate_json(read_document(files, "metadata.json"))
    assert actual == expected, f"Expected {expected}, got {actual}."


def test_dataset_card_shows_every_data_folder_to_the_dataset_viewer(
    run: GeneratedRun, stored: RepositoryMetadata, recipe: Recipe
) -> None:
    files = plan_repository_files("sentences", "run1", run, stored, recipe)

    _, header, body = read_document(files, "README.md").split("---\n", 2)
    expected = [
        {"config_name": "data", "data_files": "data/*.parquet", "default": True},
        {"config_name": "chat", "data_files": "chat/*.parquet"},
    ]
    assert yaml.safe_load(header)["configs"] == expected, f"Expected the data folders, got {header!r}."
    assert "11 rows" in body, f"Expected the row count of the repository, got {body!r}."


def test_run_without_images_or_processors_adds_only_data_files(tmp_path: Path, recipe: Recipe) -> None:
    (tmp_path / "parquet-files").mkdir()
    (tmp_path / "parquet-files" / "batch_00000.parquet").write_text("rows")
    metadata = {
        "actual_num_records": 2,
        "target_num_records": 2,
        "file_paths": {"parquet-files": ["parquet-files/batch_00000.parquet"]},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    empty = RepositoryMetadata(actual_num_records=0, target_num_records=0, file_paths=RepositoryFilePaths(data=()))

    files = plan_repository_files("sentences", "run1", read_generated_run(tmp_path), empty, recipe)

    expected = {"data/run1-batch_00000.parquet", "builder_config.json", "metadata.json", "README.md", "recipe.yaml"}
    assert set(files) == expected, f"Expected {expected}, got {set(files)}."
