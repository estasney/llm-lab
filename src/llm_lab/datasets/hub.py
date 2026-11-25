import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Protocol, assert_never

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import RepositoryNotFoundError
from pydantic import BaseModel, ConfigDict, SecretStr, StringConstraints

from llm_lab.datasets.recipe import PortableRecipe
from llm_lab.datasets.repository import (
    RepositoryFilePaths,
    RepositoryMetadata,
    plan_repository_files,
    read_generated_run,
)
from llm_lab.datasets.seeds import SeedUpload
from llm_lab.errors import CommandError
from llm_lab.jobs.protocol import JobName

type HubNamespace = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9]+([._-][A-Za-z0-9]+)*$", max_length=96)]


class HubDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: HubNamespace
    name: JobName

    @property
    def repo_id(self) -> str:
        return f"{self.namespace}/{self.name}"


def parse_hub_dataset(repo_id: str) -> HubDataset:
    match repo_id.split("/"):
        case [namespace, name]:
            return HubDataset(namespace=namespace, name=name)
        case _:
            raise ValueError(f"{repo_id!r} is not a repository id of the form <namespace>/<name>")


@dataclass(frozen=True)
class EmptyRepository:
    revision: str


@dataclass(frozen=True)
class StoredDataset:
    revision: str
    recipe_source: str
    metadata: RepositoryMetadata


type HubRepository = StoredDataset | EmptyRepository | None


class DatasetHub(Protocol):
    def read(self, dataset: HubDataset) -> HubRepository: ...

    def publish(
        self, dataset: HubDataset, repository: HubRepository, directory: Path, recipe: PortableRecipe
    ) -> str: ...

    def upload_seeds(self, upload: SeedUpload) -> None: ...


class HuggingFaceHub:
    def __init__(self, token: SecretStr | None) -> None:
        self.api = HfApi(token=None if token is None else token.get_secret_value())

    def read(self, dataset: HubDataset) -> HubRepository:
        try:
            revision = self.api.dataset_info(dataset.repo_id).sha
        except RepositoryNotFoundError:
            return None
        if revision is None:
            raise ValueError(f"The Hub did not report the revision of {dataset.repo_id}.")
        files = set(self.api.list_repo_files(dataset.repo_id, repo_type="dataset", revision=revision))
        if "recipe.yaml" in files:
            return StoredDataset(
                revision=revision,
                recipe_source=self.read_text(dataset, "recipe.yaml", revision),
                metadata=RepositoryMetadata.model_validate_json(self.read_text(dataset, "metadata.json", revision)),
            )
        if files <= {".gitattributes"}:
            return EmptyRepository(revision=revision)
        raise CommandError(f"The dataset {dataset.repo_id} has no recipe.yaml, so llm-lab cannot add rows to it.")

    def read_text(self, dataset: HubDataset, filename: str, revision: str) -> str:
        return Path(
            self.api.hf_hub_download(dataset.repo_id, filename, repo_type="dataset", revision=revision)
        ).read_text()

    def publish(self, dataset: HubDataset, repository: HubRepository, directory: Path, recipe: PortableRecipe) -> str:
        empty = RepositoryMetadata(actual_num_records=0, target_num_records=0, file_paths=RepositoryFilePaths(data=()))
        match repository:
            case StoredDataset(revision=revision, metadata=stored):
                pass
            case EmptyRepository(revision=revision):
                stored = empty
            case None:
                self.api.create_repo(dataset.repo_id, repo_type="dataset", private=True)
                revision, stored = None, empty
            case _:
                assert_never(repository)
        run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"
        files = plan_repository_files(dataset.name, run_id, read_generated_run(directory), stored, recipe)
        self.api.create_commit(
            dataset.repo_id,
            [CommitOperationAdd(path_in_repo=path, path_or_fileobj=content) for path, content in files.items()],
            commit_message=f"Add the rows of run {run_id}",
            repo_type="dataset",
            parent_commit=revision,
        )
        return f"https://huggingface.co/datasets/{dataset.repo_id}"

    def upload_seeds(self, upload: SeedUpload) -> None:
        self.api.create_bucket(upload.bucket_id, private=True, exist_ok=True)
        stored = {file.path for file in self.api.get_bucket_paths_info(upload.bucket_id, upload.files)}
        missing: list[tuple[str | Path | bytes, str]] = [
            (path, name) for name, path in upload.files.items() if name not in stored
        ]
        if missing:
            self.api.batch_bucket_files(upload.bucket_id, add=missing)
