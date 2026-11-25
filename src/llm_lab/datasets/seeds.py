import glob
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from huggingface_hub import HfApi

from llm_lab.datasets.hub import HubDataset
from llm_lab.errors import CommandError
from llm_lab.settings import Settings


class SeedStore(Protocol):
    def store(self, pattern: Path, dataset: HubDataset) -> str: ...


@dataclass(frozen=True)
class SeedFolder:
    name: str
    files: Mapping[str, Path]


def read_seed_folder(pattern: Path) -> SeedFolder:
    files = {Path(path).name: Path(path) for path in glob.glob(str(pattern))}
    if not files:
        raise CommandError(f"No seed files match {pattern}.")
    listing = hashlib.sha256()
    for name, path in sorted(files.items()):
        with path.open("rb") as file:
            listing.update(f"{name}\0{hashlib.file_digest(file, 'sha256').hexdigest()}\n".encode())
    return SeedFolder(name=f"seeds/{listing.hexdigest()}", files=files)


class HuggingFaceSeedStore:
    def __init__(self, settings: Settings) -> None:
        self.api = HfApi(token=None if settings.hf_token is None else settings.hf_token.get_secret_value())
        self.bucket_name = settings.seed_bucket_name

    def upload(self, folder: SeedFolder, bucket_id: str) -> None:
        targets = {f"{folder.name}/{name}": path for name, path in folder.files.items()}
        self.api.create_bucket(bucket_id, private=True, exist_ok=True)
        stored = {file.path for file in self.api.get_bucket_paths_info(bucket_id, targets)}
        missing: list[tuple[str | Path | bytes, str]] = [
            (path, target) for target, path in targets.items() if target not in stored
        ]
        if missing:
            self.api.batch_bucket_files(bucket_id, add=missing)

    def store(self, pattern: Path, dataset: HubDataset) -> str:
        folder = read_seed_folder(pattern)
        bucket_id = f"{dataset.namespace}/{self.bucket_name}"
        self.upload(folder, bucket_id)
        return f"buckets/{bucket_id}/{folder.name}/{pattern.name}"
