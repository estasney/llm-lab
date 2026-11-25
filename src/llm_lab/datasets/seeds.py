import glob
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import yaml
from huggingface_hub import HfApi

from llm_lab.datasets.hub import HubDataset
from llm_lab.datasets.recipe import (
    HubSeedFiles,
    LocalSeedFiles,
    Recipe,
    RecipeFile,
    RecipeSeed,
    SeedLocation,
    parse_recipe,
)
from llm_lab.errors import CommandError
from llm_lab.settings import Settings


class SeedStore(Protocol):
    def store(self, pattern: Path, dataset: HubDataset) -> HubSeedFiles: ...


def name_seed_folder(digests: Mapping[str, str]) -> str:
    listing = "".join(f"{name}\0{digest}\n" for name, digest in sorted(digests.items()))
    return f"seeds/{hashlib.sha256(listing.encode()).hexdigest()}"


class HuggingFaceSeedStore:
    def __init__(self, settings: Settings) -> None:
        self.api = HfApi(token=None if settings.hf_token is None else settings.hf_token.get_secret_value())
        self.bucket_name = settings.seed_bucket_name

    def store(self, pattern: Path, dataset: HubDataset) -> HubSeedFiles:
        files = {Path(path).name: Path(path) for path in sorted(glob.glob(str(pattern)))}
        if not files:
            raise CommandError(f"No seed files match {pattern}.")
        digests: dict[str, str] = {}
        for name, path in files.items():
            with path.open("rb") as file:
                digests[name] = hashlib.file_digest(file, "sha256").hexdigest()
        folder = name_seed_folder(digests)
        bucket_id = f"{dataset.namespace}/{self.bucket_name}"
        targets = {f"{folder}/{name}": path for name, path in files.items()}
        self.api.create_bucket(bucket_id, private=True, exist_ok=True)
        stored = {file.path for file in self.api.get_bucket_paths_info(bucket_id, targets)}
        missing: list[tuple[str | Path | bytes, str]] = [
            (path, target) for target, path in targets.items() if target not in stored
        ]
        if missing:
            self.api.batch_bucket_files(bucket_id, add=missing)
        return HubSeedFiles(kind=SeedLocation.hub, path=f"buckets/{bucket_id}/{folder}/{pattern.name}")


def resolve_recipe(source: str, directory: Path, dataset: HubDataset, store: SeedStore) -> Recipe:
    document = yaml.safe_load(source)
    match RecipeFile.model_validate(document).seed:
        case RecipeSeed(files=LocalSeedFiles(path=path)) as seed:
            document["seed"] = seed.stored_in(store.store(directory / path, dataset)).model_dump(mode="json")
            return parse_recipe(yaml.safe_dump(document, sort_keys=False, allow_unicode=True))
        case _:
            return parse_recipe(source)
