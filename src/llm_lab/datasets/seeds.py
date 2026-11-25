import glob
import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from llm_lab.datasets.recipe import PortableRecipe, Recipe, parse_portable_recipe


@dataclass(frozen=True)
class SeedUpload:
    bucket_id: str
    files: Mapping[str, Path]


@dataclass(frozen=True)
class SeedPlan:
    recipe: PortableRecipe
    upload: SeedUpload | None


def plan_seed_upload(recipe: Recipe, bucket_id: str) -> SeedPlan:
    match yaml.safe_load(recipe.source):
        case {"data_designer": {"seed_config": {"source": {"seed_type": "local", "path": str(pattern)}}}} as document:
            files = {Path(path).name: Path(path) for path in sorted(glob.glob(os.path.expanduser(pattern)))}
            digest = hashlib.sha256()
            for name, path in files.items():
                with path.open("rb") as file:
                    digest.update(f"{name}\0{hashlib.file_digest(file, 'sha256').hexdigest()}\n".encode())
            folder = f"seeds/{digest.hexdigest()}"
            document["data_designer"]["seed_config"]["source"] = {
                "seed_type": "hf",
                "path": f"buckets/{bucket_id}/{folder}/{Path(pattern).name}",
            }
            return SeedPlan(
                recipe=parse_portable_recipe(yaml.safe_dump(document, sort_keys=False, allow_unicode=True)),
                upload=SeedUpload(bucket_id, {f"{folder}/{name}": path for name, path in files.items()}),
            )
        case _:
            return SeedPlan(recipe=parse_portable_recipe(recipe.source), upload=None)
