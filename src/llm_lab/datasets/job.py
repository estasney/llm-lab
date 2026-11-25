import base64
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

import yaml

from llm_lab.datasets.hub import HubDataset
from llm_lab.datasets.recipe import HubSeed, LocalSeed, Recipe, SeedLocation
from llm_lab.datasets.seeds import SeedStore
from llm_lab.jobs.protocol import JobSpec
from llm_lab.keyring_settings import SecretName


@dataclass(frozen=True)
class JobRecipe(Recipe):
    seed: HubSeed | None


def prepare_job_recipe(recipe: Recipe, dataset: HubDataset, seeds: SeedStore) -> JobRecipe:
    match recipe.seed:
        case LocalSeed() as local:
            seed = HubSeed(
                kind=SeedLocation.hub,
                path=seeds.store(Path(local.path), dataset),
                sampling_strategy=local.sampling_strategy,
                selection_strategy=local.selection_strategy,
            )
            document = yaml.safe_load(recipe.source)
            document["seed"] = seed.model_dump(mode="json")
            source = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
        case HubSeed() | None as seed:
            source = recipe.source
        case _:
            assert_never(recipe.seed)
    return JobRecipe(
        source=source,
        job=recipe.job,
        vllm_args=recipe.vllm_args,
        seed=seed,
        rows_per_batch=recipe.rows_per_batch,
        data_designer=recipe.data_designer,
        served_model=recipe.served_model,
        provider_name=recipe.provider_name,
    )


def build_dataset_job(recipe: JobRecipe, dataset: HubDataset, rows: int, image: str) -> JobSpec:
    artifacts_directory = "/workspace/artifacts"
    return JobSpec(
        name=dataset.name,
        image=image,
        command=("llm-lab-dataset-job", "--dataset", dataset.repo_id, "--rows", str(rows)),
        environment={
            "LLM_LAB_RECIPE": base64.b64encode(gzip.compress(recipe.source.encode(), mtime=0)).decode(),
            "LLM_LAB_ARTIFACTS_DIR": artifacts_directory,
        },
        secret_environment=(SecretName.hf_token,),
        disk_gb=recipe.job.disk_gb,
        ttl_seconds_after_finished=recipe.job.ttl_seconds_after_finished,
        placement=recipe.job.placement,
        artifacts_directory=artifacts_directory,
    )
