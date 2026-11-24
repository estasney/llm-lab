import base64
import gzip
from typing import assert_never

from llm_lab.datasets.hub import HubDataset
from llm_lab.datasets.recipe import Recipe
from llm_lab.jobs.protocol import ExplicitOffer, JobSpec, ResourceRequest
from llm_lab.keyring_settings import SecretName


def require_image_cuda(
    placement: ResourceRequest | ExplicitOffer, image_min_cuda: float
) -> ResourceRequest | ExplicitOffer:
    match placement:
        case ResourceRequest():
            floors = [floor for floor in (placement.min_cuda, image_min_cuda) if floor is not None]
            return placement.model_copy(update={"min_cuda": max(floors)})
        case ExplicitOffer():
            return placement
        case _:
            assert_never(placement)


def build_dataset_job(recipe: Recipe, dataset: HubDataset, rows: int, image: str, image_min_cuda: float) -> JobSpec:
    return JobSpec(
        name=dataset.name,
        image=image,
        command=("llm-lab-dataset-job", "--dataset", dataset.repo_id, "--rows", str(rows)),
        environment={"LLM_LAB_RECIPE": base64.b64encode(gzip.compress(recipe.source.encode(), mtime=0)).decode()},
        secret_environment=(SecretName.hf_token,),
        disk_gb=recipe.job.disk_gb,
        ttl_seconds_after_finished=recipe.job.ttl_seconds_after_finished,
        placement=require_image_cuda(recipe.job.placement, image_min_cuda),
    )
