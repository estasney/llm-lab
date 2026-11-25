import base64
import gzip

from llm_lab.datasets.hub import HubDataset
from llm_lab.datasets.recipe import PortableRecipe
from llm_lab.jobs.protocol import JobSpec
from llm_lab.keyring_settings import SecretName


def build_dataset_job(recipe: PortableRecipe, dataset: HubDataset, rows: int, image: str) -> JobSpec:
    artifacts_directory = "/workspace/artifacts"
    return JobSpec(
        name=dataset.name,
        image=image,
        command=("llm-lab-dataset-job", "--dataset", dataset.repo_id, "--rows", str(rows)),
        environment={
            "LLM_LAB_RECIPE": base64.b64encode(gzip.compress(recipe.recipe.source.encode(), mtime=0)).decode(),
            "LLM_LAB_ARTIFACTS_DIR": artifacts_directory,
        },
        secret_environment=(SecretName.hf_token,),
        disk_gb=recipe.recipe.job.disk_gb,
        ttl_seconds_after_finished=recipe.recipe.job.ttl_seconds_after_finished,
        placement=recipe.recipe.job.placement,
        artifacts_directory=artifacts_directory,
    )
