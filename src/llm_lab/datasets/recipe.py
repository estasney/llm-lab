from dataclasses import dataclass
from typing import Annotated

import yaml
from data_designer.config.data_designer_config import DataDesignerConfig
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from llm_lab.errors import CommandError
from llm_lab.jobs.protocol import ExplicitOffer, ResourceRequest


class RecipeJob(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    disk_gb: PositiveFloat
    ttl_seconds_after_finished: PositiveInt
    placement: Annotated[ResourceRequest | ExplicitOffer, Field(discriminator="kind")]


class RecipeFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job: RecipeJob
    vllm_args: tuple[str, ...] = ()
    data_designer: DataDesignerConfig


@dataclass(frozen=True)
class Recipe:
    source: str
    job: RecipeJob
    vllm_args: tuple[str, ...]
    data_designer: DataDesignerConfig
    served_model: str
    provider_name: str


def parse_recipe(source: str) -> Recipe:
    recipe_file = RecipeFile.model_validate(yaml.safe_load(source))
    model_configs = recipe_file.data_designer.model_configs or []
    match sorted({config.model for config in model_configs}), sorted({config.provider for config in model_configs}):
        case [served_model], [provider_name]:
            return Recipe(
                source=source,
                job=recipe_file.job,
                vllm_args=recipe_file.vllm_args,
                data_designer=recipe_file.data_designer,
                served_model=served_model,
                provider_name=provider_name,
            )
        case models, providers:
            raise CommandError(
                "The model configs of a recipe must name one model on one provider. "
                f"They name the models {models} and the providers {providers}."
            )
