from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal, Self

import yaml
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.seed import IndexRange, PartitionBlock, SamplingStrategy
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, field_validator

from llm_lab.errors import CommandError
from llm_lab.jobs.protocol import ExplicitOffer, ResourceRequest


class RecipeJob(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    disk_gb: PositiveFloat
    ttl_seconds_after_finished: PositiveInt
    placement: Annotated[ResourceRequest | ExplicitOffer, Field(discriminator="kind")]


class SeedLocation(StrEnum):
    local = "local"
    hub = "hub"


class SeedSampling(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sampling_strategy: SamplingStrategy = SamplingStrategy.ORDERED
    selection_strategy: IndexRange | PartitionBlock | None = None


class LocalSeed(SeedSampling):
    kind: Literal[SeedLocation.local]
    path: str = Field(
        min_length=1,
        description="Seed file, or a wildcard such as seeds/*.parquet, relative to the current folder. "
        "A job reads a copy that llm-lab uploads to a Hub bucket.",
    )


class HubSeed(SeedSampling):
    kind: Literal[SeedLocation.hub]
    path: str = Field(
        min_length=1,
        description="Seed files on the Hub, such as buckets/<namespace>/<bucket>/seeds/*.parquet "
        "or datasets/<namespace>/<dataset>/data/*.parquet.",
    )


class RecipeFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job: RecipeJob
    vllm_args: tuple[str, ...] = ()
    seed: Annotated[LocalSeed | HubSeed, Field(discriminator="kind")] | None = None
    data_designer: DataDesignerConfig

    @field_validator("data_designer")
    @classmethod
    def reject_data_designer_seeds(cls, config: DataDesignerConfig) -> DataDesignerConfig:
        if config.seed_config is not None:
            raise ValueError("put the seed in the seed section of the recipe, not in data_designer")
        return config


@dataclass(frozen=True)
class Recipe:
    source: str
    job: RecipeJob
    vllm_args: tuple[str, ...]
    seed: LocalSeed | HubSeed | None
    data_designer: DataDesignerConfig
    served_model: str
    provider_name: str

    def generates_rows_like(self, other: Self) -> bool:
        return (self.data_designer, self.vllm_args, self.seed) == (other.data_designer, other.vllm_args, other.seed)


def parse_recipe(source: str) -> Recipe:
    recipe_file = RecipeFile.model_validate(yaml.safe_load(source))
    model_configs = recipe_file.data_designer.model_configs or []
    match sorted({config.model for config in model_configs}), sorted({config.provider for config in model_configs}):
        case [served_model], [provider_name]:
            return Recipe(
                source=source,
                job=recipe_file.job,
                vllm_args=recipe_file.vllm_args,
                seed=recipe_file.seed,
                data_designer=recipe_file.data_designer,
                served_model=served_model,
                provider_name=provider_name,
            )
        case models, providers:
            raise CommandError(
                "The model configs of a recipe must name one model on one provider. "
                f"They name the models {models} and the providers {providers}."
            )
