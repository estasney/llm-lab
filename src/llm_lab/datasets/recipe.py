from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal, Self, assert_never

import yaml
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.seed import IndexRange, PartitionBlock, SamplingStrategy, SeedConfig
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, field_validator

from llm_lab.errors import CommandError
from llm_lab.jobs.protocol import ExplicitOffer, ResourceRequest
from llm_lab.keyring_settings import SecretName


class RecipeJob(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    disk_gb: PositiveFloat
    ttl_seconds_after_finished: PositiveInt
    placement: Annotated[ResourceRequest | ExplicitOffer, Field(discriminator="kind")]


class SeedLocation(StrEnum):
    local = "local"
    hub = "hub"


class LocalSeedFiles(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[SeedLocation.local]
    path: str = Field(
        min_length=1,
        description="Seed file, or a wildcard such as seeds/*.parquet, relative to the folder of the recipe file. "
        "llm-lab uploads the files to a Hub bucket before a run.",
    )


class HubSeedFiles(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[SeedLocation.hub]
    path: str = Field(
        min_length=1,
        description="Seed files on the Hub, such as buckets/<namespace>/<bucket>/seeds/*.parquet "
        "or datasets/<namespace>/<dataset>/data/*.parquet.",
    )


class SeedSampling(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sampling_strategy: SamplingStrategy = SamplingStrategy.ORDERED
    selection_strategy: IndexRange | PartitionBlock | None = None


class HubSeed(SeedSampling):
    files: HubSeedFiles

    def as_seed_config(self) -> SeedConfig:
        return SeedConfig.model_validate(
            {
                "source": {"seed_type": "hf", "path": self.files.path, "token": SecretName.hf_token},
                "sampling_strategy": self.sampling_strategy,
                "selection_strategy": self.selection_strategy,
            }
        )


class RecipeSeed(SeedSampling):
    files: Annotated[LocalSeedFiles | HubSeedFiles, Field(discriminator="kind")]

    def stored_in(self, files: HubSeedFiles) -> HubSeed:
        return HubSeed(
            files=files, sampling_strategy=self.sampling_strategy, selection_strategy=self.selection_strategy
        )

    def on_hub(self) -> HubSeed:
        match self.files:
            case HubSeedFiles() as files:
                return self.stored_in(files)
            case LocalSeedFiles():
                raise CommandError("The recipe names local seed files. Upload them before the recipe runs.")
            case _:
                assert_never(self.files)


class RecipeFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job: RecipeJob
    vllm_args: tuple[str, ...] = ()
    seed: RecipeSeed | None = None
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
    seed: HubSeed | None
    data_designer: DataDesignerConfig
    served_model: str
    provider_name: str

    def generates_rows_like(self, other: Self) -> bool:
        return (self.data_designer, self.vllm_args, self.seed) == (other.data_designer, other.vllm_args, other.seed)


def parse_recipe(source: str) -> Recipe:
    recipe_file = RecipeFile.model_validate(yaml.safe_load(source))
    seed = None if recipe_file.seed is None else recipe_file.seed.on_hub()
    model_configs = recipe_file.data_designer.model_configs or []
    match sorted({config.model for config in model_configs}), sorted({config.provider for config in model_configs}):
        case [served_model], [provider_name]:
            return Recipe(
                source=source,
                job=recipe_file.job,
                vllm_args=recipe_file.vllm_args,
                seed=seed,
                data_designer=recipe_file.data_designer,
                served_model=served_model,
                provider_name=provider_name,
            )
        case models, providers:
            raise CommandError(
                "The model configs of a recipe must name one model on one provider. "
                f"They name the models {models} and the providers {providers}."
            )
