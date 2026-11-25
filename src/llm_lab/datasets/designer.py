from pathlib import Path
from typing import Protocol, assert_never

from data_designer.config.config_builder import BuilderConfig, DataDesignerConfigBuilder
from data_designer.config.errors import InvalidConfigError
from data_designer.config.models import ModelProvider
from data_designer.config.seed import SeedConfig
from data_designer.config.seed_source import HuggingFaceSeedSource, LocalFileSeedSource
from data_designer.interface import DataDesigner

from llm_lab.datasets.recipe import HubSeed, LocalSeed, Recipe
from llm_lab.errors import CommandError
from llm_lab.keyring_settings import SecretName
from llm_lab.secrets import SettingsSecretSource
from llm_lab.settings import Settings


class DatasetDesigner(Protocol):
    def validate(self, recipe: Recipe) -> None: ...

    def create(self, recipe: Recipe, name: str, rows: int) -> Path: ...


def build_seed_config(seed: LocalSeed | HubSeed) -> SeedConfig:
    match seed:
        case LocalSeed(path=path):
            source = LocalFileSeedSource(path=path)
        case HubSeed(path=path):
            source = HuggingFaceSeedSource(path=path, token=SecretName.hf_token)
        case _:
            assert_never(seed)
    return SeedConfig.model_validate(
        {
            "source": source,
            "sampling_strategy": seed.sampling_strategy,
            "selection_strategy": seed.selection_strategy,
        }
    )


class DataDesignerRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def open_designer(self, recipe: Recipe) -> DataDesigner:
        return DataDesigner(
            self.settings.artifacts_dir,
            model_providers=[ModelProvider(name=recipe.provider_name, endpoint=f"{self.settings.vllm_base_url}/v1")],
            secret_resolver=SettingsSecretSource(self.settings),
        )

    def build_config(self, recipe: Recipe) -> DataDesignerConfigBuilder:
        seed_config = None if recipe.seed is None else build_seed_config(recipe.seed)
        data_designer = recipe.data_designer.model_copy(update={"seed_config": seed_config})
        return DataDesignerConfigBuilder.from_config(BuilderConfig(data_designer=data_designer))

    def validate(self, recipe: Recipe) -> None:
        try:
            self.open_designer(recipe).validate(self.build_config(recipe))
        except InvalidConfigError as error:
            raise CommandError(f"Data Designer rejected the recipe: {error}") from error

    def create(self, recipe: Recipe, name: str, rows: int) -> Path:
        results = self.open_designer(recipe).create(self.build_config(recipe), num_records=rows, dataset_name=name)
        return results.artifact_storage.base_dataset_path
