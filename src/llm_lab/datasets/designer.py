from pathlib import Path
from typing import Protocol

from data_designer.config.config_builder import BuilderConfig, DataDesignerConfigBuilder
from data_designer.config.errors import InvalidConfigError
from data_designer.config.models import ModelProvider
from data_designer.interface import DataDesigner

from llm_lab.datasets.recipe import Recipe
from llm_lab.errors import CommandError
from llm_lab.jobs.secrets import SettingsSecretSource
from llm_lab.settings import Settings


class DatasetDesigner(Protocol):
    def validate(self, recipe: Recipe) -> None: ...

    def create(self, recipe: Recipe, name: str, rows: int) -> Path: ...


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
        seed_config = None if recipe.seed is None else recipe.seed.as_seed_config()
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
