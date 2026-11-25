from pathlib import Path
from typing import Protocol

from data_designer.config.config_builder import BuilderConfig, DataDesignerConfigBuilder
from data_designer.config.errors import InvalidConfigError
from data_designer.config.models import ModelProvider
from data_designer.engine.secret_resolver import (
    CompositeResolver,
    EnvironmentResolver,
    PlaintextResolver,
    SecretResolutionError,
)
from data_designer.interface import DataDesigner
from pydantic import SecretStr

from llm_lab.datasets.recipe import Recipe
from llm_lab.errors import CommandError
from llm_lab.keyring_settings import SecretName
from llm_lab.settings import Settings


class DatasetDesigner(Protocol):
    def validate(self, recipe: Recipe) -> None: ...

    def create(self, recipe: Recipe, name: str, rows: int) -> Path: ...


class HubTokenResolver:
    def __init__(self, token: SecretStr | None) -> None:
        self.token = token

    def resolve(self, secret: str) -> str:
        match secret, self.token:
            case SecretName.hf_token, SecretStr() as token:
                return token.get_secret_value()
            case _:
                raise SecretResolutionError(f"llm-lab holds no secret named {secret!r}")


class DataDesignerRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def open_designer(self, recipe: Recipe) -> DataDesigner:
        return DataDesigner(
            self.settings.artifacts_dir,
            model_providers=[ModelProvider(name=recipe.provider_name, endpoint=f"{self.settings.vllm_base_url}/v1")],
            secret_resolver=CompositeResolver(
                [HubTokenResolver(self.settings.hf_token), EnvironmentResolver(), PlaintextResolver()]
            ),
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
