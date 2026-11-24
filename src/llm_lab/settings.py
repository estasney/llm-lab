from functools import cache
from pathlib import Path
from typing import Annotated

from pydantic import SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from llm_lab.dependencies import Depends
from llm_lab.keyring_settings import KeyringEntry, KeyringSettingsSource


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_LAB_")

    job_database_path: Path = Path(".llm-lab/jobs.sqlite3")
    hf_token: Annotated[SecretStr | None, KeyringEntry(service="llm-lab", username="HF_TOKEN")] = None
    vast_api_key: Annotated[SecretStr | None, KeyringEntry(service="llm-lab", username="VAST_API_KEY")] = None
    vast_api_url: str = "https://console.vast.ai/api"
    vast_request_attempts: int = 5
    vast_retry_delay_seconds: float = 2
    vast_offer_limit: int = 10
    vast_log_tail_lines: int = 200

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, KeyringSettingsSource(settings_cls)


@cache
def get_settings() -> Settings:
    return Settings()


type SettingsDep = Annotated[Settings, Depends(get_settings)]
