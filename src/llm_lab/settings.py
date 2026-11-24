from functools import cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_lab.dependencies import Depends


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_LAB_")

    datasets_dir: Path = Field(default_factory=lambda: Path.home() / ".local" / "share" / "llm-lab" / "datasets")


@cache
def get_settings() -> Settings:
    return Settings()


type SettingsDep = Annotated[Settings, Depends(get_settings)]


class StudioSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_LAB_STUDIO_")

    ssh_host: str
    ssh_port: int
    ssh_user: str = "root"
    ssh_identity_file: Path | None = None
    known_hosts_file: Path = Field(default_factory=lambda: Path.home() / ".local" / "share" / "llm-lab" / "known_hosts")
    port: int = 8000
    username: str = "unsloth"
    password: SecretStr
    request_timeout_seconds: float = 60
    poll_seconds: float = 5


@cache
def get_studio_settings() -> StudioSettings:
    return StudioSettings.model_validate({})


type StudioSettingsDep = Annotated[StudioSettings, Depends(get_studio_settings)]
