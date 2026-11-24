from functools import cache
from pathlib import Path
from typing import Annotated

from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_lab.dependencies import Depends


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_LAB_")

    container_executable: str = "docker"
    job_records_directory: Path = Path(".llm-lab/jobs")


@cache
def get_settings() -> Settings:
    return Settings()


type SettingsDep = Annotated[Settings, Depends(get_settings)]
