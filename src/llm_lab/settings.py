from functools import cache
from pathlib import Path
from typing import Annotated

from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_lab.dependencies import Depends


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_LAB_")

    job_database_path: Path = Path(".llm-lab/jobs.sqlite3")
    keyring_service_name: str = "llm-lab"


@cache
def get_settings() -> Settings:
    return Settings()


type SettingsDep = Annotated[Settings, Depends(get_settings)]
