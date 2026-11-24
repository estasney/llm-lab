from functools import cache
from typing import Annotated

from pydantic_settings import BaseSettings

from llm_lab.dependencies import Depends


class Settings(BaseSettings):
    pass


@cache
def get_settings() -> Settings:
    return Settings()


type SettingsDep = Annotated[Settings, Depends(get_settings)]
