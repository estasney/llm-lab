from collections.abc import Iterator
from typing import Annotated

from sqlalchemy import Engine

from llm_lab.dependencies import Depends
from llm_lab.settings import SettingsDep


def get_database_engine(settings: SettingsDep) -> Iterator[Engine]:
    engine = settings.create_database_engine()
    try:
        yield engine
    finally:
        engine.dispose()


type DatabaseEngineDep = Annotated[Engine, Depends(get_database_engine)]
