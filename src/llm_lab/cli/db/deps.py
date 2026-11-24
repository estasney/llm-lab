from collections.abc import Iterator
from typing import Annotated

from sqlalchemy.orm import Session

from llm_lab.database import require_current_schema
from llm_lab.dependencies import Depends
from llm_lab.settings import SettingsDep


def get_database_session(settings: SettingsDep) -> Iterator[Session]:
    require_current_schema(settings.database_engine)
    with Session(settings.database_engine) as session:
        yield session


type DatabaseSessionDep = Annotated[Session, Depends(get_database_session)]
