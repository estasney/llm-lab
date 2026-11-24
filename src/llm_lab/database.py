from importlib.resources import files

from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine

from llm_lab.errors import CommandError


def build_migration_config(engine: Engine) -> Config:
    config = Config()
    config.set_main_option("script_location", str(files("llm_lab").joinpath("migrations")))
    config.attributes["engine"] = engine
    return config


def read_revision(engine: Engine) -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def require_current_schema(engine: Engine) -> None:
    head = ScriptDirectory.from_config(build_migration_config(engine)).get_current_head()
    if read_revision(engine) != head:
        raise CommandError("The job database is not current. Run `llm-lab db upgrade`.")
