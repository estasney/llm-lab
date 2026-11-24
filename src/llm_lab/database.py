from importlib.resources import files

from alembic.config import Config
from sqlalchemy import URL
from sqlalchemy.orm import DeclarativeBase

from llm_lab.settings import Settings


class Base(DeclarativeBase):
    pass


def build_database_url(settings: Settings) -> URL:
    return URL.create("sqlite", database=str(settings.job_database_path))


def build_migration_config(settings: Settings) -> Config:
    config = Config()
    config.set_main_option("script_location", str(files("llm_lab").joinpath("migrations")))
    config.set_main_option("sqlalchemy.url", build_database_url(settings).render_as_string(hide_password=False))
    return config
