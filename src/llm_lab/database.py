from importlib.resources import files

from alembic.config import Config
from sqlalchemy import Engine


def build_migration_config(engine: Engine) -> Config:
    config = Config()
    config.set_main_option("script_location", str(files("llm_lab").joinpath("migrations")))
    config.attributes["engine"] = engine
    return config
