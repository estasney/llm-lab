from alembic import command

from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.database import build_migration_config
from llm_lab.settings import SettingsDep

app = DependentTyper(name="db", no_args_is_help=True)


@app.command("upgrade")
def upgrade_database(settings: SettingsDep) -> None:
    settings.job_database_path.parent.mkdir(parents=True, exist_ok=True)
    command.upgrade(build_migration_config(settings), "head")


@app.command("revision")
def create_revision(message: str, settings: SettingsDep) -> None:
    command.revision(build_migration_config(settings), message=message, autogenerate=True)
