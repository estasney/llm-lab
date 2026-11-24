from alembic import command

from llm_lab.cli.db.deps import DatabaseEngineDep
from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.database import build_migration_config

app = DependentTyper(name="db", no_args_is_help=True)


@app.command("upgrade")
def upgrade_database(engine: DatabaseEngineDep) -> None:
    command.upgrade(build_migration_config(engine), "head")


@app.command("revision")
def create_revision(message: str, engine: DatabaseEngineDep) -> None:
    command.revision(build_migration_config(engine), message=message, autogenerate=True)
