from typing import Annotated

import typer
from alembic import command

from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.database import build_migration_config
from llm_lab.settings import SettingsDep

app = DependentTyper(name="db", no_args_is_help=True, help="Manage the job database.")


def prompt_for_revision_message() -> str:
    return typer.prompt("Revision message", type=str)


@app.command("upgrade", help="Apply migrations.")
def upgrade_database(settings: SettingsDep) -> None:
    command.upgrade(build_migration_config(settings.database_engine), "head")
    typer.echo("Database is current.")


@app.command("revision", help="Create a migration.")
def create_revision(
    message: Annotated[
        str,
        typer.Argument(
            help="Migration description.",
            default_factory=prompt_for_revision_message,
        ),
    ],
    settings: SettingsDep,
) -> None:
    command.revision(build_migration_config(settings.database_engine), message=message, autogenerate=True)
