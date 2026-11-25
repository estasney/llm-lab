from typing import Annotated

import keyring
import typer
from keyring.errors import NoKeyringError

from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.errors import CommandError
from llm_lab.keyring_settings import SecretName
from llm_lab.secrets import find_secret_entry

app = DependentTyper(name="secrets", no_args_is_help=True, help="Manage keyring secrets.")


def prompt_for_secret_name() -> SecretName:
    return typer.prompt("Secret name", type=SecretName)


@app.command("set", help="Save a secret.")
def set_secret(
    name: Annotated[
        SecretName,
        typer.Argument(help="Secret name.", default_factory=prompt_for_secret_name),
    ],
    value: Annotated[
        str,
        typer.Option(
            prompt="Secret value",
            hide_input=True,
            confirmation_prompt=True,
            help="Secret value.",
        ),
    ],
) -> None:
    entry = find_secret_entry(name)
    try:
        keyring.set_password(entry.service, entry.username, value)
    except NoKeyringError as error:
        raise CommandError(
            f"This computer has no keyring backend. Set the environment variable LLM_LAB_{name} instead."
        ) from error
    typer.echo(f"Saved {name}.")
