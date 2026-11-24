from typing import Annotated

import keyring
import typer

from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.jobs.secrets import find_secret_entry
from llm_lab.keyring_settings import SecretName

app = DependentTyper(
    name="secrets", no_args_is_help=True, help="Store the secrets that llm-lab reads from the keyring."
)


def prompt_for_secret_name() -> SecretName:
    return typer.prompt("Secret name", type=SecretName)


@app.command("set", help="Write one secret to the keyring of the operating system.")
def set_secret(
    name: Annotated[
        SecretName,
        typer.Argument(
            help="Name of the secret. A prompt asks for it when it is omitted.", default_factory=prompt_for_secret_name
        ),
    ],
    value: Annotated[
        str,
        typer.Option(
            prompt="Secret value",
            hide_input=True,
            confirmation_prompt=True,
            help="Value of the secret. Omit this option to enter the value at a hidden prompt.",
        ),
    ],
) -> None:
    entry = find_secret_entry(name)
    keyring.set_password(entry.service, entry.username, value)
    typer.echo(f"Saved {name} to the keyring service {entry.service}.")
