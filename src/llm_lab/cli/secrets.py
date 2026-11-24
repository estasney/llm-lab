from typing import Annotated

import keyring
import typer
from pydantic import SecretStr

from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.dependencies import Depends
from llm_lab.jobs.secrets import KeyringSecretSource
from llm_lab.settings import SettingsDep

app = DependentTyper(name="secrets", no_args_is_help=True)


def get_secret_source(settings: SettingsDep) -> KeyringSecretSource:
    return KeyringSecretSource(keyring.get_keyring(), settings.keyring_service_name)


type SecretSourceDep = Annotated[KeyringSecretSource, Depends(get_secret_source)]


@app.command("set")
def set_secret(name: str, secret_source: SecretSourceDep) -> None:
    secret_source.write(name, SecretStr(typer.prompt(f"Value of {name}", hide_input=True)))
