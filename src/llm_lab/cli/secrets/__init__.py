import keyring
import typer

from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.jobs.secrets import find_secret_entry

app = DependentTyper(name="secrets", no_args_is_help=True)


@app.command("set")
def set_secret(name: str) -> None:
    entry = find_secret_entry(name)
    keyring.set_password(entry.service, entry.username, typer.prompt(f"Value of {name}", hide_input=True))
