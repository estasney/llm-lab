from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.settings import SettingsDep

app = DependentTyper(name="datasets", no_args_is_help=True)


@app.command("create")
def create_dataset(settings: SettingsDep) -> None:
    pass


@app.command("list")
def list_datasets(settings: SettingsDep) -> None:
    pass


@app.command("delete")
def delete_dataset(settings: SettingsDep) -> None:
    pass
