from llm_lab.cli.dependent_typer import DependentTyper

app = DependentTyper(name="datasets", no_args_is_help=True)


@app.command("create")
def create_dataset() -> None:
    pass


@app.command("list")
def list_datasets() -> None:
    pass


@app.command("delete")
def delete_dataset() -> None:
    pass
