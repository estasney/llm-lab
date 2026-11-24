import typer

app = typer.Typer(name="llm-lab", no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command()
def datasets() -> None:
    pass
