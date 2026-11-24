import typer

from llm_lab.cli import datasets

app = typer.Typer(name="llm-lab", no_args_is_help=True)
app.add_typer(datasets.app)


@app.callback()
def main() -> None:
    pass
