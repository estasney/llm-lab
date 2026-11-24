from llm_lab.cli import datasets
from llm_lab.cli.dependent_typer import DependentTyper

app = DependentTyper(name="llm-lab", no_args_is_help=True)
app.add_typer(datasets.app)


@app.callback()
def main() -> None:
    pass
