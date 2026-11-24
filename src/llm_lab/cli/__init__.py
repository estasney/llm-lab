from llm_lab.cli import datasets, db, jobs, secrets
from llm_lab.cli.dependent_typer import DependentTyper

app = DependentTyper(name="llm-lab", no_args_is_help=True)
app.add_typer(datasets.app)
app.add_typer(db.app)
app.add_typer(jobs.app)
app.add_typer(secrets.app)


@app.callback()
def main() -> None:
    pass
