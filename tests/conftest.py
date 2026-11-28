from dataclasses import dataclass, field

import pytest
import typer
from typer.testing import CliRunner


@dataclass
class Journal:
    events: list[str] = field(default_factory=list)
    opened: int = 0

    def open_resource(self, name: str) -> int:
        self.opened += 1
        self.events.append(f"open {name} {self.opened}")
        return self.opened


def run_cli(app: typer.Typer, arguments: list[str]) -> str:
    return CliRunner().invoke(app, arguments, catch_exceptions=False).output


def invoke_cli(app: typer.Typer, arguments: list[str]) -> tuple[int, str]:
    result = CliRunner().invoke(app, arguments, catch_exceptions=False)
    return result.exit_code, result.output


def follow_context_chain(error: BaseException) -> list[BaseException]:
    chain = [error]
    while (context := chain[-1].__context__) is not None and context not in chain:
        chain.append(context)
    return chain


@pytest.fixture
def journal() -> Journal:
    return Journal()
