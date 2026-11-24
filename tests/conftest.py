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


def follow_context_chain(error: BaseException) -> list[BaseException]:
    chain = [error]
    while (context := chain[-1].__context__) is not None and context not in chain:
        chain.append(context)
    return chain


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if item.get_closest_marker("xdist_group") is None:
            item.add_marker(pytest.mark.xdist_group("other_tests"))


@pytest.fixture
def journal() -> Journal:
    return Journal()
