import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Annotated

import pytest
import typer
from typer.testing import CliRunner

from llm_lab.cli.dependent_typer import DependentCommand, DependentTyper
from llm_lab.dependencies import Depends


def return_value() -> int:
    return 7


async def await_value() -> int:
    return 7


def yield_value() -> Iterator[int]:
    yield 7


async def yield_awaited_value() -> AsyncIterator[int]:
    yield 7


def provide_unset_number() -> int:
    raise NotImplementedError


type NumberDep = Annotated[int, Depends(return_value)]
type ProvidedDep[ValueT] = Annotated[ValueT, Depends(return_value)]
type UnsetNumberDep = Annotated[int, Depends(provide_unset_number)]


def invoke(app: typer.Typer, arguments: list[str]) -> str:
    return CliRunner().invoke(app, arguments, catch_exceptions=False).output


def echo_number_app() -> DependentTyper:
    app = DependentTyper()

    @app.command()
    def echo_number(value: UnsetNumberDep) -> None:
        typer.echo(value)

    return app


@pytest.mark.parametrize("provider", [return_value, await_value, yield_value, yield_awaited_value])
def test_provider_of_each_kind_supplies_its_value(provider: Callable[..., object]) -> None:
    app = DependentTyper()

    @app.command()
    def echo_value(value: Annotated[int, Depends(provider)]) -> None:
        typer.echo(value)

    actual = invoke(app, [])

    assert actual == "7\n", f"Expected the command to receive 7 from {provider.__name__}, got {actual!r}."


def test_async_command_receives_its_dependency() -> None:
    app = DependentTyper()

    @app.command()
    async def echo_value(value: NumberDep) -> None:
        await asyncio.sleep(0)
        typer.echo(value)

    actual = invoke(app, [])

    assert actual == "7\n", f"Expected the async command to receive 7, got {actual!r}."


def test_command_keeps_its_cli_parameters_and_hides_its_dependencies() -> None:
    app = DependentTyper()

    @app.command()
    def greet(name: str, value: NumberDep, *, loud: bool = False) -> None:
        typer.echo(f"{name} {value} {loud}")

    actual = invoke(app, ["ada", "--loud"])
    parameters = {parameter.name for parameter in typer.main.get_command(app).params}

    assert actual == "ada 7 True\n", f"Expected the CLI values and the dependency, got {actual!r}."
    assert {"name", "loud"} <= parameters, f"Expected the CLI parameters name and loud, got {parameters}."
    assert "value" not in parameters, f"Expected the dependency to be hidden from the CLI, got {parameters}."


def test_provider_requested_two_times_in_one_command_runs_one_time() -> None:
    opened: list[str] = []

    def open_resource() -> int:
        opened.append("open")
        return len(opened)

    def provide_first(value: Annotated[int, Depends(open_resource)]) -> int:
        return value

    def provide_second(value: Annotated[int, Depends(open_resource)]) -> int:
        return value

    app = DependentTyper()

    @app.command()
    def echo_values(
        first: Annotated[int, Depends(provide_first)], second: Annotated[int, Depends(provide_second)]
    ) -> None:
        typer.echo(f"{first} {second}")

    actual = invoke(app, [])

    assert actual == "1 1\n", f"Expected both providers to receive the same value, got {actual!r}."
    assert opened == ["open"], f"Expected the shared provider to run one time, got {opened}."


@pytest.mark.parametrize("command_kind", ["sync", "async"])
def test_generator_provider_closes_after_its_command(command_kind: str) -> None:
    events: list[str] = []

    def open_resource() -> Iterator[int]:
        events.append("open")
        yield 1
        events.append("close")

    def record_sync(value: Annotated[int, Depends(open_resource)]) -> None:
        events.append("command")

    async def record_async(value: Annotated[int, Depends(open_resource)]) -> None:
        events.append("command")

    app = DependentTyper()
    app.command()(record_sync if command_kind == "sync" else record_async)

    invoke(app, [])

    assert events == ["open", "command", "close"], f"Expected open, command and close in order, got {events}."


@pytest.mark.parametrize("provider_kind", ["sync", "async"])
@pytest.mark.parametrize("error_type", [LookupError, ValueError, RuntimeError])
def test_provider_cleanup_receives_the_exception_of_the_command(
    provider_kind: str, error_type: type[Exception]
) -> None:
    caught: list[BaseException] = []

    def open_sync_resource() -> Iterator[int]:
        try:
            yield 1
        except error_type as error:
            caught.append(error)
            raise

    async def open_async_resource() -> AsyncIterator[int]:
        try:
            yield 1
        except error_type as error:
            caught.append(error)
            raise

    provider = open_sync_resource if provider_kind == "sync" else open_async_resource
    app = DependentTyper()

    @app.command()
    def raise_error(value: Annotated[int, Depends(provider)]) -> None:
        raise error_type("no value")

    with pytest.raises(error_type) as raised:
        invoke(app, [])

    assert caught == [raised.value], f"Expected the provider to receive {raised.value!r}, got {caught}."


@pytest.mark.parametrize("provider_kind", ["sync", "async"])
def test_provider_that_suppresses_the_exception_of_its_command_is_rejected(provider_kind: str) -> None:
    def open_sync_resource() -> Iterator[int]:
        try:
            yield 1
        except LookupError:
            return

    async def open_async_resource() -> AsyncIterator[int]:
        try:
            yield 1
        except LookupError:
            return

    provider = open_sync_resource if provider_kind == "sync" else open_async_resource
    app = DependentTyper()

    @app.command()
    def raise_error(value: Annotated[int, Depends(provider)]) -> None:
        raise LookupError

    with pytest.raises(RuntimeError, match="suppressed the exception"):
        invoke(app, [])


def test_cancellation_at_the_start_of_a_cleanup_does_not_skip_the_cleanup() -> None:
    closed: list[str] = []

    def open_resource() -> Iterator[int]:
        try:
            yield 1
        finally:
            closed.append("closed")

    async def cancel_and_fail(value: Annotated[int, Depends(open_resource)]) -> None:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("the command runs outside a task")
        task.cancel()
        raise LookupError

    with pytest.raises(LookupError):
        DependentCommand(cancel_and_fail, {})()

    assert closed == ["closed"], f"Expected the cleanup of the provider to run one time, got {closed}."


@pytest.mark.parametrize("replacement", [return_value, await_value, yield_value, yield_awaited_value])
def test_override_replaces_a_provider(replacement: Callable[..., object]) -> None:
    app = echo_number_app().with_dependency_overrides({provide_unset_number: replacement})

    actual = invoke(app, [])

    assert actual == "7\n", f"Expected the override {replacement.__name__} to supply 7, got {actual!r}."


def test_override_reaches_the_commands_of_an_added_group() -> None:
    app = DependentTyper()
    app.add_typer(echo_number_app(), name="numbers")

    @app.callback()
    def main() -> None:
        pass

    actual = invoke(app.with_dependency_overrides({provide_unset_number: return_value}), ["numbers", "echo-number"])

    assert actual == "7\n", f"Expected the override to reach the command of the group, got {actual!r}."


def test_override_leaves_the_original_app_unchanged() -> None:
    app = echo_number_app()
    app.with_dependency_overrides({provide_unset_number: return_value})

    with pytest.raises(NotImplementedError):
        invoke(app, [])


def test_type_alias_with_type_parameter_supplies_its_value() -> None:
    app = DependentTyper()

    @app.command()
    def echo_value(value: ProvidedDep[int]) -> None:
        typer.echo(value)

    actual = invoke(app, [])

    assert actual == "7\n", f"Expected the generic alias to supply 7, got {actual!r}."


def register_dependency_cycle(app: DependentTyper) -> None:
    def provide_first(value: int) -> int:
        return value

    def provide_second(value: Annotated[int, Depends(provide_first)]) -> int:
        return value

    provide_first.__annotations__["value"] = Annotated[int, Depends(provide_second)]

    @app.command()
    def use_value(value: Annotated[int, Depends(provide_first)]) -> None:
        pass


def register_plain_provider_parameter(app: DependentTyper) -> None:
    def provide_number(config: int) -> int:
        return config

    @app.command()
    def use_value(value: Annotated[int, Depends(provide_number)]) -> None:
        pass


def register_two_markers(app: DependentTyper) -> None:
    @app.command()
    def use_value(value: Annotated[int, Depends(return_value), Depends(await_value)]) -> None:
        pass


def register_variadic_parameter(app: DependentTyper) -> None:
    @app.command()
    def use_value(*values: Annotated[int, Depends(return_value)]) -> None:
        pass


@pytest.mark.parametrize(
    ("register", "error"),
    [
        (register_dependency_cycle, ValueError),
        (register_plain_provider_parameter, TypeError),
        (register_two_markers, TypeError),
        (register_variadic_parameter, TypeError),
    ],
)
def test_invalid_dependency_is_rejected_when_the_command_is_registered(
    register: Callable[[DependentTyper], None], error: type[Exception]
) -> None:
    with pytest.raises(error):
        register(DependentTyper())


def test_invalid_override_is_rejected_when_it_is_applied() -> None:
    def provide_number(config: int) -> int:
        return config

    with pytest.raises(TypeError):
        echo_number_app().with_dependency_overrides({provide_unset_number: provide_number})
