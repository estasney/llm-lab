from collections.abc import AsyncIterator, Callable, Iterator
from typing import Annotated

import pytest
import typer

from llm_lab.cli.dependent_typer import DependentTyper
from llm_lab.dependencies import Depends
from tests.conftest import Journal, follow_context_chain, run_cli


def return_value() -> int:
    return 7


def yield_value() -> Iterator[int]:
    yield 7


async def await_value() -> int:
    return 7


async def yield_awaited_value() -> AsyncIterator[int]:
    yield 7


def provide_unset_number() -> int:
    raise NotImplementedError


def provide_from_config(config: int) -> int:
    return config


type NumberDep = Annotated[int, Depends(return_value)]
type UnsetNumberDep = Annotated[int, Depends(provide_unset_number)]


def build_echo_app(provider: Callable[..., object]) -> DependentTyper:
    app = DependentTyper()

    @app.command()
    def echo_value(value: Annotated[int, Depends(provider)]) -> None:
        typer.echo(value)

    return app


@pytest.mark.parametrize("provider", [return_value, yield_value])
def test_provider_of_each_kind_supplies_its_value(provider: Callable[..., object]) -> None:
    """
    Arrange: a command that requests a value from a function provider or a generator provider.
    Act: run the command.
    Assert: the command printed the value of the provider.
    """
    app = build_echo_app(provider)

    actual = run_cli(app, [])

    assert actual == "7\n", f"Expected the command to receive 7 from {provider.__name__}, got {actual!r}."


def test_providers_open_before_and_close_after_the_command_in_reverse_order(journal: Journal) -> None:
    """
    Arrange: a command that requests a session provider, which requests a client provider.
    Act: run the command.
    Assert: the client opened first, the command ran after both providers opened, and the client closed last.
    """

    def open_client() -> Iterator[int]:
        number = journal.open_resource("client")
        yield number
        journal.events.append("close client")

    def open_session(client: Annotated[int, Depends(open_client)]) -> Iterator[int]:
        number = journal.open_resource("session")
        yield number
        journal.events.append("close session")

    app = DependentTyper()

    @app.command()
    def use_session(session: Annotated[int, Depends(open_session)]) -> None:
        journal.events.append("command")

    expected = ["open client 1", "open session 2", "command", "close session", "close client"]

    run_cli(app, [])

    assert journal.events == expected, f"Expected {expected}, got {journal.events}."


def test_provider_requested_two_times_in_one_command_runs_one_time(journal: Journal) -> None:
    """
    Arrange: a command that requests a provider directly and through a second provider.
    Act: run the command.
    Assert: the provider opened one time, and both requests received its value.
    """

    def provide_number() -> int:
        return journal.open_resource("number")

    def double_number(value: Annotated[int, Depends(provide_number)]) -> int:
        return value * 2

    app = DependentTyper()

    @app.command()
    def echo_values(
        value: Annotated[int, Depends(provide_number)], twice: Annotated[int, Depends(double_number)]
    ) -> None:
        typer.echo(f"{value} {twice}")

    actual = run_cli(app, [])

    assert actual == "1 2\n", f"Expected the two requests to share one value, got {actual!r}."
    assert journal.events == ["open number 1"], f"Expected the provider to open one time, got {journal.events}."


@pytest.mark.parametrize("error_type", [LookupError, ValueError, RuntimeError])
def test_provider_cleanup_receives_the_exception_of_the_command(error_type: type[Exception]) -> None:
    """
    Arrange: a command that raises an error and requests a generator provider that records what it catches.
    Act: run the command.
    Assert: the command raised the error, and the provider caught the same error.
    """
    caught: list[BaseException] = []

    def open_test_resource() -> Iterator[int]:
        try:
            yield 1
        except error_type as error:
            caught.append(error)
            raise

    app = DependentTyper()

    @app.command()
    def raise_error(value: Annotated[int, Depends(open_test_resource)]) -> None:
        raise error_type("no value")

    with pytest.raises(error_type) as raised:
        run_cli(app, [])

    assert caught == [raised.value], f"Expected the provider to receive {raised.value!r}, got {caught}."


def test_cleanup_errors_keep_the_error_of_the_command_in_the_context_chain() -> None:
    """
    Arrange: a command that fails and requests two generator providers whose cleanups fail.
    Act: run the command.
    Assert: the context chain of the raised error has the two cleanup errors and the error of the command.
    """

    def open_first_resource() -> Iterator[int]:
        try:
            yield 1
        finally:
            raise LookupError("first cleanup failed")

    def open_second_resource() -> Iterator[int]:
        try:
            yield 2
        finally:
            raise KeyError("second cleanup failed")

    app = DependentTyper()

    @app.command()
    def raise_error(
        first: Annotated[int, Depends(open_first_resource)], second: Annotated[int, Depends(open_second_resource)]
    ) -> None:
        raise ValueError("command failed")

    with pytest.raises(LookupError) as raised:
        run_cli(app, [])

    actual = [type(error) for error in follow_context_chain(raised.value)]
    assert actual == [LookupError, KeyError, ValueError], (
        f"Expected the cleanup errors in reverse order of opening, then the error of the command, got {actual}."
    )


def test_provider_that_suppresses_the_exception_of_its_command_is_rejected() -> None:
    """
    Arrange: a command that raises an error and requests a generator provider that suppresses it.
    Act: run the command.
    Assert: the run raises a RuntimeError in place of a silent success.
    """

    def open_test_resource() -> Iterator[int]:
        try:
            yield 1
        except LookupError:
            return

    app = DependentTyper()

    @app.command()
    def raise_error(value: Annotated[int, Depends(open_test_resource)]) -> None:
        raise LookupError

    with pytest.raises(RuntimeError, match="suppressed the exception"):
        run_cli(app, [])


@pytest.mark.parametrize("replacement", [return_value, yield_value])
def test_override_replaces_a_provider(replacement: Callable[..., object]) -> None:
    """
    Arrange: a command that requests a provider that fails, and an override of it with a provider of each kind.
    Act: run the command of the overridden app.
    Assert: the command printed the value of the replacement.
    """
    app = build_echo_app(provide_unset_number).with_dependency_overrides({provide_unset_number: replacement})

    actual = run_cli(app, [])

    assert actual == "7\n", f"Expected the override {replacement.__name__} to supply 7, got {actual!r}."


@pytest.mark.parametrize("depth", [1, 2], ids=["group", "group-of-a-group"])
def test_override_reaches_the_commands_of_added_groups(depth: int) -> None:
    """
    Arrange: a command that requests a provider that fails, in a group added at a given depth below the root app.
    Act: override the provider on the root app, and run the command.
    Assert: the command printed the value of the replacement.
    """
    app = build_echo_app(provide_unset_number)
    for _ in range(depth):
        parent = DependentTyper()
        parent.add_typer(app, name="group")

        @parent.callback()
        def select_group() -> None:
            pass

        app = parent
    overridden = app.with_dependency_overrides({provide_unset_number: return_value})

    actual = run_cli(overridden, [*["group"] * depth, "echo-value"])

    assert actual == "7\n", f"Expected the override to reach the command of the group, got {actual!r}."


def test_override_leaves_the_original_app_unchanged() -> None:
    """
    Arrange: an app with a command that requests a provider that fails, and an overridden copy of the app.
    Act: run the command of the original app.
    Assert: the original provider ran.
    """
    app = build_echo_app(provide_unset_number)
    app.with_dependency_overrides({provide_unset_number: return_value})

    with pytest.raises(NotImplementedError):
        run_cli(app, [])


@pytest.mark.parametrize(
    "replacement",
    [await_value, yield_awaited_value, provide_from_config],
    ids=["async-function", "async-generator", "plain-parameter"],
)
def test_invalid_override_is_rejected_when_it_is_applied(replacement: Callable[..., object]) -> None:
    """
    Arrange: an app with a command, and a replacement provider that is async or has a parameter without Depends.
    Act: apply the override.
    Assert: a TypeError is raised before any command runs.
    """
    with pytest.raises(TypeError):
        build_echo_app(provide_unset_number).with_dependency_overrides({provide_unset_number: replacement})


def test_type_alias_with_depends_supplies_its_value() -> None:
    """
    Arrange: a command whose parameter is annotated with a type alias that holds Depends.
    Act: run the command.
    Assert: the command printed the value of the provider.
    """
    app = DependentTyper()

    @app.command()
    def echo_value(value: NumberDep) -> None:
        typer.echo(value)

    actual = run_cli(app, [])

    assert actual == "7\n", f"Expected the alias to supply 7, got {actual!r}."


type NameArgument = Annotated[str, typer.Argument(help="Name to print.", default_factory=lambda: "from factory")]


def test_type_alias_with_a_cli_parameter_keeps_the_typer_configuration() -> None:
    """
    Arrange: a command whose parameter is annotated with a type alias that holds a Typer argument.
    Act: run the command without the argument, and show its help.
    Assert: the command printed the value of the default factory, and the help shows the help text of the argument.
    """
    app = DependentTyper()

    @app.command()
    def echo_name(name: NameArgument) -> None:
        typer.echo(name)

    actual = run_cli(app, [])
    help_text = run_cli(app, ["--help"])

    assert actual == "from factory\n", f"Expected the default factory to supply the name, got {actual!r}."
    assert "Name to print." in help_text, f"Expected the help text of the argument, got {help_text!r}."


def test_command_keeps_its_cli_parameters_and_hides_its_dependencies() -> None:
    """
    Arrange: a command with an argument, an option and a dependency.
    Act: run the command with the argument and the option, and read the parameters of its Click command.
    Assert: the command received the CLI values and the dependency, and only the CLI values are Click parameters.
    """
    app = DependentTyper()

    @app.command()
    def greet(name: str, value: NumberDep, *, loud: bool = False) -> None:
        typer.echo(f"{name} {value} {loud}")

    actual = run_cli(app, ["ada", "--loud"])
    parameters = {parameter.name for parameter in typer.main.get_command(app).params}

    assert actual == "ada 7 True\n", f"Expected the CLI values and the dependency, got {actual!r}."
    assert {"name", "loud"} <= parameters, f"Expected the CLI parameters name and loud, got {parameters}."
    assert "value" not in parameters, f"Expected the dependency to be hidden from the CLI, got {parameters}."


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
    @app.command()
    def use_value(value: Annotated[int, Depends(provide_from_config)]) -> None:
        pass


def register_two_markers(app: DependentTyper) -> None:
    @app.command()
    def use_value(value: Annotated[int, Depends(return_value), Depends(yield_value)]) -> None:
        pass


def register_variadic_parameter(app: DependentTyper) -> None:
    @app.command()
    def use_value(*values: Annotated[int, Depends(return_value)]) -> None:
        pass


def register_async_provider(app: DependentTyper) -> None:
    @app.command()
    def use_value(value: Annotated[int, Depends(await_value)]) -> None:
        pass


def register_async_command(app: DependentTyper) -> None:
    @app.command()
    async def use_value(value: NumberDep) -> None:
        pass


@pytest.mark.parametrize(
    ("register", "error"),
    [
        (register_dependency_cycle, ValueError),
        (register_plain_provider_parameter, TypeError),
        (register_two_markers, TypeError),
        (register_variadic_parameter, TypeError),
        (register_async_provider, TypeError),
        (register_async_command, TypeError),
    ],
)
def test_invalid_dependency_is_rejected_when_the_command_is_registered(
    register: Callable[[DependentTyper], None], error: type[Exception]
) -> None:
    """
    Arrange: a command that is async, whose dependencies form a cycle or are async, or that has a parameter
    that cannot receive a dependency.
    Act: register the command.
    Assert: the registration raises before the command can run.
    """
    with pytest.raises(error):
        register(DependentTyper())
