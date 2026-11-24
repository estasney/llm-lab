import asyncio
import inspect
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Annotated

import pytest
import typer
from hypothesis import given
from hypothesis import strategies as st

from llm_lab.cli.dependent_typer import DependentCommand, DependentTyper
from llm_lab.dependencies import Depends, execute_plan, plan_call
from tests.conftest import Journal, follow_context_chain, run_cli


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


def register_sync_echo(app: DependentTyper, provider: Callable[..., object]) -> None:
    @app.command()
    def echo_value(value: Annotated[int, Depends(provider)]) -> None:
        typer.echo(value)


def register_async_echo(app: DependentTyper, provider: Callable[..., object]) -> None:
    @app.command()
    async def echo_value(value: Annotated[int, Depends(provider)]) -> None:
        await asyncio.sleep(0)
        typer.echo(value)


def build_echo_app(provider: Callable[..., object]) -> DependentTyper:
    app = DependentTyper()
    register_sync_echo(app, provider)
    return app


def build_generator_provider(journal: Journal) -> Callable[..., object]:
    def open_resource() -> Iterator[int]:
        number = journal.open_resource("resource")
        yield number
        journal.events.append(f"close {number}")

    return open_resource


def build_async_generator_provider(journal: Journal) -> Callable[..., object]:
    async def open_resource() -> AsyncIterator[int]:
        number = journal.open_resource("resource")
        yield number
        journal.events.append(f"close {number}")

    return open_resource


def register_sync_recorder(app: DependentTyper, provider: Callable[..., object], journal: Journal) -> None:
    @app.command()
    def record(number: Annotated[int, Depends(provider)]) -> None:
        journal.events.append(f"command {number}")


def register_async_recorder(app: DependentTyper, provider: Callable[..., object], journal: Journal) -> None:
    @app.command()
    async def record(number: Annotated[int, Depends(provider)]) -> None:
        journal.events.append(f"command {number}")


@pytest.mark.parametrize("register", [register_sync_echo, register_async_echo], ids=["sync-command", "async-command"])
@pytest.mark.parametrize("provider", [return_value, await_value, yield_value, yield_awaited_value])
def test_provider_of_each_kind_supplies_its_value(
    provider: Callable[..., object], register: Callable[[DependentTyper, Callable[..., object]], None]
) -> None:
    """
    Arrange: a sync or async command that requests a value from a provider of each kind.
    Act: run the command.
    Assert: the command printed the value of the provider.
    """
    app = DependentTyper()
    register(app, provider)

    actual = run_cli(app, [])

    assert actual == "7\n", f"Expected the command to receive 7 from {provider.__name__}, got {actual!r}."


@pytest.mark.parametrize(
    "register", [register_sync_recorder, register_async_recorder], ids=["sync-command", "async-command"]
)
@pytest.mark.parametrize(
    "build_provider",
    [build_generator_provider, build_async_generator_provider],
    ids=["sync-generator", "async-generator"],
)
def test_generator_provider_opens_before_and_closes_after_its_command(
    journal: Journal,
    build_provider: Callable[[Journal], Callable[..., object]],
    register: Callable[[DependentTyper, Callable[..., object], Journal], None],
) -> None:
    """
    Arrange: a sync or async command that requests a value from a sync or async generator provider.
    Act: run the command.
    Assert: the provider opened before the command and closed after it.
    """
    app = DependentTyper()
    register(app, build_provider(journal), journal)
    expected = ["open resource 1", "command 1", "close 1"]

    run_cli(app, [])

    assert journal.events == expected, f"Expected {expected}, got {journal.events}."


def test_providers_close_in_the_reverse_order_of_opening(journal: Journal) -> None:
    """
    Arrange: a command that requests a session provider, which requests a client provider.
    Act: run the command.
    Assert: the client opened first and closed last.
    """

    def open_client() -> Iterator[int]:
        number = journal.open_resource("client")
        yield number
        journal.events.append("close client")

    async def open_session(client: Annotated[int, Depends(open_client)]) -> AsyncIterator[int]:
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

    async def provide_number() -> int:
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


def test_sync_command_and_sync_providers_run_on_the_calling_thread() -> None:
    """
    Arrange: a sync command that requests a sync generator provider, and both record their thread.
    Act: run the command.
    Assert: the provider and the command ran on the thread that called the command.
    """
    threads: list[threading.Thread] = []

    def open_resource() -> Iterator[int]:
        threads.append(threading.current_thread())
        yield 1

    app = DependentTyper()

    @app.command()
    def record_thread(value: Annotated[int, Depends(open_resource)]) -> None:
        threads.append(threading.current_thread())

    expected = [threading.current_thread()] * 2

    run_cli(app, [])

    assert threads == expected, (
        f"Expected the provider and the command on the calling thread, got {threads}. "
        "A sync command must not run in a worker thread, so that an interrupt stops it."
    )


def test_sync_provider_of_an_async_command_runs_outside_the_event_loop_thread() -> None:
    """
    Arrange: an async command that requests a sync provider, and both record their thread.
    Act: run the command.
    Assert: the provider ran on a different thread from the async command.
    """
    threads: list[threading.Thread] = []

    def provide_number() -> int:
        threads.append(threading.current_thread())
        return 1

    app = DependentTyper()

    @app.command()
    async def record_thread(value: Annotated[int, Depends(provide_number)]) -> None:
        threads.append(threading.current_thread())

    run_cli(app, [])

    assert len(threads) == 2, f"Expected a thread for the provider and for the command, got {threads}."
    assert threads[0] is not threads[1], (
        f"Expected the provider on a worker thread, got {threads}. A sync provider must not block the event loop."
    )


@pytest.mark.parametrize("error_type", [LookupError, ValueError, RuntimeError])
@pytest.mark.parametrize("provider_kind", ["sync", "async"])
def test_provider_cleanup_receives_the_exception_of_the_command(
    provider_kind: str, error_type: type[Exception]
) -> None:
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

    async def open_awaited_resource() -> AsyncIterator[int]:
        try:
            yield 1
        except error_type as error:
            caught.append(error)
            raise

    provider = open_test_resource if provider_kind == "sync" else open_awaited_resource
    app = DependentTyper()

    @app.command()
    def raise_error(value: Annotated[int, Depends(provider)]) -> None:
        raise error_type("no value")

    with pytest.raises(error_type) as raised:
        run_cli(app, [])

    assert caught == [raised.value], f"Expected the provider to receive {raised.value!r}, got {caught}."


@pytest.mark.parametrize("provider_kind", ["sync", "async"])
def test_cleanup_error_keeps_the_error_of_the_command(provider_kind: str) -> None:
    """
    Arrange: a command that fails and requests a generator provider whose cleanup fails.
    Act: run the command.
    Assert: the run raises the cleanup error, and the error of the command is in its context chain.
    """

    def open_test_resource() -> Iterator[int]:
        try:
            yield 1
        finally:
            raise LookupError("cleanup failed")

    async def open_awaited_resource() -> AsyncIterator[int]:
        try:
            yield 1
        finally:
            raise LookupError("cleanup failed")

    provider = open_test_resource if provider_kind == "sync" else open_awaited_resource
    app = DependentTyper()

    @app.command()
    def raise_error(value: Annotated[int, Depends(provider)]) -> None:
        raise ValueError("command failed")

    with pytest.raises(LookupError) as raised:
        run_cli(app, [])

    chain = follow_context_chain(raised.value)
    assert any(isinstance(error, ValueError) for error in chain), (
        f"Expected the error of the command in the context chain, got {chain}."
    )


@pytest.mark.parametrize("command_kind", ["sync", "async"])
def test_cleanup_errors_of_two_providers_are_both_in_the_context_chain(command_kind: str) -> None:
    """
    Arrange: a sync or async command that requests two generator providers whose cleanups fail.
    Act: run the command.
    Assert: the context chain of the raised error has the two cleanup errors.
    """

    def open_test_resource() -> Iterator[int]:
        try:
            yield 1
        finally:
            raise LookupError("first cleanup failed")

    async def open_awaited_resource() -> AsyncIterator[int]:
        try:
            yield 2
        finally:
            raise ValueError("second cleanup failed")

    def use_values_sync(
        first: Annotated[int, Depends(open_test_resource)], second: Annotated[int, Depends(open_awaited_resource)]
    ) -> None:
        pass

    async def use_values_async(
        first: Annotated[int, Depends(open_test_resource)], second: Annotated[int, Depends(open_awaited_resource)]
    ) -> None:
        pass

    app = DependentTyper()
    app.command()(use_values_sync if command_kind == "sync" else use_values_async)

    with pytest.raises((LookupError, ValueError)) as raised:
        run_cli(app, [])

    actual = sorted(type(error).__name__ for error in follow_context_chain(raised.value))
    assert actual == ["LookupError", "ValueError"], f"Expected the two cleanup errors in the chain, got {actual}."


@pytest.mark.parametrize("provider_kind", ["sync", "async"])
def test_provider_that_suppresses_the_exception_of_its_command_is_rejected(provider_kind: str) -> None:
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

    async def open_awaited_resource() -> AsyncIterator[int]:
        try:
            yield 1
        except LookupError:
            return

    provider = open_test_resource if provider_kind == "sync" else open_awaited_resource
    app = DependentTyper()

    @app.command()
    def raise_error(value: Annotated[int, Depends(provider)]) -> None:
        raise LookupError

    with pytest.raises(RuntimeError, match="suppressed the exception"):
        run_cli(app, [])


def test_cancellation_at_the_start_of_a_cleanup_does_not_skip_the_cleanup() -> None:
    """
    Arrange: an async command with a sync generator provider that cancels its own task and then fails.
    Act: run the command, so the cancellation arrives when the cleanup of the provider starts.
    Assert: the error of the command is raised, and the cleanup ran.
    """
    closed: list[str] = []

    def open_test_resource() -> Iterator[int]:
        try:
            yield 1
        finally:
            closed.append("closed")

    async def cancel_and_fail(value: Annotated[int, Depends(open_test_resource)]) -> None:
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
    """
    Arrange: a command that requests a provider that fails, and an override of it with a provider of each kind.
    Act: run the command of the overridden app.
    Assert: the command printed the value of the replacement.
    """
    app = build_echo_app(provide_unset_number).with_dependency_overrides({provide_unset_number: replacement})

    actual = run_cli(app, [])

    assert actual == "7\n", f"Expected the override {replacement.__name__} to supply 7, got {actual!r}."


@pytest.mark.parametrize("depth", [1, 2, 3], ids=["group", "group-of-a-group", "group-of-a-group-of-a-group"])
def test_override_reaches_the_commands_of_added_groups(depth: int) -> None:
    """
    Arrange: a command that requests a provider that fails, in a group added at a given depth below the root app.
    Act: override the provider on the root app, and run the command.
    Assert: the command printed the value of the replacement.
    """
    app = build_echo_app(provide_unset_number)
    for _ in range(depth):
        parent = DependentTyper(no_args_is_help=True)
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


def test_invalid_override_is_rejected_when_it_is_applied() -> None:
    """
    Arrange: an app with a command, and a replacement provider with a parameter that has no Depends annotation.
    Act: apply the override.
    Assert: a TypeError is raised before any command runs.
    """

    def provide_number(config: int) -> int:
        return config

    with pytest.raises(TypeError):
        build_echo_app(provide_unset_number).with_dependency_overrides({provide_unset_number: provide_number})


@pytest.mark.parametrize("annotation", [NumberDep, ProvidedDep[int]], ids=["alias", "alias-with-type-parameter"])
def test_type_alias_with_depends_supplies_its_value(annotation: object) -> None:
    """
    Arrange: a command whose parameter is annotated with a type alias that holds Depends.
    Act: run the command.
    Assert: the command printed the value of the provider.
    """
    app = DependentTyper()

    def echo_value(value: int) -> None:
        typer.echo(value)

    echo_value.__annotations__["value"] = annotation
    app.command()(echo_value)

    actual = run_cli(app, [])

    assert actual == "7\n", f"Expected the alias to supply 7, got {actual!r}."


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


def register_variadic_keyword_parameter(app: DependentTyper) -> None:
    @app.command()
    def use_value(**values: Annotated[int, Depends(return_value)]) -> None:
        pass


@pytest.mark.parametrize(
    ("register", "error"),
    [
        (register_dependency_cycle, ValueError),
        (register_plain_provider_parameter, TypeError),
        (register_two_markers, TypeError),
        (register_variadic_parameter, TypeError),
        (register_variadic_keyword_parameter, TypeError),
    ],
)
def test_invalid_dependency_is_rejected_when_the_command_is_registered(
    register: Callable[[DependentTyper], None], error: type[Exception]
) -> None:
    """
    Arrange: a command whose dependencies form a cycle, or a parameter that cannot receive a dependency.
    Act: register the command.
    Assert: the registration raises before the command can run.
    """
    with pytest.raises(error):
        register(DependentTyper())


@dataclass(frozen=True)
class ProviderGraph:
    requests: tuple[frozenset[int], ...]
    command_requests: frozenset[int]

    def reachable(self) -> frozenset[int]:
        found: set[int] = set()
        pending = list(self.command_requests)
        while pending:
            index = pending.pop()
            if index not in found:
                found.add(index)
                pending.extend(self.requests[index])
        return frozenset(found)


@st.composite
def provider_graphs(draw: st.DrawFn) -> ProviderGraph:
    count = draw(st.integers(min_value=1, max_value=6))
    requests = tuple(
        frozenset(draw(st.lists(st.integers(min_value=0, max_value=index - 1), max_size=3)) if index else ())
        for index in range(count)
    )
    command_requests = draw(st.frozensets(st.integers(min_value=0, max_value=count - 1), min_size=1))
    return ProviderGraph(requests, command_requests)


class RecordingFunction:
    def __init__(self, index: int, requests: dict[int, Depends], opened: list[int]) -> None:
        self.index = index
        self.opened = opened
        self.__signature__ = inspect.Signature(
            [
                inspect.Parameter(f"value_{child}", inspect.Parameter.KEYWORD_ONLY, annotation=Annotated[int, request])
                for child, request in requests.items()
            ]
        )

    def __call__(self, **values: int) -> int:
        self.opened.append(self.index)
        return self.index


@given(graph=provider_graphs())
def test_each_reachable_provider_opens_one_time_after_the_providers_it_requests(graph: ProviderGraph) -> None:
    """
    Arrange: providers that request lower-numbered providers, and a command that requests a subset of them.
    Act: plan and run the command.
    Assert: each reachable provider opened one time and after each provider it requests, and no other provider opened.
    """
    opened: list[int] = []
    requests: list[Depends] = []
    for index, children in enumerate(graph.requests):
        provider = RecordingFunction(index, {child: requests[child] for child in children}, opened)
        requests.append(Depends(provider))
    command = RecordingFunction(-1, {index: requests[index] for index in graph.command_requests}, opened)

    execute_plan(plan_call(command, {}), {})

    expected = graph.reachable()
    assert sorted(opened[:-1]) == sorted(expected), f"Expected each of {sorted(expected)} one time, got {opened}."
    assert opened[-1] == -1, f"Expected the command to run after its providers, got {opened}."
    for position, index in enumerate(opened[:-1]):
        assert graph.requests[index] <= set(opened[:position]), (
            f"Expected the providers {sorted(graph.requests[index])} to open before {index}, got {opened}."
        )
