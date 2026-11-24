import asyncio
import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Mapping
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    AsyncExitStack,
    ExitStack,
    asynccontextmanager,
    contextmanager,
)
from dataclasses import dataclass
from types import TracebackType
from typing import Annotated, Protocol, Self, TypeAliasType, TypeVar, get_args, get_origin, runtime_checkable

from typing_extensions import TypeForm


@dataclass(frozen=True)
class Depends:
    dependency: Callable[..., object]


@runtime_checkable
class SubscriptableHint(Protocol):
    def __getitem__(self, arguments: tuple[object, ...]) -> Self: ...


def substitute_type_arguments[HintT](hint: HintT, arguments: Mapping[object, HintT]) -> HintT:
    match hint:
        case TypeVar():
            return arguments.get(hint, hint)
        case SubscriptableHint():
            parameters = tuple[object, ...](getattr(hint, "__parameters__", ()))
            if not parameters:
                return hint
            return hint[tuple(arguments.get(parameter, parameter) for parameter in parameters)]
        case _:
            return hint


def expand_alias[HintT](annotation: HintT) -> HintT:
    match annotation:
        case TypeAliasType():
            return expand_alias(annotation.__value__)
        case _:
            origin = get_origin(annotation)
            match origin:
                case TypeAliasType():
                    arguments: dict[object, TypeForm[object]] = dict(
                        zip(origin.__type_params__, get_args(annotation), strict=True)
                    )
                    return expand_alias(substitute_type_arguments(origin.__value__, arguments))
                case _:
                    return annotation


@dataclass(frozen=True)
class ResolvedProvider:
    request: Depends
    function: Callable[..., object]
    parameters: Mapping[str, Depends]


@dataclass(frozen=True)
class CallPlan:
    function: Callable[..., object]
    dependencies: Mapping[str, Depends]
    providers: tuple[ResolvedProvider, ...]
    signature: inspect.Signature
    runs_async: bool


def is_async(function: Callable[..., object]) -> bool:
    return inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function)


def read_depends(function: Callable[..., object], parameter: inspect.Parameter) -> Depends | None:
    annotation = expand_alias(parameter.annotation)
    metadata = get_args(annotation)[1:] if get_origin(annotation) is Annotated else ()
    markers = [item for item in metadata if isinstance(item, Depends)]
    match markers:
        case []:
            return None
        case [Depends() as request]:
            if parameter.kind not in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}:
                raise TypeError(f"parameter {parameter.name!r} of {function!r} cannot receive a dependency by keyword")
            return request
        case _:
            raise TypeError(f"parameter {parameter.name!r} of {function!r} has more than one Depends annotation")


def read_provider_parameters(function: Callable[..., object]) -> dict[str, Depends]:
    parameters = inspect.signature(function, eval_str=True).parameters.values()
    requests = {parameter.name: read_depends(function, parameter) for parameter in parameters}
    match [name for name, request in requests.items() if request is None]:
        case []:
            return {name: request for name, request in requests.items() if request is not None}
        case missing:
            raise TypeError(f"parameters {missing!r} of dependency {function!r} are not annotated with Depends")


def resolve_provider(
    request: Depends,
    overrides: Mapping[Callable[..., object], Callable[..., object]],
    resolved: dict[Depends, ResolvedProvider],
    path: tuple[Depends, ...],
) -> ResolvedProvider:
    if request in resolved:
        return resolved[request]
    if request in path:
        raise ValueError(f"dependency cycle through {request.dependency!r}")
    function = overrides.get(request.dependency, request.dependency)
    parameters = read_provider_parameters(function)
    for child in parameters.values():
        resolve_provider(child, overrides, resolved, (*path, request))
    resolved[request] = ResolvedProvider(request, function, parameters)
    return resolved[request]


def plan_call(
    function: Callable[..., object], overrides: Mapping[Callable[..., object], Callable[..., object]]
) -> CallPlan:
    signature = inspect.signature(function, eval_str=True)
    requests = {parameter.name: read_depends(function, parameter) for parameter in signature.parameters.values()}
    dependencies = {name: request for name, request in requests.items() if request is not None}
    resolved: dict[Depends, ResolvedProvider] = {}
    for request in dependencies.values():
        resolve_provider(request, overrides, resolved, ())
    providers = tuple(resolved.values())
    return CallPlan(
        function,
        dependencies,
        providers,
        signature.replace(
            parameters=[parameter for parameter in signature.parameters.values() if requests[parameter.name] is None]
        ),
        runs_async=is_async(function) or any(is_async(provider.function) for provider in providers),
    )


def open_value(function: Callable[..., object]) -> Callable[..., AbstractContextManager[object]]:
    if inspect.isgeneratorfunction(function):
        return contextmanager(function)

    @contextmanager
    def open_returned_value(**arguments: object) -> Generator[object]:
        yield function(**arguments)

    return open_returned_value


class CommandExitStack(ExitStack[bool | None]):
    def __exit__(
        self, error_type: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> bool:
        return bool(super().__exit__(error_type, error, traceback))


def execute_plan(plan: CallPlan, arguments: Mapping[str, object]) -> object:
    with CommandExitStack() as exits:
        values: dict[Depends, object] = {}
        for provider in plan.providers:
            provider_arguments = {name: values[request] for name, request in provider.parameters.items()}
            values[provider.request] = exits.enter_context(open_value(provider.function)(**provider_arguments))
        return plan.function(**arguments, **{name: values[request] for name, request in plan.dependencies.items()})
    raise RuntimeError(f"a dependency of {plan.function!r} suppressed the exception of its command")


async def finish_despite_cancellation[ResultT](work: Awaitable[ResultT]) -> ResultT:
    future = asyncio.ensure_future(work)
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("finish_despite_cancellation works only inside a task")
    requests = 0
    while not future.done():
        try:
            await asyncio.wait([future])
        except asyncio.CancelledError:
            requests += 1
            task.uncancel()
    if not future.cancelled() and (error := future.exception()) is not None:
        context = error.__context__
        try:
            raise error
        finally:
            error.__context__ = context
    result = future.result()
    for _ in range(requests):
        task.cancel()
    return result


class ShieldedExitStack(AsyncExitStack[bool | None]):
    async def __aexit__(
        self, error_type: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> bool:
        return bool(await finish_despite_cancellation(super().__aexit__(error_type, error, traceback)))


async def run_in_thread(function: Callable[..., object], *arguments: object, **named_arguments: object) -> object:
    await asyncio.sleep(0)  # deliver a pending cancellation
    return await finish_despite_cancellation(asyncio.to_thread(function, *arguments, **named_arguments))


@dataclass(frozen=True)
class ThreadedContext(AbstractAsyncContextManager[object]):
    manager: AbstractContextManager[object]

    async def __aenter__(self) -> object:
        return await run_in_thread(self.manager.__enter__)

    async def __aexit__(
        self, error_type: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        return await finish_despite_cancellation(asyncio.to_thread(self.manager.__exit__, error_type, error, traceback))


async def call_without_blocking(
    function: Callable[..., object], *arguments: object, **named_arguments: object
) -> object:
    if inspect.iscoroutinefunction(function):
        result = function(*arguments, **named_arguments)
    else:
        result = await run_in_thread(function, *arguments, **named_arguments)
    return await result if inspect.isawaitable(result) else result


def open_value_without_blocking(function: Callable[..., object]) -> Callable[..., AbstractAsyncContextManager[object]]:
    if inspect.isasyncgenfunction(function):
        return asynccontextmanager(function)
    if inspect.isgeneratorfunction(function):
        open_in_context = contextmanager(function)

        def open_generator_value(**arguments: object) -> ThreadedContext:
            return ThreadedContext(open_in_context(**arguments))

        return open_generator_value

    @asynccontextmanager
    async def open_returned_value(**arguments: object) -> AsyncGenerator[object]:
        yield await call_without_blocking(function, **arguments)

    return open_returned_value


async def execute_plan_without_blocking(plan: CallPlan, arguments: Mapping[str, object]) -> object:
    async with ShieldedExitStack() as exits:
        values: dict[Depends, object] = {}
        for provider in plan.providers:
            provider_arguments = {name: values[request] for name, request in provider.parameters.items()}
            values[provider.request] = await exits.enter_async_context(
                open_value_without_blocking(provider.function)(**provider_arguments)
            )
        return await call_without_blocking(
            plan.function, **arguments, **{name: values[request] for name, request in plan.dependencies.items()}
        )
    raise RuntimeError(f"a dependency of {plan.function!r} suppressed the exception of its command")
