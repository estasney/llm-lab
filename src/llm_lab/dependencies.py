import inspect
from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Annotated, TypeAliasType, get_args, get_origin


@dataclass(frozen=True)
class Depends:
    dependency: Callable[..., object]


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


def reject_async(function: Callable[..., object]) -> None:
    if inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function):
        raise TypeError(f"{function!r} is async, and commands and their dependencies run synchronously")


def read_depends(function: Callable[..., object], parameter: inspect.Parameter) -> Depends | None:
    annotation = parameter.annotation
    match annotation:
        case TypeAliasType():
            annotation = annotation.__value__
        case _:
            pass
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


def expand_type_alias(parameter: inspect.Parameter) -> inspect.Parameter:
    match parameter.annotation:
        case TypeAliasType() as alias:
            return parameter.replace(annotation=alias.__value__)
        case _:
            return parameter


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
    reject_async(function)
    parameters = read_provider_parameters(function)
    for child in parameters.values():
        resolve_provider(child, overrides, resolved, (*path, request))
    resolved[request] = ResolvedProvider(request, function, parameters)
    return resolved[request]


def plan_call(
    function: Callable[..., object], overrides: Mapping[Callable[..., object], Callable[..., object]]
) -> CallPlan:
    reject_async(function)
    signature = inspect.signature(function, eval_str=True)
    requests = {parameter.name: read_depends(function, parameter) for parameter in signature.parameters.values()}
    dependencies = {name: request for name, request in requests.items() if request is not None}
    resolved: dict[Depends, ResolvedProvider] = {}
    for request in dependencies.values():
        resolve_provider(request, overrides, resolved, ())
    return CallPlan(
        function,
        dependencies,
        tuple(resolved.values()),
        signature.replace(
            parameters=[
                expand_type_alias(parameter)
                for parameter in signature.parameters.values()
                if requests[parameter.name] is None
            ]
        ),
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
