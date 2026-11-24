import inspect
from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from typing import Annotated, TypeAliasType, get_args, get_origin


@dataclass(frozen=True)
class Depends:
    dependency: Callable[..., object]


@dataclass(frozen=True)
class ResolvedProvider:
    request: Depends
    open_value: Callable[..., AbstractContextManager[object]]
    parameters: Mapping[str, Depends]


@dataclass(frozen=True)
class CallPlan:
    function: Callable[..., object]
    dependencies: Mapping[str, Depends]
    providers: tuple[ResolvedProvider, ...]
    signature: inspect.Signature


def expand_alias(annotation: object) -> object:
    match annotation:
        case TypeAliasType():
            return expand_alias(annotation.__value__)
        case _:
            return annotation


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


def build_value_opener(function: Callable[..., object]) -> Callable[..., AbstractContextManager[object]]:
    if inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function):
        raise TypeError(f"dependency {function!r} is async, and commands run synchronously")
    if inspect.isgeneratorfunction(function):
        return contextmanager(function)

    @contextmanager
    def open_returned_value(**arguments: object) -> Generator[object]:
        yield function(**arguments)

    return open_returned_value


def resolve_provider(
    request: Depends, resolved: dict[Depends, ResolvedProvider], path: tuple[Depends, ...]
) -> ResolvedProvider:
    if request in resolved:
        return resolved[request]
    if request in path:
        raise ValueError(f"dependency cycle through {request.dependency!r}")
    parameters = read_provider_parameters(request.dependency)
    for child in parameters.values():
        resolve_provider(child, resolved, (*path, request))
    resolved[request] = ResolvedProvider(request, build_value_opener(request.dependency), parameters)
    return resolved[request]


def plan_call(function: Callable[..., object]) -> CallPlan:
    signature = inspect.signature(function, eval_str=True)
    requests = {parameter.name: read_depends(function, parameter) for parameter in signature.parameters.values()}
    dependencies = {name: request for name, request in requests.items() if request is not None}
    resolved: dict[Depends, ResolvedProvider] = {}
    for request in dependencies.values():
        resolve_provider(request, resolved, ())
    return CallPlan(
        function,
        dependencies,
        tuple(resolved.values()),
        signature.replace(
            parameters=[parameter for parameter in signature.parameters.values() if requests[parameter.name] is None]
        ),
    )


def execute_plan(plan: CallPlan, arguments: Mapping[str, object]) -> object:
    with ExitStack() as exits:
        values: dict[Depends, object] = {}
        for provider in plan.providers:
            provider_arguments = {name: values[request] for name, request in provider.parameters.items()}
            values[provider.request] = exits.enter_context(provider.open_value(**provider_arguments))
        return plan.function(**arguments, **{name: values[request] for name, request in plan.dependencies.items()})
