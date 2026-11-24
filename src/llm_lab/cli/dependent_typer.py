from collections.abc import Callable, Mapping
from copy import copy
from functools import update_wrapper
from typing import Self

import typer
from typer.core import TyperCommand

from llm_lab.dependencies import execute_plan, plan_call
from llm_lab.errors import CommandError


class DependentCommand:
    function: Callable[..., object]

    def __init__(
        self, function: Callable[..., object], overrides: Mapping[Callable[..., object], Callable[..., object]]
    ) -> None:
        self.function = function
        self.plan = plan_call(function, overrides)
        self.__signature__ = self.plan.signature
        update_wrapper(self, function)

    def __call__(self, **arguments: object) -> object:
        try:
            return execute_plan(self.plan, arguments)
        except CommandError as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(1) from error


class DependentTyper(typer.Typer):
    def command[CommandT: Callable[..., object]](
        self,
        name: str | None = None,
        *,
        cls: type[TyperCommand] | None = None,
        context_settings: dict[object, object] | None = None,
        help: str | None = None,
        epilog: str | None = None,
        short_help: str | None = None,
        options_metavar: str | None = None,
        add_help_option: bool = True,
        no_args_is_help: bool = False,
        hidden: bool = False,
        deprecated: bool = False,
        rich_help_panel: str | None = None,
    ) -> Callable[[CommandT], CommandT]:
        register = super().command(
            name,
            cls=cls,
            context_settings=context_settings,
            help=help,
            epilog=epilog,
            short_help=short_help,
            options_metavar=options_metavar,
            add_help_option=add_help_option,
            no_args_is_help=no_args_is_help,
            hidden=hidden,
            deprecated=deprecated,
            rich_help_panel=rich_help_panel,
        )

        def decorator(function: CommandT) -> CommandT:
            register(DependentCommand(function, {}))
            return function

        return decorator

    def with_dependency_overrides(self, overrides: Mapping[Callable[..., object], Callable[..., object]]) -> Self:
        overridden = copy(self)
        overridden.registered_commands = [copy(info) for info in self.registered_commands]
        for info in overridden.registered_commands:
            if isinstance(info.callback, DependentCommand):
                info.callback = DependentCommand(info.callback.function, overrides)
        overridden.registered_groups = [copy(info) for info in self.registered_groups]
        for info in overridden.registered_groups:
            match info.typer_instance:
                case DependentTyper() as group:
                    info.typer_instance = group.with_dependency_overrides(overrides)
                case _:
                    pass
        return overridden
