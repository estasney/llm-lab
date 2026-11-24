from collections.abc import Callable
from functools import update_wrapper

import typer
from typer.core import TyperCommand

from llm_lab.dependencies import execute_plan, plan_call


class DependentCommand:
    def __init__(self, function: Callable[..., object]) -> None:
        self.plan = plan_call(function)
        self.__signature__ = self.plan.signature
        update_wrapper(self, function)

    def __call__(self, **arguments: object) -> object:
        return execute_plan(self.plan, arguments)


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
            register(DependentCommand(function))
            return function

        return decorator
