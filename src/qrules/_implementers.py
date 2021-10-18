"""A collection of implementation tools to can be used accross all modules."""

from typing import Any, Callable, Type, TypeVar

import attr

try:
    from IPython.lib.pretty import PrettyPrinter
except ImportError:
    PrettyPrinter = Any


_DecoratedClass = TypeVar("_DecoratedClass")


def implement_pretty_repr() -> Callable[
    [Type[_DecoratedClass]], Type[_DecoratedClass]
]:
    """Implement a pretty :code:`repr` in a `attr` decorated class."""

    def decorator(
        decorated_class: Type[_DecoratedClass],
    ) -> Type[_DecoratedClass]:
        if not attr.has(decorated_class):
            raise TypeError(
                "Can only implement a pretty repr for a class created with attrs"
            )

        def repr_pretty(self: Any, p: PrettyPrinter, cycle: bool) -> None:
            class_name = type(self).__name__
            if cycle:
                p.text(f"{class_name}(...)")
            else:
                with p.group(indent=2, open=f"{class_name}("):
                    for field in attr.fields(type(self)):
                        if not field.init:
                            continue
                        value = getattr(self, field.name)
                        p.breakable()
                        p.text(f"{field.name}=")
                        p.pretty(value)
                        p.text(",")
                p.breakable()
                p.text(")")

        # pylint: disable=protected-access
        decorated_class._repr_pretty_ = repr_pretty  # type: ignore[attr-defined]
        return decorated_class

    return decorator
