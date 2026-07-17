"""A collection of implementation tools to can be used accross all modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import attrs

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

_DecoratedClass = TypeVar("_DecoratedClass")


class PrettyPrinter(Protocol):
    def breakable(self) -> None: ...

    def group(
        self,
        indent: int = 0,
        open: str = "",  # ruff:ignore[builtin-argument-shadowing]
    ) -> AbstractContextManager: ...

    def pretty(self, obj: Any) -> None: ...

    def text(self, obj: str) -> None: ...


def implement_pretty_repr(
    decorated_class: type[_DecoratedClass],
) -> type[_DecoratedClass]:
    """Implement a pretty :code:`repr` in a class decorated by `attrs`."""
    if not attrs.has(decorated_class):
        msg = "Can only implement a pretty repr for a class created with attrs"
        raise TypeError(msg)

    def repr_pretty(self: Any, p: PrettyPrinter, cycle: bool) -> None:
        class_name = type(self).__name__
        if cycle:
            p.text(f"{class_name}(...)")
        else:
            with p.group(indent=2, open=f"{class_name}("):
                for field in attrs.fields(type(self)):
                    if not field.init:
                        continue
                    value = getattr(self, field.name)
                    p.breakable()
                    p.text(f"{field.name}=")
                    p.pretty(value)
                    p.text(",")
            p.breakable()
            p.text(")")

    decorated_class._repr_pretty_ = repr_pretty  # ty:ignore[unresolved-attribute]
    return decorated_class  # ty:ignore[invalid-return-type]
