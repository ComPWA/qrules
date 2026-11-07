from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, SupportsFloat, TypeVar

from attrs.converters import optional

from qrules.quantum_numbers import Parity

if TYPE_CHECKING:
    from collections.abc import Callable

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


def optional_converter(
    converter: Callable[[_Input], _Output],
) -> Callable[[_Input | None], _Output | None]:
    """Preserve the input and output types of an optional attrs converter."""
    return optional(converter)


def to_fraction(value: SupportsFloat) -> Fraction:
    float_value = float(value)
    if float_value == -0.0:  # ruff: ignore[float-equality-comparison]
        float_value = 0.0
    return Fraction(float_value)


def to_parity(value: Parity | int) -> Parity:
    return Parity(value)
