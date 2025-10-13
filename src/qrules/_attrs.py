from __future__ import annotations

from fractions import Fraction
from typing import SupportsFloat

from qrules.quantum_numbers import Parity


def to_fraction(value: SupportsFloat) -> Fraction:
    float_value = float(value)
    if float_value == -0.0:
        float_value = 0.0
    return Fraction(float_value)


def to_parity(value: Parity | int) -> Parity:
    return Parity(value)
