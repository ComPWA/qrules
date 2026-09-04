from copy import deepcopy
from fractions import Fraction

import pytest

from qrules.io._labels import _render_fraction
from qrules.quantum_numbers import Parity


def describe_Parity():
    def it_init_and_eq():
        parity = Parity(+1)
        assert parity == +1
        assert int(parity) == +1
        assert parity > None

    def it_comparison():
        neg = Parity(-1)
        pos = Parity(+1)
        assert pos > 0
        assert neg < 0
        assert neg < pos
        assert neg <= pos
        assert pos > neg
        assert pos >= neg
        assert pos >= 0
        assert neg <= 0
        assert pos > 0

    def it_hash():
        neg = Parity(-1)
        pos = Parity(+1)
        assert {pos, neg, deepcopy(pos)} == {neg, pos}

    def it_neg():
        parity = Parity(+1)
        flipped_parity = -parity
        assert flipped_parity.value == -parity.value

    @pytest.mark.parametrize("value", [-1, +1])
    def it_repr(value):
        parity = Parity(value)
        from_repr = eval(repr(parity))
        assert from_repr == parity

    def it_exceptions():
        with pytest.raises(TypeError):
            Parity(1.2)  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match=r"Parity can only be \+1 or -1, not 0"):
            Parity(0)


@pytest.mark.parametrize(
    ("value", "render_plus", "expected"),
    [
        (0, False, "0"),
        (0, True, "0"),
        (-1, False, "-1"),
        (-1, True, "-1"),
        (1, False, "1"),
        (1, True, "+1"),
        (1.0, True, "+1"),
        (0.5, True, "+1/2"),
        (-0.5, True, "-1/2"),
        (+1.5, False, "3/2"),
        (+1.5, True, "+3/2"),
        (-1.5, True, "-3/2"),
    ],
)
def test_to_fraction(value, render_plus: bool, expected: str):
    assert _render_fraction(Fraction(value), render_plus) == expected
