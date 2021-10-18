# pylint: disable=eval-used, no-self-use
import typing
from copy import deepcopy

import pytest

from qrules.quantum_numbers import Parity, _to_fraction


class TestParity:
    def test_init_and_eq(self):
        parity = Parity(+1)
        assert parity == +1
        assert int(parity) == +1
        assert parity > None

    @typing.no_type_check  # https://github.com/python/mypy/issues/4610
    def test_comparison(self):
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
        assert 0 < pos  # pylint: disable=misplaced-comparison-constant

    def test_hash(self):
        neg = Parity(-1)
        pos = Parity(+1)
        assert {pos, neg, deepcopy(pos)} == {neg, pos}

    def test_neg(self):
        parity = Parity(+1)
        flipped_parity = -parity
        assert flipped_parity.value == -parity.value

    @pytest.mark.parametrize("value", [-1, +1])
    def test_repr(self, value):
        parity = Parity(value)
        from_repr = eval(repr(parity))
        assert from_repr == parity

    def test_exceptions(self):
        with pytest.raises(TypeError):
            Parity(1.2)  # type: ignore[arg-type]
        with pytest.raises(
            ValueError, match=r"Parity can only be \+1 or -1, not 0"
        ):
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
    assert _to_fraction(value, render_plus) == expected
