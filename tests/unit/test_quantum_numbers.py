import pickle  # ruff: ignore[suspicious-pickle-import]
from copy import copy, deepcopy
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
        assert flipped_parity == Parity(-1)

    @pytest.mark.parametrize("value", [-1, +1])
    def it_repr(value):
        parity = Parity(value)
        from_repr = eval(repr(parity))
        assert from_repr == parity

    def it_exceptions():
        with pytest.raises(TypeError):
            Parity(1.2)  # ty: ignore[invalid-argument-type]
        with pytest.raises(
            ValueError, match=r"Parity can only be \+1, -1, or unknown, not 0"
        ):
            Parity(0)

    def describe_unknown():
        def it_is_distinct_from_known_parities():
            unknown = Parity.UNKNOWN
            assert unknown.is_unknown
            assert not Parity(+1).is_unknown
            assert not Parity(-1).is_unknown
            assert unknown != Parity(+1)
            assert unknown != Parity(-1)
            assert unknown != +1
            assert unknown != -1
            assert unknown is not None

        def it_init_from_unknown():
            assert Parity(Parity.UNKNOWN) == Parity.UNKNOWN
            assert Parity(Parity.UNKNOWN.value) == Parity.UNKNOWN

        def it_survives_copy_and_pickle():
            unknown = Parity.UNKNOWN
            for copied in [
                copy(unknown),
                deepcopy(unknown),
                pickle.loads(pickle.dumps(unknown)),  # ruff: ignore[suspicious-pickle-usage]
            ]:
                assert copied == unknown
                assert copied.is_unknown
                assert copied.value is unknown.value

        def it_hash():
            assert {Parity.UNKNOWN, deepcopy(Parity.UNKNOWN), Parity(+1)} == {
                Parity(+1),
                Parity.UNKNOWN,
            }

        def it_neg():
            assert -Parity.UNKNOWN == Parity.UNKNOWN

        def it_int_raises():
            with pytest.raises(ValueError, match=r"unknown parity"):
                int(Parity.UNKNOWN)

        def it_compares_with_int_raises():
            with pytest.raises(ValueError, match=r"unknown parity"):
                _ = Parity.UNKNOWN > 0

        def it_sorts():
            unsorted = [Parity(+1), Parity.UNKNOWN, Parity(-1)]
            assert sorted(unsorted) == [Parity.UNKNOWN, Parity(-1), Parity(+1)]
            assert Parity.UNKNOWN > None
            assert Parity(-1) > Parity.UNKNOWN

        def it_repr():
            assert repr(Parity.UNKNOWN) == "Parity.UNKNOWN"
            assert eval(repr(Parity.UNKNOWN)) == Parity.UNKNOWN


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
