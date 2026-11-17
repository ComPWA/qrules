from fractions import Fraction
from itertools import product

import pytest

from qrules.conservation_rules import (
    CParityEdgeInput,
    CParityNodeInput,
    c_parity_conservation,
)
from qrules.quantum_numbers import Parity


def describe_c_parity_conservation():
    @pytest.mark.parametrize(
        ("rule_input", "expected"),
        [
            (
                (
                    [CParityEdgeInput(spin_magnitude=0.0, pid=1, c_parity=Parity(-1))],
                    [
                        CParityEdgeInput(
                            spin_magnitude=0.0, pid=1, c_parity=Parity(-1)
                        ),
                        CParityEdgeInput(spin_magnitude=0.0, pid=1, c_parity=Parity(1)),
                    ],
                    None,
                ),
                True,
            ),
            (
                (
                    [CParityEdgeInput(spin_magnitude=0.0, pid=1, c_parity=Parity(1))],
                    [
                        CParityEdgeInput(
                            spin_magnitude=0.0, pid=1, c_parity=Parity(-1)
                        ),
                        CParityEdgeInput(spin_magnitude=0.0, pid=1, c_parity=Parity(1)),
                    ],
                    None,
                ),
                False,
            ),
        ],
    )
    def it_c_parity_all_defined(rule_input, expected):
        assert c_parity_conservation(*rule_input) is expected

    @pytest.mark.parametrize(
        ("rule_input", "expected"),
        [
            (
                (
                    [
                        CParityEdgeInput(
                            spin_magnitude=0, pid=123, c_parity=Parity(c_parity)
                        )
                    ],
                    [
                        CParityEdgeInput(spin_magnitude=0, pid=100),
                        CParityEdgeInput(spin_magnitude=0, pid=-100),
                    ],
                    CParityNodeInput(
                        l_magnitude=Fraction(l_magnitude), s_magnitude=Fraction(0)
                    ),
                ),
                (-1) ** l_magnitude == c_parity,
            )
            for c_parity, l_magnitude in product([-1, 1], range(5))
        ],
    )
    def it_c_parity_multiparticle_boson(rule_input, expected):
        assert c_parity_conservation(*rule_input) is expected

    @pytest.mark.parametrize(
        ("rule_input", "expected"),
        [
            (
                (
                    [
                        CParityEdgeInput(
                            spin_magnitude=0, pid=123, c_parity=Parity(c_parity)
                        )
                    ],
                    [
                        CParityEdgeInput(spin_magnitude=0.5, pid=100),
                        CParityEdgeInput(spin_magnitude=0.5, pid=-100),
                    ],
                    CParityNodeInput(
                        l_magnitude=Fraction(l_magnitude),
                        s_magnitude=Fraction(s_magnitude),
                    ),
                ),
                (s_magnitude + l_magnitude) % 2 == abs(c_parity - 1) / 2,
            )
            for c_parity, s_magnitude, l_magnitude in product(
                [-1, 1], range(5), range(5)
            )
        ],
    )
    def it_c_parity_multiparticle_fermion(rule_input, expected):
        assert c_parity_conservation(*rule_input) is expected


def describe_unknown_c_parity():
    @pytest.mark.parametrize("c_parity_in", [Parity(-1), Parity(+1), Parity.UNKNOWN])
    def it_passes_if_one_side_is_unknown(c_parity_in: Parity):
        ingoing = [CParityEdgeInput(spin_magnitude=0, pid=1, c_parity=c_parity_in)]
        outgoing = [
            CParityEdgeInput(spin_magnitude=0, pid=2, c_parity=Parity.UNKNOWN),
            CParityEdgeInput(spin_magnitude=0, pid=3, c_parity=Parity(+1)),
        ]
        node = CParityNodeInput(l_magnitude=0, s_magnitude=0)
        assert c_parity_conservation(ingoing, outgoing, node) is True

    @pytest.mark.parametrize(
        ("c_parity_in", "expected"),
        [(Parity(-1), True), (Parity(+1), False), (Parity.UNKNOWN, True)],
    )
    def it_uses_c_parity_of_boson_pair(c_parity_in: Parity, expected: bool):
        # C of a boson-antiboson pair is (-1)^L, independent of the individual parities
        ingoing = [CParityEdgeInput(spin_magnitude=1, pid=1, c_parity=c_parity_in)]
        outgoing = [
            CParityEdgeInput(spin_magnitude=0, pid=+5, c_parity=Parity.UNKNOWN),
            CParityEdgeInput(spin_magnitude=0, pid=-5, c_parity=Parity.UNKNOWN),
        ]
        node = CParityNodeInput(l_magnitude=1, s_magnitude=0)
        assert c_parity_conservation(ingoing, outgoing, node) is expected
