from fractions import Fraction
from itertools import product

import pytest

from qrules.conservation_rules import (
    GParityEdgeInput,
    GParityNodeInput,
    g_parity_conservation,
)
from qrules.quantum_numbers import Parity


def describe_g_parity_conservation():
    @pytest.mark.parametrize(
        ("rule_input", "expected"),
        [
            (
                (
                    [
                        GParityEdgeInput(
                            isospin_magnitude=0,
                            spin_magnitude=0,
                            pid=123,
                            g_parity=Parity(g_parity_in[0]),
                        )
                    ],
                    [
                        GParityEdgeInput(
                            isospin_magnitude=0,
                            spin_magnitude=0,
                            pid=0,
                            g_parity=Parity(g_parity_out[0][0]),
                        ),
                        GParityEdgeInput(
                            isospin_magnitude=0,
                            spin_magnitude=0,
                            pid=0,
                            g_parity=Parity(g_parity_out[0][1]),
                        ),
                    ],
                    GParityNodeInput(l_magnitude=Fraction(0), s_magnitude=Fraction(0)),
                ),
                g_parity_in[1] is g_parity_out[1],
            )
            for g_parity_in, g_parity_out in product(
                [
                    (1, True),
                    (-1, False),
                ],
                [
                    ((1, 1), True),
                    ((-1, -1), True),
                    ((-1, 1), False),
                    ((1, -1), False),
                ],
            )
        ],
    )
    def it_g_parity_all_defined(rule_input, expected):
        assert g_parity_conservation(*rule_input) is expected

    @pytest.mark.parametrize(
        ("rule_input", "expected"),
        [
            (
                (
                    [
                        GParityEdgeInput(
                            isospin_magnitude=isospin,
                            spin_magnitude=0,
                            pid=123,
                            g_parity=Parity(g_parity),
                        )
                    ],
                    [
                        GParityEdgeInput(
                            isospin_magnitude=0,
                            spin_magnitude=0,
                            pid=100,
                        ),
                        GParityEdgeInput(
                            isospin_magnitude=0,
                            spin_magnitude=0,
                            pid=-100,
                        ),
                    ],
                    GParityNodeInput(
                        l_magnitude=Fraction(l_magnitude), s_magnitude=Fraction(0)
                    ),
                ),
                (-1) ** (l_magnitude + isospin) == g_parity,
            )
            for g_parity, isospin, l_magnitude in product([-1, 1], [0, 1], range(5))
        ],
    )
    def it_g_parity_multiparticle_boson(rule_input, expected):
        assert g_parity_conservation(*rule_input) is expected

    @pytest.mark.parametrize(
        ("g_parity", "isospin", "l_magnitude"),
        list(product([-1, 1], [0, 1], range(5))),
    )
    def it_g_parity_multiparticle_boson_fusion(
        g_parity: int, isospin: int, l_magnitude: int
    ):
        # Mirror of the 1 -> 2 case: the pair is now in the initial state
        ingoing = [
            GParityEdgeInput(isospin_magnitude=0, spin_magnitude=0, pid=pid)
            for pid in (+100, -100)
        ]
        outgoing = [
            GParityEdgeInput(
                isospin_magnitude=isospin,
                spin_magnitude=0,
                pid=123,
                g_parity=Parity(g_parity),
            )
        ]
        node = GParityNodeInput(
            l_magnitude=Fraction(l_magnitude), s_magnitude=Fraction(0)
        )
        expected = (-1) ** (l_magnitude + isospin) == g_parity
        assert g_parity_conservation(ingoing, outgoing, node) is expected


def describe_unknown_g_parity():
    @pytest.mark.parametrize(
        ("g_parity_in", "g_parities_out"),
        [
            (Parity.UNKNOWN, (Parity(-1), Parity(-1))),
            (Parity(-1), (Parity.UNKNOWN, Parity(-1))),
            (Parity(+1), (Parity(-1), Parity.UNKNOWN)),
        ],
    )
    def it_passes_if_one_state_is_unknown(
        g_parity_in: Parity, g_parities_out: tuple[Parity, Parity]
    ):
        ingoing = [
            GParityEdgeInput(
                isospin_magnitude=0, spin_magnitude=0, pid=1, g_parity=g_parity_in
            )
        ]
        outgoing = [
            GParityEdgeInput(isospin_magnitude=1, spin_magnitude=0, pid=2, g_parity=g)
            for g in g_parities_out
        ]
        node = GParityNodeInput(l_magnitude=Fraction(0), s_magnitude=Fraction(0))
        assert g_parity_conservation(ingoing, outgoing, node) is True

    @pytest.mark.parametrize(
        ("g_parity_in", "expected"),
        [(Parity(-1), True), (Parity(+1), False), (Parity.UNKNOWN, True)],
    )
    def it_uses_g_parity_of_boson_pair(g_parity_in: Parity, expected: bool):
        # G of a boson-antiboson pair is (-1)^(L+I), independent of individual parities
        ingoing = [
            GParityEdgeInput(
                isospin_magnitude=1, spin_magnitude=0, pid=1, g_parity=g_parity_in
            )
        ]
        outgoing = [
            GParityEdgeInput(
                isospin_magnitude=1, spin_magnitude=0, pid=pid, g_parity=Parity.UNKNOWN
            )
            for pid in (+5, -5)
        ]
        node = GParityNodeInput(l_magnitude=Fraction(0), s_magnitude=Fraction(0))
        assert g_parity_conservation(ingoing, outgoing, node) is expected

    @pytest.mark.parametrize("l_magnitude", range(4))
    def it_passes_fusion_into_unknown_g_parity(l_magnitude: int):
        # The pair has known G-parities, but the single final state has not, so the
        # rule cannot be checked and must not compare against one of the pair
        ingoing = [
            GParityEdgeInput(
                isospin_magnitude=1, spin_magnitude=0, pid=pid, g_parity=Parity(-1)
            )
            for pid in (+5, -5)
        ]
        outgoing = [
            GParityEdgeInput(
                isospin_magnitude=0, spin_magnitude=0, pid=1, g_parity=Parity.UNKNOWN
            )
        ]
        node = GParityNodeInput(
            l_magnitude=Fraction(l_magnitude), s_magnitude=Fraction(0)
        )
        assert g_parity_conservation(ingoing, outgoing, node) is True
