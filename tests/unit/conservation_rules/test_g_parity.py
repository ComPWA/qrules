from fractions import Fraction
from itertools import product

import pytest

from qrules.conservation_rules import (
    GParityCoupling,
    GParityCouplingEdgeInput,
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


def describe_GParityCoupling():
    pi_plus = GParityCouplingEdgeInput(
        1, spin_magnitude=0, parity=-1, pid=211, g_parity=-1
    )
    pi_minus = GParityCouplingEdgeInput(
        1, spin_magnitude=0, parity=-1, pid=-211, g_parity=-1
    )
    k_plus = GParityCouplingEdgeInput(0.5, spin_magnitude=0, parity=-1, pid=321)
    k_minus = GParityCouplingEdgeInput(0.5, spin_magnitude=0, parity=-1, pid=-321)
    proton = GParityCouplingEdgeInput(0.5, spin_magnitude=0.5, parity=+1, pid=2212)
    antiproton = GParityCouplingEdgeInput(0.5, spin_magnitude=0.5, parity=-1, pid=-2212)

    @pytest.mark.parametrize(
        ("max_angular_momentum", "single", "expected"),
        [
            # phi: I=0 and G=-1 with L=1
            (1, GParityCouplingEdgeInput(0, 1, -1, 333, g_parity=-1), True),
            (0, GParityCouplingEdgeInput(0, 1, -1, 333, g_parity=-1), False),
            # rho: I=1 and G=+1 with L=1
            (1, GParityCouplingEdgeInput(1, 1, -1, 113, g_parity=+1), True),
            # f0: I=0 and G=+1 with L=0
            (0, GParityCouplingEdgeInput(0, 0, +1, 9010221, g_parity=+1), True),
            # exotic 1-+ with I=1 and G=-1: parity requires odd L, but G-parity even L
            (3, GParityCouplingEdgeInput(1, 1, -1, 1, g_parity=-1), False),
            (3, GParityCouplingEdgeInput(1, 1, -1, 1, g_parity=None), True),
        ],
    )
    def it_couples_boson_pairs(max_angular_momentum, single, expected):
        rule = GParityCoupling(max_angular_momentum)
        assert rule([single], [k_plus, k_minus]) is expected
        assert rule([k_plus, k_minus], [single]) is expected

    @pytest.mark.parametrize(
        ("max_angular_momentum", "single", "expected"),
        [
            # rho: 1-- with I=1 needs L=0, S=1, so C=-1 and G=+1
            (0, GParityCouplingEdgeInput(1, 1, -1, 113, g_parity=+1), True),
            # exotic 1-+ with I=1: C=+1 requires S=0, so J=0
            (1, GParityCouplingEdgeInput(1, 1, -1, 1, g_parity=-1), False),
        ],
    )
    def it_couples_fermion_pairs(max_angular_momentum, single, expected):
        rule = GParityCoupling(max_angular_momentum)
        assert rule([single], [proton, antiproton]) is expected

    def it_multiplies_defined_g_parities():
        rule = GParityCoupling(max_angular_momentum=0)
        rho = GParityCouplingEdgeInput(1, 1, -1, 113, g_parity=+1)
        omega = GParityCouplingEdgeInput(0, 1, -1, 223, g_parity=-1)
        assert rule([rho], [pi_plus, pi_minus]) is True
        assert rule([omega], [pi_plus, pi_minus]) is False
        assert rule([omega], [rho, pi_plus]) is True
        assert rule([rho], [rho, pi_plus]) is False

    def it_skips_half_integer_isospin():
        rule = GParityCoupling(max_angular_momentum=0)
        single = GParityCouplingEdgeInput(0.5, 1, -1, 1, g_parity=-1)
        assert rule([single], [k_plus, k_minus]) is True
