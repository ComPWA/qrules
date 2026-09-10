from fractions import Fraction
from itertools import product

import pytest

from qrules.conservation_rules import (
    CParityCoupling,
    CParityCouplingEdgeInput,
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


def describe_CParityCoupling():
    pi_plus = CParityCouplingEdgeInput(spin_magnitude=0, parity=-1, pid=211)
    pi_minus = CParityCouplingEdgeInput(spin_magnitude=0, parity=-1, pid=-211)
    proton = CParityCouplingEdgeInput(spin_magnitude=0.5, parity=+1, pid=2212)
    antiproton = CParityCouplingEdgeInput(spin_magnitude=0.5, parity=-1, pid=-2212)

    @pytest.mark.parametrize(
        ("max_angular_momentum", "single", "expected"),
        [
            (1, CParityCouplingEdgeInput(1, -1, 113, c_parity=-1), True),
            (0, CParityCouplingEdgeInput(1, -1, 113, c_parity=-1), False),
            (0, CParityCouplingEdgeInput(0, +1, 9010221, c_parity=+1), True),
            (3, CParityCouplingEdgeInput(1, -1, 1, c_parity=+1), False),
            (3, CParityCouplingEdgeInput(1, -1, 1, c_parity=None), True),
        ],
    )
    def it_couples_boson_pairs(max_angular_momentum, single, expected):
        rule = CParityCoupling(max_angular_momentum)
        assert rule([single], [pi_plus, pi_minus]) is expected
        assert rule([pi_plus, pi_minus], [single]) is expected

    @pytest.mark.parametrize(
        ("max_angular_momentum", "single", "expected"),
        [
            # eta(c): 0-+ with L=0, S=0
            (0, CParityCouplingEdgeInput(0, -1, 441, c_parity=+1), True),
            # J/psi: 1-- with L=0, S=1
            (0, CParityCouplingEdgeInput(1, -1, 443, c_parity=-1), True),
            # chi(c1): 1++ needs L=1, S=1
            (0, CParityCouplingEdgeInput(1, +1, 20443, c_parity=+1), False),
            (1, CParityCouplingEdgeInput(1, +1, 20443, c_parity=+1), True),
            # exotic 1-+: parity allows L=0, but C-parity then requires S=0, so J=0
            (1, CParityCouplingEdgeInput(1, -1, 1, c_parity=+1), False),
        ],
    )
    def it_couples_fermion_pairs(max_angular_momentum, single, expected):
        rule = CParityCoupling(max_angular_momentum)
        assert rule([single], [proton, antiproton]) is expected

    def it_multiplies_defined_c_parities():
        rule = CParityCoupling(max_angular_momentum=0)
        photon = CParityCouplingEdgeInput(1, -1, 22, c_parity=-1)
        jpsi = CParityCouplingEdgeInput(1, -1, 443, c_parity=-1)
        eta_c = CParityCouplingEdgeInput(0, -1, 441, c_parity=+1)
        chi_c1 = CParityCouplingEdgeInput(1, +1, 20443, c_parity=+1)
        assert rule([jpsi], [photon, eta_c]) is True
        assert rule([jpsi], [photon, chi_c1]) is True
        assert rule([eta_c], [photon, chi_c1]) is False

    def it_skips_pairs_that_are_not_particle_antiparticle():
        rule = CParityCoupling(max_angular_momentum=0)
        exotic = CParityCouplingEdgeInput(1, -1, 1, c_parity=+1)
        kaon = CParityCouplingEdgeInput(0, -1, pid=321)
        assert rule([exotic], [pi_plus, kaon]) is True
