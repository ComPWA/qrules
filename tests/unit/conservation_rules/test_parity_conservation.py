from fractions import Fraction
from itertools import product

import pytest

from qrules.conservation_rules import (
    HelicityParityEdgeInput,
    parity_conservation,
    parity_conservation_helicity,
)
from qrules.quantum_numbers import NodeQuantumNumbers, Parity


def describe_parity_conservation():
    @pytest.mark.parametrize(
        ("in_parities", "out_parities", "l_magnitude", "expected"),
        [
            (
                [
                    Parity(parity_in),
                ],
                [
                    Parity(parity_out1),
                    Parity(1),
                ],
                NodeQuantumNumbers.l_magnitude(Fraction(l_magnitude)),
                parity_in == parity_out1 * (-1) ** (l_magnitude),
            )
            for parity_in, parity_out1, l_magnitude in product(
                [-1, 1], [-1, 1], range(5)
            )
        ],
    )
    def it_conserves_parity(in_parities, out_parities, l_magnitude, expected):
        assert parity_conservation(in_parities, out_parities, l_magnitude) is expected

    @pytest.mark.parametrize(
        ("in_parities", "out_parities", "l_magnitude", "expected"),
        [
            (
                [
                    HelicityParityEdgeInput(
                        parity=Parity(in_parity),
                        spin_magnitude=in_spin_mag,
                        spin_projection=0,
                    )
                ],
                [
                    HelicityParityEdgeInput(
                        parity=Parity(out_parity1),
                        spin_magnitude=1,
                        spin_projection=-1,
                    ),
                    HelicityParityEdgeInput(
                        parity=Parity(out_parity2),
                        spin_magnitude=1,
                        spin_projection=-1,
                    ),
                ],
                NodeQuantumNumbers.parity_prefactor(1),
                in_parity * out_parity1 * out_parity2 * (-1) ** (in_spin_mag % 2) == 1,
            )
            for in_spin_mag, in_parity, out_parity1, out_parity2 in product(
                range(4), [-1, 1], [-1, 1], [-1, 1]
            )
        ],
    )
    def it_helicity_prefactor(in_parities, out_parities, l_magnitude, expected):
        assert (
            parity_conservation_helicity(in_parities, out_parities, l_magnitude)
            is expected
        )

    @pytest.mark.parametrize(
        ("in_parities", "out_parities", "l_magnitude", "expected"),
        [
            (
                [
                    HelicityParityEdgeInput(
                        parity=Parity(in_parity),
                        spin_magnitude=in_spin_mag,
                        spin_projection=0,
                    )
                ],
                [
                    HelicityParityEdgeInput(
                        parity=Parity(1),
                        spin_magnitude=1,
                        spin_projection=0,
                    ),
                    HelicityParityEdgeInput(
                        parity=Parity(1),
                        spin_magnitude=1,
                        spin_projection=0,
                    ),
                ],
                NodeQuantumNumbers.parity_prefactor(parity_prefactor),
                in_parity * (-1) ** (in_spin_mag % 2) == 1 and parity_prefactor == 1,
            )
            for in_spin_mag, in_parity, parity_prefactor in product(
                range(4), [-1, 1], [-1, 1]
            )
        ],
    )
    def it_helicity(in_parities, out_parities, l_magnitude, expected):
        assert (
            parity_conservation_helicity(in_parities, out_parities, l_magnitude)
            is expected
        )


def describe_unknown_parity():
    @pytest.mark.parametrize(
        ("in_parities", "out_parities"),
        [
            ([Parity.UNKNOWN], [Parity(-1), Parity(-1)]),
            ([Parity(+1)], [Parity.UNKNOWN, Parity(-1)]),
            ([Parity(+1)], [Parity(-1), None]),
        ],
    )
    @pytest.mark.parametrize("l_magnitude", range(3))
    def it_passes_parity_conservation(in_parities, out_parities, l_magnitude: int):
        l_mag = NodeQuantumNumbers.l_magnitude(Fraction(l_magnitude))
        assert parity_conservation(in_parities, out_parities, l_mag) is True

    @pytest.mark.parametrize("parity_prefactor", [-1, +1])
    @pytest.mark.parametrize("unknown_edge", range(3))
    def it_passes_parity_conservation_helicity(
        parity_prefactor: int, unknown_edge: int
    ):
        # λ1 = λ2 = 0 with prefactor -1 would normally fail independent of the prefactor
        edges = [
            HelicityParityEdgeInput(parity=-1, spin_magnitude=0, spin_projection=0),
            HelicityParityEdgeInput(parity=+1, spin_magnitude=0, spin_projection=0),
            HelicityParityEdgeInput(parity=+1, spin_magnitude=0, spin_projection=0),
        ]
        edges[unknown_edge] = HelicityParityEdgeInput(
            parity=Parity.UNKNOWN, spin_magnitude=0, spin_projection=0
        )
        prefactor = NodeQuantumNumbers.parity_prefactor(parity_prefactor)
        assert parity_conservation_helicity(edges[:1], edges[1:], prefactor) is True
