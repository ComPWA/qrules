from fractions import Fraction
from itertools import product

import pytest

from qrules.conservation_rules import HelicityFacts, helicity_conservation


def _helicity_facts(magnitude: float, projection: float) -> HelicityFacts:
    return HelicityFacts(
        spin_magnitude=Fraction(magnitude),
        spin_projection=Fraction(projection),
    )


@pytest.mark.parametrize(
    ("in_edge_qns", "out_edge_qns", "expected"),
    [
        (
            [_helicity_facts(s_magnitude, min(s_magnitude, 1))],
            [
                _helicity_facts(abs(lambda1), lambda1),
                _helicity_facts(abs(lambda2), lambda2),
            ],
            abs(lambda1 - lambda2) <= s_magnitude,
        )
        for s_magnitude, lambda1, lambda2 in product(
            [0, 0.5, 1, 1.5, 2],
            [-2, -1.5, -1.0, -0.5, 0, 0.5, 1, 1.5, 2],
            [-1, 0, 1],
        )
    ],
)
def test_helicity_conservation_decay(in_edge_qns, out_edge_qns, expected):
    assert helicity_conservation(in_edge_qns, out_edge_qns) is expected


@pytest.mark.parametrize(
    ("in_edge_qns", "out_edge_qns", "expected"),
    [
        (
            [
                _helicity_facts(abs(lambda1), lambda1),
                _helicity_facts(abs(lambda2), lambda2),
            ],
            [_helicity_facts(s_magnitude, min(s_magnitude, 1))],
            abs(lambda1 - lambda2) <= s_magnitude,
        )
        for s_magnitude, lambda1, lambda2 in product(
            [0, 0.5, 1, 1.5, 2],
            [-2, -1.5, -1.0, -0.5, 0, 0.5, 1, 1.5, 2],
            [-1, 0, 1],
        )
    ],
)
def test_helicity_conservation_production(in_edge_qns, out_edge_qns, expected):
    assert helicity_conservation(in_edge_qns, out_edge_qns) is expected
