from __future__ import annotations

from fractions import Fraction

import pytest

from qrules.conservation_rules import (
    SpinCoupling,
    SpinMagnitudeFacts,
    SpinMagnitudeNodeFacts,
    SpinParityCoupling,
    SpinParityFacts,
    spin_conservation,
    spin_magnitude_conservation,
)
from qrules.particle import Spin

from tests.unit.conservation_rules.helpers import (
    SpinRuleInputType,
    create_two_body_decay_spin_data,
)

_SpinMagnitudeRuleInputType = tuple[
    list[SpinMagnitudeFacts],
    list[SpinMagnitudeFacts],
    SpinMagnitudeNodeFacts,
]


@pytest.mark.parametrize(
    ("rule_input", "expected"),
    [
        (
            create_two_body_decay_spin_data(angular_momentum=Spin(ang_mom_mag, 0)),
            expected,
        )
        for ang_mom_mag, expected in [
            (0, True),
            (1, False),
            (2, False),
            (3, False),
        ]
    ]
    + [
        (
            create_two_body_decay_spin_data(
                in_spin=Spin(spin_magnitude, 0),
                angular_momentum=Spin(spin_magnitude, 0),
            ),
            expected,
        )
        for spin_magnitude, expected in zip([0, 1, 2], [True] * 3, strict=True)
    ]
    + [
        (
            create_two_body_decay_spin_data(
                in_spin=Spin(spin_magnitude, 0),
                out_spin1=Spin(1, -1),
                out_spin2=Spin(1, 1),
                angular_momentum=Spin(1, 0),
                coupled_spin=Spin(spin_magnitude, 0),
            ),
            expected,
        )
        for spin_magnitude, expected in [
            (0, False),
            (1, False),
            (2, False),
            (3, False),
        ]
    ]
    + [
        (
            create_two_body_decay_spin_data(
                in_spin=Spin(1, -1),
                out_spin2=Spin(1, -1),
                coupled_spin=Spin(1, -1),
            ),
            True,
        ),
        (
            create_two_body_decay_spin_data(
                in_spin=Spin(1, 0),
                out_spin1=Spin(1, 1),
                out_spin2=Spin(1, -1),
                angular_momentum=Spin(1, 0),
                coupled_spin=Spin(2, 0),
            ),
            True,
        ),
    ],
)
def test_spin_all_defined(rule_input: SpinRuleInputType, expected: bool) -> None:
    assert spin_conservation(*rule_input) is expected


@pytest.mark.parametrize(
    ("rule_input", "expected"),
    [
        (
            (
                [SpinMagnitudeFacts(spin_magnitude=Fraction(1))],
                [
                    SpinMagnitudeFacts(spin_magnitude=Fraction(spin2_mag)),
                    SpinMagnitudeFacts(spin_magnitude=Fraction(1)),
                ],
                SpinMagnitudeNodeFacts(
                    l_magnitude=Fraction(ang_mom_mag),
                    s_magnitude=Fraction(coupled_spin_mag),
                ),
            ),
            True,
        )
        for spin2_mag, ang_mom_mag, coupled_spin_mag in zip(
            (0, 0, 1), (2, 1, 2), (1, 1, 2), strict=True
        )
    ]
    + [
        (
            (
                [SpinMagnitudeFacts(spin_magnitude=Fraction(1))],
                [
                    SpinMagnitudeFacts(spin_magnitude=Fraction(spin2_mag)),
                    SpinMagnitudeFacts(spin_magnitude=Fraction(1)),
                ],
                SpinMagnitudeNodeFacts(
                    l_magnitude=Fraction(ang_mom_mag),
                    s_magnitude=Fraction(coupled_spin_mag),
                ),
            ),
            False,
        )
        for spin2_mag, ang_mom_mag, coupled_spin_mag in zip(
            (1, 0, 1), (0, 1, 2), (0, 2, 0), strict=True
        )
    ],
)
def test_spin_ignore_z_component(
    rule_input: _SpinMagnitudeRuleInputType, expected: bool
) -> None:
    assert spin_magnitude_conservation(*rule_input) is expected


def describe_SpinCoupling():
    def it_checks_integer_spin_sums_for_other_nodes():
        rule = SpinCoupling(max_angular_momentum=0)
        boson: SpinMagnitudeFacts = {"spin_magnitude": Fraction(1)}
        fermion: SpinMagnitudeFacts = {"spin_magnitude": Fraction(1, 2)}
        assert rule([boson, boson], [boson, boson]) is True
        assert rule([boson, fermion], [fermion, fermion]) is False


def describe_SpinParityCoupling():
    def it_ignores_other_nodes():
        rule = SpinParityCoupling(max_angular_momentum=0)
        state: SpinParityFacts = {"spin_magnitude": Fraction(0), "parity": -1}
        assert rule([state, state], [state, state]) is True

    def it_requires_parities():
        rule = SpinParityCoupling(max_angular_momentum=0)
        state: SpinParityFacts = {"spin_magnitude": Fraction(0), "parity": -1}
        undefined: SpinParityFacts = {
            "spin_magnitude": Fraction(0),
            "parity": None,  # ty: ignore[invalid-argument-type]
        }
        assert rule([undefined], [state, state]) is False
