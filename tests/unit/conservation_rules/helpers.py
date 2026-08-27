from __future__ import annotations

from qrules.conservation_rules import SpinEdgeInput, SpinNodeInput
from qrules.particle import Spin

SpinRuleInputType = tuple[
    list[SpinEdgeInput],
    list[SpinEdgeInput],
    SpinNodeInput,
]


def create_two_body_decay_spin_data(
    in_spin: Spin | None = None,
    out_spin1: Spin | None = None,
    out_spin2: Spin | None = None,
    angular_momentum: Spin | None = None,
    coupled_spin: Spin | None = None,
) -> SpinRuleInputType:
    spin_zero = Spin(0, 0)
    if in_spin is None:
        in_spin = spin_zero
    if out_spin1 is None:
        out_spin1 = spin_zero
    if out_spin2 is None:
        out_spin2 = spin_zero
    if angular_momentum is None:
        angular_momentum = spin_zero
    if coupled_spin is None:
        coupled_spin = spin_zero
    return (
        [SpinEdgeInput(in_spin.magnitude, in_spin.projection)],
        [
            SpinEdgeInput(out_spin1.magnitude, out_spin1.projection),
            SpinEdgeInput(out_spin2.magnitude, out_spin2.projection),
        ],
        SpinNodeInput(
            angular_momentum.magnitude,
            angular_momentum.projection,
            coupled_spin.magnitude,
            coupled_spin.projection,
        ),
    )
