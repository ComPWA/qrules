"""Test for https://github.com/ComPWA/qrules/issues/165."""

import pytest

import qrules


@pytest.mark.parametrize("formalism", ["helicity", "canonical-helicity"])
@pytest.mark.parametrize(
    "resonances",
    [
        ["h(1)(1415)"],
        ["omega(1650)"],
        ["h(1)(1415)", "omega(1650)"],
    ],
)
def test_resonances(formalism, resonances):
    reaction = qrules.generate_transitions(
        initial_state=("psi(2S)", [+1, -1]),
        final_state=["eta", "K-", "K*(892)+"],
        allowed_intermediate_particles=resonances,
        allowed_interaction_types=["em"],
        formalism=formalism,
    )
    assert reaction.get_intermediate_particles().names == resonances
