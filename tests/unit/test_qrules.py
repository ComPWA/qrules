import pytest

from qrules import generate_transitions


@pytest.mark.parametrize(
    "resonance_names",
    [
        ["Sigmabar(1660)-"],
        ["N(1650)+"],
        ["Kbar^*(1680)0"],
        ["Sigmabar(1660)-", "N(1650)+"],
        ["Sigmabar(1660)-", "Kbar^*(1680)0"],
        ["N(1650)+", "Kbar^*(1680)0"],
        ["Sigmabar(1660)-", "N(1650)+", "Kbar^*(1680)0"],
    ],
)
def test_generate_transitions(resonance_names):
    final_state_names = ["K0", "Sigma+", "pbar"]
    reaction = generate_transitions(
        initial_state="J/psi(1S)",
        final_state=final_state_names,
        allowed_intermediate_particles=resonance_names,
        allowed_interaction_types="strong",
    )
    assert len(reaction.group_by_topology()) == len(resonance_names)
    final_state = dict(enumerate(final_state_names))
    for transition in reaction.transitions:
        this_final_state = {
            i: state.particle.name for i, state in transition.final_states.items()
        }
        assert final_state == this_final_state
