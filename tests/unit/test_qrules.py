import pytest

from qrules import generate_transitions
from qrules.particle import create_particle, load_pdg


@pytest.mark.parametrize(
    "resonance_names",
    [
        ["Sigma(1660)~-"],
        ["N(1650)+"],
        ["K*(1680)~0"],
        ["Sigma(1660)~-", "N(1650)+"],
        ["Sigma(1660)~-", "K*(1680)~0"],
        ["N(1650)+", "K*(1680)~0"],
        ["Sigma(1660)~-", "N(1650)+", "K*(1680)~0"],
    ],
)
def test_generate_transitions(resonance_names):
    final_state_names = ["K0", "Sigma+", "p~"]
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
            i: state.name for i, state in transition.final_states.items()
        }
        assert final_state == this_final_state


@pytest.mark.filterwarnings(
    "ignore:There are conservation rules that were not executed"
)
def test_ls_free_solving_conserves_pair_c_parity():
    """Existence rules prune particle-antiparticle pairs without LS couplings."""
    particle_db = load_pdg()
    # 1-+ cannot decay to p pbar: parity requires L=0 and C-parity then requires S=0
    exotic = create_particle(
        particle_db["rho(770)0"],
        name="X(1-+)",
        pid=9999999,
        mass=2.5,
        width=0.1,
        c_parity=+1,
        g_parity=-1,
    )
    particle_db.add(exotic)
    for ls_couplings in (True, False):
        with pytest.raises(RuntimeError, match="No solutions were found"):
            generate_transitions(
                initial_state="J/psi(1S)",
                final_state=["gamma", "p", "p~"],
                particle_db=particle_db,
                allowed_intermediate_particles=[exotic.name],
                allowed_interaction_types=["strong", "em"],
                max_angular_momentum=1,
                ls_couplings=ls_couplings,
            )
