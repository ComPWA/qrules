"""Test 2-to-n production reactions, https://github.com/ComPWA/qrules/issues/29."""

import pytest

import qrules
from qrules.transition import SpinFormalism


@pytest.mark.parametrize("formalism", ["helicity", "canonical-helicity"])
def test_pi0_photoproduction(formalism: SpinFormalism, particle_database):
    reaction = qrules.generate_transitions(
        initial_state=["gamma", "p"],
        final_state=["p", "pi0"],
        allowed_intermediate_particles=[
            "Delta(1232)",
            "N(1440)",
            "rho(770)",
            "omega(782)",
        ],
        allowed_interaction_types=["strong", "em"],
        formalism=formalism,
        particle_db=particle_database,
    )
    assert reaction.get_intermediate_particles().names == [
        "Delta(1232)~-",  # u-channel baryon exchange
        "Delta(1232)+",  # s-channel resonance
        "N(1440)~-",
        "N(1440)+",
        "omega(782)",  # t-channel meson exchange
        "rho(770)0",
    ]
    assert {p.name for p in reaction.initial_state.values()} == {"gamma", "p"}


def test_group_by_channel(particle_database):
    reaction = qrules.generate_transitions(
        initial_state=["gamma", "p"],
        final_state=["pi0", "p"],
        allowed_intermediate_particles=[
            "Delta(1232)",
            "N(1440)",
            "rho(770)",
            "omega(782)",
        ],
        allowed_interaction_types=["strong", "em"],
        formalism="helicity",
        particle_db=particle_database,
    )
    channels = {
        channel: {
            state.particle.name
            for transition in transitions
            for state in transition.intermediate_states.values()
        }
        for channel, transitions in reaction.group_by_channel().items()
    }
    assert channels == {
        "s": {"Delta(1232)+", "N(1440)+"},
        "t": {"omega(782)", "rho(770)0"},
        "u": {"Delta(1232)+", "Delta(1232)~-", "N(1440)+", "N(1440)~-"},
    }
