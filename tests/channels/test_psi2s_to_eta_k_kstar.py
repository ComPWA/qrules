"""Test for https://github.com/ComPWA/qrules/issues/165."""

import pytest

import qrules
from qrules.particle import ParticleCollection
from qrules.transition import SpinFormalism


@pytest.mark.parametrize("formalism", ["helicity", "canonical-helicity"])
@pytest.mark.parametrize(
    "resonances",
    [
        ["h(1)(1415)"],
        ["omega(1650)"],
        ["h(1)(1415)", "omega(1650)"],
    ],
)
def test_resonances(formalism: SpinFormalism, resonances, modified_pdg):
    reaction = qrules.generate_transitions(
        initial_state=("psi(2S)", [+1, -1]),
        final_state=["eta", "K-", "K*(892)+"],
        allowed_intermediate_particles=resonances,
        allowed_interaction_types=["em"],
        formalism=formalism,
        particle_db=modified_pdg,
    )
    assert reaction.get_intermediate_particles().names == resonances


@pytest.fixture(scope="session")
def modified_pdg(particle_database) -> ParticleCollection:
    # https://github.com/scikit-hep/particle/issues/486
    # https://github.com/ComPWA/qrules/issues/165#issuecomment-1497343548
    particles = ParticleCollection(particle_database)
    original_h1415 = particles["h(1)(1415)"]
    new_h1415 = qrules.particle.create_particle(
        original_h1415,
        isospin=qrules.particle.Spin(0, 0),
    )
    particles.remove(original_h1415)
    particles.add(new_h1415)
    return particles
