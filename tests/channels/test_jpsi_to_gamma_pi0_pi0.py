import pytest

import qrules
from qrules.combinatorics import _create_edge_id_particle_mapping
from qrules.particle import ParticleWithSpin
from qrules.topology import StateTransitionGraph


@pytest.mark.parametrize(
    ("allowed_intermediate_particles", "n_topologies", "number_of_solutions"),
    [
        (["f(0)(1500)"], 1, 4),
        (["f(0)(980)", "f(0)(1500)"], 1, 8),
        (["f(2)(1270)"], 1, 12),
        (["omega(782)"], 1, 8),
        (
            [
                "f(0)(980)",
                "f(2)(1270)",
                "f(0)(1500)",
                "f(2)(1950)",
                "omega(782)",
            ],
            2,
            40,
        ),
    ],
)
@pytest.mark.slow()
def test_number_of_solutions(
    particle_database,
    allowed_intermediate_particles,
    n_topologies,
    number_of_solutions,
):
    reaction = qrules.generate_transitions(
        initial_state=("J/psi(1S)", [-1, +1]),
        final_state=["gamma", "pi0", "pi0"],
        particle_db=particle_database,
        allowed_interaction_types=["strong", "EM"],
        allowed_intermediate_particles=allowed_intermediate_particles,
        number_of_threads=1,
        formalism="helicity",
    )
    assert len(reaction.transition_groups) == n_topologies
    assert len(reaction.transitions) == number_of_solutions
    assert (
        reaction.get_intermediate_particles().names
        == allowed_intermediate_particles
    )


def test_id_to_particle_mappings(particle_database):
    reaction = qrules.generate_transitions(
        initial_state=("J/psi(1S)", [-1, +1]),
        final_state=["gamma", "pi0", "pi0"],
        particle_db=particle_database,
        allowed_interaction_types="strong",
        allowed_intermediate_particles=["f(0)(980)"],
        number_of_threads=1,
        formalism="helicity",
    )
    assert len(reaction.transition_groups) == 1
    assert len(reaction.transitions) == 4
    iter_transitions = iter(reaction.transitions)
    first_transition = next(iter_transitions)
    graph: StateTransitionGraph[ParticleWithSpin] = first_transition.to_graph()
    ref_mapping_fs = _create_edge_id_particle_mapping(
        graph, graph.topology.outgoing_edge_ids
    )
    ref_mapping_is = _create_edge_id_particle_mapping(
        graph, graph.topology.incoming_edge_ids
    )
    for transition in iter_transitions:
        graph = transition.to_graph()
        assert ref_mapping_fs == _create_edge_id_particle_mapping(
            graph, graph.topology.outgoing_edge_ids
        )
        assert ref_mapping_is == _create_edge_id_particle_mapping(
            graph, graph.topology.incoming_edge_ids
        )
