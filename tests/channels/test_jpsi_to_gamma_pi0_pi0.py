import pytest

import qrules
from qrules.combinatorics import _create_edge_id_particle_mapping


def describe_reaction_generation():
    @pytest.mark.parametrize(
        ("allowed_intermediate_particles", "n_topologies", "number_of_solutions"),
        [
            (["f_0(1500)0"], 1, 4),
            (["f_0(980)0", "f_0(1500)0"], 1, 8),
            (["f_2(1270)0"], 1, 12),
            (["omega(782)0"], 1, 8),
            (
                [
                    "f_0(980)0",
                    "f_2(1270)0",
                    "f_0(1500)0",
                    "f_2(1950)0",
                    "omega(782)0",
                ],
                2,
                40,
            ),
        ],
    )
    @pytest.mark.slow
    def it_number_of_solutions(
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
            formalism="helicity",
        )
        assert len(reaction.group_by_topology()) == n_topologies
        assert len(reaction.transitions) == number_of_solutions
        assert (
            reaction.get_intermediate_particles().names
            == allowed_intermediate_particles
        )

    def it_id_to_particle_mappings(particle_database):
        reaction = qrules.generate_transitions(
            initial_state=("J/psi(1S)", [-1, +1]),
            final_state=["gamma", "pi0", "pi0"],
            particle_db=particle_database,
            allowed_interaction_types="strong",
            allowed_intermediate_particles=["f_0(980)0"],
            formalism="helicity",
        )
        assert len(reaction.group_by_topology()) == 1
        assert len(reaction.transitions) == 4
        iter_transitions = iter(reaction.transitions)
        first_transition = next(iter_transitions)
        graph = first_transition.convert(
            lambda s: (s.particle, s.spin_projection)
        ).unfreeze()
        ref_mapping_fs = _create_edge_id_particle_mapping(
            graph, graph.topology.outgoing_edge_ids
        )
        ref_mapping_is = _create_edge_id_particle_mapping(
            graph, graph.topology.incoming_edge_ids
        )
        for transition in iter_transitions:
            graph = transition.convert(
                lambda s: (s.particle, s.spin_projection)
            ).unfreeze()
            assert ref_mapping_fs == _create_edge_id_particle_mapping(
                graph, graph.topology.outgoing_edge_ids
            )
            assert ref_mapping_is == _create_edge_id_particle_mapping(
                graph, graph.topology.incoming_edge_ids
            )
