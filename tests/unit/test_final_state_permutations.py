# cspell:ignore pbar
import itertools

import pytest

import qrules
from qrules.settings import InteractionType
from qrules.transition import StateTransitionManager


@pytest.mark.parametrize(
    "final_state_description",
    sorted({" ".join(p) for p in itertools.permutations(["p~", "Sigma+", "K0"])}),
)
def test_create_problem_sets(final_state_description: str):
    input_final_state = final_state_description.split(" ")
    stm = StateTransitionManager(
        initial_state=["J/psi(1S)"],
        final_state=input_final_state,
        allowed_intermediate_particles=["N(1440)"],
    )
    stm.set_allowed_interaction_types([InteractionType.STRONG])
    problem_sets = stm.create_problem_sets()
    for problem_set in problem_sets.values():
        for problem in problem_set:
            problem_final_state = [
                problem.initial_facts.states[i][0].name for i in range(3)
            ]
            assert problem_final_state == input_final_state


@pytest.mark.parametrize(
    "final_state_description",
    sorted({" ".join(p) for p in itertools.permutations(["gamma", "pi0", "pi0"], 3)}),
)
def test_generate_transitions(final_state_description: str):
    final_state = final_state_description.split(" ")
    reaction = qrules.generate_transitions(
        initial_state=("J/psi(1S)", [-1, +1]),
        final_state=final_state,
        allowed_intermediate_particles=["omega(782)"],
        allowed_interaction_types=["strong", "EM"],
    )
    ordered_final_state = [
        reaction.final_state[i].name for i in sorted(reaction.final_state)
    ]
    assert final_state == ordered_final_state

    assert len(reaction.transitions) == 8
    for transition in reaction.transitions:
        ordered_final_state = [
            transition.final_states[i].particle.name
            for i in sorted(transition.final_states)
        ]
        assert final_state == ordered_final_state

        topology = transition.topology
        decay_products = {
            transition.states[i].particle.name
            for i in topology.get_edge_ids_outgoing_from_node(1)
        }
        assert decay_products == {"gamma", "pi0"}
