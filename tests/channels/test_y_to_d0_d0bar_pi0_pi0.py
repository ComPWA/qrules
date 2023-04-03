import pytest

import qrules
from qrules import InteractionType, StateTransitionManager


@pytest.mark.parametrize(
    ("formalism", "n_solutions"),
    [
        ("helicity", 14),
        ("canonical-helicity", 28),  # two different LS couplings 2*14 = 28
    ],
)
def test_simple(formalism, n_solutions, particle_database):
    reaction = qrules.generate_transitions(
        initial_state=[("Y(4260)", [-1, +1])],
        final_state=["D*(2007)0", "D*(2007)~0"],
        particle_db=particle_database,
        formalism=formalism,
        allowed_interaction_types="strong",
    )
    assert len(reaction.group_by_topology()) == 1
    assert len(reaction.transitions) == n_solutions


@pytest.mark.slow()
@pytest.mark.parametrize(
    ("formalism", "n_solutions"),
    [
        ("helicity", 14),
        ("canonical-helicity", 28),  # two different LS couplings 2*14 = 28
    ],
)
def test_full(formalism, n_solutions, particle_database):
    stm = StateTransitionManager(
        initial_state=[("Y(4260)", [-1, +1])],
        final_state=["D0", "D~0", "pi0", "pi0"],
        particle_db=particle_database,
        allowed_intermediate_particles=["D*"],
        formalism=formalism,
    )
    stm.set_allowed_interaction_types([InteractionType.STRONG])
    stm.add_final_state_grouping([["D0", "pi0"], ["D~0", "pi0"]])
    problem_sets = stm.create_problem_sets()
    reaction = stm.find_solutions(problem_sets)
    assert len(reaction.group_by_topology()) == 1
    assert len(reaction.transitions) == n_solutions


def test_resonance_filter(particle_database):
    # https://github.com/ComPWA/qrules/issues/33
    stm = StateTransitionManager(
        initial_state=[("Y(4260)", [-1, +1])],
        final_state=["D0", "D~0", "pi0", "pi0"],
        allowed_intermediate_particles=["D"],
        particle_db=particle_database,
    )
    stm.set_allowed_interaction_types([InteractionType.STRONG])
    stm.add_final_state_grouping([["D0", "pi0"], ["D~0", "pi0"]])
    problem_sets = stm.create_problem_sets()
    result = stm.find_solutions(problem_sets)
    assert set(result.get_intermediate_particles().names) == {
        "D*(2007)0",
        "D*(2007)~0",
        "D(0)*(2300)0",
        "D(0)*(2300)~0",
    }
