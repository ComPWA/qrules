import pytest

import qrules as q
from qrules import InteractionType, StateTransitionManager


@pytest.mark.parametrize(
    ("formalism_type", "n_solutions"),
    [
        ("helicity", 14),
        ("canonical-helicity", 28),  # two different LS couplings 2*14 = 28
    ],
)
def test_simple(formalism_type, n_solutions, particle_database):
    result = q.generate_transitions(
        initial_state=[("Y(4260)", [-1, +1])],
        final_state=["D*(2007)0", "D*(2007)~0"],
        particle_db=particle_database,
        formalism_type=formalism_type,
        allowed_interaction_types="strong",
        number_of_threads=1,
    )
    assert len(result.transitions) == n_solutions


@pytest.mark.slow()
@pytest.mark.parametrize(
    ("formalism_type", "n_solutions"),
    [
        ("helicity", 14),
        ("canonical-helicity", 28),  # two different LS couplings 2*14 = 28
    ],
)
def test_full(formalism_type, n_solutions, particle_database):
    stm = StateTransitionManager(
        initial_state=[("Y(4260)", [-1, +1])],
        final_state=["D0", "D~0", "pi0", "pi0"],
        particle_db=particle_database,
        allowed_intermediate_particles=["D*"],
        formalism_type=formalism_type,
        number_of_threads=1,
    )
    stm.set_allowed_interaction_types([InteractionType.STRONG])
    stm.add_final_state_grouping([["D0", "pi0"], ["D~0", "pi0"]])
    problem_sets = stm.create_problem_sets()
    result = stm.find_solutions(problem_sets)
    assert len(result.transitions) == n_solutions
