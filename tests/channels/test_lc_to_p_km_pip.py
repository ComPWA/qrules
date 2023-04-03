from qrules.settings import InteractionType
from qrules.transition import StateTransitionManager


def test_resonances():
    stm = StateTransitionManager(
        initial_state=["Lambda(c)+"],
        final_state=["p", "K-", "pi+"],
        mass_conservation_factor=0,
    )
    stm.set_allowed_interaction_types([InteractionType.STRONG], node_id=1)
    problem_sets = stm.create_problem_sets()
    reaction = stm.find_solutions(problem_sets)
    resonances = reaction.get_intermediate_particles().names
    assert resonances == [
        "Delta(1232)++",
        "Delta(1600)++",
        "Delta(1620)++",
        "K(0)*(700)~0",
        "K*(892)~0",
        "Lambda(1600)",
        "Lambda(1670)",
        "Lambda(1810)",
        "Lambda(1800)",
        "Lambda(1890)",
        "Sigma(1660)0",
        "Sigma(1750)0",
    ]
