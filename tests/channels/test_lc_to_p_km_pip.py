from qrules.settings import InteractionType
from qrules.transition import StateTransitionManager


def test_resonances():
    stm = StateTransitionManager(
        initial_state=["Lambda(c)+"],
        final_state=["p", "K-", "pi+"],
        allowed_intermediate_particles=["Delta", "K", "Lambda"],
        mass_conservation_factor=0.6,
    )
    stm.set_allowed_interaction_types([InteractionType.STRONG], node_id=1)
    problem_sets = stm.create_problem_sets()
    reaction = stm.find_solutions(problem_sets)
    resonances = reaction.get_intermediate_particles().names
    assert resonances == [
        "Delta(1232)++",
        "Delta(1600)++",
        "Delta(1620)++",
        "Delta(1900)++",
        "Delta(1910)++",
        "Delta(1920)++",
        "K(0)*(700)~0",
        "K*(892)~0",
        "K*(1410)~0",
        "K(0)*(1430)~0",
        "Lambda(1405)",
        "Lambda(1600)",
        "Lambda(1670)",
        "Lambda(1810)",
        "Lambda(1800)",
        "Lambda(1890)",
    ]
