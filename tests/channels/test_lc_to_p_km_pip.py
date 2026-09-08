from qrules.settings import InteractionType
from qrules.transition import StateTransitionManager


def test_resonances():
    stm = StateTransitionManager(
        initial_state=["Lambda_c()+"],
        final_state=["p", "K-", "pi+"],
        mass_conservation_factor=0.6,
        max_angular_momentum=2,
    )
    stm.set_allowed_interaction_types([InteractionType.STRONG], node_id=1)
    stm.set_allowed_intermediate_particles([r"Delta..(?!9)", r"^K", r"^L"], regex=True)
    problem_sets = stm.create_problem_sets()
    reaction = stm.find_solutions(problem_sets)
    sorted_resonances = sorted(reaction.get_intermediate_particles().names)
    # https://lc2pkpi-polarimetry.docs.cern.ch/amplitude-model.html#resonances-and-ls-scheme
    expected = {
        "Delta(1232)++",
        "Delta(1600)++",
        "Delta(1620)++",
        "Delta(1700)++",
        "Kbar_0^*(700)0",
        "Kbar^*(892)0",
        "Kbar^*(1410)0",
        "Kbar_0^*(1430)0",
        "Lambda(1405)0",
        "Lambda(1520)0",
        "Lambda(1600)0",
        "Lambda(1670)0",
        "Lambda(1690)0",
        "Lambda(1810)0",
        "Lambda(1800)0",
        "Lambda(1890)0",
    }
    sorted_expected = sorted(expected)
    assert sorted_resonances == sorted_expected
