import qrules.particle
import qrules.solving
import qrules.system_control
import qrules.transition


def test_find_solutions():
    stm = qrules.StateTransitionManager(
        initial_state=["psi(2S)"],
        final_state=["gamma", "eta", "eta"],
        formalism="helicity",
    )
    problem_sets = stm.create_problem_sets()
    qn_problem_sets = [
        p.to_qn_problem_set() for pl in problem_sets.values() for p in pl
    ]

    # in principle the allowed intermediate states are already in the
    particles = qrules.load_pdg()
    allowed_intermediate_states = [
        qrules.system_control.create_edge_properties(part) for part in particles
    ]
    solver = qrules.solving.CSPSolver(allowed_intermediate_states)
    solutions = solver.find_solutions(qn_problem_sets[0])

    assert isinstance(solutions, qrules.solving.QNResult)
