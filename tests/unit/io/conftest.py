import pytest

from qrules.particle import ParticleCollection
from qrules.settings import InteractionType
from qrules.solving import QNProblemSet, QNResult
from qrules.transition import ProblemSet, StateTransitionManager


@pytest.fixture(scope="session")
def particle_selection(particle_database: ParticleCollection):
    selection = ParticleCollection()
    selection += particle_database.filter(lambda p: p.name.startswith("pi"))
    selection += particle_database.filter(lambda p: p.name.startswith("K"))
    selection += particle_database.filter(lambda p: p.name.startswith("D"))
    selection += particle_database.filter(lambda p: p.name.startswith("J/psi"))
    return selection


@pytest.fixture
def stm() -> StateTransitionManager:
    stm = StateTransitionManager(
        initial_state=[("J/psi(1S)", [+1])],
        final_state=["K0", ("Sigma+", [+0.5]), ("p~", [+0.5])],
        allowed_intermediate_particles=["Sigma(1750)"],
        formalism="canonical-helicity",
    )
    stm.set_allowed_interaction_types([InteractionType.STRONG, InteractionType.EM])
    return stm


@pytest.fixture
def problem_sets(stm: StateTransitionManager) -> dict[float, list[ProblemSet]]:
    return stm.create_problem_sets()


@pytest.fixture
def qn_problem_and_result(
    stm: StateTransitionManager,
    problem_sets: dict[float, list[ProblemSet]],
) -> tuple[QNProblemSet, QNResult]:
    qn_solutions = stm.find_quantum_number_transitions(problem_sets)
    strong_qn_solutions = qn_solutions[3600.0]
    return strong_qn_solutions[1]
