from __future__ import annotations

import pytest

import qrules.particle
import qrules.quantum_numbers
import qrules.system_control
import qrules.transition
from qrules.conservation_rules import (
    c_parity_conservation,
    parity_conservation,
    spin_magnitude_conservation,
    spin_validity,
)
from qrules.quantum_numbers import EdgeQuantumNumbers, NodeQuantumNumbers
from qrules.solving import CSPSolver, QNProblemSet, filter_quantum_number_problem_set


def test_solve(
    all_particles: qrules.particle.ParticleCollection,
    quantum_number_problem_set: QNProblemSet,
) -> None:
    solver = CSPSolver(all_particles)
    result = solver.find_solutions(quantum_number_problem_set)
    assert len(result.solutions) == 19


@pytest.mark.parametrize("with_spin_projection", [True, False])
def test_solve_with_filtered_quantum_number_problem_set(
    all_particles: qrules.particle.ParticleCollection,
    quantum_number_problem_set: QNProblemSet,
    with_spin_projection: bool,
) -> None:
    solver = CSPSolver(all_particles)
    parametrized_edge_properties_and_domains = {
        EdgeQuantumNumbers.pid,  # had to be added for c_parity_conservation to work
        EdgeQuantumNumbers.spin_magnitude,
        EdgeQuantumNumbers.parity,
        EdgeQuantumNumbers.c_parity,
    }
    if with_spin_projection:
        parametrized_edge_properties_and_domains.add(EdgeQuantumNumbers.spin_projection)

    new_quantum_number_problem_set = filter_quantum_number_problem_set(
        quantum_number_problem_set,
        edge_rules={spin_validity},
        node_rules={
            spin_magnitude_conservation,
            parity_conservation,
            c_parity_conservation,
        },
        edge_properties_and_domains={
            EdgeQuantumNumbers.pid,  # had to be added for c_parity_conservation to work
            EdgeQuantumNumbers.spin_magnitude,
            # EdgeQuantumNumbers.spin_projection,  # can be left out to reduce the number of solutions
            EdgeQuantumNumbers.parity,
            EdgeQuantumNumbers.c_parity,
        },
        node_properties_and_domains=(
            NodeQuantumNumbers.l_magnitude,
            NodeQuantumNumbers.s_magnitude,
        ),
    )
    result = solver.find_solutions(new_quantum_number_problem_set)

    assert len(result.solutions) == 127


@pytest.fixture(scope="session")
def all_particles():
    return [
        qrules.system_control.create_edge_properties(part)
        for part in qrules.particle.load_pdg()
    ]


@pytest.fixture(scope="session")
def quantum_number_problem_set() -> QNProblemSet:
    stm = qrules.StateTransitionManager(
        initial_state=["psi(2S)"],
        final_state=["gamma", "eta", "eta"],
        formalism="helicity",
    )
    problem_sets = stm.create_problem_sets()
    qn_problem_sets = [
        p.to_qn_problem_set()
        for strength in sorted(problem_sets)
        for p in problem_sets[strength]
    ]
    return qn_problem_sets[0]
