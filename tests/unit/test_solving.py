from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from qrules.argument_handling import GraphEdgePropertyMap


def test_solve(
    all_particles: qrules.particle.ParticleCollection,
    quantum_number_problem_set: QNProblemSet,
) -> None:
    solver = CSPSolver(all_particles)
    result = solver.find_solutions(quantum_number_problem_set)
    assert len(result.solutions) == 19


@pytest.mark.parametrize("with_spin_projection", [True, False])
def test_solve_with_filtered_quantum_number_problem_set(
    all_particles: list[GraphEdgePropertyMap],
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
        edge_properties=parametrized_edge_properties_and_domains,
        node_properties=(
            NodeQuantumNumbers.l_magnitude,
            NodeQuantumNumbers.s_magnitude,
        ),
    )
    result = solver.find_solutions(new_quantum_number_problem_set)

    if with_spin_projection:
        assert len(result.solutions) == 319
    else:
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
    return next(
        p.to_qn_problem_set()
        for strength in sorted(problem_sets)
        for p in problem_sets[strength]
    )
