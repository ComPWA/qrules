from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import qrules.particle
import qrules.system_control
from qrules.conservation_rules import (
    c_parity_conservation,
    parity_conservation,
    spin_magnitude_conservation,
    spin_validity,
)
from qrules.quantum_numbers import EdgeQuantumNumbers, NodeQuantumNumbers, Parity
from qrules.solving import CSPSolver, QNProblemSet, filter_quantum_number_problem_set

if TYPE_CHECKING:
    from qrules.argument_handling import GraphEdgePropertyMap


def describe_CSPSolver():
    def it_finds_solutions(
        all_particles: qrules.particle.ParticleCollection,
        quantum_number_problem_set: QNProblemSet,
    ) -> None:
        solver = CSPSolver(all_particles)
        result = solver.find_solutions(quantum_number_problem_set)
        assert len(result.solutions) == 19

    @pytest.mark.parametrize(
        ("with_spin_projection", "expected_solutions"),
        # Includes resonances with unknown parity, like D(s)*+, W+, and Z0
        [(True, 404), (False, 158)],
        ids=["with-spin-projection", "without-spin-projection"],
    )
    def it_with_filtered_quantum_number_problem_set(
        all_particles: list[GraphEdgePropertyMap],
        quantum_number_problem_set: QNProblemSet,
        with_spin_projection: bool,
        expected_solutions: int,
    ) -> None:
        solver = CSPSolver(all_particles)
        parametrized_edge_properties_and_domains = {
            EdgeQuantumNumbers.pid,  # had to be added for c_parity_conservation to work
            EdgeQuantumNumbers.spin_magnitude,
            EdgeQuantumNumbers.parity,
            EdgeQuantumNumbers.c_parity,
        }
        if with_spin_projection:
            parametrized_edge_properties_and_domains.add(
                EdgeQuantumNumbers.spin_projection
            )

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

        assert len(result.solutions) == expected_solutions


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


def describe_get_candidate_states():
    @pytest.mark.parametrize(
        ("solved_value", "expected"),
        [(-1, True), (+1, True), (None, False)],
    )
    def it_matches_unknown_parity_with_any_defined_value(
        solved_value: int | None, expected: bool
    ):
        get_candidate_states = getattr(qrules.solving, "__get_candidate_states")
        unknown = Parity.UNKNOWN.value
        candidate = {EdgeQuantumNumbers.pid: 23, EdgeQuantumNumbers.c_parity: unknown}
        state = {EdgeQuantumNumbers.pid: 23, EdgeQuantumNumbers.c_parity: solved_value}
        candidates = get_candidate_states(state, [candidate])
        assert (candidates == [candidate]) is expected

    def it_does_not_match_known_parity_with_other_value():
        get_candidate_states = getattr(qrules.solving, "__get_candidate_states")
        candidate = {EdgeQuantumNumbers.parity: -1}
        assert get_candidate_states({EdgeQuantumNumbers.parity: +1}, [candidate]) == []
        assert get_candidate_states({EdgeQuantumNumbers.parity: -1}, [candidate]) == [
            candidate
        ]
