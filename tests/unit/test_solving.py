from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import pytest

import qrules.particle
import qrules.system_control
from qrules.conservation_rules import (
    c_parity_conservation,
    parity_conservation,
    spin_magnitude_conservation,
    spin_validity,
)
from qrules.quantum_numbers import EdgeQuantumNumbers, NodeQuantumNumbers
from qrules.solving import (
    CSPSolver,
    QNProblemSet,
    complete_intermediate_states,
    filter_quantum_number_problem_set,
    merge_qn_problem_sets,
)

if TYPE_CHECKING:
    from qrules.argument_handling import GraphEdgePropertyMap


def describe_CSPSolver():
    def it_finds_solutions(
        all_particles: qrules.particle.ParticleCollection,
        quantum_number_problem_set: QNProblemSet,
    ) -> None:
        solver = CSPSolver()
        qn_result = solver.find_solutions(quantum_number_problem_set)
        result = complete_intermediate_states(
            qn_result, quantum_number_problem_set, all_particles
        )
        assert len(result.solutions) == 19

    @pytest.mark.parametrize(
        ("with_spin_projection", "expected_solutions"),
        [(True, 321), (False, 129)],
        ids=["with-spin-projection", "without-spin-projection"],
    )
    def it_with_filtered_quantum_number_problem_set(
        all_particles: list[GraphEdgePropertyMap],
        quantum_number_problem_set: QNProblemSet,
        with_spin_projection: bool,
        expected_solutions: int,
    ) -> None:
        solver = CSPSolver()
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
        qn_result = solver.find_solutions(new_quantum_number_problem_set)
        result = complete_intermediate_states(
            qn_result, new_quantum_number_problem_set, all_particles
        )

        assert len(result.solutions) == expected_solutions


def test_merge_qn_problem_sets() -> None:
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
    merged_problem_sets = merge_qn_problem_sets(qn_problem_sets)
    assert len(merged_problem_sets) < len(qn_problem_sets)

    def n_total_solutions(problem_sets: list[QNProblemSet]) -> int:
        solver = CSPSolver()
        return sum(len(solver.find_solutions(p).solutions) for p in problem_sets)

    assert n_total_solutions(merged_problem_sets) == n_total_solutions(qn_problem_sets)


def _with_spin_projections(
    problem_set: QNProblemSet, projections: dict[int, float]
) -> QNProblemSet:
    states = {
        edge_id: dict(prop_map)
        for edge_id, prop_map in problem_set.initial_facts.states.items()
    }
    for edge_id, projection in projections.items():
        states[edge_id][EdgeQuantumNumbers.spin_projection] = projection
    initial_facts = attrs.evolve(problem_set.initial_facts, states=states)
    return attrs.evolve(problem_set, initial_facts=initial_facts)


def describe_merge_qn_problem_sets():
    def it_keeps_a_single_problem_set(quantum_number_problem_set: QNProblemSet) -> None:
        merged = merge_qn_problem_sets([quantum_number_problem_set])
        assert merged == [quantum_number_problem_set]

    def it_does_not_merge_incomplete_products(
        quantum_number_problem_set: QNProblemSet,
    ) -> None:
        edge_a, edge_b, *_ = sorted(quantum_number_problem_set.initial_facts.states)
        group = [
            _with_spin_projections(quantum_number_problem_set, {edge_a: 0, edge_b: 0}),
            _with_spin_projections(quantum_number_problem_set, {edge_a: 1, edge_b: 0}),
            _with_spin_projections(quantum_number_problem_set, {edge_a: 0, edge_b: 1}),
        ]
        assert merge_qn_problem_sets(group) == group

    def it_does_not_merge_inhomogeneous_facts(
        quantum_number_problem_set: QNProblemSet,
    ) -> None:
        edge_id = min(quantum_number_problem_set.initial_facts.states)
        with_projection = _with_spin_projections(
            quantum_number_problem_set, {edge_id: 0}
        )
        states = {
            i: dict(prop_map)
            for i, prop_map in with_projection.initial_facts.states.items()
        }
        del states[edge_id][EdgeQuantumNumbers.spin_projection]
        without_projection = attrs.evolve(
            with_projection,
            initial_facts=attrs.evolve(with_projection.initial_facts, states=states),
        )
        group = [with_projection, without_projection]
        assert merge_qn_problem_sets(group) == group


def describe_complete_intermediate_states():
    def it_respects_node_facts(
        all_particles: list[GraphEdgePropertyMap],
        quantum_number_problem_set: QNProblemSet,
    ) -> None:
        solver = CSPSolver()
        reference = complete_intermediate_states(
            solver.find_solutions(quantum_number_problem_set),
            quantum_number_problem_set,
            all_particles,
        )
        node_id, node_props = next(iter(reference.solutions[0].interactions.items()))
        l_magnitude = node_props[NodeQuantumNumbers.l_magnitude]
        s_magnitude = node_props[NodeQuantumNumbers.s_magnitude]
        node_facts = {
            **node_props,
            NodeQuantumNumbers.s_magnitude: [s_magnitude],  # value range
        }
        initial_facts = attrs.evolve(
            quantum_number_problem_set.initial_facts,
            interactions={node_id: node_facts},
        )
        problem_set = attrs.evolve(
            quantum_number_problem_set, initial_facts=initial_facts
        )
        result = complete_intermediate_states(
            solver.find_solutions(problem_set), problem_set, all_particles
        )
        assert 0 < len(result.solutions) <= len(reference.solutions)
        for solution in result.solutions:
            props = solution.interactions[node_id]
            assert props[NodeQuantumNumbers.l_magnitude] == l_magnitude
            assert props[NodeQuantumNumbers.s_magnitude] == s_magnitude


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
