from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import attrs
import pytest

import qrules.particle
import qrules.system_control
from qrules.conservation_rules import (
    c_parity_conservation,
    helicity_conservation,
    parity_conservation,
    spin_magnitude_conservation,
)
from qrules.quantum_numbers import EdgeQuantumNumbers, NodeQuantumNumbers
from qrules.solving import (
    CSPSolver,
    QNProblemSet,
    complete_intermediate_states,
    filter_quantum_number_problem_set,
    merge_qn_problem_sets,
    remove_dominated_qn_problem_sets,
)
from qrules.topology import MutableTransition

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
        assert len(result.solutions) == 24

    def it_with_filtered_quantum_number_problem_set(
        all_particles: list[GraphEdgePropertyMap],
        quantum_number_problem_set: QNProblemSet,
    ) -> None:
        solver = CSPSolver()
        new_quantum_number_problem_set = filter_quantum_number_problem_set(
            quantum_number_problem_set,
            edge_rules=set(),
            node_rules={
                spin_magnitude_conservation,
                parity_conservation,
                c_parity_conservation,
            },
            edge_properties={
                EdgeQuantumNumbers.pid,  # had to be added for c_parity_conservation to work
                EdgeQuantumNumbers.spin_magnitude,
                EdgeQuantumNumbers.parity,
                EdgeQuantumNumbers.c_parity,
            },
            node_properties=(
                NodeQuantumNumbers.l_magnitude,
                NodeQuantumNumbers.s_magnitude,
            ),
        )
        qn_result = solver.find_solutions(new_quantum_number_problem_set)
        result = complete_intermediate_states(
            qn_result, new_quantum_number_problem_set, all_particles
        )
        assert len(result.solutions) == 129


def test_merge_qn_problem_sets(quantum_number_problem_set: QNProblemSet) -> None:
    """Merge problem sets that differ only in custom spin-projection facts."""

    def with_initial_projection(
        problem_set: QNProblemSet, projection: Fraction
    ) -> QNProblemSet:
        facts = problem_set.initial_facts
        new_states = {i: dict(m) for i, m in facts.states.items()}
        new_states[-1][EdgeQuantumNumbers.spin_projection] = projection
        new_facts = MutableTransition(
            facts.topology,
            new_states,  # type: ignore[arg-type]
            dict(facts.interactions),  # type: ignore[arg-type]
        )
        return attrs.evolve(problem_set, initial_facts=new_facts)

    projections = [Fraction(-1), Fraction(+1)]
    variants = [
        with_initial_projection(quantum_number_problem_set, p) for p in projections
    ]
    merged = merge_qn_problem_sets(
        variants, merge_qns={EdgeQuantumNumbers.spin_projection}
    )
    assert len(merged) == 1
    merged_facts = merged[0].initial_facts.states[-1]
    assert merged_facts[EdgeQuantumNumbers.spin_projection] == projections


def test_remove_dominated_qn_problem_sets(
    quantum_number_problem_set: QNProblemSet,
) -> None:
    def with_extra_node_rule(problem_set: QNProblemSet) -> QNProblemSet:
        settings = problem_set.solving_settings
        new_interactions = {
            node_id: attrs.evolve(
                node_settings,
                conservation_rules={
                    **node_settings.conservation_rules,
                    helicity_conservation: 1,
                },
            )
            for node_id, node_settings in settings.interactions.items()
        }
        new_settings = MutableTransition(
            settings.topology,
            dict(settings.states),  # type: ignore[arg-type]
            new_interactions,  # type: ignore[arg-type]
        )
        return attrs.evolve(problem_set, solving_settings=new_settings)

    stricter = with_extra_node_rule(quantum_number_problem_set)
    deduped = remove_dominated_qn_problem_sets({
        1.0: [quantum_number_problem_set],
        60.0: [stricter],
    })
    assert deduped == {1.0: [quantum_number_problem_set]}

    exact_duplicate = attrs.evolve(quantum_number_problem_set)
    deduped = remove_dominated_qn_problem_sets({
        1.0: [quantum_number_problem_set],
        60.0: [exact_duplicate],
    })
    assert sum(len(group) for group in deduped.values()) == 1


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
        merged = merge_qn_problem_sets(
            [quantum_number_problem_set], merge_qns={EdgeQuantumNumbers.spin_projection}
        )
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
        assert (
            merge_qn_problem_sets(group, merge_qns={EdgeQuantumNumbers.spin_projection})
            == group
        )

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
        assert (
            merge_qn_problem_sets(group, merge_qns={EdgeQuantumNumbers.spin_projection})
            == group
        )


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
