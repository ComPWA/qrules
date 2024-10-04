from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import attrs
import pytest

import qrules.particle
import qrules.quantum_numbers
import qrules.system_control
import qrules.transition
from qrules.conservation_rules import (
    GraphElementRule,
    c_parity_conservation,
    parity_conservation,
    spin_magnitude_conservation,
    spin_validity,
)
from qrules.quantum_numbers import (
    EdgeQuantumNumbers,
    EdgeQuantumNumberTypes,
    NodeQuantumNumbers,
    NodeQuantumNumberTypes,
)
from qrules.solving import CSPSolver, EdgeSettings, NodeSettings, QNProblemSet
from qrules.topology import MutableTransition

if TYPE_CHECKING:
    from qrules.argument_handling import Rule


def test_solve(
    all_particles: qrules.particle.ParticleCollection,
    quantum_number_problem_set: QNProblemSet,
) -> None:
    solver = CSPSolver(all_particles)
    result = solver.find_solutions(quantum_number_problem_set)
    assert len(result.solutions) == 19


def test_solve_with_filtered_quantum_number_problem_set(
    all_particles: qrules.particle.ParticleCollection,
    quantum_number_problem_set: QNProblemSet,
) -> None:
    solver = CSPSolver(all_particles)
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

    assert len(result.solutions) != 0


def filter_quantum_number_problem_set(
    quantum_number_problem_set: QNProblemSet,
    edge_rules: set[GraphElementRule],
    node_rules: set[Rule],
    edge_properties_and_domains: Iterable[EdgeQuantumNumberTypes],
    node_properties_and_domains: Iterable[NodeQuantumNumberTypes],
) -> QNProblemSet:
    old_edge_settings = quantum_number_problem_set.solving_settings.states
    old_node_settings = quantum_number_problem_set.solving_settings.interactions
    old_edge_properties = quantum_number_problem_set.initial_facts.states
    old_node_properties = quantum_number_problem_set.initial_facts.interactions
    new_edge_settings = {
        edge_id: EdgeSettings(
            conservation_rules=edge_rules,
            rule_priorities=edge_setting.rule_priorities,
            qn_domains=({
                key: val
                for key, val in edge_setting.qn_domains.items()
                if key in set(edge_properties_and_domains)
            }),
        )
        for edge_id, edge_setting in old_edge_settings.items()
    }
    new_node_settings = {
        node_id: NodeSettings(
            conservation_rules=node_rules,
            rule_priorities=node_setting.rule_priorities,
            qn_domains=({
                key: val
                for key, val in node_setting.qn_domains.items()
                if key in set(node_properties_and_domains)
            }),
        )
        for node_id, node_setting in old_node_settings.items()
    }
    new_combined_settings = MutableTransition(
        topology=quantum_number_problem_set.solving_settings.topology,
        states=new_edge_settings,
        interactions=new_node_settings,
    )
    new_edge_properties = {
        edge_id: {
            edge_quantum_number: scalar
            for edge_quantum_number, scalar in graph_edge_property_map.items()
            if edge_quantum_number in edge_properties_and_domains
        }
        for edge_id, graph_edge_property_map in old_edge_properties.items()
    }
    new_node_properties = {
        node_id: {
            node_quantum_number: scalar
            for node_quantum_number, scalar in graph_node_property_map.items()
            if node_quantum_number in node_properties_and_domains
        }
        for node_id, graph_node_property_map in old_node_properties.items()
    }
    new_combined_properties = MutableTransition(
        topology=quantum_number_problem_set.initial_facts.topology,
        states=new_edge_properties,
        interactions=new_node_properties,
    )
    return attrs.evolve(
        quantum_number_problem_set,
        solving_settings=new_combined_settings,
        initial_facts=new_combined_properties,
    )


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
