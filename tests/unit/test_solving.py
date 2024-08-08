from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

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
from qrules.quantum_numbers import EdgeQuantumNumbers, NodeQuantumNumbers
from qrules.solving import CSPSolver, EdgeSettings, NodeSettings, QNProblemSet
from qrules.topology import MutableTransition

if TYPE_CHECKING:
    from qrules.argument_handling import (
        GraphEdgePropertyMap,
        GraphNodePropertyMap,
        Rule,
    )


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
    new_quantum_number_problem_set = filter_quantum_number_problem_set_settings(
        quantum_number_problem_set,
        edge_rules={spin_validity},
        node_rules={
            spin_magnitude_conservation,
            parity_conservation,
            c_parity_conservation,
        },
        edge_domains=(
            EdgeQuantumNumbers.spin_magnitude,
            EdgeQuantumNumbers.parity,
            EdgeQuantumNumbers.c_parity,
        ),
        node_domains=(NodeQuantumNumbers.l_magnitude, NodeQuantumNumbers.s_magnitude),
    )
    new_quantum_number_problem_set = filter_quantum_number_problem_set_properties(
        new_quantum_number_problem_set,
        edge_properties=(
            EdgeQuantumNumbers.spin_magnitude,
            EdgeQuantumNumbers.parity,
            EdgeQuantumNumbers.c_parity,
        ),
        node_properties=(
            NodeQuantumNumbers.l_magnitude,
            NodeQuantumNumbers.s_magnitude,
        ),
    )
    result = solver.find_solutions(new_quantum_number_problem_set)

    assert len(result.solutions) != 0


def remove_quantum_number_problem_set_settings(
    quantum_number_problem_set: QNProblemSet,
    edge_rules_to_be_removed: set[GraphElementRule],
    node_rules_to_be_removed: set[Rule],
    edge_domains_to_be_removed: tuple[Any, ...],
    node_domains_to_be_removed: tuple[Any, ...],
) -> QNProblemSet:
    old_edge_settings = quantum_number_problem_set.solving_settings.states
    old_node_settings = quantum_number_problem_set.solving_settings.interactions
    new_edge_settings = {
        edge_id: EdgeSettings(
            conservation_rules=edge_setting.conservation_rules
            - edge_rules_to_be_removed,
            rule_priorities=edge_setting.rule_priorities,
            qn_domains={
                key: val
                for key, val in edge_setting.qn_domains.items()
                if key not in edge_domains_to_be_removed
            },
        )
        for edge_id, edge_setting in old_edge_settings.items()
    }
    new_node_settings = {
        node_id: NodeSettings(
            conservation_rules=node_setting.conservation_rules
            - node_rules_to_be_removed,
            rule_priorities=node_setting.rule_priorities,
            qn_domains={
                key: val
                for key, val in node_setting.qn_domains.items()
                if key not in node_domains_to_be_removed
            },
        )
        for node_id, node_setting in old_node_settings.items()
    }
    new_mutable_transition = MutableTransition(
        topology=quantum_number_problem_set.solving_settings.topology,
        states=new_edge_settings,
        interactions=new_node_settings,
    )
    return attrs.evolve(
        quantum_number_problem_set, solving_settings=new_mutable_transition
    )


def remove_quantum_number_problem_set_properties(
    quantum_number_problem_set: QNProblemSet,
    edge_properties_to_be_removed: tuple[EdgeQuantumNumbers],
    node_properties_to_be_removed: tuple[NodeQuantumNumbers],
) -> QNProblemSet:
    old_edge_properties = quantum_number_problem_set.initial_facts.states
    old_node_properties = quantum_number_problem_set.initial_facts.interactions
    new_edge_properties = {
        edge_id: {
            edge_quantum_number: scalar
            for edge_quantum_number, scalar in graph_edge_property_map.items()
            if edge_quantum_number not in edge_properties_to_be_removed
        }
        for edge_id, graph_edge_property_map in old_edge_properties.items()
    }
    new_node_properties = {
        node_id: {
            node_quantum_number: scalar
            for node_quantum_number, scalar in graph_node_property_map.items()
            if node_quantum_number not in node_properties_to_be_removed
        }
        for node_id, graph_node_property_map in old_node_properties.items()
    }
    new_mutable_transition = MutableTransition(
        topology=quantum_number_problem_set.initial_facts.topology,
        states=new_edge_properties,
        interactions=new_node_properties,
    )
    return attrs.evolve(
        quantum_number_problem_set, initial_facts=new_mutable_transition
    )


def test_inner_dicts_unchanged(
    quantum_number_problem_set: QNProblemSet,
) -> None:
    old_inner_graph_edge_property_map = copy.deepcopy(
        quantum_number_problem_set.initial_facts.states
    )
    old_inner_graph_node_property_map = copy.deepcopy(
        quantum_number_problem_set.initial_facts.interactions
    )
    graph_edge_property_map = {
        EdgeQuantumNumbers.spin_magnitude: 1,
        EdgeQuantumNumbers.parity: -1,
        EdgeQuantumNumbers.c_parity: 1,
    }
    graph_node_property_map = {
        NodeQuantumNumbers.s_magnitude: 1,
        NodeQuantumNumbers.l_magnitude: 0,
    }
    quantum_number_problem_set_with_new_properties(
        quantum_number_problem_set, graph_edge_property_map, graph_node_property_map
    )
    assert (
        old_inner_graph_edge_property_map
        == quantum_number_problem_set.initial_facts.states
    )
    assert (
        old_inner_graph_node_property_map
        == quantum_number_problem_set.initial_facts.interactions
    )


def filter_quantum_number_problem_set_settings(
    quantum_number_problem_set: QNProblemSet,
    edge_rules: set[GraphElementRule],
    node_rules: set[Rule],
    edge_domains: tuple[Any, ...],
    node_domains: tuple[Any, ...],
    keep_domains: bool = True,
) -> QNProblemSet:
    old_edge_settings = quantum_number_problem_set.solving_settings.states
    old_node_settings = quantum_number_problem_set.solving_settings.interactions
    new_edge_settings = {
        edge_id: EdgeSettings(
            conservation_rules=edge_rules,
            rule_priorities=edge_setting.rule_priorities,
            qn_domains=(
                edge_setting.qn_domains
                if keep_domains
                else {
                    key: val
                    for key, val in edge_setting.qn_domains.items()
                    if key in edge_domains
                }
            ),
        )
        for edge_id, edge_setting in old_edge_settings.items()
    }
    new_node_settings = {
        node_id: NodeSettings(
            conservation_rules=node_rules,
            rule_priorities=node_setting.rule_priorities,
            qn_domains=(
                node_setting.qn_domains
                if keep_domains
                else {
                    key: val
                    for key, val in node_setting.qn_domains.items()
                    if key in node_domains
                }
            ),
        )
        for node_id, node_setting in old_node_settings.items()
    }
    new_mutable_transition = MutableTransition(
        topology=quantum_number_problem_set.solving_settings.topology,
        states=new_edge_settings,
        interactions=new_node_settings,
    )
    return attrs.evolve(
        quantum_number_problem_set, solving_settings=new_mutable_transition
    )


def filter_quantum_number_problem_set_properties(
    quantum_number_problem_set: QNProblemSet,
    edge_properties: tuple[EdgeQuantumNumbers],
    node_properties: tuple[NodeQuantumNumbers],
) -> QNProblemSet:
    old_edge_properties = quantum_number_problem_set.initial_facts.states
    old_node_properties = quantum_number_problem_set.initial_facts.interactions
    new_edge_properties = {
        edge_id: {
            edge_quantum_number: scalar
            for edge_quantum_number, scalar in graph_edge_property_map.items()
            if edge_quantum_number in edge_properties
        }
        for edge_id, graph_edge_property_map in old_edge_properties.items()
    }
    new_node_properties = {
        node_id: {
            node_quantum_number: scalar
            for node_quantum_number, scalar in graph_node_property_map.items()
            if node_quantum_number in node_properties
        }
        for node_id, graph_node_property_map in old_node_properties.items()
    }
    new_mutable_transition = MutableTransition(
        topology=quantum_number_problem_set.initial_facts.topology,
        states=new_edge_properties,
        interactions=new_node_properties,
    )
    return attrs.evolve(
        quantum_number_problem_set, initial_facts=new_mutable_transition
    )


def quantum_number_problem_set_with_new_settings(
    quantum_number_problem_set: QNProblemSet,
    edge_rules: set[GraphElementRule],
    node_rules: set[Rule],
    edge_domains: dict[Any, list],
    node_domains: dict[Any, list],
) -> QNProblemSet:
    def qnp_with_new_rules(
        quantum_number_problem_set: QNProblemSet,
        edge_rules: set[GraphElementRule],
        node_rules: set[Rule],
    ) -> QNProblemSet:
        old_settings = quantum_number_problem_set.solving_settings
        new_settings = attrs.evolve(
            quantum_number_problem_set.solving_settings,
            states={
                edge_id: attrs.evolve(
                    setting, conservation_rules=setting.conservation_rules & edge_rules
                )
                for edge_id, setting in old_settings.states.items()
            },
            interactions={
                node_id: attrs.evolve(
                    setting, conservation_rules=setting.conservation_rules & node_rules
                )
                for node_id, setting in old_settings.interactions.items()
            },
        )
        return attrs.evolve(quantum_number_problem_set, solving_settings=new_settings)

    def qnp_with_new_domains(
        quantum_number_problem_set: QNProblemSet,
        edge_domains: dict[Any, list],
        node_domains: dict[Any, list],
    ) -> QNProblemSet:
        old_settings = quantum_number_problem_set.solving_settings
        new_settings = attrs.evolve(
            old_settings,
            states={
                edge_id: attrs.evolve(setting, qn_domains=edge_domains)
                for edge_id, setting in old_settings.states.items()
            },
            interactions={
                node_id: attrs.evolve(setting, qn_domains=node_domains)
                for node_id, setting in old_settings.interactions.items()
            },
        )
        return attrs.evolve(quantum_number_problem_set, solving_settings=new_settings)

    return qnp_with_new_rules(
        qnp_with_new_domains(quantum_number_problem_set, edge_domains, node_domains),
        edge_rules,
        node_rules,
    )


def quantum_number_problem_set_with_new_properties(
    quantum_number_problem_set: QNProblemSet,
    graph_edge_property_map: GraphEdgePropertyMap,
    graph_node_property_map: GraphNodePropertyMap,
) -> QNProblemSet:
    old_facts = quantum_number_problem_set.initial_facts
    new_facts = attrs.evolve(
        old_facts,
        states={
            node_id: {
                key: val
                for key, val in prop_map.items()
                if key in graph_edge_property_map
            }
            for node_id, prop_map in old_facts.states.items()
        },
        interactions={
            node_id: {
                key: val
                for key, val in prop_map.items()
                if key in graph_node_property_map
            }
            for node_id, prop_map in old_facts.interactions.items()
        },
    )
    return attrs.evolve(quantum_number_problem_set, initial_facts=new_facts)


@pytest.fixture(scope="session")
def all_particles():
    return [
        qrules.system_control.create_edge_properties(part)
        for part in qrules.particle.load_pdg()
    ]


@pytest.fixture(scope="session")
def quantum_number_problem_set(request) -> QNProblemSet:
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
