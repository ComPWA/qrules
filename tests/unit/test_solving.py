from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import attrs
import pytest

import qrules.particle
import qrules.quantum_numbers
import qrules.solving
import qrules.system_control
import qrules.transition
from qrules.conservation_rules import (
    GraphElementRule,
    isospin_validity,
    parity_conservation,
    spin_magnitude_conservation,
    spin_validity,
)
from qrules.quantum_numbers import EdgeQuantumNumbers, NodeQuantumNumbers

if TYPE_CHECKING:
    from qrules.argument_handling import (
        GraphEdgePropertyMap,
        GraphNodePropertyMap,
        Rule,
    )


def test_solve_with_new_quantum_number_problem_set(
    particle_database: qrules.particle.ParticleCollection,
    qn_problem_set: qrules.solving.QNProblemSet,
) -> None:
    allowed_intermediate_states = [
        qrules.system_control.create_edge_properties(part) for part in particle_database
    ]
    solver = qrules.solving.CSPSolver(allowed_intermediate_states)
    new_qn_problem_set = quantum_number_problem_set_with_new_properties(
        qn_problem_set,
        graph_edge_property_map={
            EdgeQuantumNumbers.spin_magnitude: 1,
            EdgeQuantumNumbers.parity: -1,
            EdgeQuantumNumbers.c_parity: 1,
        },
        graph_node_property_map={
            NodeQuantumNumbers.s_magnitude: 1,
            NodeQuantumNumbers.l_magnitude: 0,
        },
    )
    new_qn_problem_set = quantum_number_problem_set_with_new_settings(
        new_qn_problem_set,
        edge_rules={spin_validity, isospin_validity},
        node_rules={spin_magnitude_conservation, parity_conservation},
        edge_domains={"spin_magnitude": [1], "parity": [-1, 1], "c_parity": [-1, 1]},
        node_domains={"l_projection": [0]},
    )
    solutions = solver.find_solutions(new_qn_problem_set)

    assert isinstance(solutions, qrules.solving.QNResult)


def test_inner_dicts_unchanged(
    qn_problem_set: qrules.solving.QNProblemSet,
    graph_edge_property_map: GraphEdgePropertyMap,
    graph_node_property_map: GraphNodePropertyMap,
) -> None:
    old_inner_graph_edge_property_map = copy.deepcopy(
        qn_problem_set.initial_facts.states
    )
    old_inner_graph_node_property_map = copy.deepcopy(
        qn_problem_set.initial_facts.interactions
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
        qn_problem_set, graph_edge_property_map, graph_node_property_map
    )
    assert old_inner_graph_edge_property_map == qn_problem_set.initial_facts.states
    assert (
        old_inner_graph_node_property_map == qn_problem_set.initial_facts.interactions
    )


def quantum_number_problem_set_with_new_settings(
    qn_problem_set: qrules.solving.QNProblemSet,
    edge_rules: set[GraphElementRule],
    node_rules: set[Rule],
    edge_domains: dict[Any, list],
    node_domains: dict[Any, list],
) -> qrules.solving.QNProblemSet:
    def qnp_with_new_rules(
        qn_problem_set: qrules.solving.QNProblemSet,
        edge_rules: set[GraphElementRule],
        node_rules: set[Rule],
    ) -> qrules.solving.QNProblemSet:
        old_settings = qn_problem_set.solving_settings
        new_settings = attrs.evolve(
            qn_problem_set.solving_settings,
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
        return attrs.evolve(qn_problem_set, solving_settings=new_settings)

    def qnp_with_new_domains(
        qn_problem_set: qrules.solving.QNProblemSet,
        edge_domains: dict[Any, list],
        node_domains: dict[Any, list],
    ) -> qrules.solving.QNProblemSet:
        old_settings = qn_problem_set.solving_settings
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
        return attrs.evolve(qn_problem_set, solving_settings=new_settings)

    return qnp_with_new_rules(
        qnp_with_new_domains(qn_problem_set, edge_domains, node_domains),
        edge_rules,
        node_rules,
    )


def quantum_number_problem_set_with_new_properties(
    qn_problem_set: qrules.solving.QNProblemSet,
    graph_edge_property_map: GraphEdgePropertyMap,
    graph_node_property_map: GraphNodePropertyMap,
) -> qrules.solving.QNProblemSet:
    old_facts = qn_problem_set.initial_facts
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
    return attrs.evolve(qn_problem_set, initial_facts=new_facts)


@pytest.fixture(scope="session")
def qn_problem_set() -> qrules.solving.QNProblemSet:
    stm = qrules.StateTransitionManager(
        initial_state=["psi(2S)"],
        final_state=["gamma", "eta", "eta"],
        formalism="helicity",
    )
    problem_sets = stm.create_problem_sets()
    qn_problem_sets = [
        p.to_qn_problem_set() for pl in problem_sets.values() for p in pl
    ]
    return qn_problem_sets[0]
