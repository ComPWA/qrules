from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Union

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
    from qrules.argument_handling import Rule


EdgeQuantumNumberTypes = Union[
    type[EdgeQuantumNumbers.pid],
    type[EdgeQuantumNumbers.mass],
    type[EdgeQuantumNumbers.width],
    type[EdgeQuantumNumbers.spin_magnitude],
    type[EdgeQuantumNumbers.spin_projection],
    type[EdgeQuantumNumbers.charge],
    type[EdgeQuantumNumbers.isospin_magnitude],
    type[EdgeQuantumNumbers.isospin_projection],
    type[EdgeQuantumNumbers.strangeness],
    type[EdgeQuantumNumbers.charmness],
    type[EdgeQuantumNumbers.bottomness],
    type[EdgeQuantumNumbers.topness],
    type[EdgeQuantumNumbers.baryon_number],
    type[EdgeQuantumNumbers.electron_lepton_number],
    type[EdgeQuantumNumbers.muon_lepton_number],
    type[EdgeQuantumNumbers.tau_lepton_number],
    type[EdgeQuantumNumbers.parity],
    type[EdgeQuantumNumbers.c_parity],
    type[EdgeQuantumNumbers.g_parity],
]

NodeQuantumNumberTypes = Union[
    type[NodeQuantumNumbers.l_magnitude],
    type[NodeQuantumNumbers.l_projection],
    type[NodeQuantumNumbers.s_magnitude],
    type[NodeQuantumNumbers.s_projection],
    type[NodeQuantumNumbers.parity_prefactor],
]


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
            EdgeQuantumNumbers.pid,  # had to be added for c_parity_conservation to work
            EdgeQuantumNumbers.spin_magnitude,
            EdgeQuantumNumbers.spin_projection,  # had to be added for spin_magnitude_conservation to work
            EdgeQuantumNumbers.parity,
            EdgeQuantumNumbers.c_parity,
        ),
        node_domains=(NodeQuantumNumbers.l_magnitude, NodeQuantumNumbers.s_magnitude),
    )
    new_quantum_number_problem_set = filter_quantum_number_problem_set_properties(
        new_quantum_number_problem_set,
        edge_properties=(
            EdgeQuantumNumbers.pid,  # had to be added for c_parity_conservation to work
            EdgeQuantumNumbers.spin_magnitude,
            EdgeQuantumNumbers.spin_projection,  # had to be added for spin_magnitude_conservation to work
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


def filter_quantum_number_problem_set_settings(
    quantum_number_problem_set: QNProblemSet,
    edge_rules: set[GraphElementRule],
    node_rules: set[Rule],
    edge_domains: Iterable[Any],
    node_domains: Iterable[Any],
) -> QNProblemSet:
    old_edge_settings = quantum_number_problem_set.solving_settings.states
    old_node_settings = quantum_number_problem_set.solving_settings.interactions
    new_edge_settings = {
        edge_id: EdgeSettings(
            conservation_rules=edge_rules,
            rule_priorities=edge_setting.rule_priorities,
            qn_domains=({
                key: val
                for key, val in edge_setting.qn_domains.items()
                if key in set(edge_domains)
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
                if key in set(node_domains)
            }),
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
    edge_properties: Iterable[EdgeQuantumNumberTypes],
    node_properties: Iterable[NodeQuantumNumberTypes],
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
