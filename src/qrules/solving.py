"""Functions to solve a particle reaction problem.

This module is responsible for solving a particle reaction problem stated by a
`.QNProblemSet`. The `.Solver` classes (e.g. :class:`.CSPSolver`) generate new quantum
numbers (for example belonging to an intermediate state) and validate the decay
processes with the rules formulated by the :mod:`.conservation_rules` module.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from typing import Any, Callable, Generic, Iterable, Tuple, Type, TypeVar

import attrs
from attrs import define, field, frozen
from constraint import BacktrackingSolver, Constraint, Problem, Unassigned, Variable

from qrules._implementers import implement_pretty_repr
from qrules.argument_handling import (
    GraphEdgePropertyMap,
    GraphElementRule,
    GraphNodePropertyMap,
    Rule,
    RuleArgumentHandler,
    Scalar,
    get_required_qns,
)
from qrules.quantum_numbers import (
    EdgeQuantumNumber,
    EdgeQuantumNumbers,
    NodeQuantumNumber,
)
from qrules.topology import MutableTransition, Topology

_LOGGER = logging.getLogger(__name__)


@implement_pretty_repr
@define
class EdgeSettings:
    """Solver settings for a specific edge of a graph."""

    conservation_rules: set[GraphElementRule] = field(factory=set)
    rule_priorities: dict[GraphElementRule, int] = field(factory=dict)
    qn_domains: dict[Any, list] = field(factory=dict)


@implement_pretty_repr
@define
class NodeSettings:
    """Container class for the interaction settings.

    This class can be assigned to each node of a state transition graph. Hence, these
    settings contain the complete configuration information which is required for the
    solution finding, e.g:

      - set of conservation rules
      - mapping of rules to priorities (optional)
      - mapping of quantum numbers to their domains
      - strength scale parameter (higher value means stronger force)
    """

    conservation_rules: set[Rule] = field(factory=set)
    rule_priorities: dict[Rule, int] = field(factory=dict)
    qn_domains: dict[Any, list] = field(factory=dict)
    interaction_strength: float = 1.0


GraphSettings = MutableTransition[EdgeSettings, NodeSettings]
"""(Mutable) mapping of settings on a `.Topology`."""
GraphElementProperties = MutableTransition[GraphEdgePropertyMap, GraphNodePropertyMap]
"""(Mutable) mapping of edge and node properties on a `.Topology`."""


@implement_pretty_repr
@frozen
class QNProblemSet:
    """Particle reaction problem set, defined as a graph like data structure.

    Args:
      initial_facts: all of the known facts quantum numbers of the problem.
      solving_settings: solving specific settings, such as the specific rules and
        variable domains for nodes and edges of the :attr:`topology`.
    """

    initial_facts: GraphElementProperties
    solving_settings: GraphSettings

    @property
    def topology(self) -> Topology:
        return self.initial_facts.topology


QuantumNumberSolution = MutableTransition[GraphEdgePropertyMap, GraphNodePropertyMap]


def _convert_violated_rules_to_names(
    rules: dict[int, set[Rule]] | dict[int, set[GraphElementRule]],
) -> dict[int, set[str]]:
    def get_name(rule: Any) -> str:
        if inspect.isfunction(rule):
            return rule.__name__
        if isinstance(rule, str):
            return rule
        return type(rule).__name__

    converted_dict = defaultdict(set)
    for node_id, rule_set in rules.items():
        converted_dict[node_id] = {get_name(rule) for rule in rule_set}

    return converted_dict


def _convert_non_executed_rules_to_names(
    rules: dict[int, set[Rule]] | dict[int, set[GraphElementRule]],
) -> dict[int, set[str]]:
    def get_name(rule: Any) -> str:
        if inspect.isfunction(rule):
            return rule.__name__
        if isinstance(rule, str):
            return rule
        return type(rule).__name__

    converted_dict = defaultdict(set)
    for node_id, rule_set in rules.items():
        rule_name_set = set()
        for rule_tuple in rule_set:
            rule_name_set.add(get_name(rule_tuple))

        converted_dict[node_id] = rule_name_set

    return converted_dict


@implement_pretty_repr
@define(on_setattr=attrs.setters.frozen)
class QNResult:
    """Defines a result to a problem set processed by the solving code."""

    solutions: list[QuantumNumberSolution] = field(factory=list)
    not_executed_node_rules: dict[int, set[str]] = field(
        factory=lambda: defaultdict(set)
    )
    violated_node_rules: dict[int, set[str]] = field(factory=lambda: defaultdict(set))
    not_executed_edge_rules: dict[int, set[str]] = field(
        factory=lambda: defaultdict(set)
    )
    violated_edge_rules: dict[int, set[str]] = field(factory=lambda: defaultdict(set))

    def __attrs_post_init__(self) -> None:
        if self.solutions and (self.violated_node_rules or self.violated_edge_rules):
            msg = (
                f"Invalid {type(self).__name__}! Found {len(self.solutions)} solutions,"
                " but also violated rules."
            )
            raise ValueError(
                (msg),
                self.violated_node_rules,
                self.violated_edge_rules,
            )

    def extend(self, other_result: QNResult) -> None:
        if self.solutions or other_result.solutions:
            self.solutions.extend(other_result.solutions)
            self.not_executed_node_rules.clear()
            self.violated_node_rules.clear()
            self.not_executed_edge_rules.clear()
            self.violated_edge_rules.clear()
        else:
            for key, rules in other_result.not_executed_node_rules.items():
                self.not_executed_node_rules[key].update(rules)

            for key, rules in other_result.not_executed_edge_rules.items():
                self.not_executed_edge_rules[key].update(rules)

            for key, rules2 in other_result.violated_node_rules.items():
                self.violated_node_rules[key].update(rules2)

            for key, rules2 in other_result.violated_edge_rules.items():
                self.violated_edge_rules[key].update(rules2)


class Solver(ABC):
    """Interface of a Solver."""

    @abstractmethod
    def find_solutions(self, problem_set: QNProblemSet) -> QNResult:
        """Find solutions for the given input.

        It is expected that this function determines and returns all of the found
        solutions. In case no solutions are found a partial list of violated rules has
        to be given. This list of violated rules does not have to be complete.

        Args:
            problem_set (`.QNProblemSet`): states a problem set

        Returns:
            QNResult: contains possible solutions, violated rules and not executed
                rules due to requirement issues.
        """


def _insert_allowed_states(
    solutions: list[QuantumNumberSolution],
    topology: Topology,
    allowed_states: Iterable[GraphEdgePropertyMap],
) -> list[QuantumNumberSolution]:
    _LOGGER.debug("Inserting allowed states into QN solution graphs...")
    substituted_graphs: list[QuantumNumberSolution] = []
    for solution in solutions:
        current_substituted_graphs = [solution]
        for edge_id in topology.intermediate_edge_ids:
            incomplete_state = solution.states[edge_id]
            candidate_states = __get_candidate_states(incomplete_state, allowed_states)
            if len(candidate_states) == 0:
                message = f"Did not find any QN state candidate for edge id: {edge_id}"
                _LOGGER.debug(message)
                _LOGGER.debug(f"State properties: {solution.states[edge_id]}")
            graphs_with_candidates = []
            for new_solution in current_substituted_graphs:
                for candidate in candidate_states:
                    # need "shallow" copy of the nested dicts
                    new_states = {i: copy(s) for i, s in new_solution.states.items()}
                    new_states[edge_id].update(candidate)  # keep spin_projection
                    graph = attrs.evolve(new_solution, states=new_states)  # type: ignore[arg-type]
                    graphs_with_candidates.append(graph)
            current_substituted_graphs = graphs_with_candidates

        substituted_graphs.extend(current_substituted_graphs)

    return substituted_graphs


def __get_candidate_states(
    state: GraphEdgePropertyMap,
    allowed_states: Iterable[GraphEdgePropertyMap],
) -> list[GraphEdgePropertyMap]:
    candidates = []
    for candidate in allowed_states:
        if __is_sub_mapping(state, candidate):
            candidates.append(candidate)
    return candidates


def __is_sub_mapping(
    state: GraphEdgePropertyMap, reference_state: GraphEdgePropertyMap
) -> bool:
    for qn_type, qn_value in state.items():
        if qn_type is EdgeQuantumNumbers.spin_projection:
            continue
        if qn_type not in reference_state:
            return False
        if qn_value != reference_state[qn_type]:
            return False
    return True


def validate_full_solution(problem_set: QNProblemSet) -> QNResult:  # noqa: C901
    _LOGGER.debug("validating graph...")

    rule_argument_handler = RuleArgumentHandler()

    def _create_node_variables(
        node_id: int, qn_list: set[type[NodeQuantumNumber]]
    ) -> dict[type[NodeQuantumNumber], Scalar]:
        """Create variables for the quantum numbers of the specified node."""
        variables = {}
        if node_id in problem_set.initial_facts.interactions:
            interactions = problem_set.initial_facts.interactions[node_id]
            variables = interactions
            for qn_type in qn_list:
                if qn_type in interactions:
                    variables[qn_type] = interactions[qn_type]
        return variables

    def _create_edge_variables(
        edge_ids: Iterable[int],
        qn_list: set[type[EdgeQuantumNumber]],
    ) -> list[dict]:
        """Create variables for the quantum numbers of the specified edges.

        Initial and final state edges just get a single domain value. Intermediate edges
        are initialized with the default domains of that quantum number.
        """
        variables = []
        for edge_id in edge_ids:
            if edge_id in problem_set.initial_facts.states:
                states = problem_set.initial_facts.states[edge_id]
                edge_vars = {}
                for qn_type in qn_list:
                    if qn_type in states:
                        edge_vars[qn_type] = states[qn_type]
                variables.append(edge_vars)
        return variables

    def _create_variable_containers(
        node_id: int, cons_law: Rule
    ) -> tuple[list[dict], list[dict], dict]:
        topology = problem_set.topology
        in_edges = topology.get_edge_ids_ingoing_to_node(node_id)
        out_edges = topology.get_edge_ids_outgoing_from_node(node_id)

        edge_qns, node_qns = get_required_qns(cons_law)
        in_edges_vars = _create_edge_variables(in_edges, edge_qns)
        out_edges_vars = _create_edge_variables(out_edges, edge_qns)

        node_vars = _create_node_variables(node_id, node_qns)

        return (in_edges_vars, out_edges_vars, node_vars)

    edge_violated_rules: dict[int, set[GraphElementRule]] = defaultdict(set)
    edge_not_executed_rules: dict[int, set[GraphElementRule]] = defaultdict(set)
    node_violated_rules: dict[int, set[Rule]] = defaultdict(set)
    node_not_executed_rules: dict[int, set[Rule]] = defaultdict(set)
    for (
        edge_id,
        edge_settings,
    ) in problem_set.solving_settings.states.items():
        edge_rules = edge_settings.conservation_rules
        for edge_rule in edge_rules:
            # get the needed qns for this conservation law
            # for all edges and the node
            (
                check_requirements,
                create_rule_args,
            ) = rule_argument_handler.register_rule(edge_rule)

            edge_qns, _ = get_required_qns(edge_rule)
            edge_variables = _create_edge_variables([edge_id], edge_qns)[0]
            if check_requirements(
                edge_variables,
            ):
                if not edge_rule(
                    *create_rule_args(
                        edge_variables,
                    )
                ):
                    edge_violated_rules[edge_id].add(edge_rule)
            else:
                edge_not_executed_rules[edge_id].add(edge_rule)

    for (
        node_id,
        node_settings,
    ) in problem_set.solving_settings.interactions.items():
        node_rules = node_settings.conservation_rules
        for rule in node_rules:
            # get the needed qns for this conservation law
            # for all edges and the node
            (
                check_requirements,
                create_rule_args,
            ) = rule_argument_handler.register_rule(rule)

            var_containers = _create_variable_containers(node_id, rule)
            if check_requirements(
                var_containers[0],
                var_containers[1],
                var_containers[2],
            ):
                if not rule(
                    *create_rule_args(
                        var_containers[0],
                        var_containers[1],
                        var_containers[2],
                    )
                ):
                    node_violated_rules[node_id].add(rule)
            else:
                node_not_executed_rules[node_id].add(rule)
    if node_violated_rules or node_not_executed_rules:
        return QNResult(
            [],
            _convert_non_executed_rules_to_names(node_not_executed_rules),
            _convert_violated_rules_to_names(node_violated_rules),
            _convert_non_executed_rules_to_names(edge_not_executed_rules),
            _convert_violated_rules_to_names(edge_violated_rules),
        )
    return QNResult(
        [
            MutableTransition(
                topology=problem_set.topology,
                states=problem_set.initial_facts.states,  # type: ignore[arg-type]
                interactions=problem_set.initial_facts.interactions,  # type: ignore[arg-type]
            )
        ],
    )


_EdgeVariableInfo = Tuple[int, Type[EdgeQuantumNumber]]
_NodeVariableInfo = Tuple[int, Type[NodeQuantumNumber]]


def _create_variable_string(
    element_id: int,
    qn_type: type[EdgeQuantumNumber] | type[NodeQuantumNumber],
) -> str:
    return str(element_id) + "-" + qn_type.__name__


@define
class _VariableContainer:
    ingoing_edge_variables: set[_EdgeVariableInfo] = field(factory=set)
    fixed_ingoing_edge_variables: dict[int, GraphEdgePropertyMap] = field(factory=dict)
    outgoing_edge_variables: set[_EdgeVariableInfo] = field(factory=set)
    fixed_outgoing_edge_variables: dict[int, GraphEdgePropertyMap] = field(factory=dict)
    node_variables: set[_NodeVariableInfo] = field(factory=set)
    fixed_node_variables: GraphNodePropertyMap = field(factory=dict)


class CSPSolver(Solver):
    """Solver reducing the task to a Constraint Satisfaction Problem.

    Solving this done with the :doc:`constraint<constraint:reference>` package.

    The variables are the quantum numbers of particles/edges, but also some composite
    quantum numbers which are attributed to the interaction nodes (such as angular
    momentum :math:`L`). The conservation rules serve as the constraints and a special
    wrapper class serves as an adapter.
    """

    def __init__(self, allowed_intermediate_states: Iterable[GraphEdgePropertyMap]):
        self.__variables: set[_EdgeVariableInfo | _NodeVariableInfo] = set()
        self.__var_string_to_data: dict[str, _EdgeVariableInfo | _NodeVariableInfo] = {}
        self.__node_rules: dict[int, set[Rule]] = defaultdict(set)
        self.__non_executable_node_rules: dict[int, set[Rule]] = defaultdict(set)
        self.__edge_rules: dict[int, set[GraphElementRule]] = defaultdict(set)
        self.__non_executable_edge_rules: dict[int, set[GraphElementRule]] = (
            defaultdict(set)
        )
        self.__problem = Problem(BacktrackingSolver(True))
        self.__allowed_intermediate_states = tuple(allowed_intermediate_states)
        self.__scoresheet = Scoresheet()

    def find_solutions(self, problem_set: QNProblemSet) -> QNResult:  # noqa: C901
        self.__initialize_constraints(problem_set)
        solutions = self.__problem.getSolutions()

        node_not_executed_rules = self.__non_executable_node_rules
        node_not_satisfied_rules: dict[int, set[Rule]] = defaultdict(set)
        edge_not_executed_rules = self.__non_executable_edge_rules
        edge_not_satisfied_rules: dict[int, set[GraphElementRule]] = defaultdict(set)
        for node_id, rules in self.__node_rules.items():
            for rule in rules:
                if self.__scoresheet.rule_calls[(node_id, rule)] == 0:
                    node_not_executed_rules[node_id].add(rule)
                elif self.__scoresheet.rule_passes[(node_id, rule)] == 0:
                    node_not_satisfied_rules[node_id].add(rule)

        for edge_id, edge_rules in self.__edge_rules.items():
            for rule in edge_rules:
                if self.__scoresheet.rule_calls[(edge_id, rule)] == 0:
                    edge_not_executed_rules[edge_id].add(rule)
                elif self.__scoresheet.rule_passes[(edge_id, rule)] == 0:
                    edge_not_satisfied_rules[edge_id].add(rule)

        solutions = self.__convert_solution_keys(problem_set.topology, solutions)

        # insert particle instances
        if self.__node_rules or self.__edge_rules:
            selected_solutions = _insert_allowed_states(
                solutions,
                problem_set.topology,
                self.__allowed_intermediate_states,
            )
        else:
            selected_solutions = [
                QuantumNumberSolution(
                    topology=problem_set.topology,
                    interactions=problem_set.initial_facts.interactions,  # type: ignore[arg-type]
                    states=problem_set.initial_facts.states,  # type: ignore[arg-type]
                )
            ]

        if selected_solutions and (node_not_executed_rules or edge_not_executed_rules):
            # rerun solver on these graphs using not executed rules and combine results
            topology = problem_set.topology
            result = QNResult()
            for full_particle_solution in selected_solutions:
                interactions = full_particle_solution.interactions
                states = full_particle_solution.states
                interactions.update(problem_set.initial_facts.interactions)
                states.update(problem_set.initial_facts.states)
                result.extend(
                    validate_full_solution(
                        QNProblemSet(
                            initial_facts=MutableTransition(
                                topology, states, interactions
                            ),
                            solving_settings=MutableTransition(
                                topology,
                                interactions={
                                    i: NodeSettings(conservation_rules=rules)  # type: ignore[misc]
                                    for i, rules in node_not_executed_rules.items()
                                },
                                states={
                                    i: EdgeSettings(conservation_rules=rules)  # type: ignore[misc]
                                    for i, rules in edge_not_executed_rules.items()
                                },
                            ),
                        )
                    )
                )
            return result

        return QNResult(
            selected_solutions,
            _convert_non_executed_rules_to_names(node_not_executed_rules),
            _convert_violated_rules_to_names(node_not_satisfied_rules),
            _convert_non_executed_rules_to_names(edge_not_executed_rules),
            _convert_violated_rules_to_names(edge_not_satisfied_rules),
        )

    def __clear(self) -> None:
        self.__variables = set()
        self.__var_string_to_data = {}
        self.__node_rules = defaultdict(set)
        self.__edge_rules = defaultdict(set)
        self.__problem = Problem(BacktrackingSolver(True))
        self.__scoresheet = Scoresheet()

    def __initialize_constraints(  # noqa: PLR0914
        self, problem_set: QNProblemSet
    ) -> None:
        """Initialize all of the constraints for this graph.

        For each interaction node a set of independent constraints/conservation laws are
        created. For each conservation law a new CSP wrapper is created. This wrapper
        needs all of the qn numbers/variables which enter or exit the node and play a
        role for this conservation law. Hence variables are also created within this
        method.
        """
        self.__clear()

        def get_rules_by_priority(
            graph_element_settings: NodeSettings | EdgeSettings,
        ) -> list[Rule]:
            # first add priorities to the entries
            priority_list = [
                (
                    (x, graph_element_settings.rule_priorities[type(x)])  # type: ignore[index]
                    if type(x) in graph_element_settings.rule_priorities
                    else (x, 1)
                )
                for x in graph_element_settings.conservation_rules
            ]
            # then sort according to priority
            sorted_list = sorted(priority_list, key=lambda x: x[1], reverse=True)
            # and strip away the priorities again
            return [x[0] for x in sorted_list]

        arg_handler = RuleArgumentHandler()

        for edge_id in problem_set.topology.edges:
            edge_settings = problem_set.solving_settings.states[edge_id]
            for rule in get_rules_by_priority(edge_settings):
                variable_mapping = _VariableContainer()
                # from cons law and graph determine needed var lists
                edge_qns, node_qns = get_required_qns(rule)

                edge_vars, fixed_edge_vars = self.__create_edge_variables(
                    [edge_id],
                    edge_qns,
                    problem_set,
                )

                score_callback = self.__scoresheet.register_rule(edge_id, rule)
                constraint = _GraphElementConstraint[EdgeQuantumNumber](
                    rule,  # type: ignore[arg-type]
                    edge_vars,
                    fixed_edge_vars,
                    arg_handler,
                    score_callback,
                )

                if edge_vars:
                    var_strings = [_create_variable_string(*x) for x in edge_vars]
                    self.__edge_rules[edge_id].add(rule)  # type: ignore[arg-type]
                    self.__problem.addConstraint(constraint, var_strings)
                else:
                    self.__non_executable_edge_rules[edge_id].add(rule)  # type: ignore[arg-type]

        for node_id in problem_set.topology.nodes:
            for rule in get_rules_by_priority(
                problem_set.solving_settings.interactions[node_id]
            ):
                variable_mapping = _VariableContainer()
                # from cons law and graph determine needed var lists
                edge_qns, node_qns = get_required_qns(rule)

                in_edges = problem_set.topology.get_edge_ids_ingoing_to_node(node_id)
                in_edge_vars = self.__create_edge_variables(
                    in_edges, edge_qns, problem_set
                )
                variable_mapping.ingoing_edge_variables = in_edge_vars[0]
                variable_mapping.fixed_ingoing_edge_variables = in_edge_vars[1]
                var_list: list[_EdgeVariableInfo | _NodeVariableInfo] = list(
                    variable_mapping.ingoing_edge_variables
                )

                out_edges = problem_set.topology.get_edge_ids_outgoing_from_node(
                    node_id
                )
                out_edge_vars = self.__create_edge_variables(
                    out_edges, edge_qns, problem_set
                )
                variable_mapping.outgoing_edge_variables = out_edge_vars[0]
                variable_mapping.fixed_outgoing_edge_variables = out_edge_vars[1]
                var_list.extend(list(variable_mapping.outgoing_edge_variables))

                # now create variables for node/interaction qns
                int_node_vars = self.__create_node_variables(
                    node_id,
                    node_qns,
                    problem_set,
                )
                variable_mapping.node_variables = int_node_vars[0]
                variable_mapping.fixed_node_variables = int_node_vars[1]
                var_list.extend(list(variable_mapping.node_variables))

                score_callback = self.__scoresheet.register_rule(node_id, rule)
                if len(inspect.signature(rule).parameters) == 1:
                    constraint = _GraphElementConstraint[NodeQuantumNumber](
                        rule,  # type: ignore[arg-type]
                        int_node_vars[0],
                        {node_id: int_node_vars[1]},
                        arg_handler,
                        score_callback,
                    )
                else:
                    constraint = _ConservationRuleConstraintWrapper(
                        rule, variable_mapping, arg_handler, score_callback
                    )
                if var_list:
                    var_strings = [_create_variable_string(*x) for x in var_list]
                    self.__node_rules[node_id].add(rule)
                    self.__problem.addConstraint(constraint, var_strings)
                else:
                    self.__non_executable_node_rules[node_id].add(rule)

    def __create_node_variables(
        self,
        node_id: int,
        qn_list: set[type[NodeQuantumNumber]],
        problem_set: QNProblemSet,
    ) -> tuple[set[_NodeVariableInfo], GraphNodePropertyMap]:
        """Create variables for the quantum numbers of the specified node.

        If a quantum number is already defined for a node, then a fixed variable is
        created, which cannot be changed by the csp solver. Otherwise the node is
        initialized with the specified domain of that quantum number.
        """
        variables: tuple[set[_NodeVariableInfo], GraphNodePropertyMap] = (
            set(),
            {},
        )

        if node_id in problem_set.initial_facts.interactions:
            interactions = problem_set.initial_facts.interactions[node_id]
            for qn_type in qn_list:
                if qn_type in interactions:
                    variables[1].update({qn_type: interactions[qn_type]})
        else:
            node_settings = problem_set.solving_settings.interactions[node_id]
            for qn_type in qn_list:
                var_info = (node_id, qn_type)
                if qn_type in node_settings.qn_domains:
                    qn_domain = node_settings.qn_domains[qn_type]
                    self.__add_variable(var_info, qn_domain)
                    variables[0].add(var_info)
        return variables

    def __create_edge_variables(
        self,
        edge_ids: Iterable[int],
        qn_list: set[type[EdgeQuantumNumber]],
        problem_set: QNProblemSet,
    ) -> tuple[set[_EdgeVariableInfo], dict[int, GraphEdgePropertyMap]]:
        """Create variables for the quantum numbers of the specified edges.

        If a quantum number is already defined for an edge, then a fixed variable is
        created, which cannot be changed by the csp solver. This is the case for initial
        and final state edges. Otherwise the edges are initialized with the specified
        domains of that quantum number.
        """
        variables: tuple[
            set[_EdgeVariableInfo],
            dict[int, GraphEdgePropertyMap],
        ] = (
            set(),
            {},
        )

        for edge_id in edge_ids:
            variables[1][edge_id] = {}
            if edge_id in problem_set.initial_facts.states:
                states = problem_set.initial_facts.states[edge_id]
                for qn_type in qn_list:
                    if qn_type in states:
                        variables[1][edge_id].update({qn_type: states[qn_type]})
            else:
                edge_settings = problem_set.solving_settings.states[edge_id]
                for qn_type in qn_list:
                    var_info = (edge_id, qn_type)
                    if qn_type in edge_settings.qn_domains:
                        qn_domain = edge_settings.qn_domains[qn_type]
                        self.__add_variable(var_info, qn_domain)
                        variables[0].add(var_info)
        return variables

    def __add_variable(
        self,
        var_info: _EdgeVariableInfo | _NodeVariableInfo,
        domain: list[Any],
    ) -> None:
        if var_info not in self.__variables:
            self.__variables.add(var_info)
            var_string = _create_variable_string(*var_info)
            self.__var_string_to_data[var_string] = var_info
            self.__problem.addVariable(var_string, domain)

    def __convert_solution_keys(
        self, topology: Topology, solutions: list[dict[str, Scalar]]
    ) -> list[QuantumNumberSolution]:
        """Convert keys of CSP solutions from `str` to quantum number types."""
        converted_solutions: list[
            MutableTransition[GraphEdgePropertyMap, GraphNodePropertyMap]
        ] = []
        for solution in solutions:
            states: dict[int, GraphEdgePropertyMap] = defaultdict(dict)
            interactions: dict[int, GraphNodePropertyMap] = defaultdict(dict)
            for var_string, value in solution.items():
                ele_id, qn_type = self.__var_string_to_data[var_string]

                if qn_type in getattr(EdgeQuantumNumber, "__args__"):  # noqa: B009
                    states[ele_id].update({qn_type: value})  # type: ignore[dict-item]
                else:
                    interactions[ele_id].update({qn_type: value})  # type: ignore[dict-item]
            converted_solutions.append(
                MutableTransition(topology, states, interactions)  # type: ignore[arg-type]
            )
        return converted_solutions


class Scoresheet:
    def __init__(self) -> None:
        self.__rule_calls: dict[tuple[int, Rule], int] = {}
        self.__rule_passes: dict[tuple[int, Rule], int] = {}

    def register_rule(
        self, graph_element_id: int, rule: Rule
    ) -> Callable[[bool], None]:
        self.__rule_calls[(graph_element_id, rule)] = 0
        self.__rule_passes[(graph_element_id, rule)] = 0

        return self.__create_callback(graph_element_id, rule)

    def __create_callback(
        self, graph_element_id: int, rule: Rule
    ) -> Callable[[bool], None]:
        def passed_callback(passed: bool) -> None:
            if passed:
                self.__rule_passes[(graph_element_id, rule)] += 1
            self.__rule_calls[(graph_element_id, rule)] += 1

        return passed_callback

    @property
    def rule_calls(self) -> dict[tuple[int, Rule], int]:
        return self.__rule_calls

    @property
    def rule_passes(self) -> dict[tuple[int, Rule], int]:
        return self.__rule_passes


_QNType = TypeVar("_QNType", EdgeQuantumNumber, NodeQuantumNumber)


class _GraphElementConstraint(Generic[_QNType], Constraint):
    """Wrapper class of the `~constraints.Constraint` class.

    This allows a customized definition of conservation rules, and hence a cleaner user
    interface.
    """

    def __init__(
        self,
        rule: GraphElementRule,
        variables: set[tuple[int, type[_QNType]]],
        fixed_variables: dict[int, dict[type[_QNType], Scalar]],
        argument_handler: RuleArgumentHandler,
        scoresheet: Callable[[bool], None],
    ) -> None:
        if not callable(rule):
            msg = "rule argument has to be a callable"
            raise TypeError(msg)
        self.__rule = rule
        (
            self.__check_rule_requirements,
            self.__create_rule_args,
        ) = argument_handler.register_rule(rule)
        self.__score_callback = scoresheet

        self.__var_string_to_data: dict[str, type[_QNType]] = {}
        self.__qns: dict[type[_QNType], Scalar | None] = {}

        self.__initialize_variable_containers(variables, fixed_variables)

    @property
    def rule(self) -> Rule:
        return self.__rule

    def __initialize_variable_containers(
        self,
        variables: set[tuple[int, type[_QNType]]],
        fixed_variables: dict[int, dict[type[_QNType], Scalar]],
    ) -> None:
        """Fill the name decoding map.

        Also initialize the in and out particle lists. The variable names follow the
        scheme edge_id(delimiter)qn_name. This method creates a dict linking the var
        name to a list that consists of the particle list index and the qn name.
        """
        self.__qns.update(next(iter(fixed_variables.values())))
        for element_id, qn_type in variables:
            self.__var_string_to_data[_create_variable_string(element_id, qn_type)] = (
                qn_type
            )
            self.__qns.update({qn_type: None})

    def __call__(
        self,
        variables: set[str],
        domains: dict,
        assignments: dict,
        forwardcheck: bool = False,
        _unassigned: Variable = Unassigned,
    ) -> bool:
        """Perform the constraint checking.

        If the forwardcheck parameter is not false, besides telling if the constraint is
        currently broken or not, the constraint implementation may choose to hide values
        from the domains of unassigned variables to prevent them from being used, and
        thus prune the search space.

        Args:
            variables: Variables affected by that constraint, in the same order
                provided by the user.

            domains (dict): Dictionary mapping variables to their domains.

            assignments (dict): Dictionary mapping assigned variables to their
                current assumed value.

            forwardcheck (bool): Boolean value stating whether forward checking
                should be performed or not.

            _unassigned: Can be left empty

        Return:
            bool:
                Boolean value stating if this constraint is currently broken or not.
        """
        params = [(x, assignments.get(x, _unassigned)) for x in variables]
        missing = [name for (name, val) in params if val is _unassigned]
        if missing:
            return True

        self.__update_variable_lists(params)

        if not self.__check_rule_requirements(
            self.__qns,
        ):
            return True

        passed = self.__rule(*self.__create_rule_args(self.__qns))

        self.__score_callback(passed)

        return passed

    def __update_variable_lists(
        self,
        parameters: list[tuple[str, Any]],
    ) -> None:
        for var_string, value in parameters:
            qn_type = self.__var_string_to_data[var_string]
            if qn_type in self.__qns:
                self.__qns[qn_type] = value
            else:
                raise ValueError(
                    "The variable with name "
                    + qn_type.__name__
                    + "does not appear in the variable mapping"
                )


class _ConservationRuleConstraintWrapper(
    Constraint  # pyright: ignore[reportUntypedBaseClass]
):
    """Wrapper class of the `~constraints.Constraint` class.

    This allows a customized definition of conservation rules, and hence a cleaner user
    interface.
    """

    def __init__(
        self,
        rule: Rule,
        variables: _VariableContainer,
        argument_handler: RuleArgumentHandler,
        score_callback: Callable[[bool], None],
    ) -> None:
        if not callable(rule):
            msg = "rule argument has to be a callable"
            raise TypeError(msg)
        self.__rule = rule
        (
            self.__check_rule_requirements,
            self.__create_rule_args,
        ) = argument_handler.register_rule(rule)
        self.__score_callback = score_callback

        self.__var_string_to_data: dict[
            str,
            _EdgeVariableInfo | _NodeVariableInfo,
        ] = {}
        self.__in_edges_qns: dict[int, GraphEdgePropertyMap] = {}
        self.__out_edges_qns: dict[int, GraphEdgePropertyMap] = {}
        self.__node_qns: GraphNodePropertyMap = {}

        self.__initialize_variable_containers(variables)

    def __initialize_variable_containers(self, variables: _VariableContainer) -> None:
        """Fill the name decoding map.

        Also initialize the in and out particle lists. The variable names follow the
        scheme edge_id(delimiter)qn_name. This method creates a dict linking the var
        name to a list that consists of the particle list index and the qn name.
        """

        def _initialize_edge_container(
            variable_set: set[_EdgeVariableInfo],
            fixed_variables: dict[int, dict[type[EdgeQuantumNumber], Scalar]],
            container: dict[int, GraphEdgePropertyMap],
        ) -> None:
            container.update(fixed_variables)
            for element_id, qn_type in variable_set:
                self.__var_string_to_data[
                    _create_variable_string(element_id, qn_type)
                ] = (element_id, qn_type)
                if element_id not in container:
                    container[element_id] = {}
                container[element_id].update({qn_type: None})  # type: ignore[dict-item]

        _initialize_edge_container(
            variables.ingoing_edge_variables,
            variables.fixed_ingoing_edge_variables,
            self.__in_edges_qns,
        )
        _initialize_edge_container(
            variables.outgoing_edge_variables,
            variables.fixed_outgoing_edge_variables,
            self.__out_edges_qns,
        )
        # and now interaction node variables
        for var_info in variables.node_variables:
            self.__node_qns[var_info[1]] = None  # type: ignore[assignment]
            self.__var_string_to_data[_create_variable_string(*var_info)] = var_info
        self.__node_qns.update(variables.fixed_node_variables)

    def __call__(
        self,
        variables: set[str],
        domains: dict,
        assignments: dict,
        forwardcheck: bool = False,
        _unassigned: Variable = Unassigned,
    ) -> bool:
        """Perform the constraint checking.

        If the forwardcheck parameter is not false, besides telling if the constraint is
        currently broken or not, the constraint implementation may choose to hide values
        from the domains of unassigned variables to prevent them from being used, and
        thus prune the search space.

        Args:
            variables: Variables affected by that constraint, in the same order
                provided by the user.

            domains (dict): Dictionary mapping variables to their domains.

            assignments (dict): Dictionary mapping assigned variables to their
                current assumed value.

            forwardcheck (bool): Boolean value stating whether forward checking
                should be performed or not.

            _unassigned: Can be left empty

        Return:
            bool:
                Boolean value stating if this constraint is currently broken or not.
        """
        params = [(x, assignments.get(x, _unassigned)) for x in variables]
        missing = [name for (name, val) in params if val is _unassigned]
        if missing:
            return True

        self.__update_variable_lists(params)

        if not self.__check_rule_requirements(
            list(self.__in_edges_qns.values()),
            list(self.__out_edges_qns.values()),
            self.__node_qns,
        ):
            return True

        passed = self.__rule(
            *self.__create_rule_args(
                list(self.__in_edges_qns.values()),
                list(self.__out_edges_qns.values()),
                self.__node_qns,
            )
        )
        self.__score_callback(passed)
        return passed

    def __update_variable_lists(
        self,
        parameters: list[tuple[str, Any]],
    ) -> None:
        for var_string, value in parameters:
            index, qn_type = self.__var_string_to_data[var_string]
            if index in self.__in_edges_qns and qn_type in self.__in_edges_qns[index]:
                self.__in_edges_qns[index][qn_type] = value  # type: ignore[index]
            elif (
                index in self.__out_edges_qns and qn_type in self.__out_edges_qns[index]
            ):
                self.__out_edges_qns[index][qn_type] = value  # type: ignore[index]
            elif qn_type in self.__node_qns:
                self.__node_qns[qn_type] = value  # type: ignore[index]
            else:
                msg = (
                    f"The variable with name {qn_type.__name__} and a graph element"
                    f" index of {index} does not appear in the variable mapping"
                )
                raise ValueError(msg)
