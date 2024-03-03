"""Find allowed transitions between an initial and final state."""

from __future__ import annotations

import logging
import re
import sys
import warnings
from collections import defaultdict
from copy import copy, deepcopy
from enum import Enum, auto
from multiprocessing import Pool
from typing import Iterable, Sequence, overload

import attrs
from attrs import define, field, frozen
from attrs.validators import in_, instance_of
from tqdm.auto import tqdm

from qrules._implementers import implement_pretty_repr
from qrules.combinatorics import (
    InitialFacts,
    StateDefinition,
    create_initial_facts,
    ensure_nested_list,
    match_external_edges,
    permutate_topology_kinematically,
)
from qrules.particle import (
    Particle,
    ParticleCollection,
    ParticleWithSpin,
    _to_float,
    load_pdg,
)
from qrules.quantum_numbers import (
    EdgeQuantumNumber,
    EdgeQuantumNumbers,
    InteractionProperties,
    NodeQuantumNumber,
    NodeQuantumNumbers,
)
from qrules.settings import (
    DEFAULT_INTERACTION_TYPES,
    InteractionType,
    NumberOfThreads,
    create_interaction_settings,
)
from qrules.solving import (
    CSPSolver,
    EdgeSettings,
    GraphEdgePropertyMap,
    GraphSettings,
    NodeSettings,
    QNProblemSet,
    QNResult,
)
from qrules.system_control import (
    GammaCheck,
    InteractionDeterminator,
    LeptonCheck,
    create_edge_properties,
    create_interaction_properties,
    create_node_properties,
    filter_interaction_types,
    find_particle,
    remove_duplicate_solutions,
)
from qrules.topology import (
    FrozenDict,
    FrozenTransition,
    MutableTransition,
    Topology,
    create_isobar_topologies,
    create_n_body_topology,
)

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


_LOGGER = logging.getLogger(__name__)

SpinFormalism = Literal[
    "helicity",
    "canonical-helicity",
    "canonical",
]
"""Name for the spin formalism to be used.

The options :code:`"helicity"`, :code:`"canonical-helicity"`, and :code:`"canonical"`
are all used for the helicity formalism, but :code:`"canonical-helicity"` and
:code:`"canonical"` generate angular momentum and coupled spins as well on the
interaction nodes.
"""


class SolvingMode(Enum):
    """Types of modes for solving."""

    FAST = auto()
    """Find "likeliest" solutions only."""
    FULL = auto()
    """Find all possible solutions."""


@implement_pretty_repr
@define(on_setattr=attrs.setters.frozen)
class ExecutionInfo:
    not_executed_node_rules: dict[int, set[str]] = field(
        factory=lambda: defaultdict(set)
    )
    violated_node_rules: dict[int, set[str]] = field(factory=lambda: defaultdict(set))
    not_executed_edge_rules: dict[int, set[str]] = field(
        factory=lambda: defaultdict(set)
    )
    violated_edge_rules: dict[int, set[str]] = field(factory=lambda: defaultdict(set))

    def extend(
        self, other_result: ExecutionInfo, intersect_violations: bool = False
    ) -> None:
        for key, rules in other_result.not_executed_node_rules.items():
            self.not_executed_node_rules[key].update(rules)

        for key, rules in other_result.not_executed_edge_rules.items():
            self.not_executed_edge_rules[key].update(rules)

        for key, rules2 in other_result.violated_node_rules.items():
            if intersect_violations:
                self.violated_node_rules[key] &= rules2
            else:
                self.violated_node_rules[key].update(rules2)

        for key, rules2 in other_result.violated_edge_rules.items():
            if intersect_violations:
                self.violated_edge_rules[key] &= rules2
            else:
                self.violated_edge_rules[key].update(rules2)

    def clear(self) -> None:
        self.not_executed_node_rules.clear()
        self.violated_node_rules.clear()
        self.not_executed_edge_rules.clear()
        self.violated_edge_rules.clear()


@frozen
class _SolutionContainer:
    """Defines a result of a `.ProblemSet`."""

    solutions: list[MutableTransition[ParticleWithSpin, InteractionProperties]] = field(
        factory=list
    )
    execution_info: ExecutionInfo = field(default=ExecutionInfo())

    def __attrs_post_init__(self) -> None:
        if self.solutions and (
            self.execution_info.violated_node_rules
            or self.execution_info.violated_edge_rules
        ):
            msg = (
                f"Invalid {type(self).__name__}! Found {len(self.solutions)} solutions,"
                " but also violated rules."
            )
            raise ValueError(
                msg,
                self.execution_info.violated_node_rules,
                self.execution_info.violated_edge_rules,
            )

    def extend(
        self, other: _SolutionContainer, intersect_violations: bool = False
    ) -> None:
        if self.solutions or other.solutions:
            self.solutions.extend(other.solutions)
            self.execution_info.clear()
        else:
            self.execution_info.extend(other.execution_info, intersect_violations)


@implement_pretty_repr
@define
class ProblemSet:
    """Particle reaction problem set as a graph-like data structure."""

    topology: Topology
    """`.Topology` over which the problem set is defined."""
    initial_facts: InitialFacts
    """Information about the initial and final state."""
    solving_settings: GraphSettings
    """Solving settings, such as conservation rules and QN-domains."""

    def to_qn_problem_set(self) -> QNProblemSet:
        interactions = {
            k: create_node_properties(v)
            for k, v in self.initial_facts.interactions.items()
        }
        states = {
            k: create_edge_properties(v[0], v[1])
            for k, v in self.initial_facts.states.items()
        }
        return QNProblemSet(
            initial_facts=MutableTransition(self.topology, states, interactions),
            solving_settings=self.solving_settings,
        )


def _group_by_strength(
    problem_sets: list[ProblemSet],
) -> dict[float, list[ProblemSet]]:
    def calculate_strength(node_interaction_settings: dict[int, NodeSettings]) -> float:
        strength = 1.0
        for int_setting in node_interaction_settings.values():
            strength *= int_setting.interaction_strength
        return strength

    strength_sorted_problem_sets: dict[float, list[ProblemSet]] = defaultdict(list)
    for problem_set in problem_sets:
        strength = calculate_strength(problem_set.solving_settings.interactions)
        strength_sorted_problem_sets[strength].append(problem_set)
    return strength_sorted_problem_sets


class StateTransitionManager:
    """Main handler for decay topologies.

    .. seealso:: :doc:`/usage/reaction` and `.generate_transitions`
    """

    def __init__(  # noqa: C901, PLR0912, PLR0917
        self,
        initial_state: Sequence[StateDefinition],
        final_state: Sequence[StateDefinition],
        particle_db: ParticleCollection | None = None,
        allowed_intermediate_particles: list[str] | None = None,
        interaction_type_settings: dict[
            InteractionType, tuple[EdgeSettings, NodeSettings]
        ]
        | None = None,
        formalism: SpinFormalism = "helicity",
        topology_building: str = "isobar",
        solving_mode: SolvingMode = SolvingMode.FAST,
        reload_pdg: bool = False,
        mass_conservation_factor: float | None = 3.0,
        max_angular_momentum: int = 1,
        max_spin_magnitude: float = 2.0,
        number_of_threads: int | None = None,
    ) -> None:
        if number_of_threads is not None:
            NumberOfThreads.set(number_of_threads)
        self.__number_of_threads = NumberOfThreads.get()
        if interaction_type_settings is None:
            interaction_type_settings = {}
        if formalism not in set(SpinFormalism.__args__):  # type: ignore[attr-defined]
            msg = (
                f'Formalism "{formalism}" not implemented. Use one of'
                f" {', '.join(SpinFormalism.__args__)} instead."  # type: ignore[attr-defined]
            )
            raise NotImplementedError(msg)
        self.__formalism = formalism
        self.__particles = ParticleCollection()
        if particle_db is not None:
            self.__particles = particle_db
        self.reaction_mode = str(solving_mode)
        self.initial_state = list(initial_state)
        self.final_state = list(final_state)
        self.interaction_type_settings = interaction_type_settings

        self.interaction_determinators: list[InteractionDeterminator] = [
            LeptonCheck(),
            GammaCheck(),
        ]
        """Checks that are executed over selected conservation rules.

        .. seealso:: {ref}`usage/reaction:Select interaction types`
        """
        self.final_state_groupings: list[list[list[str]]] | None = None
        self.__allowed_interaction_types: (
            list[InteractionType] | dict[int, list[InteractionType]]
        ) = DEFAULT_INTERACTION_TYPES
        self.filter_remove_qns: set[type[NodeQuantumNumber]] = set()
        self.filter_ignore_qns: set[type[NodeQuantumNumber]] = set()
        if formalism == "helicity":
            self.filter_remove_qns = {
                NodeQuantumNumbers.l_magnitude,
                NodeQuantumNumbers.l_projection,
                NodeQuantumNumbers.s_magnitude,
                NodeQuantumNumbers.s_projection,
            }
        if "helicity" in formalism:
            self.filter_ignore_qns = {NodeQuantumNumbers.parity_prefactor}
        use_nbody_topology = False
        topology_building = topology_building.lower()
        if topology_building == "isobar":
            self.topologies: tuple[Topology, ...] = create_isobar_topologies(
                len(final_state)
            )
            """`.Topology` instances over which the STM propagates quantum numbers."""
        elif "n-body" in topology_building or "nbody" in topology_building:
            self.topologies = (
                create_n_body_topology(len(initial_state), len(final_state)),
            )
            use_nbody_topology = True
            # turn off mass conservation, in case more than one initial state
            # particle is present
            if len(initial_state) > 1:
                mass_conservation_factor = None

        if reload_pdg or len(self.__particles) == 0:
            self.__particles = load_pdg()

        if not self.interaction_type_settings:
            self.interaction_type_settings = create_interaction_settings(
                formalism,
                particle_db=self.__particles,
                nbody_topology=use_nbody_topology,
                mass_conservation_factor=mass_conservation_factor,
                max_angular_momentum=max_angular_momentum,
                max_spin_magnitude=max_spin_magnitude,
            )

        self.__intermediate_particle_filters = allowed_intermediate_particles
        if allowed_intermediate_particles is None:
            self.__allowed_intermediate_states: list[GraphEdgePropertyMap] = [
                create_edge_properties(x) for x in self.__particles
            ]
        else:
            self.set_allowed_intermediate_particles(allowed_intermediate_particles)

    def set_allowed_intermediate_particles(
        self, name_patterns: Iterable[str] | str, regex: bool = False
    ) -> None:
        if isinstance(name_patterns, str):
            name_patterns = [name_patterns]
        selected_particles = ParticleCollection()
        for pattern in name_patterns:
            matches = _filter_by_name_pattern(self.__particles, pattern, regex)
            if len(matches) == 0:
                msg = (
                    "Could not find any matches for allowed intermediate particle"
                    f' pattern "{pattern}"'
                )
                raise LookupError(msg)
            selected_particles.update(matches)
        self.__allowed_intermediate_states = [
            create_edge_properties(x)
            for x in sorted(selected_particles, key=lambda p: p.name)
        ]
        self.__intermediate_particle_filters = selected_particles.names

    @property
    def formalism(self) -> SpinFormalism:
        return self.__formalism

    def add_final_state_grouping(self, fs_group: list[str] | list[list[str]]) -> None:
        if not isinstance(fs_group, list):
            msg = "The final state grouping has to be of type list."
            raise TypeError(msg)
        if len(fs_group) > 0:
            if self.final_state_groupings is None:
                self.final_state_groupings = []
            nested_list = ensure_nested_list(fs_group)
            self.final_state_groupings.append(nested_list)

    @overload
    def get_allowed_interaction_types(
        self,
    ) -> list[InteractionType] | dict[int, list[InteractionType]]: ...

    @overload
    def get_allowed_interaction_types(self, node_id: int) -> list[InteractionType]: ...

    def get_allowed_interaction_types(self, node_id=None):  # type: ignore[no-untyped-def]
        if node_id is None:
            return self.__allowed_interaction_types
        if isinstance(self.__allowed_interaction_types, list):
            return self.__allowed_interaction_types
        return self.__allowed_interaction_types.get(node_id, DEFAULT_INTERACTION_TYPES)

    def set_allowed_interaction_types(
        self,
        allowed_interaction_types: Iterable[InteractionType],
        node_id: int | None = None,
    ) -> None:
        # verify order
        for allowed_types in allowed_interaction_types:
            if not isinstance(allowed_types, InteractionType):
                msg = "Allowed interaction types must be of type[InteractionType]"
                raise TypeError(msg)
            if allowed_types not in self.interaction_type_settings:
                _LOGGER.info(self.interaction_type_settings.keys())
                msg = f"Interaction {allowed_types} not found in settings"
                raise ValueError(msg)
        allowed_interaction_types = list(allowed_interaction_types)
        if node_id is None:
            self.__allowed_interaction_types = allowed_interaction_types
        else:
            if not isinstance(self.__allowed_interaction_types, dict):
                self.__allowed_interaction_types = {}
            self.__allowed_interaction_types[node_id] = allowed_interaction_types

    def create_problem_sets(self) -> dict[float, list[ProblemSet]]:
        problem_sets = [
            ProblemSet(permutation, initial_facts, settings)
            for topology in self.topologies
            for permutation in permutate_topology_kinematically(
                topology,
                self.initial_state,
                self.final_state,
                self.final_state_groupings,
            )
            for initial_facts in create_initial_facts(
                permutation, self.initial_state, self.final_state, self.__particles
            )
            for settings in self.__determine_graph_settings(permutation, initial_facts)
        ]
        return _group_by_strength(problem_sets)

    def __determine_graph_settings(  # noqa: C901, PLR0914
        self, topology: Topology, initial_facts: InitialFacts
    ) -> list[GraphSettings]:
        weak_edge_settings, _ = self.interaction_type_settings[InteractionType.WEAK]

        def create_intermediate_edge_qn_domains() -> dict:
            if self.__intermediate_particle_filters is None:
                return weak_edge_settings.qn_domains

            # if a list of intermediate states is given by user,
            # built a domain based on these states
            intermediate_edge_domains: dict[type[EdgeQuantumNumber], set] = defaultdict(
                set
            )
            intermediate_edge_domains[EdgeQuantumNumbers.spin_projection].update(
                weak_edge_settings.qn_domains[EdgeQuantumNumbers.spin_projection]
            )
            for particle_props in self.__allowed_intermediate_states:
                for edge_qn, qn_value in particle_props.items():
                    if edge_qn in {
                        EdgeQuantumNumbers.pid,
                        EdgeQuantumNumbers.mass,
                        EdgeQuantumNumbers.width,
                    }:
                        continue
                    intermediate_edge_domains[edge_qn].add(qn_value)

            return {k: list(v) for k, v in intermediate_edge_domains.items()}

        intermediate_state_edges = topology.intermediate_edge_ids
        int_edge_domains = create_intermediate_edge_qn_domains()

        def create_edge_settings(edge_id: int) -> EdgeSettings:
            settings = copy(weak_edge_settings)
            if edge_id in intermediate_state_edges:
                settings.qn_domains = int_edge_domains
            else:
                settings.qn_domains = {}
            return settings

        final_state_edges = topology.outgoing_edge_ids
        initial_state_edges = topology.incoming_edge_ids

        graph_settings: list[GraphSettings] = [
            MutableTransition(
                topology,
                states={
                    edge_id: create_edge_settings(edge_id)  # type: ignore[misc]
                    for edge_id in topology.edges
                },
            )
        ]

        for node_id in topology.nodes:
            interaction_types: list[InteractionType] = []
            out_edge_ids = topology.get_edge_ids_outgoing_from_node(node_id)
            in_edge_ids = topology.get_edge_ids_outgoing_from_node(node_id)
            in_states = [
                initial_facts.states[edge_id]
                for edge_id in [x for x in in_edge_ids if x in initial_state_edges]
            ]
            out_states = [
                initial_facts.states[edge_id]
                for edge_id in [x for x in out_edge_ids if x in final_state_edges]
            ]
            interactions = InteractionProperties()
            if node_id in initial_facts.interactions:
                interactions = initial_facts.interactions[node_id]
            for int_det in self.interaction_determinators:
                determined_interactions = int_det.check(
                    in_states, out_states, interactions
                )
                if interaction_types:
                    interaction_types = list(
                        set(determined_interactions) & set(interaction_types)
                    )
                else:
                    interaction_types = determined_interactions
            allowed_interaction_types = self.get_allowed_interaction_types(node_id)
            interaction_types = filter_interaction_types(
                interaction_types, allowed_interaction_types
            )
            _LOGGER.debug(
                "using %s interaction order for node: %s",
                str(interaction_types),
                str(node_id),
            )

            temp_graph_settings: list[GraphSettings] = graph_settings
            graph_settings = []
            for temp_setting in temp_graph_settings:
                for int_type in interaction_types:
                    updated_setting = deepcopy(temp_setting)
                    updated_setting.interactions[node_id] = deepcopy(
                        self.interaction_type_settings[int_type][1]
                    )
                    graph_settings.append(updated_setting)

        return graph_settings

    def find_solutions(  # noqa: C901
        self, problem_sets: dict[float, list[ProblemSet]]
    ) -> ReactionInfo:
        """Check for solutions for a specific set of interaction settings."""
        results = self._find_particle_transitions(problem_sets)
        for strength, result in results.items():
            _LOGGER.info(
                f"Number of solutions for strength {strength} after"
                f"QN solving: {len(result.solutions)}",
            )

        final_result = _SolutionContainer()
        for particle_result in results.values():
            final_result.extend(particle_result)

        # remove duplicate solutions, which only differ in the interaction qns
        final_solutions = remove_duplicate_solutions(
            final_result.solutions,
            self.filter_remove_qns,
            self.filter_ignore_qns,
        )

        execution_info = final_result.execution_info
        if (
            final_result.execution_info.violated_edge_rules
            or final_result.execution_info.violated_node_rules
        ):
            violated_rules: set[str] = set()
            for rules in execution_info.violated_edge_rules.values():
                violated_rules |= rules
            for rules in execution_info.violated_node_rules.values():
                violated_rules |= rules
            if violated_rules:
                msg = (
                    "There were violated conservation rules: "
                    f" {', '.join(violated_rules)}"
                )
                warnings.warn(msg, category=RuntimeWarning, stacklevel=1)
        if (
            final_result.execution_info.not_executed_edge_rules
            or final_result.execution_info.not_executed_node_rules
        ):
            not_executed_rules: set[str] = set()
            for rules in execution_info.not_executed_edge_rules.values():
                not_executed_rules |= rules
            for rules in execution_info.not_executed_node_rules.values():
                not_executed_rules |= rules
            msg = (
                "There are conservation rules that were not executed:"
                f" {', '.join(not_executed_rules)}"
            )
            warnings.warn(msg, category=RuntimeWarning, stacklevel=1)
        if not final_solutions:
            msg = "No solutions were found"
            raise RuntimeError(msg, execution_info)

        match_external_edges(final_solutions)
        final_solutions = [
            _match_final_state_ids(graph, self.final_state) for graph in final_solutions
        ]
        transitions = [
            graph.freeze().convert(lambda s: State(*s)) for graph in final_solutions
        ]
        return ReactionInfo(transitions, self.formalism)

    def _find_particle_transitions(
        self, problem_sets: dict[float, list[ProblemSet]]
    ) -> dict[float, _SolutionContainer]:
        qn_results = self.find_quantum_number_transitions(problem_sets)
        results: dict[float, _SolutionContainer] = defaultdict(_SolutionContainer)
        for strength, qn_solutions in qn_results.items():
            for qn_problem_set, qn_result in qn_solutions:
                particle_result = self.__convert_to_particle_definitions(
                    qn_problem_set.topology,
                    qn_result,
                )
                results[strength].extend(
                    particle_result,
                    intersect_violations=True,
                )
        return dict(results)

    def find_quantum_number_transitions(
        self, problem_sets: dict[float, list[ProblemSet]]
    ) -> dict[float, list[tuple[QNProblemSet, QNResult]]]:
        """Find allowed transitions purely in terms of quantum number sets."""
        qn_results: dict[float, list[tuple[QNProblemSet, QNResult]]] = defaultdict(list)
        _LOGGER.info(
            "Number of interaction settings groups being processed: %d",
            len(problem_sets),
        )
        total = sum(map(len, problem_sets.values()))
        progress_bar = tqdm(
            total=total,
            desc="Propagating quantum numbers",
            disable=_LOGGER.level > logging.WARNING,
        )
        for strength, problems in sorted(problem_sets.items(), reverse=True):
            _LOGGER.info(
                f"processing interaction settings group with strength {strength}",
            )
            _LOGGER.info(f"{len(problems)} entries in this group")
            _LOGGER.info(f"running with {self.__number_of_threads} threads...")

            qn_problems = [x.to_qn_problem_set() for x in problems]

            # Because of pickling problems of Generic classes (in this case
            # MutableTransition), multithreaded code has to work with
            # QNProblemSet's and QNResult's. So the appropriate conversions
            # have to be done before and after
            if self.__number_of_threads > 1:
                with Pool(self.__number_of_threads) as pool:
                    for qn_solution in pool.imap_unordered(
                        self._solve, qn_problems, chunksize=1
                    ):
                        qn_results[strength].append(qn_solution)
                        progress_bar.update()
            else:
                for problem in qn_problems:
                    qn_solution = self._solve(problem)
                    qn_results[strength].append(qn_solution)
                    progress_bar.update()
            if qn_results[strength] and self.reaction_mode == SolvingMode.FAST:
                break
        progress_bar.close()
        return qn_results

    def _solve(self, qn_problem_set: QNProblemSet) -> tuple[QNProblemSet, QNResult]:
        solver = CSPSolver(self.__allowed_intermediate_states)
        solutions = solver.find_solutions(qn_problem_set)
        return qn_problem_set, solutions

    def __convert_to_particle_definitions(
        self, topology: Topology, qn_result: QNResult
    ) -> _SolutionContainer:
        """Converts a `.QNResult` with a `.Topology` into `.ReactionInfo`.

        The ParticleCollection is used to retrieve a particle instance reference to
        lower the memory footprint.
        """
        solutions = []
        for solution in qn_result.solutions:
            graph = MutableTransition(  # type: ignore[var-annotated]
                topology=topology,
                interactions={
                    i: create_interaction_properties(x)  # type: ignore[misc]
                    for i, x in solution.interactions.items()
                },
                states={
                    i: find_particle(x, self.__particles)  # type: ignore[misc]
                    for i, x in solution.states.items()
                },
            )
            solutions.append(graph)

        return _SolutionContainer(
            solutions,
            ExecutionInfo(
                violated_edge_rules=qn_result.violated_edge_rules,
                violated_node_rules=qn_result.violated_node_rules,
                not_executed_node_rules=qn_result.not_executed_node_rules,
                not_executed_edge_rules=qn_result.not_executed_edge_rules,
            ),
        )


def _filter_by_name_pattern(
    particles: ParticleCollection, pattern: str, regex: bool
) -> ParticleCollection:
    def match_regex(particle: Particle) -> bool:
        return re.match(pattern, particle.name) is not None

    def match_substring(particle: Particle) -> bool:
        return pattern in particle.name

    if regex:
        return particles.filter(match_regex)
    return particles.filter(match_substring)


def _match_final_state_ids(
    graph: MutableTransition[ParticleWithSpin, InteractionProperties],
    state_definition: Sequence[StateDefinition],
) -> MutableTransition[ParticleWithSpin, InteractionProperties]:
    """Temporary fix to https://github.com/ComPWA/qrules/issues/143."""
    particle_names = _strip_spin(state_definition)
    name_to_id = {name: i for i, name in enumerate(particle_names)}
    id_remapping = {
        name_to_id[graph.states[i][0].name]: i for i in graph.topology.outgoing_edge_ids
    }
    new_topology = graph.topology.relabel_edges(id_remapping)
    return MutableTransition(
        new_topology,
        states={
            i: graph.states[id_remapping.get(i, i)]  # type: ignore[misc]
            for i in graph.topology.edges
        },
        interactions={i: graph.interactions[i] for i in graph.topology.nodes},  # type: ignore[misc]
    )


def _strip_spin(state_definition: Sequence[StateDefinition]) -> list[str]:
    particle_names = []
    for state in state_definition:
        if isinstance(state, str):
            particle_names.append(state)
        else:
            particle_names.append(state[0])
    return particle_names


@implement_pretty_repr
@frozen(order=True)
class State:
    particle: Particle = field(validator=instance_of(Particle))
    spin_projection: float = field(converter=_to_float)


StateTransition = FrozenTransition[State, InteractionProperties]
"""Transition of some initial `.State` to a final `.State`."""


def _sort_tuple(
    iterable: Iterable[StateTransition],
) -> tuple[StateTransition, ...]:
    return tuple(sorted(iterable))


@implement_pretty_repr
@frozen
class ReactionInfo:
    """Ordered collection of `StateTransition` instances."""

    transitions: tuple[StateTransition, ...] = field(converter=_sort_tuple)
    formalism: SpinFormalism = field(validator=in_(SpinFormalism.__args__))  # type: ignore[attr-defined]

    initial_state: FrozenDict[int, Particle] = field(init=False, repr=False, eq=False)
    final_state: FrozenDict[int, Particle] = field(init=False, repr=False, eq=False)

    def __attrs_post_init__(self) -> None:
        transition = self.transitions[0]
        initial = {i: s.particle for i, s in transition.initial_states.items()}
        final = {i: s.particle for i, s in transition.final_states.items()}
        object.__setattr__(self, "final_state", final)
        object.__setattr__(self, "initial_state", initial)

    def get_intermediate_particles(self) -> ParticleCollection:
        """Extract the names of the intermediate state particles."""
        particles = {
            state.particle
            for transition in self.transitions
            for state in transition.intermediate_states.values()
        }
        return ParticleCollection(particles)

    def group_by_topology(self) -> dict[Topology, list[StateTransition]]:
        groupings = defaultdict(list)
        for transition in self.transitions:
            groupings[transition.topology].append(transition)
        return dict(groupings)
