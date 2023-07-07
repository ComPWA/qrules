"""Find allowed transitions between an initial and final state."""

import logging
import re
from collections import abc, defaultdict
from copy import copy, deepcopy
from enum import Enum, auto
from itertools import zip_longest
from multiprocessing import Pool
from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    overload,
)

import attrs
from attrs import define, field, frozen
from attrs.validators import instance_of
from tqdm.auto import tqdm

from qrules._implementers import implement_pretty_repr

from ._system_control import (
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
from .combinatorics import (
    InitialFacts,
    StateDefinition,
    create_initial_facts,
    ensure_nested_list,
    match_external_edges,
    permutate_topology_kinematically,
)
from .particle import (
    Particle,
    ParticleCollection,
    ParticleWithSpin,
    _to_float,
    load_pdg,
)
from .quantum_numbers import (
    EdgeQuantumNumber,
    EdgeQuantumNumbers,
    InteractionProperties,
    NodeQuantumNumber,
    NodeQuantumNumbers,
)
from .settings import (
    DEFAULT_INTERACTION_TYPES,
    InteractionType,
    NumberOfThreads,
    create_interaction_settings,
)
from .solving import (
    CSPSolver,
    EdgeSettings,
    GraphEdgePropertyMap,
    GraphElementProperties,
    GraphSettings,
    NodeSettings,
    QNProblemSet,
    QNResult,
)
from .topology import (
    FrozenDict,
    StateTransitionGraph,
    Topology,
    create_isobar_topologies,
    create_n_body_topology,
)

if TYPE_CHECKING:
    try:
        from IPython.lib.pretty import PrettyPrinter
    except ImportError:
        PrettyPrinter = Any  # type: ignore[assignment,misc]

_LOGGER = logging.getLogger(__name__)


class SolvingMode(Enum):
    """Types of modes for solving."""

    FAST = auto()
    """Find "likeliest" solutions only."""
    FULL = auto()
    """Find all possible solutions."""


@implement_pretty_repr
@define(on_setattr=attrs.setters.frozen)
class ExecutionInfo:
    not_executed_node_rules: Dict[int, Set[str]] = field(
        factory=lambda: defaultdict(set)
    )
    violated_node_rules: Dict[int, Set[str]] = field(factory=lambda: defaultdict(set))
    not_executed_edge_rules: Dict[int, Set[str]] = field(
        factory=lambda: defaultdict(set)
    )
    violated_edge_rules: Dict[int, Set[str]] = field(factory=lambda: defaultdict(set))

    def extend(
        self, other_result: "ExecutionInfo", intersect_violations: bool = False
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

    solutions: List[StateTransitionGraph[ParticleWithSpin]] = field(factory=list)
    execution_info: ExecutionInfo = field(default=ExecutionInfo())

    def __attrs_post_init__(self) -> None:
        if self.solutions and (
            self.execution_info.violated_node_rules
            or self.execution_info.violated_edge_rules
        ):
            (
                f"Invalid {type(self).__name__}! Found {len(self.solutions)} solutions,"
                " but also violated rules."
            )
            msg = (
                f"Invalid {self.__class__.__name__}! Found"
                f" {len(self.solutions)} solutions, but also violated rules."
            )
            raise ValueError(
                msg,
                self.execution_info.violated_node_rules,
                self.execution_info.violated_edge_rules,
            )

    def extend(
        self, other: "_SolutionContainer", intersect_violations: bool = False
    ) -> None:
        if self.solutions or other.solutions:
            self.solutions.extend(other.solutions)
            self.execution_info.clear()
        else:
            self.execution_info.extend(other.execution_info, intersect_violations)


@implement_pretty_repr
@define
class ProblemSet:
    """Particle reaction problem set, defined as a graph like data structure.

    Args:
        topology: `.Topology` that contains the structure of the reaction.
        initial_facts: `~.InitialFacts` that contain the info of initial and
          final state in connection with the topology.
        solving_settings: Solving related settings such as the conservation
          rules and the quantum number domains.
    """

    topology: Topology
    initial_facts: InitialFacts
    solving_settings: GraphSettings

    def to_qn_problem_set(self) -> QNProblemSet:
        node_props = {
            k: create_node_properties(v)
            for k, v in self.initial_facts.node_props.items()
        }
        edge_props = {
            k: create_edge_properties(v[0], v[1])
            for k, v in self.initial_facts.edge_props.items()
        }
        return QNProblemSet(
            topology=self.topology,
            initial_facts=GraphElementProperties(
                node_props=node_props, edge_props=edge_props
            ),
            solving_settings=self.solving_settings,
        )


def _group_by_strength(
    problem_sets: List[ProblemSet],
) -> Dict[float, List[ProblemSet]]:
    def calculate_strength(node_interaction_settings: Dict[int, NodeSettings]) -> float:
        strength = 1.0
        for int_setting in node_interaction_settings.values():
            strength *= int_setting.interaction_strength
        return strength

    strength_sorted_problem_sets: Dict[float, List[ProblemSet]] = defaultdict(list)
    for problem_set in problem_sets:
        strength = calculate_strength(problem_set.solving_settings.node_settings)
        strength_sorted_problem_sets[strength].append(problem_set)
    return strength_sorted_problem_sets


class StateTransitionManager:
    """Main handler for decay topologies.

    .. seealso:: :doc:`/usage/reaction` and `.generate_transitions`
    """

    def __init__(  # noqa: C901, PLR0912
        self,
        initial_state: Sequence[StateDefinition],
        final_state: Sequence[StateDefinition],
        particle_db: Optional[ParticleCollection] = None,
        allowed_intermediate_particles: Optional[List[str]] = None,
        interaction_type_settings: Optional[Dict[InteractionType, Tuple[EdgeSettings, NodeSettings]]] = None,  # type: ignore[assignment]
        formalism: str = "helicity",
        topology_building: str = "isobar",
        solving_mode: SolvingMode = SolvingMode.FAST,
        reload_pdg: bool = False,
        mass_conservation_factor: Optional[float] = 3.0,
        max_angular_momentum: int = 1,
        max_spin_magnitude: float = 2.0,
        number_of_threads: Optional[int] = None,
    ) -> None:
        if number_of_threads is not None:
            NumberOfThreads.set(number_of_threads)
        self.__number_of_threads = NumberOfThreads.get()
        if interaction_type_settings is None:
            interaction_type_settings = {}
        allowed_formalisms = [
            "helicity",
            "canonical-helicity",
            "canonical",
        ]
        if formalism not in allowed_formalisms:
            msg = (
                f'Formalism "{formalism}" not implemented. Use one of'
                f" {allowed_formalisms} instead."
            )
            raise NotImplementedError(msg)
        self.__formalism = str(formalism)
        self.__particles = ParticleCollection()
        if particle_db is not None:
            self.__particles = particle_db
        self.reaction_mode = str(solving_mode)
        self.initial_state = list(initial_state)
        self.final_state = list(final_state)
        self.interaction_type_settings = interaction_type_settings

        self.interaction_determinators: List[InteractionDeterminator] = [
            LeptonCheck(),
            GammaCheck(),
        ]
        self.final_state_groupings: Optional[List[List[List[str]]]] = None
        self.__allowed_interaction_types: Union[
            List[InteractionType],
            Dict[int, List[InteractionType]],
        ] = DEFAULT_INTERACTION_TYPES
        self.filter_remove_qns: Set[Type[NodeQuantumNumber]] = set()
        self.filter_ignore_qns: Set[Type[NodeQuantumNumber]] = set()
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
            self.topologies: Tuple[Topology, ...] = create_isobar_topologies(
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
            self.__allowed_intermediate_states: List[GraphEdgePropertyMap] = [
                create_edge_properties(x) for x in self.__particles
            ]
        else:
            self.set_allowed_intermediate_particles(allowed_intermediate_particles)

    def set_allowed_intermediate_particles(
        self, name_patterns: Union[Iterable[str], str], regex: bool = False
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

    @property
    def formalism(self) -> str:
        return self.__formalism

    def add_final_state_grouping(
        self, fs_group: Union[List[str], List[List[str]]]
    ) -> None:
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
    ) -> Union[List[InteractionType], Dict[int, List[InteractionType]]]:
        ...

    @overload
    def get_allowed_interaction_types(self, node_id: int) -> List[InteractionType]:
        ...

    def get_allowed_interaction_types(self, node_id=None):  # type: ignore[no-untyped-def]
        if node_id is None:
            return self.__allowed_interaction_types
        if isinstance(self.__allowed_interaction_types, list):
            return self.__allowed_interaction_types
        return self.__allowed_interaction_types.get(node_id, DEFAULT_INTERACTION_TYPES)

    def set_allowed_interaction_types(
        self,
        allowed_interaction_types: Iterable[InteractionType],
        node_id: Optional[int] = None,
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

    def create_problem_sets(self) -> Dict[float, List[ProblemSet]]:
        problem_sets = [
            ProblemSet(permutation, initial_facts, settings)
            for initial_facts in create_initial_facts(
                self.initial_state, self.final_state, self.__particles
            )
            for topology in self.topologies
            for permutation in permutate_topology_kinematically(
                topology,
                self.initial_state,
                self.final_state,
                self.final_state_groupings,
            )
            for settings in self.__determine_graph_settings(permutation, initial_facts)
        ]
        return _group_by_strength(problem_sets)

    def __determine_graph_settings(  # noqa: C901
        self, topology: Topology, initial_facts: InitialFacts
    ) -> List[GraphSettings]:
        weak_edge_settings, _ = self.interaction_type_settings[InteractionType.WEAK]

        def create_intermediate_edge_qn_domains() -> Dict:
            if self.__intermediate_particle_filters is None:
                return weak_edge_settings.qn_domains

            # if a list of intermediate states is given by user,
            # built a domain based on these states
            intermediate_edge_domains: Dict[Type[EdgeQuantumNumber], Set] = defaultdict(
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

        graph_settings: List[GraphSettings] = [
            GraphSettings(
                edge_settings={
                    edge_id: create_edge_settings(edge_id) for edge_id in topology.edges
                },
                node_settings={},
            )
        ]

        for node_id in topology.nodes:
            interaction_types: List[InteractionType] = []
            out_edge_ids = topology.get_edge_ids_outgoing_from_node(node_id)
            in_edge_ids = topology.get_edge_ids_outgoing_from_node(node_id)
            in_edge_props = [
                initial_facts.edge_props[edge_id]
                for edge_id in [x for x in in_edge_ids if x in initial_state_edges]
            ]
            out_edge_props = [
                initial_facts.edge_props[edge_id]
                for edge_id in [x for x in out_edge_ids if x in final_state_edges]
            ]
            node_props = InteractionProperties()
            if node_id in initial_facts.node_props:
                node_props = initial_facts.node_props[node_id]
            for int_det in self.interaction_determinators:
                determined_interactions = int_det.check(
                    in_edge_props, out_edge_props, node_props
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

            temp_graph_settings: List[GraphSettings] = graph_settings
            graph_settings = []
            for temp_setting in temp_graph_settings:
                for int_type in interaction_types:
                    updated_setting = deepcopy(temp_setting)
                    updated_setting.node_settings[node_id] = deepcopy(
                        self.interaction_type_settings[int_type][1]
                    )
                    graph_settings.append(updated_setting)

        return graph_settings

    def find_solutions(  # noqa: C901, PLR0912
        self,
        problem_sets: Dict[float, List[ProblemSet]],
    ) -> "ReactionInfo":
        """Check for solutions for a specific set of interaction settings."""
        results: Dict[float, _SolutionContainer] = {}
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
            # StateTransitionGraph), multithreaded code has to work with
            # QNProblemSet's and QNResult's. So the appropriate conversions
            # have to be done before and after
            temp_qn_results: List[Tuple[QNProblemSet, QNResult]] = []
            if self.__number_of_threads > 1:
                with Pool(self.__number_of_threads) as pool:
                    for qn_result in pool.imap_unordered(self._solve, qn_problems, 1):
                        temp_qn_results.append(qn_result)
                        progress_bar.update()
            else:
                for problem in qn_problems:
                    temp_qn_results.append(self._solve(problem))
                    progress_bar.update()
            for temp_qn_result in temp_qn_results:
                temp_result = self.__convert_result(
                    temp_qn_result[0].topology,
                    temp_qn_result[1],
                )
                if strength not in results:
                    results[strength] = temp_result
                else:
                    results[strength].extend(temp_result, True)
            if results[strength].solutions and self.reaction_mode == SolvingMode.FAST:
                break
        progress_bar.close()

        for key, result in results.items():
            _LOGGER.info(
                (
                    f"number of solutions for strength ({key}) "
                    f"after qn solving: {len(result.solutions)}"
                ),
            )

        final_result = _SolutionContainer()
        for temp_result in results.values():
            final_result.extend(temp_result)

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
            violated_rules: Set[str] = set()
            for rules in execution_info.violated_edge_rules.values():
                violated_rules |= rules
            for rules in execution_info.violated_node_rules.values():
                violated_rules |= rules
            if violated_rules:
                raise RuntimeError(
                    "There were violated conservation rules: "
                    + ", ".join(violated_rules)
                )
        if (
            final_result.execution_info.not_executed_edge_rules
            or final_result.execution_info.not_executed_node_rules
        ):
            not_executed_rules: Set[str] = set()
            for rules in execution_info.not_executed_edge_rules.values():
                not_executed_rules |= rules
            for rules in execution_info.not_executed_node_rules.values():
                not_executed_rules |= rules
            raise RuntimeWarning(
                "There are conservation rules that were not executed: "
                + ", ".join(not_executed_rules)
            )
        if not final_solutions:
            msg = "No solutions were found"
            raise ValueError(msg)

        match_external_edges(final_solutions)
        final_solutions = [
            _match_final_state_ids(graph, self.final_state) for graph in final_solutions
        ]
        return ReactionInfo.from_graphs(final_solutions, self.formalism)

    def _solve(self, qn_problem_set: QNProblemSet) -> Tuple[QNProblemSet, QNResult]:
        solver = CSPSolver(self.__allowed_intermediate_states)
        solutions = solver.find_solutions(qn_problem_set)
        return qn_problem_set, solutions

    def __convert_result(
        self, topology: Topology, qn_result: QNResult
    ) -> _SolutionContainer:
        """Converts a `.QNResult` with a `.Topology` into `.ReactionInfo`.

        The ParticleCollection is used to retrieve a particle instance reference to
        lower the memory footprint.
        """
        solutions = []
        for solution in qn_result.solutions:
            graph = StateTransitionGraph[ParticleWithSpin](
                topology=topology,
                node_props={
                    i: create_interaction_properties(x)
                    for i, x in solution.node_quantum_numbers.items()
                },
                edge_props={
                    i: find_particle(x, self.__particles)  # type: ignore[misc]
                    for i, x in solution.edge_quantum_numbers.items()
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
    graph: StateTransitionGraph[ParticleWithSpin],
    state_definition: Sequence[StateDefinition],
) -> StateTransitionGraph[ParticleWithSpin]:
    """Temporary fix to https://github.com/ComPWA/qrules/issues/143."""
    particle_names = _strip_spin(state_definition)
    name_to_id = {name: i for i, name in enumerate(particle_names)}
    id_remapping = {
        name_to_id[graph.get_edge_props(i)[0].name]: i
        for i in graph.topology.outgoing_edge_ids
    }
    new_topology = graph.topology.relabel_edges(id_remapping)
    return StateTransitionGraph(
        new_topology,
        edge_props={
            i: graph.get_edge_props(id_remapping.get(i, i))
            for i in graph.topology.edges
        },
        node_props={i: graph.get_node_props(i) for i in graph.topology.nodes},
    )


def _strip_spin(state_definition: Sequence[StateDefinition]) -> List[str]:
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


@implement_pretty_repr
@frozen(order=True)
class StateTransition:
    """Frozen instance of a `.StateTransitionGraph` of a particle with spin."""

    topology: Topology = field(validator=instance_of(Topology))
    states: FrozenDict[int, State] = field(converter=FrozenDict)
    interactions: FrozenDict[int, InteractionProperties] = field(converter=FrozenDict)

    def __attrs_post_init__(self) -> None:
        _assert_defined(self.topology.edges, self.states)
        _assert_defined(self.topology.nodes, self.interactions)

    @staticmethod
    def from_graph(
        graph: StateTransitionGraph[ParticleWithSpin],
    ) -> "StateTransition":
        return StateTransition(
            topology=graph.topology,
            states=FrozenDict(
                {i: State(*graph.get_edge_props(i)) for i in graph.topology.edges}
            ),
            interactions=FrozenDict(
                {i: graph.get_node_props(i) for i in graph.topology.nodes}
            ),
        )

    def to_graph(self) -> StateTransitionGraph[ParticleWithSpin]:
        return StateTransitionGraph[ParticleWithSpin](
            topology=self.topology,
            edge_props={
                i: (state.particle, state.spin_projection)
                for i, state in self.states.items()
            },
            node_props=self.interactions,
        )

    @property
    def initial_states(self) -> Dict[int, State]:
        return self.filter_states(self.topology.incoming_edge_ids)

    @property
    def final_states(self) -> Dict[int, State]:
        return self.filter_states(self.topology.outgoing_edge_ids)

    @property
    def intermediate_states(self) -> Dict[int, State]:
        return self.filter_states(self.topology.intermediate_edge_ids)

    def filter_states(self, edge_ids: Iterable[int]) -> Dict[int, State]:
        return {i: self.states[i] for i in edge_ids}

    @property
    def particles(self) -> Dict[int, Particle]:
        return {i: edge_prop.particle for i, edge_prop in self.states.items()}


def _assert_defined(items: Collection, properties: Mapping) -> None:
    existing = set(items)
    defined = set(properties)
    if existing & defined != existing:
        msg = (
            "Some items have no property assigned to them. Available items:"
            f" {existing}, items with property: {defined}"
        )
        raise ValueError(msg)


def _to_sorted_tuple(
    iterable: Iterable[StateTransition],
) -> Tuple[StateTransition, ...]:
    if any(not isinstance(t, StateTransition) for t in iterable):
        msg = f"Not all instances are of type {StateTransition.__name__}"
        raise TypeError(msg)
    return tuple(sorted(iterable))


@frozen
class StateTransitionCollection(abc.Sequence):
    """`.StateTransition` instances with the same `.Topology` and edge IDs."""

    transitions: Tuple[StateTransition, ...] = field(converter=_to_sorted_tuple)
    topology: Topology = field(init=False, repr=False)
    initial_state: FrozenDict[int, Particle] = field(init=False, repr=False)
    final_state: FrozenDict[int, Particle] = field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        if len(self.transitions) == 0:
            msg = f"At least one {StateTransition.__name__} required"
            raise ValueError(msg)
        some_transition = next(iter(self.transitions))
        topology = some_transition.topology
        if any(t.topology != topology for t in self.transitions):
            msg = (
                f"Not all {StateTransition.__name__} items have the same underlying"
                f" topology. Expecting: {topology}"
            )
            raise TypeError(msg)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(
            self,
            "initial_state",
            FrozenDict(
                {
                    i: s.particle
                    for i, s in some_transition.states.items()
                    if i in some_transition.topology.incoming_edge_ids
                }
            ),
        )
        object.__setattr__(
            self,
            "final_state",
            FrozenDict(
                {
                    i: s.particle
                    for i, s in some_transition.states.items()
                    if i in some_transition.topology.outgoing_edge_ids
                }
            ),
        )

    def _repr_pretty_(self, p: "PrettyPrinter", cycle: bool) -> None:
        class_name = type(self).__name__
        if cycle:
            p.text(f"{class_name}(...)")
        else:
            with p.group(indent=2, open=f"{class_name}(transitions=("):
                for transition in self:
                    p.breakable()
                    p.pretty(transition)  # type: ignore[attr-defined]
                    p.text(",")
            p.breakable()
            p.text("))")

    def __contains__(self, item: object) -> bool:
        return item in self.transitions

    @overload
    def __getitem__(self, idx: int) -> StateTransition:
        ...

    @overload
    def __getitem__(self, idx: slice) -> Tuple[StateTransition]:
        ...

    def __getitem__(self, idx: Any) -> Any:
        return self.transitions[idx]

    def __iter__(self) -> Iterator[StateTransition]:
        return iter(self.transitions)

    def __len__(self) -> int:
        return len(self.transitions)

    @staticmethod
    def from_graphs(
        graphs: Iterable[StateTransitionGraph[ParticleWithSpin]],
    ) -> "StateTransitionCollection":
        transitions = [StateTransition.from_graph(g) for g in graphs]
        return StateTransitionCollection(transitions)

    def to_graphs(self) -> List[StateTransitionGraph[ParticleWithSpin]]:
        return [transition.to_graph() for transition in sorted(self)]

    def get_intermediate_particles(self) -> ParticleCollection:
        """Extract the particle names of the intermediate states."""
        intermediate_states = ParticleCollection()
        for transition in self.transitions:
            for state in transition.intermediate_states.values():
                if state.particle not in intermediate_states:
                    intermediate_states.add(state.particle)
        return intermediate_states


def _to_tuple(
    iterable: Iterable[StateTransitionCollection],
) -> Tuple[StateTransitionCollection, ...]:
    if any(not isinstance(t, StateTransitionCollection) for t in iterable):
        msg = f"Not all instances are of type {StateTransitionCollection.__name__}"
        raise TypeError(msg)
    return tuple(iterable)


@frozen(eq=False, hash=True)
class ReactionInfo:
    """`StateTransitionCollection` instances, grouped by `.Topology`."""

    transition_groups: Tuple[StateTransitionCollection, ...] = field(
        converter=_to_tuple
    )
    transitions: Tuple[StateTransition, ...] = field(init=False, repr=False, eq=False)
    initial_state: FrozenDict[int, Particle] = field(init=False, repr=False)
    final_state: FrozenDict[int, Particle] = field(init=False, repr=False)
    formalism: str = field(validator=instance_of(str))

    def __attrs_post_init__(self) -> None:
        if len(self.transition_groups) == 0:
            msg = f"At least one {StateTransitionCollection.__name__} required"
            raise ValueError(msg)
        transitions: List[StateTransition] = []
        for grouping in self.transition_groups:
            transitions.extend(sorted(grouping))
        first_grouping = self.transition_groups[0]
        object.__setattr__(self, "transitions", tuple(transitions))
        object.__setattr__(self, "final_state", first_grouping.final_state)
        object.__setattr__(self, "initial_state", first_grouping.initial_state)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ReactionInfo):
            for own_grouping, other_grouping in zip_longest(
                self.transition_groups, other.transition_groups
            ):
                if own_grouping != other_grouping:
                    return False
            return True
        msg = (
            f"Cannot compare {self.__class__.__name__} with {other.__class__.__name__}"
        )
        raise NotImplementedError(msg)

    def _repr_pretty_(self, p: "PrettyPrinter", cycle: bool) -> None:
        class_name = type(self).__name__
        if cycle:
            p.text(f"{class_name}(...)")
        else:
            with p.group(indent=2, open=f"{class_name}("):
                p.breakable()
                p.text("transition_groups=")
                with p.group(indent=2, open="("):
                    for transition_grouping in self.transition_groups:
                        p.breakable()
                        p.pretty(transition_grouping)  # type: ignore[attr-defined]
                        p.text(",")
                p.breakable()
                p.text("),")
                p.breakable()
                p.text("formalism=")
                p.pretty(self.formalism)  # type: ignore[attr-defined]
                p.text(",")
            p.breakable()
            p.text(")")

    def get_intermediate_particles(self) -> ParticleCollection:
        """Extract the names of the intermediate state particles."""
        return ParticleCollection(
            set().union(
                *[
                    grouping.get_intermediate_particles()
                    for grouping in self.transition_groups
                ]
            )
        )

    @staticmethod
    def from_graphs(
        graphs: Iterable[StateTransitionGraph[ParticleWithSpin]],
        formalism: str,
    ) -> "ReactionInfo":
        transition_mapping: DefaultDict[Topology, List[StateTransition]] = defaultdict(
            list
        )
        for graph in graphs:
            transition_mapping[graph.topology].append(StateTransition.from_graph(graph))
        transition_groups = tuple(
            StateTransitionCollection(transitions)
            for transitions in transition_mapping.values()
        )
        return ReactionInfo(transition_groups, formalism)

    def to_graphs(self) -> List[StateTransitionGraph[ParticleWithSpin]]:
        graphs: List[StateTransitionGraph[ParticleWithSpin]] = []
        for grouping in self.transition_groups:
            graphs.extend(grouping.to_graphs())
        return graphs
