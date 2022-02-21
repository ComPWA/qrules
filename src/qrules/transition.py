"""Find allowed transitions between an initial and final state."""

import logging
from collections import defaultdict
from copy import copy, deepcopy
from enum import Enum, auto
from multiprocessing import Pool
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
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
    create_particle,
    filter_interaction_types,
    remove_duplicate_solutions,
)
from .combinatorics import (
    InitialFacts,
    StateDefinition,
    create_initial_facts,
    match_external_edges,
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
    MutableTransition,
    Topology,
    _assert_all_defined,
    create_isobar_topologies,
    create_n_body_topology,
)


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
    violated_node_rules: Dict[int, Set[str]] = field(
        factory=lambda: defaultdict(set)
    )
    not_executed_edge_rules: Dict[int, Set[str]] = field(
        factory=lambda: defaultdict(set)
    )
    violated_edge_rules: Dict[int, Set[str]] = field(
        factory=lambda: defaultdict(set)
    )

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

    solutions: List[
        MutableTransition[ParticleWithSpin, InteractionProperties]
    ] = field(factory=list)
    execution_info: ExecutionInfo = field(default=ExecutionInfo())

    def __attrs_post_init__(self) -> None:
        if self.solutions and (
            self.execution_info.violated_node_rules
            or self.execution_info.violated_edge_rules
        ):
            raise ValueError(
                f"Invalid {self.__class__.__name__}! Found"
                f" {len(self.solutions)} solutions, but also violated rules.",
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
            self.execution_info.extend(
                other.execution_info, intersect_violations
            )


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
    def calculate_strength(
        node_interaction_settings: Dict[int, NodeSettings]
    ) -> float:
        strength = 1.0
        for int_setting in node_interaction_settings.values():
            strength *= int_setting.interaction_strength
        return strength

    strength_sorted_problem_sets: Dict[float, List[ProblemSet]] = defaultdict(
        list
    )
    for problem_set in problem_sets:
        strength = calculate_strength(
            problem_set.solving_settings.node_settings
        )
        strength_sorted_problem_sets[strength].append(problem_set)
    return strength_sorted_problem_sets


class StateTransitionManager:  # pylint: disable=too-many-instance-attributes
    """Main handler for decay topologies.

    .. seealso:: :doc:`/usage/reaction` and `.generate_transitions`
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-branches, too-many-locals
        self,
        initial_state: Sequence[StateDefinition],
        final_state: Sequence[StateDefinition],
        particle_db: Optional[ParticleCollection] = None,
        allowed_intermediate_particles: Optional[List[str]] = None,
        interaction_type_settings: Dict[
            InteractionType, Tuple[EdgeSettings, NodeSettings]
        ] = None,
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
            raise NotImplementedError(
                f'Formalism "{formalism}" not implemented.'
                f" Use one of {allowed_formalisms} instead."
            )
        self.__formalism = str(formalism)
        self.__particles = ParticleCollection()
        if particle_db is not None:
            self.__particles = particle_db
        self.reaction_mode = str(solving_mode)
        self.initial_state = initial_state
        self.final_state = final_state
        self.interaction_type_settings = interaction_type_settings

        self.interaction_determinators: List[InteractionDeterminator] = [
            LeptonCheck(),
            GammaCheck(),
        ]
        self.final_state_groupings: Optional[List[List[List[str]]]] = None
        self.allowed_interaction_types: List[InteractionType] = [
            InteractionType.STRONG,
            InteractionType.EM,
            InteractionType.WEAK,
        ]
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

        self.__user_allowed_intermediate_particles = (
            allowed_intermediate_particles
        )
        self.__allowed_intermediate_particles: List[GraphEdgePropertyMap] = []
        if allowed_intermediate_particles is not None:
            self.set_allowed_intermediate_particles(
                allowed_intermediate_particles
            )
        else:
            self.__allowed_intermediate_particles = [
                create_edge_properties(x) for x in self.__particles
            ]

    def set_allowed_intermediate_particles(
        self, particle_names: List[str]
    ) -> None:
        self.__allowed_intermediate_particles = []
        for particle_name in particle_names:
            matches = self.__particles.filter(
                lambda p: particle_name  # pylint: disable=cell-var-from-loop
                in p.name
            )
            if len(matches) == 0:
                raise LookupError(
                    "Could not find any matches for allowed intermediate"
                    f' particle "{particle_name}"'
                )
            self.__allowed_intermediate_particles += [
                create_edge_properties(x) for x in matches
            ]

    @property
    def formalism(self) -> str:
        return self.__formalism

    def add_final_state_grouping(
        self, fs_group: Union[List[str], List[List[str]]]
    ) -> None:
        if not isinstance(fs_group, list):
            raise ValueError(
                "The final state grouping has to be of type list."
            )
        if len(fs_group) > 0:
            if self.final_state_groupings is None:
                self.final_state_groupings = []
            nested_list = _safe_wrap_list(fs_group)
            self.final_state_groupings.append(nested_list)

    def set_allowed_interaction_types(
        self, allowed_interaction_types: Iterable[InteractionType]
    ) -> None:
        # verify order
        for allowed_types in allowed_interaction_types:
            if not isinstance(allowed_types, InteractionType):
                raise TypeError(
                    "allowed interaction types must be of type"
                    "[InteractionType]"
                )
            if allowed_types not in self.interaction_type_settings:
                logging.info(self.interaction_type_settings.keys())
                raise ValueError(
                    f"interaction {allowed_types} not found in settings"
                )
        self.allowed_interaction_types = list(allowed_interaction_types)

    def create_problem_sets(self) -> Dict[float, List[ProblemSet]]:
        problem_sets = []
        for topology in self.topologies:
            for initial_facts in create_initial_facts(
                topology=topology,
                particle_db=self.__particles,
                initial_state=self.initial_state,
                final_state=self.final_state,
                final_state_groupings=self.final_state_groupings,
            ):
                problem_sets.extend(
                    [
                        ProblemSet(
                            topology=topology,
                            initial_facts=initial_facts,
                            solving_settings=x,
                        )
                        for x in self.__determine_graph_settings(
                            topology, initial_facts
                        )
                    ]
                )
        # create groups of settings ordered by "probability"
        return _group_by_strength(problem_sets)

    def __determine_graph_settings(
        self, topology: Topology, initial_facts: InitialFacts
    ) -> List[GraphSettings]:
        # pylint: disable=too-many-locals
        def create_intermediate_edge_qn_domains() -> Dict:
            # if a list of intermediate states is given by user,
            # built a domain based on these states
            if self.__user_allowed_intermediate_particles:
                intermediate_edge_domains: Dict[
                    Type[EdgeQuantumNumber], Set
                ] = defaultdict(set)
                intermediate_edge_domains[
                    EdgeQuantumNumbers.spin_projection
                ].update(
                    self.interaction_type_settings[InteractionType.WEAK][
                        0
                    ].qn_domains[EdgeQuantumNumbers.spin_projection]
                )
                for particle_props in self.__allowed_intermediate_particles:
                    for edge_qn, qn_value in particle_props.items():
                        intermediate_edge_domains[edge_qn].add(qn_value)

                return {
                    k: list(v)
                    for k, v in intermediate_edge_domains.items()
                    if k is not EdgeQuantumNumbers.pid
                    and k is not EdgeQuantumNumbers.mass
                    and k is not EdgeQuantumNumbers.width
                }

            return self.interaction_type_settings[InteractionType.WEAK][
                0
            ].qn_domains

        intermediate_state_edges = topology.intermediate_edge_ids
        int_edge_domains = create_intermediate_edge_qn_domains()

        def create_edge_settings(edge_id: int) -> EdgeSettings:
            settings = copy(
                self.interaction_type_settings[InteractionType.WEAK][0]
            )
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
                    edge_id: create_edge_settings(edge_id)
                    for edge_id in topology.edges
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
                for edge_id in [
                    x for x in in_edge_ids if x in initial_state_edges
                ]
            ]
            out_edge_props = [
                initial_facts.edge_props[edge_id]
                for edge_id in [
                    x for x in out_edge_ids if x in final_state_edges
                ]
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
            interaction_types = filter_interaction_types(
                interaction_types, self.allowed_interaction_types
            )
            logging.debug(
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

    def find_solutions(  # pylint: disable=too-many-branches
        self,
        problem_sets: Dict[float, List[ProblemSet]],
    ) -> "ReactionInfo":
        # pylint: disable=too-many-locals
        """Check for solutions for a specific set of interaction settings."""
        results: Dict[float, _SolutionContainer] = {}
        logging.info(
            "Number of interaction settings groups being processed: %d",
            len(problem_sets),
        )
        total = sum(map(len, problem_sets.values()))
        progress_bar = tqdm(
            total=total,
            desc="Propagating quantum numbers",
            disable=logging.getLogger().level > logging.WARNING,
        )
        for strength, problems in sorted(problem_sets.items(), reverse=True):
            logging.info(
                "processing interaction settings group with "
                f"strength {strength}",
            )
            logging.info(f"{len(problems)} entries in this group")
            logging.info(f"running with {self.__number_of_threads} threads...")

            qn_problems = [x.to_qn_problem_set() for x in problems]

            # Because of pickling problems of Generic classes (in this case
            # MutableTransition), multithreaded code has to work with
            # QNProblemSet's and QNResult's. So the appropriate conversions
            # have to be done before and after
            temp_qn_results: List[Tuple[QNProblemSet, QNResult]] = []
            if self.__number_of_threads > 1:
                with Pool(self.__number_of_threads) as pool:
                    for qn_result in pool.imap_unordered(
                        self._solve, qn_problems, 1
                    ):
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
            if (
                results[strength].solutions
                and self.reaction_mode == SolvingMode.FAST
            ):
                break
        progress_bar.close()

        for key, result in results.items():
            logging.info(
                f"number of solutions for strength ({key}) "
                f"after qn solving: {len(result.solutions)}",
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

        if (
            final_result.execution_info.violated_edge_rules
            or final_result.execution_info.violated_node_rules
        ):
            execution_info = final_result.execution_info
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
            raise ValueError("No solutions were found")

        match_external_edges(final_solutions)
        final_solutions = [
            _match_final_state_ids(graph, self.final_state)
            for graph in final_solutions
        ]
        return ReactionInfo.from_graphs(final_solutions, self.formalism)

    def _solve(
        self, qn_problem_set: QNProblemSet
    ) -> Tuple[QNProblemSet, QNResult]:
        solver = CSPSolver(self.__allowed_intermediate_particles)

        return (qn_problem_set, solver.find_solutions(qn_problem_set))

    def __convert_result(
        self, topology: Topology, qn_result: QNResult
    ) -> _SolutionContainer:
        """Converts a `.QNResult` with a `.Topology` into `.ReactionInfo`.

        The ParticleCollection is used to retrieve a particle instance
        reference to lower the memory footprint.
        """
        solutions = []
        for solution in qn_result.solutions:
            graph = MutableTransition[ParticleWithSpin, InteractionProperties](
                topology=topology,
                node_props={
                    i: create_interaction_properties(x)
                    for i, x in solution.node_quantum_numbers.items()
                },
                edge_props={
                    i: create_particle(x, self.__particles)
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


def _safe_wrap_list(
    nested_list: Union[List[str], List[List[str]]]
) -> List[List[str]]:
    if all(map(lambda i: isinstance(i, list), nested_list)):
        return nested_list  # type: ignore[return-value]
    if all(map(lambda i: isinstance(i, str), nested_list)):
        return [nested_list]  # type: ignore[list-item]
    raise TypeError(
        f"Input final state grouping {nested_list} is not a list of lists of"
        " strings"
    )


def _match_final_state_ids(
    graph: MutableTransition[ParticleWithSpin, InteractionProperties],
    state_definition: Sequence[StateDefinition],
) -> MutableTransition[ParticleWithSpin, InteractionProperties]:
    """Temporary fix to https://github.com/ComPWA/qrules/issues/143."""
    particle_names = _strip_spin(state_definition)
    name_to_id = {name: i for i, name in enumerate(particle_names)}
    id_remapping = {
        name_to_id[graph.edge_props[i][0].name]: i
        for i in graph.topology.outgoing_edge_ids
    }
    new_topology = graph.topology.relabel_edges(id_remapping)
    return MutableTransition(
        new_topology,
        edge_props={
            i: graph.edge_props[id_remapping.get(i, i)]
            for i in graph.topology.edges
        },
        node_props={i: graph.node_props[i] for i in graph.topology.nodes},
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
    """Frozen instance of a `.MutableTransition` of a particle with spin."""

    topology: Topology = field(validator=instance_of(Topology))
    states: FrozenDict[int, State] = field(converter=FrozenDict)
    interactions: FrozenDict[int, InteractionProperties] = field(
        converter=FrozenDict
    )

    def __attrs_post_init__(self) -> None:
        _assert_all_defined(self.topology.edges, self.states)
        _assert_all_defined(self.topology.nodes, self.interactions)

    @staticmethod
    def from_graph(
        graph: MutableTransition[ParticleWithSpin, InteractionProperties],
    ) -> "StateTransition":
        return StateTransition(
            topology=graph.topology,
            states=FrozenDict(
                {i: State(*graph.edge_props[i]) for i in graph.topology.edges}
            ),
            interactions=FrozenDict(
                {i: graph.node_props[i] for i in graph.topology.nodes}
            ),
        )

    def to_graph(
        self,
    ) -> MutableTransition[ParticleWithSpin, InteractionProperties]:
        return MutableTransition[ParticleWithSpin, InteractionProperties](
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


def _sort_tuple(
    iterable: Iterable[StateTransition],
) -> Tuple[StateTransition, ...]:
    if not all(map(lambda t: isinstance(t, StateTransition), iterable)):
        raise TypeError(
            f"Not all instances are of type {StateTransition.__name__}"
        )
    return tuple(sorted(iterable))


@implement_pretty_repr
@frozen
class ReactionInfo:
    """Ordered collection of `StateTransition` instances."""

    transitions: Tuple[StateTransition, ...] = field(converter=_sort_tuple)
    formalism: str = field(validator=instance_of(str))

    initial_state: FrozenDict[int, Particle] = field(
        init=False, repr=False, eq=False
    )
    final_state: FrozenDict[int, Particle] = field(
        init=False, repr=False, eq=False
    )

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

    @staticmethod
    def from_graphs(
        graphs: Iterable[
            MutableTransition[ParticleWithSpin, InteractionProperties]
        ],
        formalism: str,
    ) -> "ReactionInfo":
        transitions = [StateTransition.from_graph(g) for g in graphs]
        return ReactionInfo(transitions, formalism)

    def to_graphs(
        self,
    ) -> List[MutableTransition[ParticleWithSpin, InteractionProperties]]:
        return [transition.to_graph() for transition in self.transitions]

    def group_by_topology(self) -> Dict[Topology, List[StateTransition]]:
        groupings = defaultdict(list)
        for transition in self.transitions:
            groupings[transition.topology].append(transition)
        return dict(groupings)
