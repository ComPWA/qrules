"""Find allowed transitions with a pipeline of functions instead of the STM.

This module decomposes the `.StateTransitionManager` into free functions that exchange
explicit data structures, so that intermediate results — most notably the
`.QNProblemSet` collections — can be inspected, modified, and fed back into the
pipeline. The default use-case is covered by two functions:

1. `create_qn_problem_sets`, which turns initial and final state definitions into
   `.QNProblemSet` collections, grouped by interaction strength.
2. `find_solutions`, which solves them and summarizes the solutions as a
   `.ReactionInfo` object.

The remaining functions implement the individual stages of the pipeline, so that each
intermediate result can be worked with directly.

.. seealso:: :doc:`/usage/reaction`
"""

from __future__ import annotations

import logging
import re
import warnings
from collections import defaultdict
from copy import copy, deepcopy
from functools import partial
from multiprocessing import Pool
from typing import TYPE_CHECKING, overload

from attrs import define, field, frozen
from tqdm.auto import tqdm

from qrules._implementers import implement_pretty_repr
from qrules.combinatorics import (
    as_state_definition,
    create_initial_facts,
    match_external_edges,
    permutate_topology_kinematically,
)
from qrules.particle import ParticleCollection, load_pdg
from qrules.quantum_numbers import (
    EdgeQuantumNumbers,
    InteractionProperties,
    NodeQuantumNumbers,
)
from qrules.settings import (
    DEFAULT_INTERACTION_TYPES,
    InteractionType,
    NumberOfThreads,
    create_interaction_settings,
)
from qrules.solving import CSPSolver
from qrules.system_control import (
    GammaCheck,
    InteractionDeterminator,
    LeptonCheck,
    create_edge_properties,
    create_interaction_properties,
    filter_interaction_types,
    find_particle,
    remove_duplicate_solutions,
)
from qrules.topology import (
    MutableTransition,
    create_isobar_topologies,
    create_n_body_topology,
)
from qrules.transition import (
    ExecutionInfo,
    ProblemSet,
    ReactionInfo,
    SolvingMode,
    SpinFormalism,
    State,
    _SolutionContainer,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from qrules.combinatorics import InitialFacts, StateDefinition, StateDefinitionInput
    from qrules.particle import Particle, ParticleWithSpin
    from qrules.quantum_numbers import EdgeQuantumNumber, NodeQuantumNumber
    from qrules.solving import (
        EdgeSettings,
        GraphEdgePropertyMap,
        GraphSettings,
        NodeSettings,
        QNProblemSet,
        QNResult,
    )
    from qrules.topology import Topology

_LOGGER = logging.getLogger(__name__)


@implement_pretty_repr
@frozen
class AllowedIntermediateParticles:
    """Selection of the particles that are allowed as intermediate states.

    Create an instance with `filter_intermediate_particles`. The selection is fed both
    to `create_problem_sets` (to build the quantum number domains of the intermediate
    edges) and to `solve` (to match solutions against the particle database).
    """

    particles: tuple[GraphEdgePropertyMap, ...]
    """Quantum number property maps of the selected particles."""
    names: tuple[str, ...] | None = None
    """Names of the selected particles, or `None` if no filter was applied."""


def filter_intermediate_particles(
    particle_db: ParticleCollection,
    name_patterns: Iterable[str] | str | None = None,
    regex: bool = False,
) -> AllowedIntermediateParticles:
    """Select allowed intermediate particles from a database by name pattern.

    Raises:
        LookupError: If a name pattern does not match any particle in the database.
    """
    if name_patterns is None:
        return AllowedIntermediateParticles(
            particles=tuple(create_edge_properties(x) for x in particle_db),
        )
    if isinstance(name_patterns, str):
        name_patterns = [name_patterns]
    selected_particles = ParticleCollection()
    for pattern in name_patterns:
        matches = _filter_by_name_pattern(particle_db, pattern, regex)
        if len(matches) == 0:
            msg = (
                "Could not find any matches for allowed intermediate particle"
                f' pattern "{pattern}"'
            )
            raise LookupError(msg)
        selected_particles.update(matches)
    return AllowedIntermediateParticles(
        particles=tuple(
            create_edge_properties(x)
            for x in sorted(selected_particles, key=lambda p: p.name)
        ),
        names=tuple(selected_particles.names),
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


@implement_pretty_repr
@define
class InteractionConfig:
    """Configuration of the interaction types over the nodes of a `.Topology`."""

    type_settings: dict[InteractionType, tuple[EdgeSettings, NodeSettings]]
    """Interaction settings, e.g. from `.create_interaction_settings`."""
    allowed_types: list[InteractionType] | dict[int, list[InteractionType]] = field(
        factory=lambda: list(DEFAULT_INTERACTION_TYPES)
    )
    """Allowed `.InteractionType` values, optionally per node ID."""
    determinators: list[InteractionDeterminator] = field(
        factory=lambda: [LeptonCheck(), GammaCheck()]
    )
    """Checks that are executed over selected conservation rules.

    .. seealso:: {ref}`usage/reaction:Select interaction types`
    """

    @overload
    def get_allowed_interaction_types(
        self,
    ) -> list[InteractionType] | dict[int, list[InteractionType]]: ...

    @overload
    def get_allowed_interaction_types(self, node_id: int) -> list[InteractionType]: ...

    def get_allowed_interaction_types(self, node_id=None):  # type: ignore[no-untyped-def]
        if node_id is None:
            return self.allowed_types
        if isinstance(self.allowed_types, list):
            return self.allowed_types
        return self.allowed_types.get(node_id, DEFAULT_INTERACTION_TYPES)

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
            if allowed_types not in self.type_settings:
                _LOGGER.info(self.type_settings.keys())
                msg = f"Interaction {allowed_types} not found in settings"
                raise ValueError(msg)
        allowed_interaction_types = list(allowed_interaction_types)
        if node_id is None:
            self.allowed_types = allowed_interaction_types
        else:
            if not isinstance(self.allowed_types, dict):
                self.allowed_types = {}
            self.allowed_types[node_id] = allowed_interaction_types


def create_graph_settings(  # noqa: C901, PLR0914
    topology: Topology,
    initial_facts: InitialFacts,
    interaction_config: InteractionConfig,
    intermediate_particles: AllowedIntermediateParticles,
) -> list[GraphSettings]:
    """Determine the solving settings for each edge and node of a `.Topology`."""
    weak_edge_settings, _ = interaction_config.type_settings[InteractionType.WEAK]

    def create_intermediate_edge_qn_domains() -> dict:
        if intermediate_particles.names is None:
            return weak_edge_settings.qn_domains

        # if a list of intermediate states is given by user,
        # built a domain based on these states
        intermediate_edge_domains: dict[type[EdgeQuantumNumber], set] = defaultdict(set)
        intermediate_edge_domains[EdgeQuantumNumbers.spin_projection].update(
            weak_edge_settings.qn_domains[EdgeQuantumNumbers.spin_projection]
        )
        for particle_props in intermediate_particles.particles:
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
        for int_det in interaction_config.determinators:
            determined_interactions = int_det.check(in_states, out_states, interactions)
            if interaction_types:
                interaction_types = list(
                    set(determined_interactions) & set(interaction_types)
                )
            else:
                interaction_types = determined_interactions
        allowed_interaction_types = interaction_config.get_allowed_interaction_types(
            node_id
        )
        interaction_types = filter_interaction_types(
            interaction_types, allowed_interaction_types
        )
        _LOGGER.debug(
            "using %s interaction order for node: %s",
            interaction_types,
            node_id,
        )

        temp_graph_settings: list[GraphSettings] = graph_settings
        graph_settings = []
        for temp_setting in temp_graph_settings:
            for int_type in interaction_types:
                updated_setting = deepcopy(temp_setting)
                updated_setting.interactions[node_id] = deepcopy(
                    interaction_config.type_settings[int_type][1]
                )
                graph_settings.append(updated_setting)

    return graph_settings


def create_problem_sets(  # noqa: PLR0917
    initial_state: Sequence[StateDefinitionInput],
    final_state: Sequence[StateDefinitionInput],
    particle_db: ParticleCollection,
    interaction_config: InteractionConfig,
    intermediate_particles: AllowedIntermediateParticles,
    topologies: Iterable[Topology],
    final_state_groupings: list[list[list[str]]] | None = None,
) -> dict[float, list[ProblemSet]]:
    """Create a `.ProblemSet` collection over all topologies, grouped by strength."""
    initial_state = list(map(as_state_definition, initial_state))
    final_state = list(map(as_state_definition, final_state))
    problem_sets = [
        ProblemSet(permutation, initial_facts, settings)
        for topology in topologies
        for permutation in permutate_topology_kinematically(
            topology,
            initial_state,
            final_state,
            final_state_groupings,
        )
        for initial_facts in create_initial_facts(
            permutation, initial_state, final_state, particle_db
        )
        for settings in create_graph_settings(
            permutation, initial_facts, interaction_config, intermediate_particles
        )
    ]
    return _group_by_strength(problem_sets)


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


def _solve_single_problem(
    qn_problem_set: QNProblemSet,
    allowed_intermediate_states: Iterable[GraphEdgePropertyMap],
) -> tuple[QNProblemSet, QNResult]:
    solver = CSPSolver(allowed_intermediate_states)
    solutions = solver.find_solutions(qn_problem_set)
    return qn_problem_set, solutions


def solve(
    qn_problem_sets: dict[float, list[QNProblemSet]],
    intermediate_particles: AllowedIntermediateParticles,
    solving_mode: SolvingMode = SolvingMode.FULL,
    number_of_threads: int | None = None,
) -> dict[float, list[tuple[QNProblemSet, QNResult]]]:
    """Find allowed transitions purely in terms of quantum number sets.

    Each `.QNProblemSet` is solved in decreasing order of interaction strength. With
    `.SolvingMode.FAST`, solving stops after the strongest strength group that yields
    solutions.
    """
    if number_of_threads is None:
        number_of_threads = NumberOfThreads.get()
    qn_results: dict[float, list[tuple[QNProblemSet, QNResult]]] = defaultdict(list)
    _LOGGER.info(
        "Number of interaction settings groups being processed: %d",
        len(qn_problem_sets),
    )
    total = sum(map(len, qn_problem_sets.values()))
    progress_bar = tqdm(
        total=total,
        desc="Propagating quantum numbers",
        disable=_LOGGER.level > logging.WARNING,
    )
    solve_single = partial(
        _solve_single_problem,
        allowed_intermediate_states=intermediate_particles.particles,
    )
    for strength, qn_problems in sorted(qn_problem_sets.items(), reverse=True):
        _LOGGER.info(
            f"processing interaction settings group with strength {strength}",
        )
        _LOGGER.info(f"{len(qn_problems)} entries in this group")
        _LOGGER.info(f"running with {number_of_threads} threads...")

        # Because of pickling problems of Generic classes (in this case
        # MutableTransition), multithreaded code has to work with
        # QNProblemSet's and QNResult's.
        if number_of_threads > 1:
            with Pool(number_of_threads) as pool:
                for qn_solution in pool.imap_unordered(
                    solve_single, qn_problems, chunksize=1
                ):
                    qn_results[strength].append(qn_solution)
                    progress_bar.update()
        else:
            for problem in qn_problems:
                qn_solution = solve_single(problem)
                qn_results[strength].append(qn_solution)
                progress_bar.update()
        if qn_results[strength] and solving_mode == SolvingMode.FAST:
            break
    progress_bar.close()
    return qn_results


def convert_to_particle_transitions(
    qn_results: dict[float, list[tuple[QNProblemSet, QNResult]]],
    particle_db: ParticleCollection,
) -> dict[float, _SolutionContainer]:
    """Match pure quantum number solutions to particles from a database."""
    results: dict[float, _SolutionContainer] = defaultdict(_SolutionContainer)
    for strength, qn_solutions in qn_results.items():
        for qn_problem_set, qn_result in qn_solutions:
            particle_result = _convert_to_particle_definitions(
                qn_problem_set.topology,
                qn_result,
                particle_db,
            )
            results[strength].extend(
                particle_result,
                intersect_violations=True,
            )
    return dict(results)


def _convert_to_particle_definitions(
    topology: Topology, qn_result: QNResult, particle_db: ParticleCollection
) -> _SolutionContainer:
    """Convert a `.QNResult` with a `.Topology` into a `._SolutionContainer`.

    The ParticleCollection is used to retrieve a particle instance reference to lower
    the memory footprint.
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
                i: find_particle(x, particle_db)  # type: ignore[misc]
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


def collect_reaction_info(  # noqa: C901, PLR0912
    results: dict[float, _SolutionContainer],
    final_state: Sequence[StateDefinitionInput] | None = None,
    formalism: SpinFormalism = "helicity",
    filter_remove_qns: set[type[NodeQuantumNumber]] | None = None,
    filter_ignore_qns: set[type[NodeQuantumNumber]] | None = None,
) -> ReactionInfo:
    """Summarize particle-level solutions as a `.ReactionInfo` object.

    Removes duplicate solutions, warns about violated and non-executed conservation
    rules, and matches the external edge IDs over all solutions. If a
    :code:`final_state` is given, the final state edge IDs are reordered to match it.
    The quantum number filters default to those for the given :code:`formalism`.
    """
    for strength, result in results.items():
        _LOGGER.info(
            f"Number of solutions for strength {strength} after"
            f"QN solving: {len(result.solutions)}",
        )

    final_result = _SolutionContainer()
    for particle_result in results.values():
        final_result.extend(particle_result)

    if filter_remove_qns is None or filter_ignore_qns is None:
        default_remove_qns, default_ignore_qns = _create_qn_filters(formalism)
        if filter_remove_qns is None:
            filter_remove_qns = default_remove_qns
        if filter_ignore_qns is None:
            filter_ignore_qns = default_ignore_qns

    # remove duplicate solutions, which only differ in the interaction qns
    final_solutions = remove_duplicate_solutions(
        final_result.solutions,
        filter_remove_qns,
        filter_ignore_qns,
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
                f"There were violated conservation rules:  {', '.join(violated_rules)}"
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
    if final_state is not None:
        state_definitions = list(map(as_state_definition, final_state))
        final_solutions = [
            _match_final_state_ids(graph, state_definitions)
            for graph in final_solutions
        ]
    transitions = [
        graph.freeze().convert(lambda s: State(*s)) for graph in final_solutions
    ]
    return ReactionInfo(transitions, formalism)


def create_qn_problem_sets(  # noqa: PLR0917
    initial_state: Sequence[StateDefinitionInput],
    final_state: Sequence[StateDefinitionInput],
    particle_db: ParticleCollection | None = None,
    allowed_intermediate_particles: AllowedIntermediateParticles
    | Iterable[str]
    | str
    | None = None,
    interaction_config: InteractionConfig | None = None,
    formalism: SpinFormalism = "helicity",
    topology_building: str = "isobar",
    mass_conservation_factor: float | None = 3.0,
    max_angular_momentum: int = 1,
    max_spin_magnitude: float = 2,
    final_state_groupings: list[list[list[str]]] | None = None,
) -> dict[float, list[QNProblemSet]]:
    """Create a `.QNProblemSet` collection for a reaction, grouped by strength.

    This function covers the default use-case in a single call: it fans the initial
    and final state definitions out into `.QNProblemSet` objects over all topologies,
    kinematic permutations, and allowed interaction types. Solve the returned problem
    sets with `find_solutions`, optionally after inspecting or modifying them (see e.g.
    `.filter_quantum_number_problem_set`).
    """
    _validate_formalism(formalism)
    if particle_db is None:
        particle_db = load_pdg()
    topologies, use_nbody_topology = _create_topologies(
        len(initial_state), len(final_state), topology_building
    )
    # turn off mass conservation, in case more than one initial state
    # particle is present
    if use_nbody_topology and len(initial_state) > 1:
        mass_conservation_factor = None
    if interaction_config is None:
        interaction_config = InteractionConfig(
            type_settings=create_interaction_settings(
                formalism,
                particle_db=particle_db,
                nbody_topology=use_nbody_topology,
                mass_conservation_factor=mass_conservation_factor,
                max_angular_momentum=max_angular_momentum,
                max_spin_magnitude=max_spin_magnitude,
            )
        )
    problem_sets = create_problem_sets(
        initial_state,
        final_state,
        particle_db,
        interaction_config,
        _resolve_intermediate_particles(allowed_intermediate_particles, particle_db),
        topologies,
        final_state_groupings,
    )
    return {
        strength: [problem_set.to_qn_problem_set() for problem_set in problems]
        for strength, problems in problem_sets.items()
    }


def find_solutions(  # noqa: PLR0917
    qn_problem_sets: dict[float, list[QNProblemSet]],
    particle_db: ParticleCollection,
    final_state: Sequence[StateDefinitionInput] | None = None,
    formalism: SpinFormalism = "helicity",
    allowed_intermediate_particles: AllowedIntermediateParticles
    | Iterable[str]
    | str
    | None = None,
    solving_mode: SolvingMode = SolvingMode.FULL,
    number_of_threads: int | None = None,
) -> ReactionInfo:
    """Solve a `.QNProblemSet` collection and summarize it as `.ReactionInfo`.

    This function covers the default use-case in a single call: it chains `solve`,
    `convert_to_particle_transitions`, and `collect_reaction_info`. The
    :code:`particle_db` is required, because a `.QNProblemSet` contains quantum
    numbers only — particles are re-matched by PID via `.find_particle`.
    """
    _validate_formalism(formalism)
    qn_results = solve(
        qn_problem_sets,
        _resolve_intermediate_particles(allowed_intermediate_particles, particle_db),
        solving_mode,
        number_of_threads,
    )
    results = convert_to_particle_transitions(qn_results, particle_db)
    return collect_reaction_info(results, final_state, formalism)


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


def _resolve_intermediate_particles(
    selection: AllowedIntermediateParticles | Iterable[str] | str | None,
    particle_db: ParticleCollection,
) -> AllowedIntermediateParticles:
    if isinstance(selection, AllowedIntermediateParticles):
        return selection
    return filter_intermediate_particles(particle_db, selection)


def _create_topologies(
    number_of_initial_states: int,
    number_of_final_states: int,
    topology_building: str,
) -> tuple[tuple[Topology, ...], bool]:
    topology_building = topology_building.lower()
    if topology_building == "isobar":
        return create_isobar_topologies(number_of_final_states), False
    if "n-body" in topology_building or "nbody" in topology_building:
        return (
            create_n_body_topology(number_of_initial_states, number_of_final_states),
        ), True
    msg = f'Topology building method "{topology_building}" not implemented'
    raise NotImplementedError(msg)


def _create_qn_filters(
    formalism: SpinFormalism,
) -> tuple[set[type[NodeQuantumNumber]], set[type[NodeQuantumNumber]]]:
    filter_remove_qns: set[type[NodeQuantumNumber]] = set()
    filter_ignore_qns: set[type[NodeQuantumNumber]] = set()
    if formalism == "helicity":
        filter_remove_qns = {
            NodeQuantumNumbers.l_magnitude,
            NodeQuantumNumbers.l_projection,
            NodeQuantumNumbers.s_magnitude,
            NodeQuantumNumbers.s_projection,
        }
    if "helicity" in formalism:
        filter_ignore_qns = {NodeQuantumNumbers.parity_prefactor}
    return filter_remove_qns, filter_ignore_qns


def _validate_formalism(formalism: str) -> None:
    spin_formalisms = SpinFormalism.__args__  # type: ignore[attr-defined]
    if formalism not in set(spin_formalisms):
        msg = (
            f'Formalism "{formalism}" not implemented. Use one of'
            f" {', '.join(spin_formalisms)} instead."
        )
        raise NotImplementedError(msg)
