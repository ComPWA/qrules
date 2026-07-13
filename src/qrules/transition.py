"""Find allowed transitions between an initial and final state."""

from __future__ import annotations

import logging
from collections import defaultdict
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal, overload

import attrs
from attrs import define, field, frozen
from attrs.validators import in_, instance_of

from qrules._attrs import to_fraction
from qrules._implementers import implement_pretty_repr
from qrules.combinatorics import (
    InitialFacts,
    StateDefinitionInput,
    as_state_definition,
    ensure_nested_list,
)
from qrules.particle import Particle, ParticleCollection, ParticleWithSpin, load_pdg
from qrules.quantum_numbers import InteractionProperties, NodeQuantumNumber
from qrules.settings import (
    DEFAULT_INTERACTION_TYPES,
    InteractionType,
    NumberOfThreads,
    create_interaction_settings,
)
from qrules.solving import (
    EdgeSettings,
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
    create_node_properties,
)
from qrules.topology import FrozenDict, FrozenTransition, MutableTransition, Topology

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from fractions import Fraction

    from qrules.workflow import InteractionConfig


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


class StateTransitionManager:
    """Main handler for decay topologies.

    .. seealso:: :doc:`/usage/reaction` and `.generate_transitions`
    """

    def __init__(  # noqa: PLR0917
        self,
        initial_state: Sequence[StateDefinitionInput],
        final_state: Sequence[StateDefinitionInput],
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
        max_spin_magnitude: float = 2,
        number_of_threads: int | None = None,
    ) -> None:
        if number_of_threads is not None:
            NumberOfThreads.set(number_of_threads)
        self.__number_of_threads = NumberOfThreads.get()
        from qrules.workflow import (  # noqa: PLC0415
            _create_qn_filters,
            _create_topologies,
            _validate_formalism,
            filter_intermediate_particles,
        )

        if interaction_type_settings is None:
            interaction_type_settings = {}
        _validate_formalism(formalism)
        self.__formalism = formalism
        self.__particles = ParticleCollection()
        if particle_db is not None:
            self.__particles = particle_db
        self.reaction_mode = str(solving_mode)
        self.initial_state = list(map(as_state_definition, initial_state))
        self.final_state = list(map(as_state_definition, final_state))
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
        filter_remove_qns, filter_ignore_qns = _create_qn_filters(formalism)
        self.filter_remove_qns: set[type[NodeQuantumNumber]] = filter_remove_qns
        self.filter_ignore_qns: set[type[NodeQuantumNumber]] = filter_ignore_qns
        topologies, use_nbody_topology = _create_topologies(
            len(initial_state), len(final_state), topology_building
        )
        self.topologies: tuple[Topology, ...] = topologies
        """`.Topology` instances over which the STM propagates quantum numbers."""
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

        if allowed_intermediate_particles is None:
            self.__intermediate_particles = filter_intermediate_particles(
                self.__particles
            )
        else:
            self.set_allowed_intermediate_particles(allowed_intermediate_particles)

    def set_allowed_intermediate_particles(
        self, name_patterns: Iterable[str] | str, regex: bool = False
    ) -> None:
        from qrules.workflow import filter_intermediate_particles  # noqa: PLC0415

        self.__intermediate_particles = filter_intermediate_particles(
            self.__particles, name_patterns, regex
        )

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
        return self.__create_interaction_config().get_allowed_interaction_types(node_id)

    def set_allowed_interaction_types(
        self,
        allowed_interaction_types: Iterable[InteractionType],
        node_id: int | None = None,
    ) -> None:
        config = self.__create_interaction_config()
        config.set_allowed_interaction_types(allowed_interaction_types, node_id)
        self.__allowed_interaction_types = config.allowed_types

    def __create_interaction_config(self) -> InteractionConfig:
        from qrules.workflow import InteractionConfig  # noqa: PLC0415

        return InteractionConfig(
            type_settings=self.interaction_type_settings,
            allowed_types=self.__allowed_interaction_types,
            determinators=self.interaction_determinators,
        )

    def create_problem_sets(self) -> dict[float, list[ProblemSet]]:
        from qrules.workflow import create_problem_sets  # noqa: PLC0415

        return create_problem_sets(
            self.initial_state,
            self.final_state,
            self.__particles,
            interaction_config=self.__create_interaction_config(),
            intermediate_particles=self.__intermediate_particles,
            topologies=self.topologies,
            final_state_groupings=self.final_state_groupings,
        )

    def find_solutions(
        self, problem_sets: dict[float, list[ProblemSet]]
    ) -> ReactionInfo:
        """Check for solutions for a specific set of interaction settings."""
        from qrules.workflow import collect_reaction_info  # noqa: PLC0415

        results = self._find_particle_transitions(problem_sets)
        return collect_reaction_info(
            results,
            final_state=self.final_state,
            formalism=self.formalism,
            filter_remove_qns=self.filter_remove_qns,
            filter_ignore_qns=self.filter_ignore_qns,
        )

    def _find_particle_transitions(
        self, problem_sets: dict[float, list[ProblemSet]]
    ) -> dict[float, _SolutionContainer]:
        from qrules.workflow import convert_to_particle_transitions  # noqa: PLC0415

        qn_results = self.find_quantum_number_transitions(problem_sets)
        return convert_to_particle_transitions(qn_results, self.__particles)

    def find_quantum_number_transitions(
        self, problem_sets: dict[float, list[ProblemSet]]
    ) -> dict[float, list[tuple[QNProblemSet, QNResult]]]:
        """Find allowed transitions purely in terms of quantum number sets."""
        from qrules.workflow import _to_qn_problem_sets, solve  # noqa: PLC0415

        qn_problem_sets = _to_qn_problem_sets(problem_sets)
        return solve(
            qn_problem_sets,
            self.__intermediate_particles,
            # reaction_mode is a str, so the FAST-mode break never triggers here
            # (pre-existing behavior, kept for backwards compatibility)
            solving_mode=self.reaction_mode,  # type: ignore[arg-type]
            number_of_threads=self.__number_of_threads,
        )


@implement_pretty_repr
@frozen(order=True)
class State:
    particle: Particle = field(validator=instance_of(Particle))
    spin_projection: Fraction = field(converter=to_fraction)


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
