"""Perform permutations on the edges of a `.MutableTransition`.

In a `.MutableTransition`, the edges represent quantum states, while the nodes represent
interactions. This module provides tools to permutate, modify or extract these edge and
node properties.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

from qrules.particle import ParticleWithSpin
from qrules.quantum_numbers import InteractionProperties, arange
from qrules.topology import MutableTransition, Topology, get_originating_node_list

if TYPE_CHECKING:
    from qrules.particle import ParticleCollection


StateWithSpins = Tuple[str, Sequence[float]]
StateDefinition = Union[str, StateWithSpins]
"""Particle name, optionally with a list of spin projections."""
InitialFacts = MutableTransition[ParticleWithSpin, InteractionProperties]
"""A `.Transition` with only initial and final state information."""


class _KinematicRepresentation:  # noqa: PLW1641
    def __init__(
        self,
        final_state: list[list[str]] | list[str] | None = None,
        initial_state: list[list[str]] | list[str] | None = None,
    ) -> None:
        self.__initial_state: list[list[str]] | None = None
        self.__final_state: list[list[str]] | None = None
        if initial_state is not None:
            self.__initial_state = _sort_nested(ensure_nested_list(initial_state))
        if final_state is not None:
            self.__final_state = _sort_nested(ensure_nested_list(final_state))

    @property
    def initial_state(self) -> list[list[str]] | None:
        return self.__initial_state

    @property
    def final_state(self) -> list[list[str]] | None:
        return self.__final_state

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _KinematicRepresentation):
            return (
                self.initial_state == other.initial_state
                and self.final_state == other.final_state
            )
        msg = f"Cannot compare {type(self).__name__} with {type(other).__name__}"
        raise ValueError(msg)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"initial_state={self.initial_state}, "
            f"final_state={self.final_state})"
        )

    def __contains__(self, other: object) -> bool:
        """Check if a `KinematicRepresentation` is contained within another.

        You can also compare with a `list` of `list` instances, such as:

        .. code-block::

            [["gamma", "pi0"], ["gamma", "pi0", "pi0"]]

        This list will be compared **only** with the
        `~KinematicRepresentation.final_state`!
        """

        def is_sublist(
            sub_representation: list[list[str]] | None,
            main_representation: list[list[str]] | None,
        ) -> bool:
            if main_representation is None:
                return sub_representation is None
            if sub_representation is None:
                return True
            return all(group in main_representation for group in sub_representation)

        if isinstance(other, _KinematicRepresentation):
            return is_sublist(other.initial_state, self.initial_state) and is_sublist(
                other.final_state, self.final_state
            )
        if isinstance(other, list):
            for item in other:
                if not isinstance(item, list):
                    msg = "Comparison representation needs to be a list of lists"
                    raise TypeError(msg)
            return is_sublist(other, self.final_state)
        msg = f"Cannot compare {type(self).__name__} with {type(other).__name__}"
        raise ValueError(msg)


def _sort_nested(nested_list: list[list[str]]) -> list[list[str]]:
    return sorted(sorted(sub_list) for sub_list in nested_list)


def ensure_nested_list(
    nested_list: list[str] | list[list[str]],
) -> list[list[str]]:
    if any(not isinstance(item, list) for item in nested_list):
        nested_list = [nested_list]  # type: ignore[assignment]
    if any(not isinstance(i, str) for lst in nested_list for i in lst):
        msg = "Not all grouping items are particle names"
        raise ValueError(msg)
    return nested_list  # type: ignore[return-value]


def _get_kinematic_representation(
    topology: Topology, particle_names: Mapping[int, str]
) -> _KinematicRepresentation:
    r"""Group final or initial states by node, sorted by length of the group.

    The resulting sorted groups can be used to check whether two `.MutableTransition`
    instances are kinematically identical. For instance, the following two graphs:

    .. code-block::

        J/psi -- 0 -- pi0
                  \
                   1 -- gamma
                    \
                     2 -- gamma
                      \
                       pi0

        J/psi -- 0 -- pi0
                  \
                   1 -- gamma
                    \
                     2 -- pi0
                      \
                       gamma

    both result in:

    .. code-block::

        kinematic_representation.final_state == \
            [["gamma", "gamma"], ["gamma", "gamma", "pi0"], \
             ["gamma", "gamma", "pi0", "pi0"]]
        kinematic_representation.initial_state == \
            [["J/psi"], ["J/psi"]]

    and are therefore kinematically identical. The nested lists are sorted (by `list`
    length and element content) for comparisons.
    """

    def get_state_groupings(get_edge: Callable[[int], set[int]]) -> list[list[int]]:
        return [sorted(get_edge(i)) for i in topology.nodes]

    def fill_groupings(
        edge_id_groupings: Iterable[Iterable[int]],
    ) -> list[list[str]]:
        return [
            [particle_names[edge_id] for edge_id in group]
            for group in edge_id_groupings
        ]

    initial_state_edge_groups = fill_groupings(
        get_state_groupings(topology.get_originating_initial_state_edge_ids)
    )
    final_state_edge_groups = fill_groupings(
        get_state_groupings(topology.get_originating_final_state_edge_ids)
    )
    return _KinematicRepresentation(
        initial_state=initial_state_edge_groups,
        final_state=final_state_edge_groups,
    )


def create_initial_facts(
    topology: Topology,
    initial_state: Sequence[StateDefinition],
    final_state: Sequence[StateDefinition],
    particle_db: ParticleCollection,
) -> list[InitialFacts]:
    states = __create_states_with_spin_projections(
        list(topology.incoming_edge_ids) + list(topology.outgoing_edge_ids),
        list(initial_state) + list(final_state),
        particle_db,
    )
    spin_states = __generate_spin_combinations(states, particle_db)
    return [MutableTransition(topology, state) for state in spin_states]  # type: ignore[arg-type]


def __create_states_with_spin_projections(
    edge_ids: Sequence[int],
    state_definitions: Sequence[StateDefinition],
    particle_db: ParticleCollection,
) -> dict[int, StateWithSpins]:
    if len(edge_ids) != len(state_definitions):
        msg = "Number of state definitions is not same as number of edge IDs"
        raise ValueError(msg)
    states = __safe_set_spin_projections(state_definitions, particle_db)
    return dict(zip(edge_ids, states))


def __safe_set_spin_projections(
    state_definitions: Sequence[StateDefinition],
    particle_db: ParticleCollection,
) -> Sequence[StateWithSpins]:
    def fill_spin_projections(state: StateDefinition) -> StateWithSpins:
        if isinstance(state, str):
            particle_name = state
            particle = particle_db[particle_name]
            spin_projections = set(arange(-particle.spin, particle.spin + 1, 1.0))
            if particle.mass == 0.0 and 0.0 in spin_projections:
                spin_projections.remove(0.0)
            return particle_name, sorted(spin_projections)
        return state

    return [fill_spin_projections(state) for state in state_definitions]


def __generate_spin_combinations(
    states_with_spin_projections: dict[int, StateWithSpins],
    particle_db: ParticleCollection,
) -> list[dict[int, ParticleWithSpin]]:
    def populate_edge_with_spin_projections(
        permutation: dict[int, ParticleWithSpin],
        edge_id: int,
        state: StateWithSpins,
    ) -> list[dict[int, ParticleWithSpin]]:
        particle_name, spin_projections = state
        particle = particle_db[particle_name]
        new_permutations = []
        for projection in spin_projections:
            temp_permutation = deepcopy(permutation)
            temp_permutation.update({edge_id: (particle, projection)})
            new_permutations.append(temp_permutation)
        return new_permutations

    initial_facts_permutations: list[dict[int, ParticleWithSpin]] = [{}]
    for edge_id, state in states_with_spin_projections.items():
        temp_permutations = initial_facts_permutations
        initial_facts_permutations = []
        for temp_permutation in temp_permutations:
            initial_facts_permutations.extend(
                populate_edge_with_spin_projections(temp_permutation, edge_id, state)
            )

    return initial_facts_permutations


def permutate_topology_kinematically(
    topology: Topology,
    initial_state: list[StateDefinition],
    final_state: list[StateDefinition],
    final_state_groupings: list[list[list[str]]]
    | list[list[str]]
    | list[str]
    | None = None,
) -> list[Topology]:
    def strip_spin(state: StateDefinition) -> str:
        if isinstance(state, tuple):
            return state[0]
        return state

    edge_ids = sorted(topology.incoming_edge_ids) + sorted(topology.outgoing_edge_ids)
    states = initial_state + final_state
    return _generate_kinematic_permutations(
        topology,
        particle_names={i: strip_spin(s) for i, s in zip(edge_ids, states)},
        allowed_kinematic_groupings=__get_kinematic_groupings(final_state_groupings),
    )


def _generate_kinematic_permutations(
    topology: Topology,
    particle_names: dict[int, str],
    allowed_kinematic_groupings: list[_KinematicRepresentation] | None = None,
) -> list[Topology]:
    def is_allowed_grouping(kinematic_representation: _KinematicRepresentation) -> bool:
        if allowed_kinematic_groupings is None:
            return True
        for allowed_kinematic_grouping in allowed_kinematic_groupings:
            if allowed_kinematic_grouping in kinematic_representation:
                return True
        return False

    permuted_topologies: list[Topology] = []
    kinematic_representations: list[_KinematicRepresentation] = []
    for permutation in _permutate_outer_edges(topology):
        kinematic_representation = _get_kinematic_representation(
            permutation, particle_names
        )
        if kinematic_representation in kinematic_representations:
            continue
        if not is_allowed_grouping(kinematic_representation):
            continue
        kinematic_representations.append(kinematic_representation)
        permuted_topologies.append(permutation)
    return permuted_topologies


def _permutate_outer_edges(topology: Topology) -> list[Topology]:
    initial_state_ids = sorted(topology.incoming_edge_ids)
    final_state_ids = sorted(topology.outgoing_edge_ids)
    topologies = set()
    for initial_state_permutation in itertools.permutations(initial_state_ids):
        for final_state_permutation in itertools.permutations(final_state_ids):
            permutation = zip(
                initial_state_ids + final_state_ids,
                initial_state_permutation + final_state_permutation,
            )
            new_topology = topology.relabel_edges(dict(permutation))
            topologies.add(new_topology)
    return sorted(topologies)


def __get_kinematic_groupings(
    final_state_groupings: list[list[list[str]]] | list[list[str]] | list[str] | None,
) -> list[_KinematicRepresentation] | None:
    if final_state_groupings is None:
        return None

    def embed_in_list(some_list: list[Any]) -> list[list[Any]]:
        if not isinstance(some_list[0], list):
            return [some_list]
        return some_list

    final_state_groupings = embed_in_list(final_state_groupings)
    final_state_groupings = embed_in_list(final_state_groupings)
    return [_KinematicRepresentation(grouping) for grouping in final_state_groupings]


def match_external_edges(
    graphs: list[MutableTransition[ParticleWithSpin, InteractionProperties]],
) -> None:
    if not isinstance(graphs, list):
        msg = "graphs argument is not of type list"
        raise TypeError(msg)
    if not graphs:
        return
    ref_graph_id = 0
    _match_external_edge_ids(graphs, ref_graph_id, __get_final_state_edge_ids)
    _match_external_edge_ids(graphs, ref_graph_id, __get_initial_state_edge_ids)


def _match_external_edge_ids(
    graphs: list[MutableTransition[ParticleWithSpin, InteractionProperties]],
    ref_graph_id: int,
    external_edge_getter_function: Callable[[MutableTransition], Iterable[int]],
) -> None:
    ref_graph = graphs[ref_graph_id]
    # create external edge to particle mapping
    ref_edge_id_particle_mapping = _create_edge_id_particle_mapping(
        ref_graph, external_edge_getter_function(ref_graph)
    )

    for graph in graphs[:ref_graph_id] + graphs[ref_graph_id + 1 :]:
        edge_id_particle_mapping = _create_edge_id_particle_mapping(
            graph, external_edge_getter_function(graph)
        )
        # remove matching entries
        ref_mapping_copy = deepcopy(ref_edge_id_particle_mapping)
        edge_ids_mapping = {}
        for key, value in edge_id_particle_mapping.items():
            if key in ref_mapping_copy and value == ref_mapping_copy[key]:
                del ref_mapping_copy[key]
            else:
                for key_2, value_2 in ref_mapping_copy.items():
                    if value == value_2:
                        edge_ids_mapping[key] = key_2
                        del ref_mapping_copy[key_2]
                        break
        if len(ref_mapping_copy) != 0:
            msg = "Unable to match graphs, due to inherent graph structure mismatch"
            raise ValueError(msg)
        swappings = _calculate_swappings(edge_ids_mapping)
        for edge_id1, edge_id2 in swappings.items():
            graph.swap_edges(edge_id1, edge_id2)


def __get_initial_state_edge_ids(
    graph: MutableTransition[ParticleWithSpin, InteractionProperties],
) -> Iterable[int]:
    return graph.topology.incoming_edge_ids


def __get_final_state_edge_ids(
    graph: MutableTransition[ParticleWithSpin, InteractionProperties],
) -> Iterable[int]:
    return graph.topology.outgoing_edge_ids


def perform_external_edge_identical_particle_combinatorics(
    graph: MutableTransition,
) -> list[MutableTransition]:
    """Create combinatorics clones of the `.MutableTransition`.

    In case of identical particles in the initial or final state. Only identical
    particles, which do not enter or exit the same node allow for combinatorics!
    """
    if not isinstance(graph, MutableTransition):
        msg = f"graph argument is not of type {MutableTransition.__name__}"
        raise TypeError(msg)
    temp_new_graphs = _external_edge_identical_particle_combinatorics(
        graph, __get_final_state_edge_ids
    )
    new_graphs = []
    for new_graph in temp_new_graphs:
        new_graphs.extend(
            _external_edge_identical_particle_combinatorics(
                new_graph, __get_initial_state_edge_ids
            )
        )
    return new_graphs


def _external_edge_identical_particle_combinatorics(
    graph: MutableTransition[ParticleWithSpin, InteractionProperties],
    external_edge_getter_function: Callable[[MutableTransition], Iterable[int]],
) -> list[MutableTransition]:
    new_graphs = [graph]
    edge_particle_mapping = _create_edge_id_particle_mapping(
        graph, external_edge_getter_function(graph)
    )
    identical_particle_groups: dict[str, set[int]] = {}
    for key, value in edge_particle_mapping.items():
        if value not in identical_particle_groups:
            identical_particle_groups[value] = set()
        identical_particle_groups[value].add(key)
    identical_particle_groups = {
        key: value for key, value in identical_particle_groups.items() if len(value) > 1
    }
    # now for each identical particle group perform all permutations
    for edge_group in identical_particle_groups.values():
        combinations = itertools.permutations(edge_group)
        graph_combinations = set()
        ext_edge_combinations = []
        ref_node_origin = get_originating_node_list(graph.topology, edge_group)
        for comb in combinations:
            temp_edge_node_mapping = tuple(sorted(zip(comb, ref_node_origin)))
            if temp_edge_node_mapping not in graph_combinations:
                graph_combinations.add(temp_edge_node_mapping)
                ext_edge_combinations.append(dict(zip(edge_group, comb)))
        temp_new_graphs = []
        for new_graph in new_graphs:
            for combination in ext_edge_combinations:
                graph_copy = deepcopy(new_graph)
                swappings = _calculate_swappings(combination)
                for edge_id1, edge_id2 in swappings.items():
                    graph_copy.swap_edges(edge_id1, edge_id2)
                temp_new_graphs.append(graph_copy)
        new_graphs = temp_new_graphs
    return new_graphs


def _calculate_swappings(id_mapping: dict[int, int]) -> OrderedDict:
    """Calculate edge id swappings.

    Its important to use an ordered dict as the swappings do not commute!
    """
    swappings: OrderedDict = OrderedDict()
    for key, value in id_mapping.items():
        # go through existing swappings and use them
        newkey = key
        while newkey in swappings:
            newkey = swappings[newkey]
        if value != newkey:
            swappings[value] = newkey
    return swappings


def _create_edge_id_particle_mapping(
    graph: MutableTransition[ParticleWithSpin, InteractionProperties],
    edge_ids: Iterable[int],
) -> dict[int, str]:
    return {i: graph.states[i][0].name for i in edge_ids}
