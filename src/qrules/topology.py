"""Functionality for `Topology` and `Transition` instances.

.. rubric:: Main interfaces

- `Topology` and its builder functions :func:`create_isobar_topologies` and
  :func:`create_n_body_topology`.
- `Transition` and its two implementations `MutableTransition` and `FrozenTransition`.

.. autolink-preface::

    from qrules.topology import (
        create_isobar_topologies,
        create_n_body_topology,
    )
"""

from __future__ import annotations

import copy
import itertools
import logging
import sys
from abc import ABC, abstractmethod
from collections import abc
from functools import total_ordering
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    TypeVar,
    ValuesView,
    overload,
)

import attrs
from attrs import define, field, frozen
from attrs.validators import deep_iterable, deep_mapping, instance_of

from qrules._implementers import implement_pretty_repr

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol

if TYPE_CHECKING:
    from IPython.lib.pretty import PrettyPrinter

_LOGGER = logging.getLogger(__name__)


class _Comparable(Protocol):
    @abstractmethod
    def __lt__(self, other: Any) -> bool: ...


KT = TypeVar("KT", bound=_Comparable)
VT = TypeVar("VT")


@total_ordering
class FrozenDict(abc.Hashable, abc.Mapping, Generic[KT, VT]):
    """An **immutable** and **hashable** version of a `dict`.

    `FrozenDict` makes it possible to make classes hashable if they are decorated with
    :func:`attr.frozen` and contain `~typing.Mapping`-like attributes. If these
    attributes were to be implemented with a normal `dict`, the instance is strictly
    speaking still mutable (even if those attributes are a `property`) and the class is
    therefore not safely hashable.

    .. warning:: The keys have to be comparable, that is, they need to have a
        :meth:`~object.__lt__` method.
    """

    def __init__(self, mapping: Mapping | None = None):
        self.__mapping: dict[KT, VT] = {}
        if mapping is not None:
            self.__mapping = dict(mapping)
        self.__hash = hash(None)
        if len(self.__mapping) != 0:
            self.__hash = 0
            for key_value_pair in self.items():
                self.__hash ^= hash(key_value_pair)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__mapping})"

    def _repr_pretty_(self, p: PrettyPrinter, cycle: bool) -> None:
        class_name = type(self).__name__
        if cycle:
            p.text(f"{class_name}(...)")
        else:
            with p.group(indent=2, open=f"{class_name}({{"):
                for key, value in self.items():
                    p.breakable()
                    p.text(f"{key}: ")
                    p.pretty(value)  # type: ignore[attr-defined]
                    p.text(",")
            p.breakable()
            p.text("})")

    def __iter__(self) -> Iterator[KT]:
        return iter(self.__mapping)

    def __len__(self) -> int:
        return len(self.__mapping)

    def __getitem__(self, key: KT) -> VT:
        return self.__mapping[key]

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, abc.Mapping):
            sorted_self = _convert_mapping_to_sorted_tuple(self)
            sorted_other = _convert_mapping_to_sorted_tuple(other)
            return sorted_self > sorted_other

        msg = (
            f"Can only compare {type(self).__name__} with a mapping, not with"
            f" {type(other).__name__}"
        )
        raise NotImplementedError(msg)

    def __hash__(self) -> int:
        return self.__hash

    def keys(self) -> KeysView[KT]:
        return self.__mapping.keys()

    def items(self) -> ItemsView[KT, VT]:
        return self.__mapping.items()

    def values(self) -> ValuesView[VT]:
        return self.__mapping.values()


def _convert_mapping_to_sorted_tuple(
    mapping: Mapping[KT, VT],
) -> tuple[tuple[KT, VT], ...]:
    return tuple((key, mapping[key]) for key in sorted(mapping.keys()))


def _to_optional_int(optional_int: int | None) -> int | None:
    if optional_int is None:
        return None
    return int(optional_int)


@frozen(order=True)
class Edge:
    """Struct-like definition of an edge, used in `Topology.edges`."""

    originating_node_id: int | None = field(default=None, converter=_to_optional_int)
    """Node ID where the `Edge` **starts**.

    An `Edge` is **incoming to** a `Topology` if its `originating_node_id` is `None`
    (see `~Topology.incoming_edge_ids`).
    """
    ending_node_id: int | None = field(default=None, converter=_to_optional_int)
    """Node ID where the `Edge` **ends**.

    An `Edge` is **outgoing from** a `Topology` if its `ending_node_id` is `None` (see
    `~Topology.outgoing_edge_ids`).
    """

    def get_connected_nodes(self) -> set[int]:
        """Get all node IDs to which the `Edge` is connected."""
        connected_nodes = {self.ending_node_id, self.originating_node_id}
        connected_nodes.discard(None)
        return connected_nodes  # type: ignore[return-value]


def _to_topology_nodes(inst: Iterable[int]) -> frozenset[int]:
    return frozenset(inst)


def _to_topology_edges(inst: Mapping[int, Edge]) -> FrozenDict[int, Edge]:
    return FrozenDict(inst)


@implement_pretty_repr
@frozen(order=True)
class Topology:
    """Directed Feynman-like graph without edge or node properties.

    A `Topology` is **directed** in the sense that its edges are ingoing and outgoing to
    specific nodes. This is to mimic Feynman graphs, which have a time axis. Note that a
    `Topology` is not strictly speaking a graph from graph theory, because it allows
    open edges, like a Feynman-diagram.

    The edges and nodes can be provided with properties with a `Transition`, which
    contains a `~Transition.topology`.

    As opposed to a `MutableTopology`, a `Topology` is frozen, hashable, and ordered, so
    that it can be used as a kind of fingerprint for a `Transition`. In addition, the
    IDs of `edges` are guaranteed to be sequential integers and follow a specific
    pattern:

    - `incoming_edge_ids` (`~Transition.initial_states`) are always negative.
    - `outgoing_edge_ids` (`~Transition.final_states`) lie in the range :code:`0...n-1`
      with :code:`n` the number of final states.
    - `intermediate_edge_ids` continue counting from :code:`n`.

    See also :meth:`MutableTopology.organize_edge_ids`.

    Example
    -------
    **Isobar decay** topologies can best be created as follows:

    >>> topologies = create_isobar_topologies(number_of_final_states=3)
    >>> len(topologies)
    1
    >>> topologies[0]
    Topology(nodes=..., edges=...)
    """

    nodes: frozenset[int] = field(
        converter=_to_topology_nodes,
        validator=deep_iterable(member_validator=instance_of(int)),
    )
    """A node is a point where different `edges` connect."""
    edges: FrozenDict[int, Edge] = field(
        converter=_to_topology_edges,
        validator=deep_mapping(
            key_validator=instance_of(int), value_validator=instance_of(Edge)
        ),
    )
    """Mapping of edge IDs to their corresponding `Edge` definition."""

    incoming_edge_ids: frozenset[int] = field(init=False, repr=False)
    """Edge IDs of edges that have no `~Edge.originating_node_id`.

    `Transition.initial_states` provide properties for these edges.
    """
    outgoing_edge_ids: frozenset[int] = field(init=False, repr=False)
    """Edge IDs of edges that have no `~Edge.ending_node_id`.

    `Transition.final_states` provide properties for these edges.
    """
    intermediate_edge_ids: frozenset[int] = field(init=False, repr=False)
    """Edge IDs of edges that connect two `nodes`."""

    def __attrs_post_init__(self) -> None:
        self.__verify()
        incoming = {
            edge_id
            for edge_id, edge in self.edges.items()
            if edge.originating_node_id is None
        }
        outgoing = {
            edge_id
            for edge_id, edge in self.edges.items()
            if edge.ending_node_id is None
        }
        if incoming & outgoing:
            msg = "Topology has both incoming and outgoing edges. This is not allowed."
            raise ValueError(msg)
        intermediate = set(self.edges) - incoming - outgoing
        object.__setattr__(self, "incoming_edge_ids", frozenset(incoming))
        object.__setattr__(self, "outgoing_edge_ids", frozenset(outgoing))
        object.__setattr__(self, "intermediate_edge_ids", frozenset(intermediate))

    def __verify(self) -> None:
        """Verify if there are no dangling edges or nodes."""
        for edge_id, edge in self.edges.items():
            connected_nodes = edge.get_connected_nodes()
            if not connected_nodes:
                msg = f"Edge nr. {edge_id} is not connected to any other node ({edge})"
                raise ValueError(msg)
            if not connected_nodes <= self.nodes:
                msg = (
                    f"{edge} (ID: {edge_id}) has non-existing node IDs.\nAvailable node"
                    f" IDs: {self.nodes}"
                )
                raise ValueError(msg)
        self.__check_isolated_nodes()

    def __check_isolated_nodes(self) -> None:
        if len(self.nodes) < 2:
            return
        for node_id in self.nodes:
            surrounding_nodes = self.__get_surrounding_nodes(node_id)
            if not surrounding_nodes:
                msg = f"Node {node_id} is not connected to any other node"
                raise ValueError(msg)

    def __get_surrounding_nodes(self, node_id: int) -> set[int]:
        surrounding_nodes = set()
        for edge in self.edges.values():
            connected_nodes = edge.get_connected_nodes()
            if node_id in connected_nodes:
                surrounding_nodes |= connected_nodes
        surrounding_nodes.discard(node_id)
        return surrounding_nodes

    def is_isomorphic(self, other: Topology) -> bool:
        """Check if two graphs are isomorphic.

        Returns `True` if the two graphs have a one-to-one mapping of the node IDs and
        edge IDs.

        .. warning:: Not yet implemented.
        """
        raise NotImplementedError

    def get_edge_ids_ingoing_to_node(self, node_id: int) -> set[int]:
        return {
            edge_id
            for edge_id, edge in self.edges.items()
            if edge.ending_node_id == node_id
        }

    def get_edge_ids_outgoing_from_node(self, node_id: int) -> set[int]:
        return {
            edge_id
            for edge_id, edge in self.edges.items()
            if edge.originating_node_id == node_id
        }

    def get_originating_final_state_edge_ids(self, node_id: int) -> set[int]:
        fs_edges = self.outgoing_edge_ids
        edge_ids = set()
        temp_edge_list = self.get_edge_ids_outgoing_from_node(node_id)
        while temp_edge_list:
            new_temp_edge_list = set()
            for edge_id in temp_edge_list:
                if edge_id in fs_edges:
                    edge_ids.add(edge_id)
                else:
                    new_node_id = self.edges[edge_id].ending_node_id
                    if new_node_id is not None:
                        new_temp_edge_list.update(
                            self.get_edge_ids_outgoing_from_node(new_node_id)
                        )
            temp_edge_list = new_temp_edge_list
        return edge_ids

    def get_originating_initial_state_edge_ids(self, node_id: int) -> set[int]:
        is_edges = self.incoming_edge_ids
        edge_ids: set[int] = set()
        temp_edge_list = self.get_edge_ids_ingoing_to_node(node_id)
        while temp_edge_list:
            new_temp_edge_list = set()
            for edge_id in temp_edge_list:
                if edge_id in is_edges:
                    edge_ids.add(edge_id)
                else:
                    new_node_id = self.edges[edge_id].originating_node_id
                    if new_node_id is not None:
                        new_temp_edge_list.update(
                            self.get_edge_ids_ingoing_to_node(new_node_id)
                        )
            temp_edge_list = new_temp_edge_list
        return edge_ids

    def relabel_edges(self, old_to_new: Mapping[int, int]) -> Topology:
        """Create a new `Topology` with new edge IDs.

        This method is particularly useful when creating permutations of a `Topology`,
        e.g.:

        >>> topologies = create_isobar_topologies(3)
        >>> len(topologies)
        1
        >>> topology = topologies[0]
        >>> final_state_ids = topology.outgoing_edge_ids
        >>> permuted_topologies = {
        ...     topology.relabel_edges(dict(zip(final_state_ids, permutation)))
        ...     for permutation in itertools.permutations(final_state_ids)
        ... }
        >>> len(permuted_topologies)
        3
        """
        new_to_old = {j: i for i, j in old_to_new.items()}
        new_edges = {
            old_to_new.get(i, new_to_old.get(i, i)): edge
            for i, edge in self.edges.items()
        }
        return attrs.evolve(self, edges=new_edges)

    def swap_edges(self, edge_id1: int, edge_id2: int) -> Topology:
        return self.relabel_edges({edge_id1: edge_id2, edge_id2: edge_id1})


def get_originating_node_list(topology: Topology, edge_ids: Iterable[int]) -> list[int]:
    """Get list of node ids from which the supplied edges originate from.

    Args:
        topology: The `Topology` on which to perform the search.
        edge_ids ([int]): A list of edge ids for which the origin node is searched for.
    """

    def __get_originating_node(edge_id: int) -> int | None:
        return topology.edges[edge_id].originating_node_id

    return [node_id for node_id in map(__get_originating_node, edge_ids) if node_id]


def _to_mutable_topology_nodes(inst: Iterable[int]) -> set[int]:
    return set(inst)


def _to_mutable_topology_edges(inst: Mapping[int, Edge]) -> dict[int, Edge]:
    return dict(inst)


@define
class MutableTopology:
    """Mutable version of a `Topology`.

    A `MutableTopology` can be used to conveniently build up a `Topology` (see e.g.
    `SimpleStateTransitionTopologyBuilder`). It does not have restrictions on the
    numbering of edge and node IDs.
    """

    nodes: set[int] = field(
        converter=_to_mutable_topology_nodes,
        factory=set,
        on_setattr=deep_iterable(member_validator=instance_of(int)),
    )
    """See `Topology.nodes`."""
    edges: dict[int, Edge] = field(
        converter=_to_mutable_topology_edges,
        factory=dict,
        on_setattr=deep_mapping(
            key_validator=instance_of(int), value_validator=instance_of(Edge)
        ),
    )
    """See `Topology.edges`."""

    def add_node(self, node_id: int) -> None:
        """Adds a node with number :code:`node_id`.

        Raises:
            ValueError: if :code:`node_id` already exists in `nodes`.
        """
        if node_id in self.nodes:
            msg = f"Node nr. {node_id} already exists"
            raise ValueError(msg)
        self.nodes.add(node_id)

    def add_edges(self, edge_ids: Iterable[int]) -> None:
        """Add edges with the ids in the :code:`edge_ids` list.

        Raises:
            ValueError: if :code:`edge_ids` already exist in `edges`.
        """
        for edge_id in edge_ids:
            if edge_id in self.edges:
                msg = f"Edge nr. {edge_id} already exists"
                raise ValueError(msg)
            self.edges[edge_id] = Edge()

    def attach_edges_to_node_ingoing(
        self, ingoing_edge_ids: Iterable[int], node_id: int
    ) -> None:
        """Attach existing edges to nodes.

        So that the are ingoing to these nodes.

        Args:
            ingoing_edge_ids ([int]): list of edge ids, that will be attached
            node_id (int): id of the node to which the edges will be attached

        Raises:
            ValueError: if an edge not doesn't exist.
            ValueError: if an edge ID is already an ingoing node.
        """
        # first check if the ingoing edges are all available
        for edge_id in ingoing_edge_ids:
            if edge_id not in self.edges:
                msg = f"Edge nr. {edge_id} does not exist"
                raise ValueError(msg)
            if self.edges[edge_id].ending_node_id is not None:
                msg = (
                    f"Edge nr. {edge_id} is already ingoing to node"
                    f" {self.edges[edge_id].ending_node_id}"
                )
                raise ValueError(msg)

        # update the newly connected edges
        for edge_id in ingoing_edge_ids:
            edge = self.edges[edge_id]
            self.edges[edge_id] = Edge(
                ending_node_id=node_id,
                originating_node_id=edge.originating_node_id,
            )

    def attach_edges_to_node_outgoing(
        self, outgoing_edge_ids: Iterable[int], node_id: int
    ) -> None:
        # first check if the ingoing edges are all available
        for edge_id in outgoing_edge_ids:
            if edge_id not in self.edges:
                msg = f"Edge nr. {edge_id} does not exist"
                raise ValueError(msg)
            if self.edges[edge_id].originating_node_id is not None:
                msg = (
                    f"Edge nr. {edge_id} is already outgoing from node"
                    f" {self.edges[edge_id].originating_node_id}"
                )
                raise ValueError(msg)

        # update the edges
        for edge_id in outgoing_edge_ids:
            edge = self.edges[edge_id]
            self.edges[edge_id] = Edge(
                ending_node_id=edge.ending_node_id,
                originating_node_id=node_id,
            )

    def organize_edge_ids(self) -> MutableTopology:
        """Organize edge IDS so that they lie in range :code:`[-m, n+i]`.

        Here, :code:`m` is the number of `.incoming_edge_ids`, :code:`n` is the number
        of `.outgoing_edge_ids`, and :code:`i` is the number of
        `.intermediate_edge_ids`.

        In other words, relabel the edges so that:

        - incoming edge IDs lie in the range :code:`[-1, -2, ...]`,
        - outgoing edge IDs lie in the range :code:`[0, 1, ..., n]`,
        - intermediate edge IDs lie in the range :code:`[n+1, n+2, ...]`.
        """
        incoming = {
            i for i, edge in self.edges.items() if edge.originating_node_id is None
        }
        outgoing = {
            edge_id
            for edge_id, edge in self.edges.items()
            if edge.ending_node_id is None
        }
        intermediate = set(self.edges) - incoming - outgoing
        new_to_old_id = enumerate(
            list(incoming) + list(outgoing) + list(intermediate),
            start=-len(incoming),
        )
        old_to_new_id = {j: i for i, j in new_to_old_id}
        new_edges = {old_to_new_id.get(i, i): edge for i, edge in self.edges.items()}
        return attrs.evolve(self, edges=new_edges)

    def freeze(self) -> Topology:
        """Create an immutable `Topology` from this `MutableTopology`.

        You may need to call :meth:`organize_edge_ids` first.
        """
        return Topology(self.nodes, self.edges)


@define
class InteractionNode:
    """Helper class for the `.SimpleStateTransitionTopologyBuilder`."""

    number_of_ingoing_edges: int = field(validator=instance_of(int))
    number_of_outgoing_edges: int = field(validator=instance_of(int))

    def __attrs_post_init__(self) -> None:
        if self.number_of_ingoing_edges < 1:
            msg = "Number of incoming edges has to be larger than 0"
            raise ValueError(msg)
        if self.number_of_outgoing_edges < 1:
            msg = "Number of outgoing edges has to be larger than 0"
            raise ValueError(msg)


class SimpleStateTransitionTopologyBuilder:
    """Simple topology builder.

    Recursively tries to add the interaction nodes to available open end edges/lines in
    all combinations until the number of open end lines matches the final state lines.
    """

    def __init__(self, interaction_node_set: Iterable[InteractionNode]) -> None:
        if not isinstance(interaction_node_set, list):
            msg = "interaction_node_set must be a list"
            raise TypeError(msg)
        self.interaction_node_set: list[InteractionNode] = list(interaction_node_set)

    def build(
        self, number_of_initial_edges: int, number_of_final_edges: int
    ) -> tuple[Topology, ...]:
        number_of_initial_edges = int(number_of_initial_edges)
        number_of_final_edges = int(number_of_final_edges)
        if number_of_initial_edges < 1:
            msg = "number_of_initial_edges has to be larger than 0"
            raise ValueError(msg)
        if number_of_final_edges < 1:
            msg = "number_of_final_edges has to be larger than 0"
            raise ValueError(msg)

        _LOGGER.info("building topology graphs...")
        # result list
        graph_tuple_list: list[tuple[MutableTopology, list[int]]] = []
        # create seed graph
        seed_graph = MutableTopology()
        current_open_end_edges = list(range(number_of_initial_edges))
        seed_graph.add_edges(current_open_end_edges)
        extendable_graph_list = [(seed_graph, current_open_end_edges)]

        while extendable_graph_list:
            active_graph_list = extendable_graph_list
            extendable_graph_list = []
            for active_graph in active_graph_list:
                # check if finished
                if (
                    len(active_graph[1]) == number_of_final_edges
                    and len(active_graph[0].nodes) > 0
                ):
                    graph_tuple_list.append(active_graph)
                    continue

                extendable_graph_list.extend(self._extend_graph(active_graph))

        _LOGGER.info("finished building topology graphs...")
        # strip the current open end edges list from the result graph tuples
        topologies = []
        for graph_tuple in graph_tuple_list:
            topology = graph_tuple[0]
            topology = topology.organize_edge_ids()
            topologies.append(topology.freeze())
        return tuple(topologies)

    def _extend_graph(
        self, pair: tuple[MutableTopology, Sequence[int]]
    ) -> list[tuple[MutableTopology, list[int]]]:
        extended_graph_list: list[tuple[MutableTopology, list[int]]] = []

        topology, current_open_end_edges = pair

        # Try to extend the graph with interaction nodes
        # that have equal or less ingoing lines than active lines
        for interaction_node in self.interaction_node_set:
            if interaction_node.number_of_ingoing_edges <= len(current_open_end_edges):
                # make all combinations
                combis = list(
                    itertools.combinations(
                        current_open_end_edges,
                        interaction_node.number_of_ingoing_edges,
                    )
                )
                # remove all combinations that originate from the same nodes
                for comb1, comb2 in itertools.combinations(combis, 2):
                    if get_originating_node_list(
                        topology,  # type: ignore[arg-type]
                        comb1,
                    ) == get_originating_node_list(
                        topology,  # type: ignore[arg-type]
                        comb2,
                    ):
                        combis.remove(comb2)

                for combi in combis:
                    new_graph = _attach_node_to_edges(pair, interaction_node, combi)
                    extended_graph_list.append(new_graph)

        return extended_graph_list


def create_isobar_topologies(
    number_of_final_states: int,
) -> tuple[Topology, ...]:
    """Builder function to create a set of unique isobar decay topologies.

    Args:
        number_of_final_states: The number of `~Topology.outgoing_edge_ids`
            (`~.Transition.final_states`).

    Returns:
        A sorted `tuple` of non-isomorphic `Topology` instances, all with the same
        number of final states.

    Example:
        >>> topologies = create_isobar_topologies(number_of_final_states=4)
        >>> len(topologies)
        2
        >>> len(topologies[0].outgoing_edge_ids)
        4
        >>> len(set(topologies))  # hashable
        2
        >>> list(topologies) == sorted(topologies)  # ordered
        True
    """
    if number_of_final_states < 2:
        msg = "At least two final states required for an isobar decay"
        raise ValueError(msg)
    builder = SimpleStateTransitionTopologyBuilder([InteractionNode(1, 2)])
    topologies = builder.build(
        number_of_initial_edges=1,
        number_of_final_edges=number_of_final_states,
    )
    return tuple(sorted(topologies))


def create_n_body_topology(
    number_of_initial_states: int, number_of_final_states: int
) -> Topology:
    """Create a `Topology` that connects all edges through a single node.

    These types of ":math:`n`-body topologies" are particularly important for
    :func:`.check_reaction_violations` and :mod:`.conservation_rules`.

    Args:
        number_of_initial_states: The number of `~Topology.incoming_edge_ids`
            (`~.Transition.initial_states`).
        number_of_final_states: The number of `~Topology.outgoing_edge_ids`
            (`~.Transition.final_states`).

    Example:
        >>> topology = create_n_body_topology(
        ...     number_of_initial_states=2,
        ...     number_of_final_states=5,
        ... )
        >>> topology
        Topology(nodes=..., edges...)
        >>> len(topology.nodes)
        1
        >>> len(topology.incoming_edge_ids)
        2
        >>> len(topology.outgoing_edge_ids)
        5
    """
    n_in = number_of_initial_states
    n_out = number_of_final_states
    builder = SimpleStateTransitionTopologyBuilder([
        InteractionNode(
            number_of_ingoing_edges=n_in,
            number_of_outgoing_edges=n_out,
        )
    ])
    topologies = builder.build(
        number_of_initial_edges=n_in,
        number_of_final_edges=n_out,
    )
    decay_name = f"{n_in} to {n_out}"
    if len(topologies) == 0:
        msg = f"Could not create n-body decay for {decay_name}"
        raise ValueError(msg)
    if len(topologies) > 1:
        msg = f"Several n-body decays for {decay_name}"
        raise RuntimeError(msg)
    return next(iter(topologies))


def _attach_node_to_edges(
    graph: tuple[MutableTopology, Sequence[int]],
    interaction_node: InteractionNode,
    ingoing_edge_ids: Iterable[int],
) -> tuple[MutableTopology, list[int]]:
    temp_graph = copy.deepcopy(graph[0])
    new_open_end_lines = list(copy.deepcopy(graph[1]))

    # add node
    new_node_id = len(temp_graph.nodes)
    temp_graph.add_node(new_node_id)

    # attach the edges to the node
    temp_graph.attach_edges_to_node_ingoing(ingoing_edge_ids, new_node_id)
    # update the newly connected edges
    for edge_id in ingoing_edge_ids:
        new_open_end_lines.remove(edge_id)

    # make new edges for the outgoing lines
    new_edge_start_id = len(temp_graph.edges)
    new_edge_ids = list(
        range(
            new_edge_start_id,
            new_edge_start_id + interaction_node.number_of_outgoing_edges,
        )
    )
    temp_graph.add_edges(new_edge_ids)
    temp_graph.attach_edges_to_node_outgoing(new_edge_ids, new_node_id)
    for edge_id in new_edge_ids:
        new_open_end_lines.append(edge_id)

    return (temp_graph, new_open_end_lines)


EdgeType = TypeVar("EdgeType")
NodeType = TypeVar("NodeType")
NewEdgeType = TypeVar("NewEdgeType")
NewNodeType = TypeVar("NewNodeType")


class Transition(ABC, Generic[EdgeType, NodeType]):
    """Mapping of edge and node properties over a `.Topology`.

    This **interface** class describes a transition from an initial state to a final
    state by providing a mapping of properties over the `~Topology.edges` and
    `~Topology.nodes` of its `topology`. Since a `Topology` behaves like a Feynman
    graph, **edges** are considered as "`states`" and **nodes** are considered as
    `interactions` between those states.

    There are two implementation classes:

    - `FrozenTransition`: a complete, hashable and ordered mapping of properties over
      the `~Topology.edges` and `~Topology.nodes` in its `~FrozenTransition.topology`.
    - `MutableTransition`: comparable to `MutableTopology` in that it is used internally
      when finding solutions through the `.StateTransitionManager` etc.

    These classes are also provided with **mixin** attributes `initial_states`,
    `final_states`, `intermediate_states`, and :meth:`filter_states`.
    """

    @property
    @abstractmethod
    def topology(self) -> Topology:
        """`Topology` over which `states` and `interactions` are defined."""

    @property
    @abstractmethod
    def states(self) -> Mapping[int, EdgeType]:
        """Mapping of properties over its `topology` `~Topology.edges`."""

    @property
    @abstractmethod
    def interactions(self) -> Mapping[int, NodeType]:
        """Mapping of properties over its `topology` `~Topology.nodes`."""

    @property
    def initial_states(self) -> dict[int, EdgeType]:
        """Properties for the `~Topology.incoming_edge_ids`."""
        return self.filter_states(self.topology.incoming_edge_ids)

    @property
    def final_states(self) -> dict[int, EdgeType]:
        """Properties for the `~Topology.outgoing_edge_ids`."""
        return self.filter_states(self.topology.outgoing_edge_ids)

    @property
    def intermediate_states(self) -> dict[int, EdgeType]:
        """Properties for the intermediate edges (connecting two nodes)."""
        return self.filter_states(self.topology.intermediate_edge_ids)

    def filter_states(self, edge_ids: Iterable[int]) -> dict[int, EdgeType]:
        """Filter `states` by a selection of :code:`edge_ids`."""
        return {i: self.states[i] for i in edge_ids}


@implement_pretty_repr
@frozen(order=True)
class FrozenTransition(Transition, Generic[EdgeType, NodeType]):
    """Defines a frozen mapping of edge and node properties on a `Topology`."""

    topology: Topology = field(validator=instance_of(Topology))
    states: FrozenDict[int, EdgeType] = field(converter=FrozenDict)
    interactions: FrozenDict[int, NodeType] = field(converter=FrozenDict)

    def __attrs_post_init__(self) -> None:
        _assert_all_defined(self.topology.nodes, self.interactions)
        _assert_all_defined(self.topology.edges, self.states)

    def unfreeze(self) -> MutableTransition[EdgeType, NodeType]:
        """Convert into a `MutableTransition`."""
        return MutableTransition(self.topology, self.states, self.interactions)

    @overload
    def convert(self) -> FrozenTransition[EdgeType, NodeType]: ...

    @overload
    def convert(
        self, state_converter: Callable[[EdgeType], NewEdgeType]
    ) -> FrozenTransition[NewEdgeType, NodeType]: ...

    @overload
    def convert(
        self, *, interaction_converter: Callable[[NodeType], NewNodeType]
    ) -> FrozenTransition[EdgeType, NewNodeType]: ...

    @overload
    def convert(
        self,
        state_converter: Callable[[EdgeType], NewEdgeType],
        interaction_converter: Callable[[NodeType], NewNodeType],
    ) -> FrozenTransition[NewEdgeType, NewNodeType]: ...

    def convert(self, state_converter=None, interaction_converter=None):  # type: ignore[no-untyped-def]
        """Cast the edge and/or node properties to another type."""
        if state_converter is None:
            state_converter = _identity_function
        if interaction_converter is None:
            interaction_converter = _identity_function
        return FrozenTransition(
            self.topology,
            states={i: state_converter(state) for i, state in self.states.items()},
            interactions={
                i: interaction_converter(interaction)
                for i, interaction in self.interactions.items()
            },
        )


def _identity_function(obj: Any) -> Any:
    return obj


def _cast_states(obj: Mapping[int, EdgeType]) -> dict[int, EdgeType]:
    return dict(obj)


def _cast_interactions(obj: Mapping[int, NodeType]) -> dict[int, NodeType]:
    return dict(obj)


@implement_pretty_repr
@define
class MutableTransition(Transition, Generic[EdgeType, NodeType]):
    """Mutable implementation of a `Transition`.

    Mainly used internally by the `.StateTransitionManager` to build solutions.
    """

    topology: Topology = field(validator=instance_of(Topology))
    states: dict[int, EdgeType] = field(converter=_cast_states, factory=dict)
    interactions: dict[int, NodeType] = field(
        converter=_cast_interactions, factory=dict
    )

    def compare(
        self,
        other: MutableTransition,
        state_comparator: Callable[[EdgeType, EdgeType], bool] | None = None,
        interaction_comparator: Callable[[NodeType, NodeType], bool] | None = None,
    ) -> bool:
        if self.topology != other.topology:
            return False
        if state_comparator is not None:
            for i in self.topology.edges:
                if not state_comparator(self.states[i], other.states[i]):
                    return False
        if interaction_comparator is not None:
            for i in self.topology.nodes:
                if not interaction_comparator(
                    self.interactions[i], other.interactions[i]
                ):
                    return False
        return True

    def swap_edges(self, edge_id1: int, edge_id2: int) -> None:
        self.topology = self.topology.swap_edges(edge_id1, edge_id2)
        value1: EdgeType | None = None
        value2: EdgeType | None = None
        if edge_id1 in self.states:
            value1 = self.states.pop(edge_id1)
        if edge_id2 in self.states:
            value2 = self.states.pop(edge_id2)
        if value1 is not None:
            self.states[edge_id2] = value1
        if value2 is not None:
            self.states[edge_id1] = value2

    def freeze(self) -> FrozenTransition[EdgeType, NodeType]:
        """Convert into a `FrozenTransition`."""
        return FrozenTransition(self.topology, self.states, self.interactions)


def _assert_all_defined(items: Iterable, properties: Iterable) -> None:
    existing = set(items)
    defined = set(properties)
    if existing & defined != existing:
        msg = (
            "Some items have no property assigned to them. Available items:"
            f" {existing}, items with property: {defined}"
        )
        raise ValueError(msg)


# pyright: reportUnusedFunction=false
def _assert_not_overdefined(items: Iterable, properties: Iterable) -> None:
    existing = set(items)
    defined = set(properties)
    over_defined = defined - existing
    if over_defined:
        msg = (
            "Properties have been defined for items that don't exist. Available items:"
            f" {existing}, over-defined: {over_defined}"
        )
        raise ValueError(msg)
