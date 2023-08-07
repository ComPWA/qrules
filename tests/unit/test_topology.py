import typing

import pytest
from attrs.exceptions import FrozenInstanceError
from IPython.lib.pretty import pretty

from qrules.topology import (
    Edge,
    FrozenDict,  # noqa: F401  # pyright: ignore[reportUnusedImport]
    InteractionNode,
    MutableTopology,
    SimpleStateTransitionTopologyBuilder,
    Topology,
    create_isobar_topologies,
    create_n_body_topology,
    get_originating_node_list,
)


class TestEdge:
    def test_get_connected_nodes(self):
        edge = Edge(1, 2)
        assert edge.get_connected_nodes() == {1, 2}
        edge = Edge(originating_node_id=3)
        assert edge.get_connected_nodes() == {3}
        edge = Edge(ending_node_id=4)
        assert edge.get_connected_nodes() == {4}

    @typing.no_type_check
    def test_immutability(self):
        edge = Edge(1, 2)
        with pytest.raises(FrozenInstanceError):
            edge.originating_node_id = None
        with pytest.raises(FrozenInstanceError):
            edge.originating_node_id += 1
        with pytest.raises(FrozenInstanceError):
            edge.ending_node_id = None
        with pytest.raises(FrozenInstanceError):
            edge.ending_node_id += 1


class TestInteractionNode:
    def test_constructor_exceptions(self):
        with pytest.raises(TypeError):
            assert InteractionNode(
                number_of_ingoing_edges="has to be int",  # type: ignore[arg-type]
                number_of_outgoing_edges=2,
            )
        with pytest.raises(TypeError):
            assert InteractionNode(
                number_of_outgoing_edges="has to be int",  # type: ignore[arg-type]
                number_of_ingoing_edges=2,
            )
        with pytest.raises(
            ValueError,
            match=r"Number of outgoing edges has to be larger than 0",
        ):
            assert InteractionNode(
                number_of_outgoing_edges=0,
                number_of_ingoing_edges=1,
            )
        with pytest.raises(
            ValueError,
            match=r"Number of incoming edges has to be larger than 0",
        ):
            assert InteractionNode(
                number_of_outgoing_edges=1,
                number_of_ingoing_edges=0,
            )


class TestMutableTopology:
    def test_add_and_attach(self, two_to_three_decay: Topology):
        topology = MutableTopology(
            edges=two_to_three_decay.edges,
            nodes=two_to_three_decay.nodes,
        )
        topology.add_node(3)
        topology.add_edges([5, 6])
        topology.attach_edges_to_node_outgoing([5, 6], 3)
        with pytest.raises(
            ValueError,
            match=r"Node 3 is not connected to any other node",
        ):
            topology.freeze()
        topology.attach_edges_to_node_ingoing([2], 3)
        assert isinstance(topology.organize_edge_ids().freeze(), Topology)

    def test_add_exceptions(self, two_to_three_decay: Topology):
        topology = MutableTopology(
            edges=two_to_three_decay.edges,
            nodes=two_to_three_decay.nodes,
        )
        with pytest.raises(ValueError, match=r"Node nr. 0 already exists"):
            topology.add_node(0)
        with pytest.raises(ValueError, match=r"Edge nr. 0 already exists"):
            topology.add_edges([0])
        with pytest.raises(
            ValueError, match=r"Edge nr. -2 is already ingoing to node 0"
        ):
            topology.attach_edges_to_node_ingoing([-2], 0)
        with pytest.raises(ValueError, match=r"Edge nr. 7 does not exist"):
            topology.attach_edges_to_node_ingoing([7], 2)
        with pytest.raises(
            ValueError, match=r"Edge nr. 2 is already outgoing from node 2"
        ):
            topology.attach_edges_to_node_outgoing([2], 2)
        with pytest.raises(ValueError, match=r"Edge nr. 5 does not exist"):
            topology.attach_edges_to_node_outgoing([5], 2)

    def test_organize_edge_ids(self):
        topology = MutableTopology(
            nodes={0, 1, 2},
            edges={
                0: Edge(None, 0),
                1: Edge(None, 0),
                2: Edge(1, None),
                3: Edge(2, None),
                4: Edge(2, None),
                5: Edge(0, 1),
                6: Edge(1, 2),
            },
        )
        assert sorted(topology.edges) == [0, 1, 2, 3, 4, 5, 6]
        topology = topology.organize_edge_ids()
        assert sorted(topology.edges) == [-2, -1, 0, 1, 2, 3, 4]


class TestSimpleStateTransitionTopologyBuilder:
    def test_two_body_states(self):
        two_body_decay_node = InteractionNode(1, 2)
        simple_builder = SimpleStateTransitionTopologyBuilder([two_body_decay_node])
        all_graphs = simple_builder.build(1, 3)
        assert len(all_graphs) == 1


class TestTopology:
    @pytest.mark.parametrize(
        ("nodes", "edges"),
        [
            ({1}, {}),
            (
                {0, 1},
                {
                    -1: Edge(None, 0),
                    0: Edge(1, None),
                    1: Edge(1, None),
                    2: Edge(0, 1),
                },
            ),
            (
                {0, 1, 2},
                {
                    -1: Edge(None, 0),
                    0: Edge(1, None),
                    1: Edge(1, None),
                    2: Edge(2, None),
                    3: Edge(2, None),
                    4: Edge(0, 1),
                    5: Edge(0, 2),
                },
            ),
        ],
    )
    def test_constructor(self, nodes, edges):
        topology = Topology(nodes, edges)
        if nodes is None:
            nodes = set()
        if edges is None:
            edges = {}
        assert topology.nodes == nodes
        assert topology.edges == edges

    @pytest.mark.parametrize(
        ("nodes", "edges"),
        [
            ([], {0: Edge()}),
            ([], {0: Edge(None, 1)}),
            ({0}, {0: Edge(1, None)}),
            ([], {0: Edge(1, None)}),
            ({0, 1}, {0: Edge(0, None), 1: Edge(None, 1)}),
        ],
    )
    def test_constructor_exceptions(self, nodes, edges):
        with pytest.raises(
            ValueError,
            match=r"(not connected to any other node|has non-existing node IDs)",
        ):
            assert Topology(nodes, edges)

    @pytest.mark.parametrize("repr_method", [repr, pretty])
    def test_repr_and_eq(self, repr_method, two_to_three_decay: Topology):
        topology = eval(repr_method(two_to_three_decay))
        assert topology == two_to_three_decay
        assert topology != 0.0

    def test_getters(self, two_to_three_decay: Topology):
        topology = two_to_three_decay  # shorter name
        assert topology.incoming_edge_ids == {-2, -1}
        assert topology.outgoing_edge_ids == {0, 1, 2}
        assert topology.intermediate_edge_ids == {3, 4}
        assert get_originating_node_list(topology, edge_ids=[-1]) == []
        assert get_originating_node_list(topology, edge_ids=[1, 2]) == [2, 2]

    @typing.no_type_check
    def test_immutability(self, two_to_three_decay: Topology):
        with pytest.raises(FrozenInstanceError):
            two_to_three_decay.edges = {0: Edge(None, None)}
        with pytest.raises(TypeError):
            two_to_three_decay.edges[0] = Edge(None, None)
        with pytest.raises(FrozenInstanceError):
            two_to_three_decay.edges[0].ending_node_id = None
        with pytest.raises(FrozenInstanceError):
            two_to_three_decay.nodes = {0, 1}
        with pytest.raises(AttributeError):
            two_to_three_decay.nodes.add(2)
        for node in two_to_three_decay.nodes:
            node += 666
        assert two_to_three_decay.nodes == {0, 1, 2}

    def test_relabel_edges(self, two_to_three_decay: Topology):
        edge_ids = set(two_to_three_decay.edges)
        relabeled_topology = two_to_three_decay.relabel_edges({0: 1, 1: 0})
        assert set(relabeled_topology.edges) == edge_ids
        relabeled_topology = relabeled_topology.relabel_edges({3: 4, 4: 3, 1: 2, 2: 1})
        assert set(relabeled_topology.edges) == edge_ids
        relabeled_topology = relabeled_topology.relabel_edges({3: 4})
        assert set(relabeled_topology.edges) == edge_ids

    def test_swap_edges(self, two_to_three_decay: Topology):
        original_topology = two_to_three_decay
        topology = original_topology.swap_edges(-2, -1)
        assert topology == original_topology
        topology = topology.swap_edges(1, 2)
        assert topology == original_topology
        topology = topology.swap_edges(0, 1)
        assert topology != original_topology

    @pytest.mark.parametrize("n_final_states", [2, 3, 4, 5, 6])
    def test_unique_ordering(self, n_final_states):
        topologies = create_isobar_topologies(n_final_states)
        assert sorted(topologies) == list(topologies)


@pytest.mark.parametrize(
    ("n_final", "n_topologies", "exception"),
    [
        (0, None, ValueError),
        (1, None, ValueError),
        (2, 1, None),
        (3, 1, None),
        (4, 2, None),
        (5, 5, None),
        (6, 16, None),
        (7, 61, None),
        (8, 272, None),
    ],
)
def test_create_isobar_topologies(
    n_final: int,
    n_topologies: int,
    exception,
):
    if exception is not None:
        with pytest.raises(exception):
            create_isobar_topologies(n_final)
    else:
        topologies = create_isobar_topologies(n_final)
        assert len(topologies) == n_topologies
        n_expected_nodes = n_final - 1
        n_intermediate_edges = n_final - 2
        for topology in topologies:
            assert len(topology.outgoing_edge_ids) == n_final
            assert len(topology.intermediate_edge_ids) == n_intermediate_edges
            assert len(topology.nodes) == n_expected_nodes


@pytest.mark.parametrize(
    ("n_initial", "n_final", "exception"),
    [
        (1, 0, ValueError),
        (0, 1, ValueError),
        (0, 0, ValueError),
        (1, 1, None),
        (2, 1, None),
        (3, 1, None),
        (1, 2, None),
        (1, 3, None),
        (2, 4, None),
    ],
)
def test_create_n_body_topology(n_initial: int, n_final: int, exception):
    if exception is not None:
        with pytest.raises(exception):
            create_n_body_topology(n_initial, n_final)
    else:
        topology = create_n_body_topology(n_initial, n_final)
        assert len(topology.incoming_edge_ids) == n_initial
        assert len(topology.outgoing_edge_ids) == n_final
        assert len(topology.intermediate_edge_ids) == 0
        assert len(topology.nodes) == 1
