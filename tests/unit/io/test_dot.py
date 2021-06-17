# pylint: disable=no-self-use
from typing import Any, Callable, Optional, Union

import pydot
import pytest

from qrules import Result, io
from qrules.io._dot import _collapse_graphs, _get_particle_graphs
from qrules.particle import ParticleCollection
from qrules.topology import (
    Edge,
    Topology,
    create_isobar_topologies,
    create_n_body_topology,
)
from qrules.transition import ReactionInfo, StateTransitionCollection


@pytest.mark.parametrize(
    "converter",
    [ReactionInfo.from_result, StateTransitionCollection.from_graphs, None],
)
def test_asdot(
    converter: Optional[
        Callable[[Any], Union[ReactionInfo, StateTransitionCollection]]
    ],
    result: Result,
):
    transitions: object = None
    if converter is not None:
        try:
            transitions = converter(result.transitions)
        except AttributeError:
            transitions = converter(result)
    else:
        transitions = result.transitions
    for transition in transitions:
        dot_data = io.asdot(transition)
        assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.asdot(result)
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.asdot(result, strip_spin=True)
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.asdot(result, collapse_graphs=True)
    assert pydot.graph_from_dot_data(dot_data) is not None


def test_asdot_topology():
    dot_data = io.asdot(create_n_body_topology(3, 4))
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.asdot(create_isobar_topologies(2))
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.asdot(create_isobar_topologies(3))
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.asdot(create_isobar_topologies(4))
    assert pydot.graph_from_dot_data(dot_data) is not None


class TestWrite:
    def test_write_topology(self, output_dir):
        output_file = output_dir + "two_body_decay_topology.gv"
        topology = Topology(
            nodes={0},
            edges={0: Edge(0, None), 1: Edge(None, 0), 2: Edge(None, 0)},
        )
        io.write(
            instance=topology,
            filename=output_file,
        )
        with open(output_file, "r") as stream:
            dot_data = stream.read()
        assert pydot.graph_from_dot_data(dot_data) is not None

    def test_write_single_graph(self, output_dir: str, result: Result):
        output_file = output_dir + "test_single_graph.gv"
        io.write(
            instance=result.transitions[0],
            filename=output_file,
        )
        with open(output_file, "r") as stream:
            dot_data = stream.read()
        assert pydot.graph_from_dot_data(dot_data) is not None

    def test_write_graph_list(self, output_dir: str, result: Result):
        output_file = output_dir + "test_graph_list.gv"
        io.write(
            instance=result.transitions,
            filename=output_file,
        )
        with open(output_file, "r") as stream:
            dot_data = stream.read()
        assert pydot.graph_from_dot_data(dot_data) is not None

    def test_write_strip_spin(self, output_dir: str, result: Result):
        output_file = output_dir + "test_particle_graphs.gv"
        io.write(
            instance=io.asdot(result, strip_spin=True),
            filename=output_file,
        )
        with open(output_file, "r") as stream:
            dot_data = stream.read()
        assert pydot.graph_from_dot_data(dot_data) is not None


def test_collapse_graphs(
    result: Result,
    particle_database: ParticleCollection,
):
    pdg = particle_database
    particle_graphs = _get_particle_graphs(result.transitions)
    assert len(particle_graphs) == 2
    collapsed_graphs = _collapse_graphs(result.transitions)
    assert len(collapsed_graphs) == 1
    graph = next(iter(collapsed_graphs))
    edge_id = next(iter(graph.topology.intermediate_edge_ids))
    f_resonances = pdg.filter(lambda p: p.name in ["f(0)(980)", "f(0)(1500)"])
    intermediate_states = graph.get_edge_props(edge_id)
    assert isinstance(intermediate_states, ParticleCollection)
    assert intermediate_states == f_resonances


def test_get_particle_graphs(
    result: Result, particle_database: ParticleCollection
):
    pdg = particle_database
    particle_graphs = _get_particle_graphs(result.transitions)
    assert len(particle_graphs) == 2
    assert particle_graphs[0].get_edge_props(3) == pdg["f(0)(980)"]
    assert particle_graphs[1].get_edge_props(3) == pdg["f(0)(1500)"]
    assert len(particle_graphs[0].topology.edges) == 5
    for edge_id in range(-1, 3):
        assert particle_graphs[0].get_edge_props(edge_id) is particle_graphs[
            1
        ].get_edge_props(edge_id)
