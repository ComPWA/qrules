# pylint: disable=no-self-use
import pydot
import pytest

import qrules
from qrules import io
from qrules.io._dot import _collapse_graphs, _get_particle_graphs
from qrules.particle import ParticleCollection
from qrules.topology import (
    Edge,
    Topology,
    create_isobar_topologies,
    create_n_body_topology,
)
from qrules.transition import ReactionInfo


def test_asdot(reaction: ReactionInfo):
    for grouping in reaction.transition_groups:
        for transition in grouping:
            dot_data = io.asdot(transition)
        assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.asdot(reaction)
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.asdot(reaction, strip_spin=True)
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.asdot(reaction, collapse_graphs=True)
    assert pydot.graph_from_dot_data(dot_data) is not None


@pytest.mark.parametrize(
    "formalism",
    ["canonical", "canonical-helicity", "helicity"],
)
def test_asdot_problemset(formalism: str):
    stm = qrules.StateTransitionManager(
        initial_state=[("J/psi(1S)", [+1])],
        final_state=["gamma", "pi0", "pi0"],
        formalism=formalism,
    )
    problem_sets = stm.create_problem_sets()
    for problem_set_list in problem_sets.values():
        for problem_set in problem_set_list:
            dot_data = io.asdot(problem_set)
            assert pydot.graph_from_dot_data(dot_data) is not None
            topology = problem_set.topology
            initial_facts = problem_set.initial_facts
            settings = problem_set.solving_settings
            dot_data = io.asdot([(topology, initial_facts)])
            assert pydot.graph_from_dot_data(dot_data) is not None
            dot_data = io.asdot([(topology, settings)])
            assert pydot.graph_from_dot_data(dot_data) is not None
        dot_data = io.asdot(problem_set_list)
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

    def test_write_single_graph(self, output_dir: str, reaction: ReactionInfo):
        for i, transition in enumerate(reaction.transitions):
            output_file = output_dir + f"test_single_graph_{i}.gv"
            io.write(
                instance=transition,
                filename=output_file,
            )
            with open(output_file, "r") as stream:
                dot_data = stream.read()
            assert pydot.graph_from_dot_data(dot_data) is not None

    def test_write_graph_list(self, output_dir: str, reaction: ReactionInfo):
        for i, grouping in enumerate(reaction.transition_groups):
            output_file = output_dir + f"test_graph_list_{i}.gv"
            io.write(
                instance=grouping,
                filename=output_file,
            )
            with open(output_file, "r") as stream:
                dot_data = stream.read()
            assert pydot.graph_from_dot_data(dot_data) is not None

    def test_write_strip_spin(self, output_dir: str, reaction: ReactionInfo):
        output_file = output_dir + "test_particle_graphs.gv"
        io.write(
            instance=io.asdot(reaction, strip_spin=True),
            filename=output_file,
        )
        with open(output_file, "r") as stream:
            dot_data = stream.read()
        assert pydot.graph_from_dot_data(dot_data) is not None


def test_collapse_graphs(
    reaction: ReactionInfo,
    particle_database: ParticleCollection,
):
    pdg = particle_database
    particle_graphs = _get_particle_graphs(reaction.to_graphs())
    assert len(particle_graphs) == 2
    collapsed_graphs = _collapse_graphs(reaction.to_graphs())
    assert len(collapsed_graphs) == 1
    graph = next(iter(collapsed_graphs))
    edge_id = next(iter(graph.topology.intermediate_edge_ids))
    f_resonances = pdg.filter(lambda p: p.name in ["f(0)(980)", "f(0)(1500)"])
    intermediate_states = graph.get_edge_props(edge_id)
    assert isinstance(intermediate_states, ParticleCollection)
    assert intermediate_states == f_resonances


def test_get_particle_graphs(
    reaction: ReactionInfo, particle_database: ParticleCollection
):
    pdg = particle_database
    particle_graphs = _get_particle_graphs(reaction.to_graphs())
    assert len(particle_graphs) == 2
    assert particle_graphs[0].get_edge_props(3) == pdg["f(0)(980)"]
    assert particle_graphs[1].get_edge_props(3) == pdg["f(0)(1500)"]
    assert len(particle_graphs[0].topology.edges) == 5
    for edge_id in range(-1, 3):
        assert particle_graphs[0].get_edge_props(edge_id) is particle_graphs[
            1
        ].get_edge_props(edge_id)
