# pylint: disable=no-self-use
import pydot
import pytest

import qrules
from qrules import io
from qrules.io._dot import (
    _collapse_graphs,
    _get_particle_graphs,
    _strip_projections,
)
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


def test_asdot_exact_format(reaction: ReactionInfo):
    dot = io.asdot(reaction.transitions[0], render_node=True)
    if reaction.formalism == "helicity":
        expected_dot = """
digraph {
    rankdir=LR;
    node [shape=point, width=0];
    edge [arrowhead=none];
    bgcolor=none;
    "edge0" [shape=none, label="0: gamma[-1]"];
    "edge1" [shape=none, label="1: pi0[0]"];
    "edge2" [shape=none, label="2: pi0[0]"];
    "edge-1" [shape=none, label="J/psi(1S)[-1]"];
    { rank=same "edge-1" };
    { rank=same "edge0", "edge1", "edge2" };
    "edge-1" -> "node0";
    "node0" -> "node1" [label="f(0)(980)[0]"];
    "node0" -> "edge0";
    "node1" -> "edge1";
    "node1" -> "edge2";
    "node0" [shape=none, label="P=+1"];
    "node1" [shape=none, label="P=+1"];
}
        """
    else:
        expected_dot = """
digraph {
    rankdir=LR;
    node [shape=point, width=0];
    edge [arrowhead=none];
    bgcolor=none;
    "edge0" [shape=none, label="0: gamma[-1]"];
    "edge1" [shape=none, label="1: pi0[0]"];
    "edge2" [shape=none, label="2: pi0[0]"];
    "edge-1" [shape=none, label="J/psi(1S)[-1]"];
    { rank=same "edge-1" };
    { rank=same "edge0", "edge1", "edge2" };
    "edge-1" -> "node0";
    "node0" -> "node1" [label="f(0)(980)[0]"];
    "node0" -> "edge0";
    "node1" -> "edge1";
    "node1" -> "edge2";
    "node0" [shape=none, label="l=(0, 0)\ns=(1, -1)\nP=+1"];
    "node1" [shape=none, label="l=(0, 0)\ns=(0, 0)\nP=+1"];
}
        """
    assert dot.strip() == expected_dot.strip()


def test_asdot_graphviz_attrs(reaction: ReactionInfo):
    dot_data = io.asdot(reaction, size=12)
    assert pydot.graph_from_dot_data(dot_data) is not None
    dot_data = io.asdot(reaction, bgcolor="red", size=12)
    assert pydot.graph_from_dot_data(dot_data) is not None
    assert '\n    bgcolor="red";\n' in dot_data
    assert "\n    size=12;\n" in dot_data
    assert "bgcolor=none" not in dot_data


def test_asdot_with_styled_edges_and_nodes(reaction: ReactionInfo, output_dir):
    transition = reaction.transitions[0]
    dot = io.asdot(
        transition,
        edge_style={"fontcolor": "blue"},
        node_style={"fontcolor": "darkgreen", "shape": "ellipse"},
    )
    assert pydot.graph_from_dot_data(dot) is not None
    with open(output_dir + f"styled_{reaction.formalism}.gv", "w") as stream:
        stream.write(dot)
    assert (
        '"edge0" [shape=none, fontcolor="blue", label="0: gamma[-1]"];' in dot
    )
    assert (
        '"node0" -> "node1" [fontcolor="blue", label="f(0)(980)[0]"];' in dot
    )
    assert '"node0" [shape="ellipse", fontcolor="darkgreen", label=""];' in dot


def test_asdot_no_label_overwriting(reaction: ReactionInfo):
    transition = reaction.transitions[0]
    label = "should be ignored"
    dot_data = io.asdot(
        transition,
        edge_style={"label": label},
        node_style={"label": label},
    )
    assert pydot.graph_from_dot_data(dot_data) is not None
    assert label not in dot_data


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
        with open(output_file) as stream:
            dot_data = stream.read()
        assert pydot.graph_from_dot_data(dot_data) is not None

    def test_write_single_graph(self, output_dir: str, reaction: ReactionInfo):
        for i, transition in enumerate(reaction.transitions):
            output_file = output_dir + f"test_single_graph_{i}.gv"
            io.write(
                instance=transition,
                filename=output_file,
            )
            with open(output_file) as stream:
                dot_data = stream.read()
            assert pydot.graph_from_dot_data(dot_data) is not None

    def test_write_graph_list(self, output_dir: str, reaction: ReactionInfo):
        for i, grouping in enumerate(reaction.transition_groups):
            output_file = output_dir + f"test_graph_list_{i}.gv"
            io.write(
                instance=grouping,
                filename=output_file,
            )
            with open(output_file) as stream:
                dot_data = stream.read()
            assert pydot.graph_from_dot_data(dot_data) is not None

    def test_write_strip_spin(self, output_dir: str, reaction: ReactionInfo):
        output_file = output_dir + "test_particle_graphs.gv"
        io.write(
            instance=io.asdot(reaction, strip_spin=True),
            filename=output_file,
        )
        with open(output_file) as stream:
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


def test_strip_projections():
    resonance = "Sigma(1670)~-"
    reaction = qrules.generate_transitions(
        initial_state=[("J/psi(1S)", [+1])],
        final_state=["K0", ("Sigma+", [+0.5]), ("p~", [+0.5])],
        allowed_intermediate_particles=[resonance],
        allowed_interaction_types="strong",
    )

    assert len(reaction.transitions) == 5
    transition = reaction.transitions[0]

    assert transition.intermediate_states[3].particle.name == resonance
    assert transition.intermediate_states[3].spin_projection == -0.5
    assert len(transition.interactions) == 2
    assert transition.interactions[0].s_projection == 1
    assert transition.interactions[0].l_projection == 0
    assert transition.interactions[1].s_projection == -0.5
    assert transition.interactions[1].l_projection == 0

    stripped_transition = _strip_projections(transition)  # type: ignore[arg-type]
    assert stripped_transition.get_edge_props(3).name == resonance
    assert stripped_transition.get_node_props(0).s_projection is None
    assert stripped_transition.get_node_props(0).l_projection is None
    assert stripped_transition.get_node_props(1).s_projection is None
    assert stripped_transition.get_node_props(1).l_projection is None
