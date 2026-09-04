import pydot
import pytest

import qrules
from qrules import io
from qrules.solving import QNProblemSet, QNResult
from qrules.topology import (
    Edge,
    Topology,
    create_isobar_topologies,
    create_n_body_topology,
)
from qrules.transition import ReactionInfo, SpinFormalism, StateTransitionManager


def describe_asdot():
    def it_serializes_a_reaction(reaction: ReactionInfo):
        for transition in reaction.transitions:
            src = io.asdot(transition)
            assert is_valid_dot(src)
        src = io.asdot(reaction)
        assert is_valid_dot(src)
        src = io.asdot(reaction, strip_spin=True)
        assert is_valid_dot(src)
        src = io.asdot(reaction, collapse_graphs=True)
        assert is_valid_dot(src)

    def it_exact_format(reaction: ReactionInfo):
        src = io.asdot(reaction.transitions[0], render_node=True)
        if reaction.formalism == "helicity":
            expected_dot = """
    digraph {
        rankdir=LR
        node [shape=none, width=0]
        edge [arrowhead=none]
        bgcolor=none
        0 [label="0: gamma[-1]"]
        1 [label="1: pi0[0]"]
        2 [label="2: pi0[0]"]
        A [label="J/psi(1S)[-1]"]
        { rank=same; A }
        { rank=same; 0 1 2 }
        A -> N0
        N0 -> N1 [label="f(0)(980)[0]"]
        N0 -> 0
        N1 -> 1
        N1 -> 2
        N0 [label="P=+1"]
        N1 [label="P=+1"]
    }
            """.replace("\n    ", "\n")
        else:
            expected_dot = """
    digraph {
        rankdir=LR
        node [shape=none, width=0]
        edge [arrowhead=none]
        bgcolor=none
        0 [label="0: gamma[-1]"]
        1 [label="1: pi0[0]"]
        2 [label="2: pi0[0]"]
        A [label="J/psi(1S)[-1]"]
        { rank=same; A }
        { rank=same; 0 1 2 }
        A -> N0
        N0 -> N1 [label="f(0)(980)[0]"]
        N0 -> 0
        N1 -> 1
        N1 -> 2
        N0 [label="L=|0,0⟩\nS=|1,-1⟩\nP=+1"]
        N1 [label="L=|0,0⟩\nS=|0,0⟩\nP=+1"]
    }
            """.replace("\n    ", "\n")
        assert src.strip() == expected_dot.strip()

    def it_graphviz_attrs(reaction: ReactionInfo):
        src = io.asdot(reaction, size=12)
        assert is_valid_dot(src)
        src = io.asdot(reaction, bgcolor="red", size=12)
        assert is_valid_dot(src)
        assert '\n    bgcolor="red"\n' in src
        assert "\n    size=12\n" in src
        assert "bgcolor=none" not in src

    def it_qn_problem_set(qn_problem_and_result: tuple[QNProblemSet, QNResult]):
        qn_problem_set, _ = qn_problem_and_result
        src = qrules.io.asdot(qn_problem_set, render_node=True)
        assert is_valid_dot(src)

    def it_with_styled_edges_and_nodes(reaction: ReactionInfo, output_dir):
        transition = reaction.transitions[0]
        src = io.asdot(
            transition,
            edge_style={"fontcolor": "blue"},
            node_style={"fontcolor": "darkgreen", "shape": "ellipse"},
        )
        assert is_valid_dot(src)
        with open(output_dir + f"styled_{reaction.formalism}.gv", "w") as stream:
            stream.write(src)
        assert '0 [fontcolor="blue", label="0: gamma[-1]"]' in src
        assert 'N0 -> N1 [fontcolor="blue", label="f(0)(980)[0]"]' in src
        assert 'N0 [fontcolor="darkgreen", shape="ellipse", label=""]' in src

    def it_no_label_overwriting(reaction: ReactionInfo):
        transition = reaction.transitions[0]
        label = "should be ignored"
        src = io.asdot(
            transition,
            edge_style={"label": label},
            node_style={"label": label},
        )
        assert is_valid_dot(src)
        assert label not in src

    @pytest.mark.parametrize(
        "formalism",
        ["canonical", "canonical-helicity", "helicity"],
    )
    def it_problemset(formalism: SpinFormalism):
        stm = StateTransitionManager(
            initial_state=[("J/psi(1S)", [+1])],
            final_state=["gamma", "pi0", "pi0"],
            formalism=formalism,
        )
        problem_sets = stm.create_problem_sets()
        for problem_set_list in problem_sets.values():
            for problem_set in problem_set_list:
                src = io.asdot(problem_set)
                assert is_valid_dot(src)
                topology = problem_set.topology
                initial_facts = problem_set.initial_facts
                settings = problem_set.solving_settings
                src = io.asdot([(topology, initial_facts)])
                assert is_valid_dot(src)
                src = io.asdot([(topology, settings)])
                assert is_valid_dot(src)
            src = io.asdot(problem_set_list)
            assert is_valid_dot(src)

    def it_topology():
        src = io.asdot(create_n_body_topology(3, 4))
        assert is_valid_dot(src)
        src = io.asdot(create_isobar_topologies(2))
        assert is_valid_dot(src)
        src = io.asdot(create_isobar_topologies(3))
        assert is_valid_dot(src)
        src = io.asdot(create_isobar_topologies(4))
        assert is_valid_dot(src)


def describe_write():
    def it_write_topology(output_dir):
        output_file = output_dir + "two_body_decay_topology.gv"
        topology = Topology(
            nodes={0},
            edges={
                -1: Edge(None, 0),
                0: Edge(0, None),
                1: Edge(0, None),
            },
        )
        io.write(
            instance=topology,
            filename=output_file,
        )
        with open(output_file) as stream:
            src = stream.read()
        assert is_valid_dot(src)

    def it_write_single_graph(output_dir: str, reaction: ReactionInfo):
        for i, transition in enumerate(reaction.transitions):
            output_file = output_dir + f"test_single_graph_{i}.gv"
            io.write(
                instance=transition,
                filename=output_file,
            )
            with open(output_file) as stream:
                src = stream.read()
            assert is_valid_dot(src)

    def it_write_graph_list(output_dir: str, reaction: ReactionInfo):
        for i, transition in enumerate(reaction.transitions):
            output_file = output_dir + f"test_graph_list_{i}.gv"
            io.write(transition, filename=output_file)
            with open(output_file) as stream:
                src = stream.read()
            assert is_valid_dot(src)

    def it_write_strip_spin(output_dir: str, reaction: ReactionInfo):
        output_file = output_dir + "test_particle_graphs.gv"
        io.write(
            instance=io.asdot(reaction, strip_spin=True),
            filename=output_file,
        )
        with open(output_file) as stream:
            src = stream.read()
        assert is_valid_dot(src)


def is_valid_dot(src: str) -> bool:
    try:
        graphs = pydot.graph_from_dot_data(src)
        if graphs is None:
            return False
        return len(graphs) > 0
    except pydot.Error:
        return False
