import pytest

from qrules import io
from qrules.io._mermaid import MermaidPrinter
from qrules.solving import QNProblemSet, QNResult
from qrules.topology import create_isobar_topologies, create_n_body_topology
from qrules.transition import (
    ProblemSet,
    ReactionInfo,
    SpinFormalism,
    StateTransitionManager,
)


def test_mermaid_topology():
    topology = create_n_body_topology(3, 4)
    src = MermaidPrinter()(topology)
    assert src.startswith("flowchart LR\n")
    assert "    n_0" in src
    assert "    n_1" in src
    assert "    n_2" in src
    assert " --- " in src


def test_mermaid_generated_topology():
    topologies = create_isobar_topologies(3)
    src = MermaidPrinter()(topologies)
    assert src.startswith("flowchart LR\n")
    assert " --- " in src
    assert "T0_" in src


def test_asmermaid_api():
    topology = create_n_body_topology(3, 4)
    src = io.asmermaid(topology)
    assert src.startswith("flowchart LR\n")
    assert "    n_0" in src
    assert " --- " in src


def test_asmermaid_accepts_style_parameters():
    topology = create_n_body_topology(3, 4)
    src = io.asmermaid(
        topology,
        figure_style={"bgcolor": "white"},
        edge_style={"color": "blue"},
        node_style={"fill": "green"},
    )
    assert src.startswith("flowchart LR\n")
    assert "style n_0 fill:green" in src
    assert "linkStyle 0 stroke:blue" in src


def test_write_mermaid_file(tmp_path):
    topology = create_n_body_topology(3, 4)
    source_file = tmp_path / "topology.mmd"
    io.write(topology, source_file)
    source = source_file.read_text()
    assert source.startswith("flowchart LR\n")
    assert not source.startswith("```mermaid")


def test_asmermaid_reaction(reaction: ReactionInfo):
    for transition in reaction.transitions:
        src = io.asmermaid(transition)
        assert src.startswith("flowchart LR\n")
        assert " --- " in src
        initial_state_id = next(iter(transition.topology.incoming_edge_ids))
        initial_node_id = transition.topology.edges[initial_state_id].ending_node_id
        initial_state = transition.states[initial_state_id]
        assert f'N{initial_node_id}["{initial_state.particle.name}' in src
        assert f"    A --- N{initial_node_id}" not in src
    src = io.asmermaid(reaction)
    assert src.startswith("flowchart LR\n")
    assert " --- " in src
    src = io.asmermaid(reaction, strip_spin=True)
    assert src.startswith("flowchart LR\n")
    assert " --- " in src
    src = io.asmermaid(reaction, collapse_graphs=True)
    assert src.startswith("flowchart LR\n")
    assert " --- " in src


def test_asmermaid_reaction_with_node_labels(reaction: ReactionInfo):
    src = io.asmermaid(reaction.transitions[0], render_node=True)
    assert src.startswith("flowchart LR\n")
    assert "gamma[-1]" in src
    assert "f(0)(980)[0]" in src
    assert "P=+1" in src
    assert "    A --- N0" in src


def test_asmermaid_keeps_multiple_initial_states_separate():
    topology = create_n_body_topology(2, 2)
    src = io.asmermaid(topology, render_node=False)
    assert "    A --- N0" in src
    assert "    B --- N0" in src


def test_asmermaid_edge_id_options():
    topology = create_isobar_topologies(5)[0]
    src = io.asmermaid(
        topology,
        render_final_state_id=False,
        render_resonance_id=True,
        render_node=False,
    )
    assert src.startswith("flowchart LR\n")
    assert any(label in src for label in (" ---|5| ", " ---|6| ", " ---|7| "))
    assert " ---|0| " not in src
    assert " ---|1| " not in src
    assert " ---|2| " not in src


def test_asmermaid_renders_unlabeled_nodes_without_boxes():
    topology = create_isobar_topologies(5)[0]
    src = io.asmermaid(
        topology,
        render_final_state_id=False,
        render_node=False,
    )
    node_declarations = set(src.splitlines())
    assert '    n_0@{ shape: text, label: " " }' in node_declarations
    assert '    N0@{ shape: text, label: " " }' in node_declarations
    assert "    n_0" not in node_declarations
    assert "    N0" not in node_declarations


def test_asmermaid_qn_problem_set(qn_problem_and_result: tuple[QNProblemSet, QNResult]):
    qn_problem_set, _ = qn_problem_and_result
    src = io.asmermaid(qn_problem_set, render_node=True)
    assert src.startswith("flowchart LR\n")
    assert "RULES" in src
    assert "DOMAINS" in src


def test_asmermaid_qn_result(qn_problem_and_result: tuple[QNProblemSet, QNResult]):
    _, qn_result = qn_problem_and_result
    src = io.asmermaid(qn_result, render_node=True)
    assert src.startswith("flowchart LR\n")
    assert " --- " in src
    assert "parity_prefactor =" in src


@pytest.mark.parametrize(
    "formalism",
    ["canonical", "canonical-helicity", "helicity"],
)
def test_asmermaid_problemset(formalism: SpinFormalism):
    stm = StateTransitionManager(
        initial_state=[("J/psi(1S)", [+1])],
        final_state=["gamma", "pi0", "pi0"],
        formalism=formalism,
    )
    problem_sets = stm.create_problem_sets()
    for problem_set_list in problem_sets.values():
        for problem_set in problem_set_list:
            src = io.asmermaid(problem_set)
            assert src.startswith("flowchart LR\n")
            assert " --- " in src

            topology = problem_set.topology
            initial_facts = problem_set.initial_facts
            settings = problem_set.solving_settings

            src = io.asmermaid([(topology, initial_facts)])
            assert src.startswith("flowchart LR\n")
            assert " --- " in src

            src = io.asmermaid([(topology, settings)])
            assert src.startswith("flowchart LR\n")
            assert " --- " in src
        src = io.asmermaid(problem_set_list)
        assert src.startswith("flowchart LR\n")
        assert " --- " in src


def test_mermaid_labels_are_escaped():
    printer = MermaidPrinter()
    node_line = printer._create_mermaid_node("A", 'value with "quotes" and\nline break')
    edge_line = printer._create_mermaid_edge(
        "A", "B", 'value with "quotes" and\nline break'
    )
    assert 'value with \\"quotes\\" and<br/>line break' in node_line
    assert 'value with \\"quotes\\" and<br/>line break' in edge_line


def test_mermaid_edge_labels_with_state_brackets_are_quoted():
    edge_line = MermaidPrinter()._create_mermaid_edge("A", "B", "f(2)(2340)[-2]")
    assert edge_line == '    A ---|"f(2)(2340)[-2]"| B'


def test_mermaid_edge_labels_with_ket_vectors_are_quoted():
    edge_line = MermaidPrinter()._create_mermaid_edge("A", "B", "|1,-1⟩")
    assert edge_line == '    A --"|1,-1⟩"--- B'


@pytest.fixture
def qn_problem_and_result(
    stm: StateTransitionManager,
    problem_sets: dict[float, list[ProblemSet]],
) -> tuple[QNProblemSet, QNResult]:
    qn_solutions = stm.find_quantum_number_transitions(problem_sets)
    strong_qn_solutions = qn_solutions[3600.0]
    for qn_problem, qn_result in strong_qn_solutions:
        if qn_result.solutions:
            return qn_problem, qn_result
    return strong_qn_solutions[0]
