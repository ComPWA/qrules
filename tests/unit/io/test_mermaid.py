import pytest

from qrules import io
from qrules.io._mermaid import MermaidPrinter
from qrules.settings import InteractionType
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
    assert " --> " in src


def test_mermaid_generated_topology():
    topologies = create_isobar_topologies(3)
    src = MermaidPrinter()(topologies)

    assert src.startswith("flowchart LR\n")
    assert " --> " in src
    assert "T0_" in src


def test_asmermaid_api():
    topology = create_n_body_topology(3, 4)
    src = io.asmermaid_source(topology)

    assert src.startswith("flowchart LR\n")
    assert "    n_0" in src
    assert " --> " in src


def test_asmermaid_accepts_style_parameters():
    topology = create_n_body_topology(3, 4)
    src = io.asmermaid_source(
        topology,
        figure_style={"bgcolor": "white"},
        edge_style={"color": "blue"},
        node_style={"fill": "green"},
    )

    assert src.startswith("flowchart LR\n")
    assert "style n_0 fill:green" in src
    assert "linkStyle 0 stroke:blue" in src


def test_asmermaid_markdown_api():
    topology = create_n_body_topology(3, 4)
    src = io.asmermaid(topology)

    assert src.startswith("```mermaid\nflowchart LR\n")
    assert src.rstrip().endswith("```")


def test_show_mermaid_markdown(monkeypatch):
    import IPython.display as ipdisplay

    displayed = []

    def fake_markdown(source):
        return {"markdown": source}

    def fake_display(obj):
        displayed.append(obj)

    monkeypatch.setattr(ipdisplay, "Markdown", fake_markdown)
    monkeypatch.setattr(ipdisplay, "display", fake_display)

    topology = create_n_body_topology(3, 4)
    markdown = io.show_mermaid_markdown(topology)

    assert markdown == displayed[0]
    assert markdown["markdown"].startswith("```mermaid\nflowchart LR\n")
    assert markdown["markdown"].rstrip().endswith("```")


def test_write_mermaid_files(tmp_path):
    topology = create_n_body_topology(3, 4)
    source_file = tmp_path / "topology.mmd"
    markdown_file = tmp_path / "topology.md"

    io.write(topology, source_file)
    io.write(topology, markdown_file)

    source = source_file.read_text()
    markdown = markdown_file.read_text()
    assert source.startswith("flowchart LR\n")
    assert not source.startswith("```mermaid")
    assert markdown.startswith("```mermaid\nflowchart LR\n")
    assert markdown.rstrip().endswith("```")


def test_write_mmd_removes_markdown_fence(tmp_path):
    output_file = tmp_path / "topology.mmd"

    with pytest.warns(UserWarning, match="fence removed"):
        io.write("```mermaid\nflowchart LR\nA --> B\n```", output_file)

    assert output_file.read_text() == "flowchart LR\nA --> B\n"


def test_write_md_adds_missing_markdown_fence(tmp_path):
    output_file = tmp_path / "topology.md"

    with pytest.warns(UserWarning, match="fence added"):
        io.write("flowchart LR\nA --> B\n", output_file)

    assert output_file.read_text() == "```mermaid\nflowchart LR\nA --> B\n```\n"


def test_asmermaid_reaction(reaction: ReactionInfo):
    for transition in reaction.transitions:
        src = io.asmermaid_source(transition)
        assert src.startswith("flowchart LR\n")
        assert " --> " in src
    src = io.asmermaid_source(reaction)
    assert src.startswith("flowchart LR\n")
    assert " --> " in src
    src = io.asmermaid_source(reaction, strip_spin=True)
    assert src.startswith("flowchart LR\n")
    assert " --> " in src
    src = io.asmermaid_source(reaction, collapse_graphs=True)
    assert src.startswith("flowchart LR\n")
    assert " --> " in src


def test_asmermaid_reaction_with_node_labels(reaction: ReactionInfo):
    src = io.asmermaid_source(reaction.transitions[0], render_node=True)

    assert src.startswith("flowchart LR\n")
    assert "gamma[-1]" in src
    assert "f(0)(980)[0]" in src
    assert "P=+1" in src


def test_asmermaid_edge_id_options():
    topology = create_isobar_topologies(5)[0]
    src = io.asmermaid_source(
        topology,
        render_final_state_id=False,
        render_resonance_id=True,
        render_node=False,
    )

    assert src.startswith("flowchart LR\n")
    assert any(label in src for label in (" -->|5| ", " -->|6| ", " -->|7| "))
    assert " -->|0| " not in src
    assert " -->|1| " not in src
    assert " -->|2| " not in src


def test_asmermaid_qn_problem_set(qn_problem_and_result: tuple[QNProblemSet, QNResult]):
    qn_problem_set, _ = qn_problem_and_result
    src = io.asmermaid_source(qn_problem_set, render_node=True)

    assert src.startswith("flowchart LR\n")
    assert "RULES" in src
    assert "DOMAINS" in src


def test_asmermaid_qn_result(qn_problem_and_result: tuple[QNProblemSet, QNResult]):
    _, qn_result = qn_problem_and_result
    src = io.asmermaid_source(qn_result, render_node=True)

    assert src.startswith("flowchart LR\n")
    assert " --> " in src
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
            src = io.asmermaid_source(problem_set)
            assert src.startswith("flowchart LR\n")
            assert " --> " in src

            topology = problem_set.topology
            initial_facts = problem_set.initial_facts
            settings = problem_set.solving_settings

            src = io.asmermaid_source([(topology, initial_facts)])
            assert src.startswith("flowchart LR\n")
            assert " --> " in src

            src = io.asmermaid_source([(topology, settings)])
            assert src.startswith("flowchart LR\n")
            assert " --> " in src
        src = io.asmermaid_source(problem_set_list)
        assert src.startswith("flowchart LR\n")
        assert " --> " in src


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

    assert edge_line == '    A -->|"f(2)(2340)[-2]"| B'


def test_mermaid_edge_labels_with_ket_vectors_are_quoted():
    edge_line = MermaidPrinter()._create_mermaid_edge("A", "B", "|1,-1⟩")

    assert edge_line == '    A --"|1,-1⟩"--> B'


@pytest.fixture
def stm() -> StateTransitionManager:
    stm = StateTransitionManager(
        initial_state=[("J/psi(1S)", [+1])],
        final_state=["K0", ("Sigma+", [+0.5]), ("p~", [+0.5])],
        allowed_intermediate_particles=["Sigma(1750)"],
        formalism="canonical-helicity",
    )
    stm.set_allowed_interaction_types([InteractionType.STRONG, InteractionType.EM])
    return stm


@pytest.fixture
def problem_sets(stm: StateTransitionManager) -> dict[float, list[ProblemSet]]:
    return stm.create_problem_sets()


@pytest.fixture
def qn_problem_and_result(
    stm: StateTransitionManager,
    problem_sets: dict[float, list[ProblemSet]],
) -> tuple[QNProblemSet, QNResult]:
    qn_solutions = stm.find_quantum_number_transitions(problem_sets)
    strong_qn_solutions = qn_solutions[3600.0]
    for pair in strong_qn_solutions:
        if pair[1].solutions:
            return pair
    return strong_qn_solutions[0]


# if __name__ == "__main__":
#     topology_src = MermaidPrinter()(create_n_body_topology(2, 4))
#     generated_topology_src = MermaidPrinter()(create_isobar_topologies(3))
#     printer = MermaidPrinter()
#     node_line = printer._create_mermaid_node("A", 'value with "quotes" and\nline break')
#     edge_line = printer._create_mermaid_edge(
#         "A", "B", 'value with "quotes" and\nline break'
#     )

#     print(topology_src)
#     print("\n---\n")
#     print(generated_topology_src)
#     print("\n---\n")
#     print(node_line)
#     print(edge_line)
