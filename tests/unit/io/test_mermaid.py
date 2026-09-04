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


def describe_MermaidPrinter():
    def it_topology():
        topology = create_n_body_topology(3, 4)
        src = MermaidPrinter()(topology)
        assert src.startswith("flowchart LR\n")
        assert "    n_0" in src
        assert "    n_1" in src
        assert "    n_2" in src
        assert " --- " in src

    def it_generated_topology():
        topologies = create_isobar_topologies(3)
        src = MermaidPrinter()(topologies)
        assert src.startswith("flowchart LR\n")
        assert " --- " in src
        assert "T0_" in src

    def it_latex_default_matches_public_api():
        topology = create_n_body_topology(3, 4)
        assert MermaidPrinter()(topology) == io.asmermaid(topology)

    def it_labels_are_escaped():
        printer = MermaidPrinter(latex=False)
        node_line = printer._create_mermaid_node(
            "A", 'value with "quotes" and\nline break'
        )
        edge_line = printer._create_mermaid_edge(
            "A", "B", 'value with "quotes" and\nline break'
        )
        assert 'value with \\"quotes\\" and<br/>line break' in node_line
        assert 'value with \\"quotes\\" and<br/>line break' in edge_line

    def it_latex_labels_are_wrapped_and_escaped():
        printer = MermaidPrinter(latex=True)
        node_line = printer._create_mermaid_node("A", '\\alpha + "quoted"\n+ \\beta')
        edge_line = printer._create_mermaid_edge("A", "B", R"\gamma")
        ket_edge_line = printer._create_mermaid_edge(
            "A", "B", R"\left|\frac{1}{2},+\frac{1}{2}\right\rangle"
        )
        multiline_node_line = printer._create_mermaid_node(
            "A", R"\begin{gathered} L = 0 \\ S = 1 \end{gathered}"
        )
        intermediate_state_line = printer._create_mermaid_node(
            "A", R"\gamma", shape="rounded"
        )
        interaction_node_line = printer._create_mermaid_node(
            "A", "L = 0", shape="circle"
        )

        assert node_line == R'    A["$$\alpha + \"quoted\" + \beta$$"]'
        assert edge_line == R'    A ---|"$$\gamma$$"| B'
        assert ket_edge_line == (
            R'    A --"$$\left|\frac{1}{2},+\frac{1}{2}\right\rangle$$"--- B'
        )
        assert multiline_node_line == (
            R'    A["$$\begin{gathered} L = 0 \\\ S = 1 \end{gathered}$$"]'
        )
        assert intermediate_state_line == R'    A("$$\gamma$$")'
        assert interaction_node_line == R'    A(("$$L = 0$$"))'

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            (R"\alpha", R"$$\alpha$$"),
            (R"L = 0 \\ S = 1", R"$$L = 0 \\\ S = 1$$"),
            ('value with "quotes"', R"$$value with \"quotes\"$$"),
            ("first\nsecond", "$$first second$$"),
            (R"\$100", R"$$\$100$$"),
            (R"\{x\}", R"$$\{x\}$$"),
            (R"\"o", R"$$\\\"o$$"),
        ],
    )
    def it_latex_label_transport(label: str, expected: str):
        assert MermaidPrinter(latex=True)._escape_label(label) == expected

    def it_latex_label_colors_are_applied():
        printer = MermaidPrinter(
            latex=True,
            figure_style={"fontcolor": "black"},
            edge_style={"color": "red", "fontcolor": "blue"},
            node_style={"fontcolor": "gray"},
        )
        node_line = printer._create_mermaid_node("A", R"\alpha")
        edge_line = printer._create_mermaid_edge("A", "B", R"\gamma")

        assert node_line == R'    A["$$\textcolor{gray}{\alpha}$$"]'
        assert edge_line == R'    A ---|"$$\textcolor{blue}{\gamma}$$"| B'

    @pytest.mark.parametrize("color", ["red", "gray", "#123456"])
    def it_latex_supported_label_colors(color: str):
        printer = MermaidPrinter(latex=True, node_style={"fontcolor": color})
        assert printer._create_mermaid_node("A", R"\alpha") == (
            Rf'    A["$$\textcolor{{{color}}}{{\alpha}}$$"]'
        )

    @pytest.mark.parametrize(
        ("font_size", "expected"),
        [
            (25, "font-size:25px"),
            (12.5, "font-size:12.5px"),
            ("12pt", "font-size:12pt"),
            (None, ""),
        ],
    )
    def it_font_size_formatting(font_size: object, expected: str):
        printer = MermaidPrinter()
        assert printer._format_style_dict({"fontsize": font_size}) == expected

    def it_edge_labels_with_state_brackets_are_quoted():
        edge_line = MermaidPrinter(latex=False)._create_mermaid_edge(
            "A", "B", "f(2)(2340)[-2]"
        )
        assert edge_line == '    A ---|"f(2)(2340)[-2]"| B'

    def it_edge_labels_with_ket_vectors_are_quoted():
        edge_line = MermaidPrinter(latex=False)._create_mermaid_edge("A", "B", "|1,-1⟩")
        assert edge_line == '    A --"|1,-1⟩"--- B'


def describe_asmermaid():
    def it_exposes_the_public_api():
        topology = create_n_body_topology(3, 4)
        src = io.asmermaid(topology)
        assert src.startswith("flowchart LR\n")
        assert "    n_0" in src
        assert " --- " in src
        assert "$$" in src

    def it_markdown_fence():
        topology = create_n_body_topology(3, 4)
        src = io.asmermaid(topology, markdown=True)
        assert src.startswith("```mermaid\nflowchart LR\n")
        assert src.endswith("\n```\n")

    def it_latex_markdown(reaction: ReactionInfo):
        src = io.asmermaid(reaction.transitions[0], latex=True, markdown=True)
        assert src.startswith("```mermaid\nflowchart LR\n")
        assert R"$$J/\psi(1S)" in src
        assert src.endswith("\n```\n")

    def it_latex_reaction(reaction: ReactionInfo):
        src = io.asmermaid(
            reaction.transitions[0],
            render_node=True,
            render_resonance_id=True,
            latex=True,
        )
        assert src.startswith("flowchart LR\n")
        assert not src.startswith("```mermaid")
        assert R"J/\psi(1S)\left[" in src
        assert R"f_{0}(980)\left[" in src
        assert R"P = \text{+}1" in src
        assert "<br/>" not in src
        if reaction.formalism == "canonical-helicity":
            assert R"$$\begin{gathered} L =" in src

        labeled_lines = [
            line for line in src.splitlines() if '["' in line or '("' in line
        ]
        assert labeled_lines
        assert all(line.count("$$") == 2 for line in labeled_lines)

    def it_latex_collapsed_graph(reaction: ReactionInfo):
        src = io.asmermaid(reaction, collapse_graphs=True, latex=True)
        assert R"$$\begin{gathered} f_{0}(980)" in src
        assert R"\\\ f_{0}(1500) \end{gathered}$$" in src

    def it_latex_strip_spin(reaction: ReactionInfo):
        src = io.asmermaid(reaction, strip_spin=True, latex=True)
        assert R"J/\psi(1S)$$" in src
        assert R"\gamma$$" in src
        assert R"\left[" not in src

    def it_accepts_style_parameters():
        topology = create_n_body_topology(3, 4)
        src = io.asmermaid(
            topology,
            figure_style={"bgcolor": "white"},
            edge_style={"color": "blue", "fontsize": 25},
            node_style={"fill": "green"},
        )
        source_lines = set(src.splitlines())
        assert src.startswith("flowchart LR\n")
        assert "    style n_0 fill:green" in source_lines
        assert "linkStyle 0 stroke:blue,font-size:25px" in src

    def it_reaction(reaction: ReactionInfo):
        for transition in reaction.transitions:
            src = io.asmermaid(transition, latex=False)
            assert src.startswith("flowchart LR\n")
            assert " --- " in src
            initial_state_id = next(iter(transition.topology.incoming_edge_ids))
            initial_node_id = transition.topology.edges[initial_state_id].ending_node_id
            initial_state = transition.states[initial_state_id]
            assert f'N{initial_node_id}["{initial_state.particle.name}' in src
            assert f"    A --- N{initial_node_id}" not in src
        src = io.asmermaid(reaction, latex=False)
        assert src.startswith("flowchart LR\n")
        assert " --- " in src
        src = io.asmermaid(reaction, strip_spin=True, latex=False)
        assert src.startswith("flowchart LR\n")
        assert " --- " in src
        src = io.asmermaid(reaction, collapse_graphs=True, latex=False)
        assert src.startswith("flowchart LR\n")
        assert " --- " in src

    def it_reaction_with_node_labels(reaction: ReactionInfo):
        src = io.asmermaid(reaction.transitions[0], render_node=True, latex=False)
        assert src.startswith("flowchart LR\n")
        assert "gamma[-1]" in src
        assert "f(0)(980)[0]" in src
        assert "P=+1" in src
        assert "    A --- N0" in src

    def it_latex_can_be_disabled(reaction: ReactionInfo):
        src = io.asmermaid(reaction.transitions[0], render_node=True, latex=False)
        assert "$$" not in src
        assert R"\text" not in src
        assert R"\frac" not in src
        assert R"\left" not in src

    def it_keeps_multiple_initial_states_separate():
        topology = create_n_body_topology(2, 2)
        src = io.asmermaid(topology, render_node=False)
        assert "    A --- N0" in src
        assert "    B --- N0" in src

    def it_keeps_blank_initial_state_before_initial_split():
        topology = create_isobar_topologies(5)[0]
        src = io.asmermaid(
            topology,
            render_final_state_id=False,
            render_node=False,
            render_resonance_id=True,
        )
        assert '    A@{ shape: text, label: " " }' in src
        assert "    A --- N0" in src

    def it_edge_id_options():
        topology = create_isobar_topologies(5)[0]
        src = io.asmermaid(
            topology,
            render_final_state_id=False,
            render_resonance_id=True,
            render_node=False,
            latex=False,
        )
        assert src.startswith("flowchart LR\n")
        node_declarations = set(src.splitlines())
        assert any(
            f'    n_{edge_id}("{edge_id}")' in node_declarations
            for edge_id in range(5, 8)
        )
        assert all(
            f'    n_{edge_id}("{edge_id}")' not in node_declarations
            for edge_id in range(3)
        )

    def it_styles_intermediate_states_as_edges():
        topology = create_isobar_topologies(5)[0]
        src = io.asmermaid(
            topology,
            edge_style={"color": "blue", "fontcolor": "red", "fontsize": 25},
            latex=False,
            render_resonance_id=True,
        )
        source_lines = src.splitlines()
        assert "    style n_5 stroke:blue,color:red,font-size:25px" in source_lines
        assert len([line for line in source_lines if "linkStyle" in line]) == len([
            line for line in source_lines if " --- " in line
        ])

    def it_distinguishes_state_and_interaction_nodes():
        topology = create_isobar_topologies(5)[0]
        src = io.asmermaid(
            topology,
            latex=False,
            render_final_state_id=True,
            render_initial_state_id=True,
            render_node=True,
        )
        node_declarations = set(src.splitlines())
        assert '    A["-1"]' in node_declarations
        assert '    n_0["0"]' in node_declarations
        assert '    N0(("(0)"))' in node_declarations
        assert not any("stroke:transparent" in line for line in node_declarations)

    def it_keeps_folded_initial_state_box(reaction: ReactionInfo):
        transition = reaction.transitions[0]
        src = io.asmermaid(transition, latex=False)
        initial_state_id = next(iter(transition.topology.incoming_edge_ids))
        node_id = transition.topology.edges[initial_state_id].ending_node_id
        assert f'    N{node_id}["' in src

    def it_qn_problem_set(qn_problem_and_result: tuple[QNProblemSet, QNResult]):
        qn_problem_set, _ = qn_problem_and_result
        src = io.asmermaid(qn_problem_set, render_node=True)
        assert src.startswith("flowchart LR\n")
        assert "RULES" in src
        assert "DOMAINS" in src
        interaction_node_lines = [
            line
            for line in src.splitlines()
            if line.lstrip().startswith("N") and "RULES" in line
        ]
        assert interaction_node_lines
        assert all('["' in line for line in interaction_node_lines)

    def it_latex_qn_problem_set(
        qn_problem_and_result: tuple[QNProblemSet, QNResult],
    ):
        qn_problem_set, _ = qn_problem_and_result
        src = io.asmermaid(qn_problem_set, render_node=True, latex=True)
        assert R"$$\begin{gathered} \text{RULES}" in src
        assert R"\text{DOMAINS}" in src
        assert R"\text{spin\_magnitude} \in" in src
        assert "<br/>" not in src

    def it_qn_result(qn_problem_and_result: tuple[QNProblemSet, QNResult]):
        _, qn_result = qn_problem_and_result
        src = io.asmermaid(qn_result, render_node=True, latex=False)
        assert src.startswith("flowchart LR\n")
        assert " --- " in src
        assert "parity_prefactor =" in src

        src = io.asmermaid(qn_result, render_node=True, latex=True)
        assert R"$$\begin{gathered}" in src
        assert R"\text{parity\_prefactor} = \text{+}1" in src

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


def test_write_mermaid_file(tmp_path):
    topology = create_n_body_topology(3, 4)
    source_file = tmp_path / "topology.mmd"
    io.write(topology, source_file)
    source = source_file.read_text()
    assert source.startswith("flowchart LR\n")
    assert not source.startswith("```mermaid")


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
