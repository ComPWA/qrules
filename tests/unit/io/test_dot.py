from fractions import Fraction
from textwrap import dedent

import attrs
import pydot
import pytest

import qrules
from qrules import io
from qrules.io._dot import (
    _collapse_graphs,
    _get_particle_graphs,
    _strip_projections,
    as_string,
)
from qrules.particle import Particle, ParticleCollection
from qrules.settings import InteractionType
from qrules.solving import QNProblemSet, QNResult
from qrules.topology import (
    Edge,
    Topology,
    create_isobar_topologies,
    create_n_body_topology,
)
from qrules.transition import (
    ProblemSet,
    ReactionInfo,
    SpinFormalism,
    StateTransitionManager,
)


def test_asdot(reaction: ReactionInfo):
    for transition in reaction.transitions:
        src = io.asdot(transition)
        assert is_valid_dot(src)
    src = io.asdot(reaction)
    assert is_valid_dot(src)
    src = io.asdot(reaction, strip_spin=True)
    assert is_valid_dot(src)
    src = io.asdot(reaction, collapse_graphs=True)
    assert is_valid_dot(src)


def test_asdot_exact_format(reaction: ReactionInfo):
    src = io.asdot(reaction.transitions[0], render_node=True)
    if reaction.formalism == "helicity":
        expected_dot = """
digraph {
    rankdir=LR
    node [shape=none, width=0]
    edge [arrowhead=none]
    bgcolor=none
    0 [label="0: gamma"]
    1 [label="1: pi0"]
    2 [label="2: pi0"]
    A [label="J/psi(1S)"]
    { rank=same; A }
    { rank=same; 0 1 2 }
    A -> N0
    N0 -> N1 [label="f(0)(980)"]
    N0 -> 0
    N1 -> 1
    N1 -> 2
    N0 [label=""]
    N1 [label=""]
}
        """
    else:
        expected_dot = """
digraph {
    rankdir=LR
    node [shape=none, width=0]
    edge [arrowhead=none]
    bgcolor=none
    0 [label="0: gamma"]
    1 [label="1: pi0"]
    2 [label="2: pi0"]
    A [label="J/psi(1S)"]
    { rank=same; A }
    { rank=same; 0 1 2 }
    A -> N0
    N0 -> N1 [label="f(0)(980)"]
    N0 -> 0
    N1 -> 1
    N1 -> 2
    N0 [label="L=0\nS=1"]
    N1 [label="L=0\nS=0"]
}
        """
    assert src.strip() == expected_dot.strip()


def test_asdot_graphviz_attrs(reaction: ReactionInfo):
    src = io.asdot(reaction, size=12)
    assert is_valid_dot(src)
    src = io.asdot(reaction, bgcolor="red", size=12)
    assert is_valid_dot(src)
    assert '\n    bgcolor="red"\n' in src
    assert "\n    size=12\n" in src
    assert "bgcolor=none" not in src


def test_asdot_qn_problem_set(qn_problem_and_result: tuple[QNProblemSet, QNResult]):
    qn_problem_set, _ = qn_problem_and_result
    src = qrules.io.asdot(qn_problem_set, render_node=True)
    assert is_valid_dot(src)


def test_asdot_with_styled_edges_and_nodes(reaction: ReactionInfo, output_dir):
    transition = reaction.transitions[0]
    src = io.asdot(
        transition,
        edge_style={"fontcolor": "blue"},
        node_style={"fontcolor": "darkgreen", "shape": "ellipse"},
    )
    assert is_valid_dot(src)
    with open(output_dir + f"styled_{reaction.formalism}.gv", "w") as stream:
        stream.write(src)
    assert '0 [fontcolor="blue", label="0: gamma"]' in src
    assert 'N0 -> N1 [fontcolor="blue", label="f(0)(980)"]' in src
    assert 'N0 [fontcolor="darkgreen", shape="ellipse", label=""]' in src


def test_asdot_no_label_overwriting(reaction: ReactionInfo):
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
def test_asdot_problemset(formalism: SpinFormalism):
    stm = StateTransitionManager(
        initial_state=["J/psi(1S)"],
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


def test_asdot_topology():
    src = io.asdot(create_n_body_topology(3, 4))
    assert is_valid_dot(src)
    src = io.asdot(create_isobar_topologies(2))
    assert is_valid_dot(src)
    src = io.asdot(create_isobar_topologies(3))
    assert is_valid_dot(src)
    src = io.asdot(create_isobar_topologies(4))
    assert is_valid_dot(src)


def test_as_string_dict(
    problem_sets: dict[float, list[ProblemSet]],
    qn_problem_and_result: tuple[QNProblemSet, QNResult],
):
    _, qn_result = qn_problem_and_result
    problem_set = problem_sets[3600.0][0]
    interaction = qn_result.solutions[1].interactions[1]
    intermediate_state, *_ = qn_result.solutions[0].intermediate_states.values()
    node_setting = problem_set.solving_settings.interactions[0]
    intermediate_setting, *_ = problem_set.solving_settings.intermediate_states.values()

    src = as_string(intermediate_setting).strip()
    print()
    print(src)
    expected_dot = dedent("""
        RULES
        isospin_validity - 61
        g_parity_validity - 60
        gellmann_nishijima - 50
        DOMAINS
        baryon_number ∊ [-1, +1]
        bottomness ∊ [0]
        c_parity ∊ [None]
        charge ∊ [-1, 0, +1]
        charmness ∊ [0]
        electron_lepton_number ∊ [0]
        g_parity ∊ [None]
        isospin_magnitude ∊ [1]
        isospin_projection ∊ [-1, 0, +1]
        muon_lepton_number ∊ [0]
        parity ∊ [-1, +1]
        spin_magnitude ∊ [1/2]
        strangeness ∊ [-1, +1]
        tau_lepton_number ∊ [0]
        topness ∊ [0]
    """).strip()
    assert src == expected_dot

    src = as_string(node_setting).strip()
    print()
    print(src)
    expected_dot = dedent("""
        RULES
        BaryonNumberConservation - 90
        spin_magnitude_conservation - 8
        CharmConservation - 70
        StrangenessConservation - 69
        BottomnessConservation - 68
        isospin_conservation - 60
        parity_conservation - 6
        c_parity_conservation - 5
        ElectronLNConservation - 45
        MuonLNConservation - 44
        TauLNConservation - 43
        g_parity_conservation - 3
        ChargeConservation - 100
        MassConservation - 10
        DOMAINS
        l_magnitude ∊ [0, 1]
        s_magnitude ∊ [0, 1/2, 1, 3/2, 2]
    """).strip()
    assert src == expected_dot

    src = as_string(interaction).strip()
    print()
    print(src)
    expected_dot = dedent("""
        l_magnitude = 0
        s_magnitude = 1/2
    """).strip()
    assert src == expected_dot

    src = as_string(intermediate_state).strip()
    lines = set(src.splitlines())
    expected_lines = {
        "spin_magnitude = 1/2",
        "parity = +1",
        "isospin_magnitude = 1",
        "isospin_projection = -1",
        "baryon_number = -1",
        "charge = -1",
        "strangeness = +1",
        "pid = -23222",
        "mass = 1.75",
        "width = 0.15",
    }
    assert lines == expected_lines


def test_as_string_spin_tuple():
    # non-spin
    src = as_string(("a", "b", "c"))
    assert src == "a\nb\nc"
    src = as_string(("a", "b"))
    assert src == "a\nb"

    # spin
    src = as_string((2, 1))
    assert src == "|2,+1⟩"
    src = as_string((Fraction(1, 2), Fraction(-1, 2)))
    assert src == "|1/2,-1/2⟩"


class TestWrite:
    def test_write_topology(self, output_dir):
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

    def test_write_single_graph(self, output_dir: str, reaction: ReactionInfo):
        for i, transition in enumerate(reaction.transitions):
            output_file = output_dir + f"test_single_graph_{i}.gv"
            io.write(
                instance=transition,
                filename=output_file,
            )
            with open(output_file) as stream:
                src = stream.read()
            assert is_valid_dot(src)

    def test_write_graph_list(self, output_dir: str, reaction: ReactionInfo):
        for i, transition in enumerate(reaction.transitions):
            output_file = output_dir + f"test_graph_list_{i}.gv"
            io.write(transition, filename=output_file)
            with open(output_file) as stream:
                src = stream.read()
            assert is_valid_dot(src)

    def test_write_strip_spin(self, output_dir: str, reaction: ReactionInfo):
        output_file = output_dir + "test_particle_graphs.gv"
        io.write(
            instance=io.asdot(reaction, strip_spin=True),
            filename=output_file,
        )
        with open(output_file) as stream:
            src = stream.read()
        assert is_valid_dot(src)


def test_collapse_graphs(
    reaction: ReactionInfo,
    particle_database: ParticleCollection,
):
    pdg = particle_database
    particle_graphs = _get_particle_graphs(reaction.transitions)  # type: ignore[arg-type]
    assert len(particle_graphs) == 2

    collapsed_graphs = _collapse_graphs(reaction.transitions)  # type: ignore[arg-type]
    assert len(collapsed_graphs) == 1
    graph = next(iter(collapsed_graphs))
    edge_id = next(iter(graph.topology.intermediate_edge_ids))
    f_resonances = pdg.filter(lambda p: p.name in {"f(0)(980)", "f(0)(1500)"})
    intermediate_states = graph.states[edge_id]
    assert isinstance(intermediate_states, tuple)
    assert all(isinstance(i, Particle) for i in intermediate_states)
    assert intermediate_states == f_resonances


def test_get_particle_graphs(
    reaction: ReactionInfo, particle_database: ParticleCollection
):
    pdg = particle_database
    graphs = _get_particle_graphs(reaction.transitions)  # type: ignore[arg-type]
    assert len(graphs) == 2
    assert graphs[0].states[3] == pdg["f(0)(980)"]
    assert graphs[1].states[3] == pdg["f(0)(1500)"]
    assert len(graphs[0].topology.edges) == 5
    for i in range(-1, 3):
        assert graphs[0].states[i] is graphs[1].states[i]


def test_strip_projections(skh_particle_version: str):
    assert skh_particle_version is not None  # skips test if particle version too low
    resonance = "Sigma(1670)~-"
    reaction = qrules.generate_transitions(
        initial_state="J/psi(1S)",
        final_state=["K0", "Sigma+", "p~"],
        allowed_intermediate_particles=[resonance],
        allowed_interaction_types="strong",
    )
    transition = reaction.transitions[0]
    assert transition.intermediate_states[3].name == resonance
    assert len(transition.interactions) == 2

    # attach projections to the interactions, as a spin-projection extension would
    transition_with_projections = transition.convert(
        interaction_converter=lambda interaction: attrs.evolve(
            interaction, l_projection=0, s_projection=interaction.s_magnitude
        )
    )
    stripped_transition = _strip_projections(transition_with_projections)
    assert stripped_transition.states[3].name == resonance
    for interaction in stripped_transition.interactions.values():
        assert interaction.l_projection is None
        assert interaction.s_projection is None
        assert interaction.l_magnitude is not None
        assert interaction.s_magnitude is not None


@pytest.fixture
def stm() -> StateTransitionManager:
    stm = StateTransitionManager(
        initial_state=["J/psi(1S)"],
        final_state=["K0", "Sigma+", "p~"],
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
    return next(pair for pair in strong_qn_solutions if pair[1].solutions)


def is_valid_dot(src: str) -> bool:
    try:
        graphs = pydot.graph_from_dot_data(src)
        if graphs is None:
            return False
        return len(graphs) > 0
    except pydot.Error:
        return False
