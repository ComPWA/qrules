import logging
from fractions import Fraction
from textwrap import dedent

import attrs

import qrules
from qrules.io._labels import (
    as_latex,
    as_string,
    collapse_graphs,
    create_edge_label,
    get_particle_graphs,
    strip_projections,
)
from qrules.particle import Particle, ParticleCollection
from qrules.quantum_numbers import InteractionProperties
from qrules.solving import QNProblemSet, QNResult
from qrules.transition import ProblemSet, ReactionInfo


def describe_as_latex():
    def it_is_extensible():
        class CustomLabel:
            pass

        as_latex.register(CustomLabel, lambda _: R"\alpha")
        assert as_latex(CustomLabel()) == R"\alpha"

    def it_label_renderer_dispatch_is_independent():
        class CustomLabel:
            pass

        as_string.register(CustomLabel, lambda _: "plain")
        as_latex.register(CustomLabel, lambda _: R"\mathrm{latex}")

        label = CustomLabel()
        assert as_string(label) == "plain"
        assert as_latex(label) == R"\mathrm{latex}"

    def it_fallback(caplog):
        class UnsupportedLabel:
            def __str__(self) -> str:
                return "unsupported"

        with caplog.at_level(logging.WARNING):
            assert as_latex(UnsupportedLabel()) == "unsupported"
        assert (
            "No LaTeX label renderer implemented type UnsupportedLabel" in caplog.text
        )

    def it_particle(particle_database: ParticleCollection):
        particle = particle_database["J/psi(1S)"]
        assert as_latex(particle) == R"J/\psi(1S)"

        particle_with_custom_latex = attrs.evolve(
            particle,
            name="this_name_is_not_rendered",
            latex=R"\mathrm{x}_{100\%}",
        )
        assert as_latex(particle_with_custom_latex) == R"\mathrm{x}_{100\%}"

        particle_without_latex = attrs.evolve(particle, name="custom_name", latex=None)
        assert as_latex(particle_without_latex) == R"\text{custom\_name}"

        special_name = R"\{}$&#_%~^"
        particle_without_latex = attrs.evolve(particle, name=special_name, latex=None)
        assert as_latex(particle_without_latex) == (
            R"\text{\textbackslash{}\{\}\$\&\#\_\%"
            R"\textasciitilde{}\textasciicircum{}}"
        )

    def it_spin_and_interaction():
        assert as_latex((Fraction(1, 2), Fraction(1, 2))) == (
            R"\left|\frac{1}{2},\text{+}\frac{1}{2}\right\rangle"
        )
        interaction = InteractionProperties(
            l_magnitude=1,
            l_projection=0,
            s_magnitude=Fraction(1, 2),
            parity_prefactor=1,
        )
        src = as_latex(interaction)
        assert src.startswith(R"\begin{gathered}")
        assert R"L = \left|1,0\right\rangle" in src
        assert R"S = \frac{1}{2}" in src
        assert R"P = \text{+}1" in src

        assert not as_latex(InteractionProperties())
        assert as_latex(InteractionProperties(l_magnitude=1)) == "L = 1"
        assert (
            as_latex(
                InteractionProperties(
                    s_magnitude=Fraction(1, 2), s_projection=Fraction(-1, 2)
                )
            )
            == R"S = \left|\frac{1}{2},\text{-}\frac{1}{2}\right\rangle"
        )

    def it_dict_and_basic_values():
        assert as_latex(1) == "1"
        assert as_latex(1.5) == "1.5"
        assert as_latex(R"\alpha") == R"\alpha"
        src = as_latex({"spin_magnitude": Fraction(1, 2), "parity": 1})
        assert R"\text{spin\_magnitude} = \frac{1}{2}" in src
        assert R"\text{parity} = \text{+}1" in src
        assert as_latex(Fraction(-1, 2)) == R"\text{-}\frac{1}{2}"
        assert as_latex(None) == R"\mathrm{None}"
        assert not as_latex({})
        assert as_latex({"pid": 1}) == R"\text{pid} = 1"

    def it_collapsed_particle_tuple(particle_database: ParticleCollection):
        particles = (
            particle_database["f(0)(980)"],
            particle_database["f(0)(1500)"],
        )
        assert as_latex(particles) == (
            R"\begin{gathered} f_{0}(980) \\ f_{0}(1500) \end{gathered}"
        )

    def it_keeps_six_particles_in_one_column(particle_database: ParticleCollection):
        particle = particle_database["f(0)(980)"]
        assert as_latex((particle,) * 6).startswith(R"\begin{gathered}")

    def it_uses_columns_for_a_long_particle_tuple(
        particle_database: ParticleCollection,
    ):
        particle = particle_database["f(0)(980)"]
        particles = tuple(
            attrs.evolve(particle, name=f"x{i}", latex=Rf"x_{{{i}}}") for i in range(7)
        )
        assert as_latex(particles) == (
            R"\begin{array}{ll} x_{0} & x_{4} \\ x_{1} & x_{5} \\ "
            R"x_{2} & x_{6} \\ x_{3} &  \end{array}"
        )

    def it_adds_columns_to_a_longer_particle_tuple(
        particle_database: ParticleCollection,
    ):
        particle = particle_database["f(0)(980)"]
        particles = tuple(
            attrs.evolve(particle, name=f"x{i}", latex=Rf"x_{{{i}}}") for i in range(13)
        )
        assert as_latex(particles) == (
            R"\begin{array}{lll} x_{0} & x_{5} & x_{10} \\ "
            R"x_{1} & x_{6} & x_{11} \\ x_{2} & x_{7} & x_{12} \\ "
            R"x_{3} & x_{8} &  \\ x_{4} & x_{9} &  \end{array}"
        )


def test_create_edge_label_accepts_renderer(reaction: ReactionInfo):
    transition = reaction.transitions[0]
    edge_id = next(iter(transition.topology.incoming_edge_ids))
    state = transition.states[edge_id]

    plain_label = create_edge_label(transition, edge_id, render_edge_id=False)
    latex_label = create_edge_label(
        transition,
        edge_id,
        render_edge_id=False,
        render_label=as_latex,
    )

    assert plain_label.startswith(state.name)
    assert state.latex is not None
    assert latex_label.startswith(state.latex)

    multiline_label = create_edge_label(
        transition,
        edge_id,
        render_edge_id=True,
        render_label=lambda _: "first\nsecond",
    )
    assert multiline_label == f"{edge_id}:\nfirst\nsecond"


def describe_as_string():
    def it_dict(
        problem_sets: dict[float, list[ProblemSet]],
        qn_problem_and_result: tuple[QNProblemSet, QNResult],
    ):
        _, qn_result = qn_problem_and_result
        problem_set = problem_sets[3600.0][0]
        interaction = qn_result.solutions[1].interactions[1]
        intermediate_state, *_ = qn_result.solutions[0].intermediate_states.values()
        node_setting = problem_set.solving_settings.interactions[0]
        intermediate_setting, *_ = (
            problem_set.solving_settings.intermediate_states.values()
        )

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
            ChargeConservation - 100
            BaryonNumberConservation - 90
            CharmConservation - 70
            StrangenessConservation - 69
            BottomnessConservation - 68
            isospin_conservation - 60
            ElectronLNConservation - 45
            MuonLNConservation - 44
            TauLNConservation - 43
            MassConservation - 10
            spin_magnitude_conservation - 8
            parity_conservation - 6
            c_parity_conservation - 5
            g_parity_conservation - 3
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

        latex = as_latex(intermediate_setting)
        assert R"\text{RULES}" in latex
        assert R"\text{isospin\_validity - 61}" in latex
        assert R"\text{DOMAINS}" in latex
        assert R"\text{spin\_magnitude} \in \left[\frac{1}{2}\right]" in latex

        latex = as_latex(node_setting)
        assert R"\text{ChargeConservation - 100}" in latex
        assert R"\text{l\_magnitude} \in \left[0, 1\right]" in latex
        assert (
            R"\text{s\_magnitude} \in \left[0, \frac{1}{2}, 1, \frac{3}{2}, 2\right]"
            in latex
        )

        latex = as_latex(intermediate_state)
        assert R"\text{spin\_magnitude} = \frac{1}{2}" in latex
        assert R"\text{parity} = \text{+}1" in latex

    def it_spin_tuple():
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


def test_collapse_graphs(
    reaction: ReactionInfo,
    particle_database: ParticleCollection,
):
    pdg = particle_database
    particle_graphs = get_particle_graphs(reaction.transitions)
    assert len(particle_graphs) == 2

    collapsed_graphs = collapse_graphs(reaction.transitions)
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
    graphs = get_particle_graphs(reaction.transitions)
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
    stripped_transition = strip_projections(transition_with_projections)
    assert stripped_transition.states[3].name == resonance
    for interaction in stripped_transition.interactions.values():
        assert interaction.l_projection is None
        assert interaction.s_projection is None
        assert interaction.l_magnitude is not None
        assert interaction.s_magnitude is not None
