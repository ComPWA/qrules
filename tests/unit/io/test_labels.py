from fractions import Fraction
from textwrap import dedent

import qrules
from qrules.io._labels import (
    as_string,
    collapse_graphs,
    get_particle_graphs,
    strip_projections,
)
from qrules.particle import Particle, ParticleCollection
from qrules.solving import QNProblemSet, QNResult
from qrules.transition import ProblemSet, ReactionInfo


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
        spin_validity - 62
        isospin_validity - 61
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
        spin_projection ∊ [-4, -7/2, -3, -5/2, -2, -3/2, -1, -1/2, 0, +1/2, +1, +3/2, +2, +5/2, +3, +7/2, +4]
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
        ls_spin_validity - 89
        CharmConservation - 70
        StrangenessConservation - 69
        BottomnessConservation - 68
        isospin_conservation - 60
        ElectronLNConservation - 45
        MuonLNConservation - 44
        TauLNConservation - 43
        MassConservation - 10
        spin_magnitude_conservation - 8
        helicity_conservation - 7
        parity_conservation - 6
        c_parity_conservation - 5
        parity_conservation_helicity - 4
        g_parity_conservation - 3
        identical_particle_symmetrization - 2
        clebsch_gordan_helicity_to_canonical - NA
        DOMAINS
        l_magnitude ∊ [0, 1]
        l_projection ∊ [0]
        parity_prefactor ∊ [-1, +1]
        s_magnitude ∊ [0, 1/2, 1, 3/2, 2]
        s_projection ∊ [-2, -3/2, -1, -1/2, 0, +1/2, +1, +3/2, +2]
    """).strip()
    assert src == expected_dot

    src = as_string(interaction).strip()
    print()
    print(src)
    expected_dot = dedent("""
        l_magnitude = 0
        s_magnitude = 1/2
        l_projection = 0
        s_projection = -1/2
        parity_prefactor = +1
    """).strip()
    assert src == expected_dot

    src = as_string(intermediate_state).strip()
    lines = set(src.splitlines())
    expected_lines = {
        "spin_magnitude = 1/2",
        "spin_projection = +1/2",
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


def test_as_string_spin_tuple(particle_database: ParticleCollection):
    # non-spin
    src = as_string(("a", "b", "c"))
    assert src == "a\nb\nc"
    src = as_string(("a", "b"))
    assert src == "a\nb"

    # spin
    src = as_string((2, 1))
    assert src == "|2,+1⟩"

    # particle with spin projection
    pion = particle_database["J/psi(1S)"]
    src = as_string((pion, 1))
    assert src == "J/psi(1S)[+1]"
    src = as_string((pion, Fraction(-1)))
    assert src == "J/psi(1S)[-1]"


def test_collapse_graphs(
    reaction: ReactionInfo,
    particle_database: ParticleCollection,
):
    pdg = particle_database
    particle_graphs = get_particle_graphs(reaction.transitions)  # type: ignore[arg-type]
    assert len(particle_graphs) == 2

    collapsed_graphs = collapse_graphs(reaction.transitions)  # type: ignore[arg-type]
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
    graphs = get_particle_graphs(reaction.transitions)  # type: ignore[arg-type]
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

    stripped_transition = strip_projections(transition)  # type: ignore[arg-type]
    assert stripped_transition.states[3].name == resonance
    assert stripped_transition.interactions[0].s_projection is None
    assert stripped_transition.interactions[0].l_projection is None
    assert stripped_transition.interactions[1].s_projection is None
    assert stripped_transition.interactions[1].l_projection is None
