import json

import pytest

from qrules.io import asdict, asdot, asmermaid
from qrules.particle import ParticleCollection, load_pdg
from qrules.quantum_numbers import EdgeQuantumNumbers
from qrules.settings import (
    DEFAULT_INTERACTION_TYPES,
    InteractionType,
    create_interaction_settings,
)
from qrules.solving import _create_merge_key
from qrules.transition import ReactionInfo, SolvingMode
from qrules.workflow import (
    InteractionConfig,
    QNProblemSetCollection,
    QNReactionInfo,
    create_qn_problem_sets,
    filter_intermediate_particles,
    find_qn_transitions,
    find_solutions,
    generate_qn_transitions,
    strip_spin_projections,
)


def describe_AllowedIntermediateParticles():
    def describe_exclude():
        def it_drops_matching_names(particle_database: ParticleCollection):
            selection = filter_intermediate_particles(
                particle_database, ["f(0)(980)", "f(0)(1500)"]
            )
            assert selection.exclude("f(0)(1500)").names == ("f(0)(980)",)

    def describe_select():
        def it_narrows_a_filtered_selection(particle_database: ParticleCollection):
            selection = filter_intermediate_particles(particle_database, "f(0)")
            narrowed = selection.select(["f(0)(980)", "f(0)(1500)"])
            assert narrowed.names == ("f(0)(980)", "f(0)(1500)")
            assert len(narrowed.particles) == 2
            assert narrowed.is_filtered

        def it_narrows_an_unfiltered_selection(particle_database: ParticleCollection):
            selection = filter_intermediate_particles(particle_database)
            narrowed = selection.select(r"^f\(0\)\(9\d0\)", regex=True)
            assert narrowed.names == ("f(0)(980)",)
            assert narrowed.is_filtered

        def it_raises_on_unmatched_pattern(particle_database: ParticleCollection):
            selection = filter_intermediate_particles(particle_database, "f(0)")
            with pytest.raises(LookupError, match="Delta"):
                selection.select("Delta")


def describe_filter_intermediate_particles():
    def it_selects_all_without_filter(particle_database: ParticleCollection):
        selection = filter_intermediate_particles(particle_database)
        assert not selection.is_filtered
        assert len(selection.particles) == len(particle_database)
        assert len(selection.names) == len(particle_database)

    def it_matches_substring_pattern(particle_database: ParticleCollection):
        selection = filter_intermediate_particles(particle_database, "f(0)(98")
        assert selection.names == ("f(0)(980)",)
        assert len(selection.particles) == 1

    def it_matches_regex_pattern(particle_database: ParticleCollection):
        selection = filter_intermediate_particles(
            particle_database, r"f\(0\)\(9\d0\)", regex=True
        )
        assert selection.names == ("f(0)(980)",)

    def it_raises_on_unmatched_pattern(particle_database: ParticleCollection):
        with pytest.raises(LookupError, match="no such particle"):
            filter_intermediate_particles(particle_database, "no such particle")

    def it_aligns_names_and_particles(particle_database: ParticleCollection):
        selection = filter_intermediate_particles(
            particle_database, ["f(0)(980)", "f(0)(1500)", "a(2)(1320)0"]
        )
        assert [p[EdgeQuantumNumbers.pid] for p in selection.particles] == [
            particle_database.find(name).pid for name in selection.names
        ]


def describe_find_solutions():
    def it_requires_formalism(particle_database: ParticleCollection):
        with pytest.raises(ValueError, match="Cannot infer the spin formalism"):
            find_solutions(qn_problem_sets={}, particle_db=particle_database)

    def it_requires_intermediate_particles(particle_database: ParticleCollection):
        with pytest.raises(ValueError, match="Cannot infer the allowed intermediate"):
            find_solutions(
                qn_problem_sets={},
                particle_db=particle_database,
                formalism="helicity",
            )

    def it_honors_fast_solving_mode(particle_database: ParticleCollection):
        def count_transitions(solving_mode: SolvingMode) -> int:
            qn_problem_sets = create_qn_problem_sets(
                initial_state=["J/psi(1S)"],
                final_state=["gamma", "pi0", "pi0"],
                particle_db=particle_database,
            )
            reaction = find_solutions(
                qn_problem_sets, particle_database, solving_mode=solving_mode
            )
            return len(reaction.transitions)

        assert count_transitions(SolvingMode.FULL) == 294
        assert count_transitions(SolvingMode.FAST) == 90


def describe_InteractionConfig():
    @pytest.fixture
    def config(particle_database: ParticleCollection) -> InteractionConfig:
        return InteractionConfig(
            type_settings=create_interaction_settings(
                "helicity", particle_db=particle_database
            )
        )

    def it_allows_all_types_by_default(config: InteractionConfig):
        assert config.get_allowed_interaction_types() == list(DEFAULT_INTERACTION_TYPES)

    def it_sets_types_globally_and_per_node(config: InteractionConfig):
        config.set_allowed_interaction_types([InteractionType.STRONG])
        assert config.get_allowed_interaction_types(node_id=0) == [
            InteractionType.STRONG
        ]
        config.set_allowed_interaction_types([InteractionType.EM], node_id=1)
        assert config.get_allowed_interaction_types(node_id=1) == [InteractionType.EM]
        assert config.get_allowed_interaction_types(node_id=0) == list(
            DEFAULT_INTERACTION_TYPES
        )

    def it_raises_on_non_interaction_type(config: InteractionConfig):
        with pytest.raises(TypeError, match="must be of type"):
            config.set_allowed_interaction_types(["strong"])  # ty: ignore[invalid-argument-type]

    def it_raises_on_unknown_interaction_type():
        config = InteractionConfig(type_settings={})
        with pytest.raises(ValueError, match="not found in settings"):
            config.set_allowed_interaction_types([InteractionType.WEAK])


@pytest.mark.parametrize(
    ("initial_state", "final_state", "expected_strengths"),
    [
        (["gamma"], ["pi0", "pi0", "pi0"], [0.0001, 1.0, 60.0]),
        (["nu(e)"], ["e-", "pi0", "pi+"], [1e-08, 0.0001, 0.006]),
    ],
)
def test_initial_state_restricts_interaction_types(
    initial_state: list[str],
    final_state: list[str],
    expected_strengths: list[float],
    particle_database: ParticleCollection,
):
    qn_problem_sets = create_qn_problem_sets(
        initial_state, final_state, particle_database
    )
    assert sorted(qn_problem_sets.problem_sets) == expected_strengths


@pytest.mark.parametrize("merge_spin_projections", [False, True])
def test_pipeline_reproduces_state_transition_manager(
    reaction: ReactionInfo, merge_spin_projections: bool
):
    particle_db = load_pdg()
    qn_problem_sets = create_qn_problem_sets(
        initial_state=[("J/psi(1S)", [-1, 1])],
        final_state=["gamma", "pi0", "pi0"],
        particle_db=particle_db,
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        interaction_config=InteractionConfig(
            type_settings=create_interaction_settings(
                reaction.formalism,
                particle_db=particle_db,
                max_angular_momentum=2,
            ),
            allowed_types=[InteractionType.STRONG],
        ),
        formalism=reaction.formalism,
        merge_spin_projections=merge_spin_projections,
    )
    assert isinstance(qn_problem_sets, QNProblemSetCollection)
    assert qn_problem_sets.formalism == reaction.formalism
    assert qn_problem_sets.final_state == ["gamma", "pi0", "pi0"]
    assert qn_problem_sets.intermediate_particles.names == ("f(0)(980)", "f(0)(1500)")
    assert all(
        strength > 0 and len(problems) > 0
        for strength, problems in qn_problem_sets.problem_sets.items()
    )
    workflow_reaction = find_solutions(qn_problem_sets, particle_db)
    assert workflow_reaction == reaction


def test_projection_free_qn_transitions():
    particle_db = load_pdg()
    collection = create_qn_problem_sets(
        initial_state=[("J/psi(1S)", [-1, 1])],
        final_state=["gamma", "pi0", "pi0"],
        particle_db=particle_db,
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        interaction_config=InteractionConfig(
            type_settings=create_interaction_settings(
                "helicity", particle_db=particle_db, max_angular_momentum=2
            ),
            allowed_types=[InteractionType.STRONG],
        ),
    )
    stripped = strip_spin_projections(collection)
    assert isinstance(stripped, QNProblemSetCollection)
    n_original = sum(map(len, collection.problem_sets.values()))
    n_stripped = sum(map(len, stripped.problem_sets.values()))
    assert n_stripped < n_original

    qn_transitions = find_qn_transitions(stripped)
    assert len(qn_transitions) > 0
    qn_names = {
        qn_type.__name__
        for transition in qn_transitions
        for prop_map in [*transition.states.values(), *transition.interactions.values()]
        for qn_type in prop_map
    }
    assert "spin_projection" not in qn_names
    assert {"spin_magnitude", "parity", "l_magnitude", "s_magnitude"} <= qn_names

    intermediate_signatures = {
        (
            state[EdgeQuantumNumbers.spin_magnitude],
            int(state[EdgeQuantumNumbers.parity]),
            int(state[EdgeQuantumNumbers.c_parity]),
        )
        for transition in qn_transitions
        for state in transition.intermediate_states.values()
    }
    assert intermediate_signatures == {(0, +1, +1)}  # both f0 resonances are 0^{++}
    collapsed = {
        transition.convert(interaction_converter=lambda _: None)
        for transition in qn_transitions
    }
    assert len(collapsed) == 1

    serialized = json.dumps(asdict(qn_transitions[0]))
    assert '"spin_projection"' not in serialized
    assert '"spin_magnitude"' in serialized

    unexpanded = create_qn_problem_sets(
        initial_state=[("J/psi(1S)", [-1, 1])],
        final_state=["gamma", "pi0", "pi0"],
        particle_db=particle_db,
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        interaction_config=InteractionConfig(
            type_settings=create_interaction_settings(
                "helicity", particle_db=particle_db, max_angular_momentum=2
            ),
            allowed_types=[InteractionType.STRONG],
        ),
        spin_projections=False,
    )
    assert _to_merge_keys(unexpanded) == _to_merge_keys(stripped)
    assert find_qn_transitions(unexpanded) == qn_transitions


def _to_merge_keys(collection: QNProblemSetCollection) -> set[tuple]:
    return {
        (strength, _create_merge_key(problem_set, set()))
        for strength, problem_sets in collection.problem_sets.items()
        for problem_set in problem_sets
    }


def test_generate_qn_transitions():
    particle_db = load_pdg()
    reaction = generate_qn_transitions(
        initial_state="J/psi(1S)",
        final_state=["gamma", "pi0", "pi0"],
        particle_db=particle_db,
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        allowed_interaction_types=["strong", "em"],
    )
    assert isinstance(reaction, QNReactionInfo)
    assert len(reaction.transitions) > 0
    assert {p.name for p in reaction.initial_state.values()} == {"J/psi(1S)"}
    assert [p.name for _, p in sorted(reaction.final_state.items())] == [
        "gamma",
        "pi0",
        "pi0",
    ]
    for qn_set in reaction.get_intermediate_quantum_numbers():
        assert qn_set[EdgeQuantumNumbers.spin_magnitude] == 0
        assert qn_set[EdgeQuantumNumbers.parity] == +1
        assert qn_set[EdgeQuantumNumbers.c_parity] == +1
    assert len(reaction.group_by_topology()) == 1

    dot = asdot(reaction)
    assert dot.startswith("digraph {")
    assert "J/psi(1S)" in dot
    assert "spin_projection" not in dot.replace("isospin_projection", "")

    mermaid = asmermaid(reaction, latex=False)
    assert mermaid.startswith("flowchart LR")
    assert "J/psi(1S)" in mermaid
    assert "spin_projection" not in mermaid.replace("isospin_projection", "")


def test_qn_reaction_info_requires_particle_states():
    particle_db = load_pdg()
    qn_problem_sets = create_qn_problem_sets(
        initial_state=["J/psi(1S)"],
        final_state=["gamma", "pi0", "pi0"],
        particle_db=particle_db,
        allowed_intermediate_particles=["f(0)(980)"],
        spin_projections=False,
    )
    qn_transitions = find_qn_transitions(qn_problem_sets)
    with pytest.raises(TypeError, match="is of type FrozenDict, not Particle"):
        QNReactionInfo(qn_transitions)


def test_incompatible_spin_projection_flags_raise():
    with pytest.raises(ValueError, match="merge_spin_projections has no effect"):
        create_qn_problem_sets(
            initial_state=["J/psi(1S)"],
            final_state=["gamma", "pi0", "pi0"],
            merge_spin_projections=True,
            spin_projections=False,
        )
