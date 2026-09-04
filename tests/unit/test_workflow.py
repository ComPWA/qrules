import pytest

from qrules.particle import ParticleCollection, load_pdg
from qrules.quantum_numbers import EdgeQuantumNumbers
from qrules.settings import (
    DEFAULT_INTERACTION_TYPES,
    InteractionType,
    create_interaction_settings,
)
from qrules.transition import ReactionInfo, SolvingMode
from qrules.workflow import (
    InteractionConfig,
    QNProblemSetCollection,
    create_qn_problem_sets,
    filter_intermediate_particles,
    find_solutions,
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


def test_pipeline_reproduces_state_transition_manager(reaction: ReactionInfo):
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
