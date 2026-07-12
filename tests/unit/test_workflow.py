import json

import pytest

from qrules.io import asdict
from qrules.particle import ParticleCollection, load_pdg
from qrules.quantum_numbers import EdgeQuantumNumbers
from qrules.settings import (
    DEFAULT_INTERACTION_TYPES,
    InteractionType,
    create_interaction_settings,
)
from qrules.transition import ReactionInfo
from qrules.workflow import (
    InteractionConfig,
    QNProblemSetCollection,
    create_qn_problem_sets,
    filter_intermediate_particles,
    find_qn_transitions,
    find_solutions,
    strip_spin_projections,
)


class TestFilterIntermediateParticles:
    def test_no_filter_selects_all(self, particle_database: ParticleCollection):
        selection = filter_intermediate_particles(particle_database)
        assert selection.names is None
        assert len(selection.particles) == len(particle_database)

    def test_substring_pattern(self, particle_database: ParticleCollection):
        selection = filter_intermediate_particles(particle_database, "f(0)(98")
        assert selection.names == ("f(0)(980)",)
        assert len(selection.particles) == 1

    def test_regex_pattern(self, particle_database: ParticleCollection):
        selection = filter_intermediate_particles(
            particle_database, r"f\(0\)\(9\d0\)", regex=True
        )
        assert selection.names == ("f(0)(980)",)

    def test_unmatched_pattern_raises(self, particle_database: ParticleCollection):
        with pytest.raises(LookupError, match="no such particle"):
            filter_intermediate_particles(particle_database, "no such particle")


class TestInteractionConfig:
    @pytest.fixture
    def config(self, particle_database: ParticleCollection) -> InteractionConfig:
        return InteractionConfig(
            type_settings=create_interaction_settings(
                "helicity", particle_db=particle_database
            )
        )

    def test_default_allowed_types(self, config: InteractionConfig):
        assert config.get_allowed_interaction_types() == list(DEFAULT_INTERACTION_TYPES)

    def test_set_globally_and_per_node(self, config: InteractionConfig):
        config.set_allowed_interaction_types([InteractionType.STRONG])
        assert config.get_allowed_interaction_types(node_id=0) == [
            InteractionType.STRONG
        ]
        config.set_allowed_interaction_types([InteractionType.EM], node_id=1)
        assert config.get_allowed_interaction_types(node_id=1) == [InteractionType.EM]
        assert config.get_allowed_interaction_types(node_id=0) == list(
            DEFAULT_INTERACTION_TYPES
        )

    def test_non_interaction_type_raises(self, config: InteractionConfig):
        with pytest.raises(TypeError, match="must be of type"):
            config.set_allowed_interaction_types(["strong"])  # type: ignore[list-item]

    def test_unknown_interaction_type_raises(self):
        config = InteractionConfig(type_settings={})
        with pytest.raises(ValueError, match="not found in settings"):
            config.set_allowed_interaction_types([InteractionType.WEAK])


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
