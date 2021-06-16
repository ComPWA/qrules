# pylint: disable=no-self-use
import pytest

from qrules.particle import ParticleCollection
from qrules.transition import (
    ReactionInfo,
    Result,
    StateTransition,
    StateTransitionCollection,
    StateTransitionManager,
)


class TestReactionInfo:
    def test_from_graphs(self, result: Result):
        reaction_info = ReactionInfo.from_graphs(result.transitions)
        assert reaction_info.initial_state[-1].name == "J/psi(1S)"
        assert reaction_info.final_state[0].name == "gamma"
        assert reaction_info.final_state[1].name == "pi0"
        assert reaction_info.final_state[2].name == "pi0"
        assert len(reaction_info) == 1
        first_group = reaction_info[0]
        assert isinstance(first_group, StateTransitionCollection)
        assert len(first_group) == len(result.transitions)


class TestStateTransition:
    def test_from_graph(
        self, particle_database: ParticleCollection, result: Result
    ):
        pdg = particle_database
        graphs = result.transitions
        some_graph = next(iter(graphs))
        transition = StateTransition.from_graph(some_graph)
        assert transition.topology == some_graph.topology
        assert len(transition.initial_states) == 1
        assert len(transition.intermediate_states) == 1
        assert len(transition.final_states) == 3
        initial_state = transition.initial_states[-1]
        assert initial_state.particle == pdg["J/psi(1S)"]
        assert initial_state.spin_projection in {-1, +1}
        assert set(transition.particles.values()) > {
            pdg["J/psi(1S)"],
            pdg["gamma"],
            pdg["pi0"],
        }
        intermediate_state = transition.intermediate_states[3]
        assert intermediate_state.spin_projection == 0
        assert intermediate_state.particle in pdg.filter(
            lambda p: p.name.startswith("f(0)")
        )


class TestStateTransitionCollection:
    def test_from_graphs(self, result: Result):
        transitions = StateTransitionCollection.from_graphs(result.transitions)
        assert result.formalism is not None
        if result.formalism.startswith("cano"):
            assert len(transitions) == 16
        elif result.formalism.startswith("heli"):
            assert len(transitions) == 8
        else:
            raise NotImplementedError
        assert len({t.topology for t in transitions}) == 1
        assert {t.topology for t in transitions} == {transitions.topology}


class TestStateTransitionManager:
    def test_allowed_intermediate_particles(self):
        stm = StateTransitionManager(
            initial_state=[("J/psi(1S)", [-1, +1])],
            final_state=["p", "p~", "eta"],
            number_of_threads=1,
        )
        particle_name = "N(753)"
        with pytest.raises(
            LookupError,
            match=r"Could not find any matches for allowed intermediate particle",
        ):
            stm.set_allowed_intermediate_particles([particle_name])
