# pyright: reportUnusedImport=false
# pylint: disable=eval-used, no-self-use
import pytest
from IPython.lib.pretty import pretty

from qrules.particle import (  # noqa: F401
    Parity,
    Particle,
    ParticleCollection,
    ParticleWithSpin,
    Spin,
)
from qrules.quantum_numbers import InteractionProperties  # noqa: F401
from qrules.topology import (  # noqa: F401
    Edge,
    FrozenDict,
    StateTransitionGraph,
    Topology,
)
from qrules.transition import (  # noqa: F401
    ReactionInfo,
    Result,
    State,
    StateTransition,
    StateTransitionCollection,
    StateTransitionManager,
    _sort_graphs,
)


class TestReactionInfo:
    def test_formalism(self, result: Result):
        assert result.formalism is not None
        reaction_info = ReactionInfo.from_result(result)
        assert reaction_info.formalism == result.formalism

    def test_from_graphs(self, result: Result):
        reaction_info = ReactionInfo.from_result(result)
        assert reaction_info.initial_state[-1].name == "J/psi(1S)"
        assert reaction_info.final_state[0].name == "gamma"
        assert reaction_info.final_state[1].name == "pi0"
        assert reaction_info.final_state[2].name == "pi0"
        assert len(reaction_info) == 1
        first_group = reaction_info[0]
        assert isinstance(first_group, StateTransitionCollection)
        assert len(first_group) == len(result.transitions)

    @pytest.mark.parametrize("repr_method", [repr, pretty])
    def test_repr(self, repr_method, result: Result):
        reaction_info = ReactionInfo.from_result(result)
        instance = reaction_info
        from_repr = eval(repr_method(instance))
        assert from_repr == instance

    def test_to_graphs(self, result: Result):
        original_graphs = result.transitions
        reaction_info = ReactionInfo.from_result(result)
        converted_graphs = reaction_info.to_graphs()
        assert len(converted_graphs) == len(original_graphs)
        original_graphs = _sort_graphs(original_graphs)
        assert converted_graphs == original_graphs


class TestStateTransition:
    @pytest.fixture(scope="session")
    def some_graph(
        self, result: Result
    ) -> StateTransitionGraph[ParticleWithSpin]:
        graphs = result.transitions
        return next(iter(graphs))

    def test_from_graph(
        self,
        particle_database: ParticleCollection,
        some_graph: StateTransitionGraph[ParticleWithSpin],
    ):
        pdg = particle_database
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

    @pytest.mark.parametrize("repr_method", [repr, pretty])
    def test_repr(
        self, repr_method, some_graph: StateTransitionGraph[ParticleWithSpin]
    ):
        instance = StateTransition.from_graph(some_graph)
        from_repr = eval(repr_method(instance))
        assert from_repr == instance

    def test_to_graph(self, result: Result):
        assert len(result.transitions) in {8, 16}
        for graph in result.transitions:
            transition = StateTransition.from_graph(graph)
            from_transition = transition.to_graph()
            assert from_transition == graph


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

    @pytest.mark.parametrize("repr_method", [repr, pretty])
    def test_repr(self, result: Result, repr_method):
        instance = StateTransitionCollection.from_graphs(result.transitions)
        from_repr = eval(repr_method(instance))
        assert from_repr == instance

    def test_to_graphs(self, result: Result):
        original_graphs = result.transitions
        transitions = StateTransitionCollection.from_graphs(original_graphs)
        converted_graphs = transitions.to_graphs()
        assert len(converted_graphs) == len(original_graphs)
        original_graphs = _sort_graphs(original_graphs)
        assert converted_graphs == original_graphs


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
