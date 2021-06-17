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
from qrules.transition import State  # noqa: F401
from qrules.transition import (
    ReactionInfo,
    StateTransition,
    StateTransitionCollection,
    StateTransitionManager,
)


class TestReactionInfo:
    def test_properties(self, reaction: ReactionInfo):
        assert reaction.initial_state[-1].name == "J/psi(1S)"
        assert reaction.final_state[0].name == "gamma"
        assert reaction.final_state[1].name == "pi0"
        assert reaction.final_state[2].name == "pi0"
        assert len(reaction) == 1
        first_group = reaction[0]
        assert isinstance(first_group, StateTransitionCollection)

    @pytest.mark.parametrize("repr_method", [repr, pretty])
    def test_repr(self, repr_method, reaction: ReactionInfo):
        for instance in reaction:
            from_repr = eval(repr_method(instance))
            assert from_repr == instance
        instance = reaction
        from_repr = eval(repr_method(instance))
        assert from_repr == instance

    def test_from_to_graphs(self, reaction: ReactionInfo):
        graphs = reaction.to_graphs()
        from_graphs = ReactionInfo.from_graphs(graphs, reaction.formalism)
        assert from_graphs == reaction


class TestStateTransition:
    @pytest.mark.parametrize("repr_method", [repr, pretty])
    def test_repr(self, repr_method, reaction: ReactionInfo):
        for grouping in reaction.transition_groups:
            for instance in grouping:
                from_repr = eval(repr_method(instance))
                assert from_repr == instance

    def test_from_to_graph(self, reaction: ReactionInfo):
        assert len(reaction) == 1
        transitions = reaction.transition_groups[0]
        assert len(transitions) in {8, 16}
        for transition in transitions:
            graph = transition.to_graph()
            from_graph = StateTransition.from_graph(graph)
            assert transition == from_graph


class TestStateTransitionCollection:
    @pytest.mark.parametrize("repr_method", [repr, pretty])
    def test_repr(self, reaction: ReactionInfo, repr_method):
        for instance in reaction.transition_groups:
            from_repr = eval(repr_method(instance))
            assert from_repr == instance

    def test_from_to_graphs(self, reaction: ReactionInfo):
        assert len(reaction) == 1
        transition_grouping = reaction[0]
        graphs = transition_grouping.to_graphs()
        from_graphs = StateTransitionCollection.from_graphs(graphs)
        assert transition_grouping == from_graphs


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
