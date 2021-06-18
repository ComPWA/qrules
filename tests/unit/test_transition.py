# pyright: reportUnusedImport=false
# pylint: disable=eval-used, no-self-use
from operator import itemgetter
from typing import List

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
        assert len(reaction.transition_groups) == 1
        for grouping in reaction.transition_groups:
            assert isinstance(grouping, StateTransitionCollection)
        if reaction.formalism.startswith("cano"):
            assert len(reaction.transitions) == 16
        else:
            assert len(reaction.transitions) == 8
        for transition in reaction.transitions:
            assert isinstance(transition, StateTransition)

    @pytest.mark.parametrize("repr_method", [repr, pretty])
    def test_repr(self, repr_method, reaction: ReactionInfo):
        instance = reaction
        from_repr = eval(repr_method(instance))
        assert from_repr == instance

    def test_from_to_graphs(self, reaction: ReactionInfo):
        graphs = reaction.to_graphs()
        from_graphs = ReactionInfo.from_graphs(graphs, reaction.formalism)
        assert from_graphs == reaction


class TestState:
    @pytest.mark.parametrize(
        ("state_def_1", "state_def_2"),
        [
            (("a", -1), ("a", +1)),
            (("a", 0), ("a", 0)),
            (("a", 0), ("b", 0)),
            (("a", -1), ("b", +1)),
        ],
    )
    def test_ordering(self, state_def_1, state_def_2):
        def create_state(state_def) -> State:
            return State(
                particle=Particle(name=state_def[0], pid=0, spin=0, mass=0),
                spin_projection=state_def[1],
            )

        state1 = create_state(state_def_1)
        state2 = create_state(state_def_2)
        assert state2 >= state1


class TestStateTransition:
    def test_ordering(self, reaction: ReactionInfo):
        sorted_transitions: List[StateTransition] = sorted(
            reaction.transitions
        )
        if reaction.formalism.startswith("cano"):
            first = sorted_transitions[0]
            second = sorted_transitions[1]
            assert first.interactions[0].l_magnitude == 0.0
            assert second.interactions[0].l_magnitude == 2.0
            assert first.interactions[1] == second.interactions[1]
            transition_selection = sorted_transitions[::2]
        else:
            transition_selection = sorted_transitions

        simplified_rendering = [
            tuple(
                (
                    transition.states[state_id].particle.name,
                    int(transition.states[state_id].spin_projection),
                )
                for state_id in sorted(transition.states)
            )
            for transition in transition_selection
        ]

        assert simplified_rendering[:3] == [
            (
                ("J/psi(1S)", -1),
                ("gamma", -1),
                ("pi0", 0),
                ("pi0", 0),
                ("f(0)(980)", 0),
            ),
            (
                ("J/psi(1S)", -1),
                ("gamma", -1),
                ("pi0", 0),
                ("pi0", 0),
                ("f(0)(1500)", 0),
            ),
            (
                ("J/psi(1S)", -1),
                ("gamma", +1),
                ("pi0", 0),
                ("pi0", 0),
                ("f(0)(980)", 0),
            ),
        ]
        assert simplified_rendering[-1] == (
            ("J/psi(1S)", +1),
            ("gamma", +1),
            ("pi0", 0),
            ("pi0", 0),
            ("f(0)(1500)", 0),
        )

        # J/psi
        first_half = slice(0, int(len(simplified_rendering) / 2))
        for item in simplified_rendering[first_half]:
            assert item[0] == ("J/psi(1S)", -1)
        second_half = slice(int(len(simplified_rendering) / 2), None)
        for item in simplified_rendering[second_half]:
            assert item[0] == ("J/psi(1S)", +1)
        second_half = slice(int(len(simplified_rendering) / 2), None)
        # gamma
        for item in itemgetter(0, 1, 4, 5)(simplified_rendering):
            assert item[1] == ("gamma", -1)
        for item in itemgetter(2, 3, 6, 7)(simplified_rendering):
            assert item[1] == ("gamma", +1)
        # pi0
        for item in simplified_rendering:
            assert item[2] == ("pi0", 0)
            assert item[3] == ("pi0", 0)
        # f0
        for item in simplified_rendering[::2]:
            assert item[4] == ("f(0)(980)", 0)
        for item in simplified_rendering[1::2]:
            assert item[4] == ("f(0)(1500)", 0)

    @pytest.mark.parametrize("repr_method", [repr, pretty])
    def test_repr(self, repr_method, reaction: ReactionInfo):
        for instance in reaction.transitions:
            from_repr = eval(repr_method(instance))
            assert from_repr == instance

    def test_from_to_graph(self, reaction: ReactionInfo):
        assert len(reaction.transition_groups) == 1
        assert len(reaction.transitions) in {8, 16}
        for transition in reaction.transitions:
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
        assert len(reaction.transition_groups) == 1
        transition_grouping = reaction.transition_groups[0]
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
