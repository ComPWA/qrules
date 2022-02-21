# pyright: reportUnusedImport=false
# pylint: disable=eval-used, no-self-use
from copy import deepcopy

import pytest
from IPython.lib.pretty import pretty

from qrules.particle import (  # noqa: F401
    Parity,
    Particle,
    ParticleCollection,
    Spin,
)
from qrules.quantum_numbers import InteractionProperties  # noqa: F401
from qrules.topology import (  # noqa: F401
    Edge,
    FrozenDict,
    FrozenTransition,
    MutableTransition,
    Topology,
)
from qrules.transition import ReactionInfo, State, StateTransitionManager


class TestReactionInfo:
    def test_properties(self, reaction: ReactionInfo):
        assert reaction.initial_state[-1].name == "J/psi(1S)"
        assert reaction.final_state[0].name == "gamma"
        assert reaction.final_state[1].name == "pi0"
        assert reaction.final_state[2].name == "pi0"
        assert len(reaction.group_by_topology()) == 1
        if reaction.formalism.startswith("cano"):
            assert len(reaction.transitions) == 16
        else:
            assert len(reaction.transitions) == 8
        for transition in reaction.transitions:
            assert isinstance(transition, FrozenTransition)

    @pytest.mark.parametrize("repr_method", [repr, pretty])
    def test_repr(self, repr_method, reaction: ReactionInfo):
        instance = reaction
        from_repr = eval(repr_method(instance))
        assert from_repr == instance

    def test_hash(self, reaction: ReactionInfo):
        assert hash(deepcopy(reaction)) == hash(reaction)


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


class TestStateTransitionManager:
    def test_allowed_intermediate_particles(self):
        stm = StateTransitionManager(
            initial_state=[("J/psi(1S)", [-1, +1])],
            final_state=["p", "p~", "eta"],
        )
        particle_name = "N(753)"
        with pytest.raises(
            LookupError,
            match=(
                r"Could not find any matches for allowed intermediate particle"
            ),
        ):
            stm.set_allowed_intermediate_particles([particle_name])
