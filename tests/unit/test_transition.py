# pyright: reportUnusedImport=false
import hashlib
import pickle  # noqa: S403
import sys
from copy import deepcopy
from fractions import Fraction

import pytest
from IPython.lib.pretty import pretty

from qrules.particle import Parity, Particle, ParticleCollection, Spin  # noqa: F401
from qrules.quantum_numbers import InteractionProperties  # noqa: F401
from qrules.settings import InteractionType
from qrules.topology import (  # noqa: F401
    Edge,
    FrozenDict,
    FrozenTransition,
    MutableTransition,
    Topology,
)
from qrules.transition import ReactionInfo, State, StateTransitionManager

NAMESPACE_WITH_FRACTIONS = globals()
NAMESPACE_WITH_FRACTIONS["Fraction"] = Fraction


class TestMutableTransition:
    def test_intermediate_states(self):
        stm = StateTransitionManager(
            initial_state=[("J/psi(1S)", [-1, +1])],
            final_state=["K0", "Sigma+", "p~"],
            allowed_intermediate_particles=["N(1700)", "Sigma(1750)"],
            formalism="helicity",
            mass_conservation_factor=0,
        )
        stm.set_allowed_interaction_types([InteractionType.STRONG, InteractionType.EM])
        problem_sets = stm.create_problem_sets()
        some_problem_set = problem_sets[3600.0][0]
        assert set(some_problem_set.initial_facts.initial_states) == {-1}
        assert set(some_problem_set.initial_facts.final_states) == {0, 1, 2}
        assert set(some_problem_set.initial_facts.intermediate_states) == set()


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
        from_repr = eval(repr_method(instance), NAMESPACE_WITH_FRACTIONS)
        assert from_repr == instance

    def test_hash(self, reaction: ReactionInfo):
        assert hash(deepcopy(reaction)) == hash(reaction)

    def test_hash_value(self, reaction: ReactionInfo):
        if sys.version_info >= (3, 11):
            expected_hash = {
                "canonical-helicity": "65106a44301f9340e633d09f66ad7d17",
                "helicity": "9646d3ee5c5e8534deb8019435161f2e",
            }[reaction.formalism]
        else:
            expected_hash = {
                "canonical-helicity": "0d8bc378677986e0dc2d3b02f5627e0b",
                "helicity": "71404ad43550850a02109e8db044bd28",
            }[reaction.formalism]

        assert _compute_hash(reaction) == expected_hash


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
            initial_state=[("J/psi(1S)", list(map(Fraction, [-1, +1])))],
            final_state=["p", "p~", "eta"],
        )
        particle_name = "N(753)"
        with pytest.raises(
            LookupError,
            match=r"Could not find any matches for allowed intermediate particle",
        ):
            stm.set_allowed_intermediate_particles(particle_name)

    def test_regex_pattern(self):
        stm = StateTransitionManager(
            initial_state=["Lambda(c)+"],
            final_state=["p", "K-", "pi+"],
            allowed_intermediate_particles=["Delta"],
        )
        stm.set_allowed_interaction_types([InteractionType.STRONG], node_id=1)
        problem_sets = stm.create_problem_sets()
        reaction = stm.find_solutions(problem_sets)
        assert reaction.get_intermediate_particles().names == [
            "Delta(1232)++",
            "Delta(1600)++",
            "Delta(1620)++",
            "Delta(1900)++",
            "Delta(1910)++",
            "Delta(1920)++",
        ]

        stm.set_allowed_intermediate_particles(r"^Delta\(\d(60|9[02])0\)", regex=True)
        problem_sets = stm.create_problem_sets()
        reaction = stm.find_solutions(problem_sets)
        assert reaction.get_intermediate_particles().names == [
            "Delta(1600)++",
            "Delta(1900)++",
            "Delta(1920)++",
        ]


def _compute_hash(obj) -> str:
    b = _to_bytes(obj)
    h = hashlib.md5(b)  # noqa: S324
    return h.hexdigest()


def _to_bytes(obj) -> bytes:
    if isinstance(obj, (bytes, bytearray)):
        return obj
    return pickle.dumps(obj)
