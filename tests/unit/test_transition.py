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
from qrules.transition import ReactionInfo, StateTransitionManager

NAMESPACE_WITH_FRACTIONS = globals()
NAMESPACE_WITH_FRACTIONS["Fraction"] = Fraction


class TestMutableTransition:
    def test_intermediate_states(self):
        stm = StateTransitionManager(
            initial_state=["J/psi(1S)"],
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
            assert len(reaction.transitions) == 4
        else:
            assert len(reaction.transitions) == 2
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
        if sys.version_info >= (3, 11) and not sys.version_info >= (3, 14):
            expected_hash = {
                "canonical-helicity": "75f6d331aceefda11d14e61bea24b076",
                "helicity": "9a5dad546caec7d2873ba7c745e8b321",
            }[reaction.formalism]
        elif sys.version_info >= (3, 14):
            expected_hash = {
                "canonical-helicity": "9e6b4b5ea854785ac33e6049b6ab86d1",
                "helicity": "a6880e15ca68c5d728574ffc3b5c59e4",
            }[reaction.formalism]
        else:
            expected_hash = {
                "canonical-helicity": "995925fb32a00be871211bf9d3ac78bc",
                "helicity": "1c1441c16cfbec426239b9c0e92d19c9",
            }[reaction.formalism]

        assert _compute_hash(reaction) == expected_hash


class TestStateTransitionManager:
    def test_allowed_intermediate_particles(self):
        stm = StateTransitionManager(
            initial_state=["J/psi(1S)"],
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
    if isinstance(obj, bytearray):
        return bytes(obj)
    if isinstance(obj, bytes):
        return obj
    return pickle.dumps(obj)
