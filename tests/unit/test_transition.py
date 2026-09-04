import hashlib
import pickle  # ruff: ignore[suspicious-pickle-import]
import sys
from copy import deepcopy
from fractions import Fraction

import pytest
from IPython.lib.pretty import pretty

from qrules.particle import Parity, Particle, ParticleCollection, Spin  # ruff: ignore[unused-import]
from qrules.quantum_numbers import InteractionProperties  # ruff: ignore[unused-import]
from qrules.settings import InteractionType
from qrules.topology import (  # ruff: ignore[unused-import]
    Edge,
    FrozenDict,
    FrozenTransition,
    MutableTransition,
    Topology,
)
from qrules.transition import ReactionInfo, SolvingMode, State, StateTransitionManager

NAMESPACE_WITH_FRACTIONS = globals()
NAMESPACE_WITH_FRACTIONS["Fraction"] = Fraction


def describe_MutableTransition():
    def it_intermediate_states():
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


def describe_ReactionInfo():
    def it_properties(reaction: ReactionInfo):
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
    def it_repr(repr_method, reaction: ReactionInfo):
        instance = reaction
        from_repr = eval(repr_method(instance), NAMESPACE_WITH_FRACTIONS)
        assert from_repr == instance

    def it_hash(reaction: ReactionInfo):
        assert hash(deepcopy(reaction)) == hash(reaction)

    def it_hash_value(reaction: ReactionInfo):
        if sys.version_info >= (3, 11) and not sys.version_info >= (3, 14):
            expected_hash = {
                "canonical-helicity": "65106a44301f9340e633d09f66ad7d17",
                "helicity": "9646d3ee5c5e8534deb8019435161f2e",
            }[reaction.formalism]
        elif sys.version_info >= (3, 14):
            expected_hash = {
                "canonical-helicity": "762cc006a8c4c0a0a88fce934a32577d",
                "helicity": "17fefe55a7da0810371e90bd762a176a",
            }[reaction.formalism]
        else:
            expected_hash = {
                "canonical-helicity": "0d8bc378677986e0dc2d3b02f5627e0b",
                "helicity": "71404ad43550850a02109e8db044bd28",
            }[reaction.formalism]

        assert _compute_hash(reaction) == expected_hash


def describe_State():
    @pytest.mark.parametrize(
        ("state_def_1", "state_def_2"),
        [
            (("a", -1), ("a", +1)),
            (("a", 0), ("a", 0)),
            (("a", 0), ("b", 0)),
            (("a", -1), ("b", +1)),
        ],
    )
    def it_ordering(state_def_1, state_def_2):
        def create_state(state_def) -> State:
            return State(
                particle=Particle(name=state_def[0], pid=0, spin=0, mass=0),
                spin_projection=state_def[1],
            )

        state1 = create_state(state_def_1)
        state2 = create_state(state_def_2)
        assert state2 >= state1


def describe_StateTransitionManager():
    def it_allowed_intermediate_particles():
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

    @pytest.mark.parametrize(
        ("initial_state", "expected_strengths"),
        [
            (["gamma"], [0.0001, 1.0, 60.0]),
            (["nu(e)"], [1e-08, 0.0001, 0.006]),
        ],
    )
    def it_initial_state_restricts_interaction_types(
        initial_state: list[str], expected_strengths: list[float]
    ):
        stm = StateTransitionManager(initial_state, final_state=["pi0", "pi0", "pi0"])
        problem_sets = stm.create_problem_sets()
        assert sorted(problem_sets) == expected_strengths

    def it_fast_solving_mode():
        def count_transitions(solving_mode: SolvingMode) -> int:
            stm = StateTransitionManager(
                initial_state=["J/psi(1S)"],
                final_state=["gamma", "pi0", "pi0"],
                solving_mode=solving_mode,
            )
            reaction = stm.find_solutions(stm.create_problem_sets())
            return len(reaction.transitions)

        assert count_transitions(SolvingMode.FULL) == 294
        assert count_transitions(SolvingMode.FAST) == 90

    def it_regex_pattern():
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
    h = hashlib.md5(b)  # ruff: ignore[hashlib-insecure-hash-function]
    return h.hexdigest()


def _to_bytes(obj) -> bytes:
    if isinstance(obj, bytearray):
        return bytes(obj)
    if isinstance(obj, bytes):
        return obj
    return pickle.dumps(obj)
