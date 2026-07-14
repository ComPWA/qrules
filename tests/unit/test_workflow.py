import json
from fractions import Fraction
from typing import Any

import attrs
import pytest

from qrules.conservation_rules import helicity_conservation, spin_validity
from qrules.io import asdict, asdot
from qrules.particle import ParticleCollection, load_pdg
from qrules.quantum_numbers import EdgeQuantumNumbers
from qrules.settings import (
    CONSERVATION_LAW_PRIORITIES,
    DEFAULT_INTERACTION_TYPES,
    EDGE_RULE_PRIORITIES,
    InteractionType,
    create_interaction_settings,
)
from qrules.solving import QNProblemSet
from qrules.topology import MutableTransition
from qrules.transition import ReactionInfo
from qrules.workflow import (
    InteractionConfig,
    QNProblemSetCollection,
    QNReactionInfo,
    create_qn_problem_sets,
    filter_intermediate_particles,
    find_qn_transitions,
    find_solutions,
    generate_qn_transitions,
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
            type_settings=create_interaction_settings(particle_db=particle_database)
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


def test_pipeline_reproduces_state_transition_manager(reaction: ReactionInfo):
    particle_db = load_pdg()
    qn_problem_sets = create_qn_problem_sets(
        initial_state=["J/psi(1S)"],
        final_state=["gamma", "pi0", "pi0"],
        particle_db=particle_db,
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        interaction_config=InteractionConfig(
            type_settings=create_interaction_settings(
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


def test_qn_transitions_are_projection_free():
    particle_db = load_pdg()
    collection = create_qn_problem_sets(
        initial_state=["J/psi(1S)"],
        final_state=["gamma", "pi0", "pi0"],
        particle_db=particle_db,
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        interaction_config=InteractionConfig(
            type_settings=create_interaction_settings(
                particle_db=particle_db, max_angular_momentum=2
            ),
            allowed_types=[InteractionType.STRONG],
        ),
    )
    qn_transitions = find_qn_transitions(collection)
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


def test_spin_projections_reenabled_through_problem_sets():
    """Spin projections remain available as an extension of the QN problem sets.

    Spin projections are no longer part of the default workflow, but the `.CSPSolver`
    handles any quantum number that the problem sets declare through facts and
    domains. This test re-enables them for :math:`J/\\psi \\to \\gamma f_2(1270)` by
    adding `~.EdgeQuantumNumbers.spin_projection` facts to the external edges (a
    `list` fact is solved as a variable over that range), a projection domain plus
    `.spin_validity` to the intermediate edges, and `.helicity_conservation` to the
    interaction nodes, which prunes the helicity combinations to
    :math:`|\\lambda_\\gamma - \\lambda_{f_2}| \\leq 1`.
    """
    particle_db = load_pdg()
    collection = create_qn_problem_sets(
        initial_state=["J/psi(1S)"],
        final_state=["gamma", "pi0", "pi0"],
        particle_db=particle_db,
        allowed_intermediate_particles=["f(2)(1270)"],
        allowed_interaction_types=["strong", "EM"],
        max_angular_momentum=2,
        final_state_groupings=[[["pi0", "pi0"]]],
    )
    spin_projections: dict[str, Fraction | list[Fraction]] = {
        "J/psi(1S)": [Fraction(-1), Fraction(+1)],  # from e+e- collision
        "gamma": [Fraction(-1), Fraction(+1)],
        "pi0": Fraction(0),
    }

    def add_spin_projections(problem_set: QNProblemSet) -> QNProblemSet:
        facts = problem_set.initial_facts
        new_states = {
            edge_id: {
                **prop_map,
                EdgeQuantumNumbers.spin_projection: spin_projections[
                    particle_db.find(int(prop_map[EdgeQuantumNumbers.pid])).name
                ],
            }
            for edge_id, prop_map in facts.states.items()
        }
        new_facts = MutableTransition(
            facts.topology,
            new_states,  # type: ignore[arg-type]
            dict(facts.interactions),  # type: ignore[arg-type]
        )
        settings = problem_set.solving_settings
        new_edge_settings = {}
        for edge_id, edge_settings in settings.states.items():
            if edge_id not in facts.topology.intermediate_edge_ids:
                new_edge_settings[edge_id] = edge_settings
                continue
            max_spin = max(edge_settings.qn_domains[EdgeQuantumNumbers.spin_magnitude])
            projection_domain = [
                Fraction(x, 2) for x in range(-int(2 * max_spin), int(2 * max_spin) + 1)
            ]
            new_edge_settings[edge_id] = attrs.evolve(
                edge_settings,
                conservation_rules={
                    **edge_settings.conservation_rules,
                    spin_validity: EDGE_RULE_PRIORITIES[spin_validity],
                },
                qn_domains={
                    **edge_settings.qn_domains,
                    EdgeQuantumNumbers.spin_projection: projection_domain,
                },
            )
        new_node_settings = {
            node_id: attrs.evolve(
                node_settings,
                conservation_rules={
                    **node_settings.conservation_rules,
                    helicity_conservation: CONSERVATION_LAW_PRIORITIES[
                        helicity_conservation
                    ],
                },
            )
            for node_id, node_settings in settings.interactions.items()
        }
        new_settings = MutableTransition(
            settings.topology,
            new_edge_settings,  # type: ignore[arg-type]
            new_node_settings,  # type: ignore[arg-type]
        )
        return QNProblemSet(initial_facts=new_facts, solving_settings=new_settings)  # type: ignore[arg-type]

    collection.problem_sets = {
        strength: [add_spin_projections(p) for p in problem_sets]
        for strength, problem_sets in collection.problem_sets.items()
    }
    qn_transitions = find_qn_transitions(collection)
    assert len(qn_transitions) == 36
    assert all(
        EdgeQuantumNumbers.spin_projection in state
        for transition in qn_transitions
        for state in transition.states.values()
    )

    def get_projection(transition, edge_id: int) -> Fraction:
        return transition.states[edge_id][EdgeQuantumNumbers.spin_projection]

    helicity_combinations = set()
    for transition in qn_transitions:
        topology = transition.topology
        resonance_edge = next(iter(topology.intermediate_edge_ids))
        gamma_edge = next(
            i
            for i in topology.outgoing_edge_ids
            if transition.states[i][EdgeQuantumNumbers.pid] == 22
        )
        helicity_combinations.add((
            get_projection(transition, gamma_edge),
            get_projection(transition, resonance_edge),
        ))
    assert helicity_combinations == {
        (Fraction(-1), Fraction(-2)),
        (Fraction(-1), Fraction(-1)),
        (Fraction(-1), Fraction(0)),
        (Fraction(+1), Fraction(0)),
        (Fraction(+1), Fraction(+1)),
        (Fraction(+1), Fraction(+2)),
    }
    initial_state_projections = {
        get_projection(transition, edge_id)
        for transition in qn_transitions
        for edge_id in transition.topology.incoming_edge_ids
    }
    assert initial_state_projections == {Fraction(-1), Fraction(+1)}


def test_generate_qn_transitions():
    particle_db = load_pdg()
    reaction = generate_qn_transitions(
        initial_state="J/psi(1S)",
        final_state=["gamma", "pi0", "pi0"],
        particle_db=particle_db,
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        allowed_interaction_types=["strong", "em"],
    )
    assert isinstance(reaction, QNReactionInfo)
    assert len(reaction.transitions) > 0
    assert {p.name for p in reaction.initial_state.values()} == {"J/psi(1S)"}
    assert [p.name for _, p in sorted(reaction.final_state.items())] == [
        "gamma",
        "pi0",
        "pi0",
    ]
    for qn_set in reaction.get_intermediate_quantum_numbers():
        assert qn_set[EdgeQuantumNumbers.spin_magnitude] == 0
        assert qn_set[EdgeQuantumNumbers.parity] == +1
        assert qn_set[EdgeQuantumNumbers.c_parity] == +1
    assert len(reaction.group_by_topology()) == 1

    dot = asdot(reaction)
    assert dot.startswith("digraph {")
    assert "J/psi(1S)" in dot
    assert "spin_projection" not in dot.replace("isospin_projection", "")

    collapsed_dot = asdot(reaction, collapse_graphs=True)
    assert "0⁺(0⁺⁺)" in collapsed_dot


def test_generate_qn_transitions_two_to_n():
    """Production reactions with two initial states (ComPWA/qrules#29)."""
    particle_db = load_pdg()
    reaction = generate_qn_transitions(
        initial_state=["gamma", "p"],
        final_state=["p", "pi0"],
        particle_db=particle_db,
        allowed_intermediate_particles=[
            "Delta(1232)",
            "N(1440)",
            "rho(770)",
            "omega(782)",
        ],
        allowed_interaction_types=["strong", "em"],
    )
    assert len(reaction.transitions) > 0
    assert {p.name for p in reaction.initial_state.values()} == {"gamma", "p"}
    assert {p.name for p in reaction.final_state.values()} == {"p", "pi0"}
    intermediate_signatures = {
        (
            state[EdgeQuantumNumbers.baryon_number],
            state[EdgeQuantumNumbers.spin_magnitude],
        )
        for transition in reaction.transitions
        for state in transition.intermediate_states.values()
    }
    assert (1, Fraction(3, 2)) in intermediate_signatures  # s-channel Delta(1232)
    assert (0, Fraction(1)) in intermediate_signatures  # t-channel vector exchange
    assert len(reaction.group_by_topology()) > 1


def test_generate_qn_transitions_without_ls_couplings():
    """LS-free solving must allow the same intermediate states (ComPWA/qrules#19)."""
    particle_db = load_pdg()
    reaction_kwargs: dict[str, Any] = dict(
        initial_state=["gamma", "p"],
        final_state=["p", "pi0"],
        particle_db=particle_db,
        allowed_intermediate_particles=[
            "Delta(1232)",
            "N(1440)",
            "rho(770)",
            "omega(782)",
        ],
        allowed_interaction_types=["strong", "em"],
        max_angular_momentum=2,
    )
    signatures_by_mode = {}
    n_transitions_by_mode = {}
    for ls_couplings in (True, False):
        reaction = generate_qn_transitions(ls_couplings=ls_couplings, **reaction_kwargs)
        n_transitions_by_mode[ls_couplings] = len(reaction.transitions)
        signatures_by_mode[ls_couplings] = {
            (
                state[EdgeQuantumNumbers.spin_magnitude],
                state[EdgeQuantumNumbers.parity],
            )
            for transition in reaction.transitions
            for state in transition.intermediate_states.values()
        }
        if not ls_couplings:
            for transition in reaction.transitions:
                for interactions in transition.interactions.values():
                    assert not interactions
    assert signatures_by_mode[False] == signatures_by_mode[True]
    assert n_transitions_by_mode[False] < n_transitions_by_mode[True]


def test_group_by_channel_and_channel_selection():
    """Mandelstam channel encoding for 2-to-n reactions (ComPWA/qrules#29)."""
    particle_db = load_pdg()
    reaction_kwargs: dict[str, Any] = dict(
        initial_state=["gamma", "p"],
        final_state=["pi0", "p"],
        particle_db=particle_db,
        allowed_intermediate_particles=[
            "Delta(1232)",
            "N(1440)",
            "rho(770)",
            "omega(782)",
        ],
        allowed_interaction_types=["strong", "em"],
    )
    reaction = generate_qn_transitions(**reaction_kwargs)
    channels = reaction.group_by_channel()
    assert sorted(channels) == ["s", "t", "u"]
    assert sum(map(len, channels.values())) == len(reaction.transitions)

    def get_baryon_numbers(channel: str) -> set:
        return {
            state[EdgeQuantumNumbers.baryon_number]
            for transition in channels[channel]
            for state in transition.intermediate_states.values()
        }

    assert get_baryon_numbers("s") == {+1}  # Delta/N* resonances
    assert get_baryon_numbers("t") == {0}  # meson exchange
    assert get_baryon_numbers("u") == {-1, +1}  # baryon exchange

    t_channel_only = generate_qn_transitions(**reaction_kwargs, allowed_channels="t")
    assert sorted(t_channel_only.group_by_channel()) == ["t"]
    assert len(t_channel_only.transitions) == len(channels["t"])

    with pytest.raises(ValueError, match="Invalid Mandelstam channel 'x'"):
        generate_qn_transitions(**reaction_kwargs, allowed_channels=["x"])


def test_qn_reaction_info_requires_particle_states():
    particle_db = load_pdg()
    qn_problem_sets = create_qn_problem_sets(
        initial_state=["J/psi(1S)"],
        final_state=["gamma", "pi0", "pi0"],
        particle_db=particle_db,
        allowed_intermediate_particles=["f(0)(980)"],
    )
    qn_transitions = find_qn_transitions(qn_problem_sets)
    with pytest.raises(TypeError, match="is of type FrozenDict, not Particle"):
        QNReactionInfo(qn_transitions)
