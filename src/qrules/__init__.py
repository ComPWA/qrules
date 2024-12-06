"""A rule based system that facilitates particle reaction analysis.

QRules generates allowed particle transitions from a set of conservation rules and
boundary conditions as specified by the user. The user boundary conditions for a
particle reaction problem are for example the initial state, final state, and allowed
interactions.

The core of `qrules` computes which transitions (represented by a `.MutableTransition`)
are allowed between a certain initial and final state. Internally, the system propagates
the quantum numbers defined by the `particle` module through the `.MutableTransition`,
while satisfying the rules define by the :mod:`.conservation_rules` module. See
:doc:`/usage/reaction` and :doc:`/usage/particle`.

Finally, the `.io` module provides tools that can read and write the objects of this
framework.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import attrs

from qrules import io
from qrules.combinatorics import (
    InitialFacts,
    StateDefinition,
    StateDefinitionInput,
    create_initial_facts,
)
from qrules.conservation_rules import (
    BaryonNumberConservation,
    BottomnessConservation,
    ChargeConservation,
    CharmConservation,
    ElectronLNConservation,
    GraphElementRule,
    MassConservation,
    MuonLNConservation,
    StrangenessConservation,
    TauLNConservation,
    c_parity_conservation,
    clebsch_gordan_helicity_to_canonical,
    g_parity_conservation,
    gellmann_nishijima,
    identical_particle_symmetrization,
    isospin_conservation,
    isospin_validity,
    parity_conservation,
    spin_magnitude_conservation,
)
from qrules.particle import ParticleCollection, load_pdg
from qrules.quantum_numbers import InteractionProperties
from qrules.settings import (
    ADDITIONAL_PARTICLES_DEFINITIONS_PATH,
    InteractionType,
    _halves_domain,
    _int_domain,
)
from qrules.solving import NodeSettings, QNResult, Rule, validate_full_solution
from qrules.topology import MutableTransition, create_n_body_topology
from qrules.transition import (
    EdgeSettings,
    ProblemSet,
    ReactionInfo,
    SpinFormalism,
    StateTransitionManager,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def check_reaction_violations(  # noqa: C901, PLR0917
    initial_state: StateDefinition | Sequence[StateDefinition],
    final_state: Sequence[StateDefinition],
    mass_conservation_factor: float | None = 3.0,
    particle_db: ParticleCollection | None = None,
    max_angular_momentum: int = 1,
    max_spin_magnitude: float = 2,
) -> set[frozenset[str]]:
    """Determine violated interaction rules for a given particle reaction.

    .. warning:: This function only guarantees to find P, C and G parity
      violations, if it's a two body decay. If all initial and final states
      have the C/G parity defined, then these violations are also determined
      correctly.

    Args:
      initial_state: Shortform description of the initial state w/o spin
        projections.
      final_state: Shortform description of the final state w/o spin
        projections.
      mass_conservation_factor: Factor with which the width is multiplied when
        checking for `.MassConservation`. Set to `None` in order to deactivate mass
        conservation.
      particle_db (Optional): Custom `.ParticleCollection` object.  Defaults to
        the `.ParticleCollection` returned by `.load_pdg`.
      max_angular_momentum: Maximum angular momentum over which to generate
        :math:`LS`-couplings.
      max_spin_magnitude: Maximum spin magnitude over which to generate
        :math:`LS`-couplings.

    Returns:
      Set of least violating rules. The set can have multiple entries, as several
      quantum numbers can be violated. Each entry in the frozenset represents a group of
      rules that together violate all possible quantum number configurations.

    Example:
        >>> import qrules
        >>> qrules.check_reaction_violations(
        ...     initial_state="pi0",
        ...     final_state=["gamma", "gamma", "gamma"],
        ... )
        {frozenset({'c_parity_conservation'})}

    .. seealso:: :ref:`usage:Check allowed reactions`
    """
    if not isinstance(initial_state, (list, tuple)):
        initial_state = [initial_state]  # type: ignore[list-item]

    if particle_db is None:
        particle_db = load_pdg()

    def _check_violations(
        facts: InitialFacts,
        node_rules: dict[int, set[Rule]],
        edge_rules: dict[int, set[GraphElementRule]],
    ) -> QNResult:
        problem_set = ProblemSet(
            topology=topology,
            initial_facts=facts,
            solving_settings=MutableTransition(
                facts.topology,
                interactions={
                    i: NodeSettings(conservation_rules=rules)  # type: ignore[misc]
                    for i, rules in node_rules.items()
                },
                states={
                    i: EdgeSettings(conservation_rules=rules)  # type: ignore[misc]
                    for i, rules in edge_rules.items()
                },
            ),
        )
        return validate_full_solution(problem_set.to_qn_problem_set())

    def check_pure_edge_rules() -> None:
        pure_edge_rules: set[GraphElementRule] = {
            gellmann_nishijima,
            isospin_validity,
        }

        edge_check_result = _check_violations(
            initial_facts[0],
            node_rules={},
            edge_rules=dict.fromkeys(
                topology.incoming_edge_ids | topology.outgoing_edge_ids, pure_edge_rules
            ),
        )

        if edge_check_result.violated_edge_rules:
            msg = f"Some edges violate {edge_check_result.violated_edge_rules.values()}"
            raise ValueError(msg)

    def check_edge_qn_conservation() -> set[frozenset[str]]:
        """Check if edge quantum numbers are conserved.

        Those rules give the same results, independent on the node and spin props. Note
        they are also independent of the topology and hence their results are always
        correct.
        """
        edge_qn_conservation_rules: set[Rule] = {
            BaryonNumberConservation(),  # type: ignore[abstract]
            BottomnessConservation(),  # type: ignore[abstract]
            ChargeConservation(),  # type: ignore[abstract]
            CharmConservation(),  # type: ignore[abstract]
            ElectronLNConservation(),  # type: ignore[abstract]
            MuonLNConservation(),  # type: ignore[abstract]
            StrangenessConservation(),  # type: ignore[abstract]
            TauLNConservation(),  # type: ignore[abstract]
            isospin_conservation,
        }
        if len(initial_state) == 1 and mass_conservation_factor is not None:
            edge_qn_conservation_rules.add(MassConservation(mass_conservation_factor))

        return {
            frozenset((x,))
            for x in _check_violations(
                initial_facts[0],
                node_rules=dict.fromkeys(topology.nodes, edge_qn_conservation_rules),
                edge_rules={},
            ).violated_node_rules[node_id]
        }

    # Using a n-body topology is enough, to determine the violations reliably
    # since only certain spin rules require the isobar model. These spin rules
    # are not required here though.
    topology = create_n_body_topology(len(initial_state), len(final_state))
    node_id = next(iter(topology.nodes))

    initial_facts = create_initial_facts(
        topology, initial_state, final_state, particle_db
    )

    check_pure_edge_rules()
    violations = check_edge_qn_conservation()

    # Create combinations of graphs for magnitudes of S and L, but only
    # if it is a two body reaction
    ls_combinations = [
        InteractionProperties(l_magnitude=l_magnitude, s_magnitude=s_magnitude)
        for l_magnitude, s_magnitude in product(
            _int_domain(0, max_angular_momentum),
            _halves_domain(0, max_spin_magnitude),
        )
    ]

    initial_facts_list = []
    for ls_combi in ls_combinations:
        for facts_combination in initial_facts:
            new_facts = attrs.evolve(
                facts_combination,  # type: ignore[arg-type]
                interactions={node_id: ls_combi},  # type: ignore[dict-item]
            )
            initial_facts_list.append(new_facts)

    # Verify each graph with the interaction rules.
    # Spin projection rules are skipped as they can only be checked reliably
    # for a isobar topology (too difficult to solve)
    conservation_rules: dict[int, set[Rule]] = {
        node_id: {
            c_parity_conservation,
            clebsch_gordan_helicity_to_canonical,
            g_parity_conservation,
            parity_conservation,
            spin_magnitude_conservation,
            identical_particle_symmetrization,
        }
    }

    conservation_rule_violations: list[set[str]] = []
    for facts in initial_facts_list:
        rule_violations = _check_violations(
            facts=facts, node_rules=conservation_rules, edge_rules={}
        ).violated_node_rules[node_id]
        conservation_rule_violations.append(rule_violations)

    # first add rules which consistently fail
    common_ruleset = set(conservation_rule_violations[0])
    for rule_set in conservation_rule_violations[1:]:
        common_ruleset &= rule_set

    violations.update({frozenset((x,)) for x in common_ruleset})

    conservation_rule_violations = [
        x - common_ruleset for x in conservation_rule_violations
    ]

    # if there is not non-violated graph with the remaining violations then
    # the collection of violations also violate everything as a group.
    if all(map(len, conservation_rule_violations)):
        rule_group: set[str] = set()
        for graph_violations in conservation_rule_violations:
            rule_group.update(graph_violations)
        violations.add(frozenset(rule_group))

    return violations


def generate_transitions(  # noqa: PLR0917
    initial_state: StateDefinitionInput | Sequence[StateDefinitionInput],
    final_state: Sequence[StateDefinitionInput],
    allowed_intermediate_particles: list[str] | None = None,
    allowed_interaction_types: str | Iterable[str] | None = None,
    formalism: SpinFormalism = "canonical-helicity",
    particle_db: ParticleCollection | None = None,
    mass_conservation_factor: float | None = 3.0,
    max_angular_momentum: int = 2,
    max_spin_magnitude: float = 2,
    topology_building: str = "isobar",
    number_of_threads: int | None = None,
) -> ReactionInfo:
    """Generate allowed transitions between an initial and final state.

    Serves as a facade to the `.StateTransitionManager` (see :doc:`/usage/reaction`).

    Arguments:
        initial_state (list): A list of particle names in the initial
            state. You can specify spin projections for these particles with a `tuple`,
            e.g. :code:`("J/psi(1S)", [-1, 0, +1])`. If spin projections are not
            specified, all projections are taken, so the example here would be
            equivalent to :code:`"J/psi(1S)"`.

        final_state (list): Same as :code:`initial_state`, but for final state
            particles.

        allowed_intermediate_particles (`list`, optional): A list of particle
            states that you want to allow as intermediate states. This helps (1) filter
            out resonances and (2) speed up computation time.

        allowed_interaction_types: Interaction types you want to consider. For
            instance, :code:`["s", "em"]` results in `~.InteractionType.EM` and
            `~.InteractionType.STRONG` and :code:`["strong"]` results in
            `~.InteractionType.STRONG`.

        formalism (`str`, optional): Formalism that you intend to use in
            the eventual amplitude model.

        particle_db (`.ParticleCollection`, optional): The particles that you
            want to be involved in the reaction. Uses `.load_pdg` by default. It's
            better to use a subset for larger reactions, because of the computation
            times. This argument is especially useful when you want to use your own
            particle definitions (see :doc:`/usage/particle`).

        mass_conservation_factor: Width factor that is taken into account for
            for the `.MassConservation` rule.

        max_angular_momentum: Maximum angular momentum over which to generate
            angular momenta.

        max_spin_magnitude: Maximum spin magnitude over which to generate
            spins.

        topology_building (str): Technique with which to build the `.Topology`
            instances. Allowed values are:

            - :code:`"isobar"`: Isobar model (each state decays into two states)
            - :code:`"nbody"`: Use one central node and connect initial and final
              states to it

        number_of_threads: Number of cores with which to compute the allowed
            transitions. Defaults to the current value returned by
            :meth:`.settings.NumberOfThreads.get`.

    An example (where, for illustrative purposes only, we specify all arguments) would
    be:

    >>> import qrules
    >>> reaction = qrules.generate_transitions(
    ...     initial_state="D0",
    ...     final_state=["K~0", "K+", "K-"],
    ...     allowed_intermediate_particles=["a(0)(980)", "a(2)(1320)-"],
    ...     allowed_interaction_types=["e", "w"],
    ...     formalism="helicity",
    ...     particle_db=qrules.load_pdg(),
    ...     topology_building="isobar",
    ... )
    >>> len(reaction.transitions)
    4
    >>> len(reaction.group_by_topology())
    3
    """
    if isinstance(initial_state, str) or (
        isinstance(initial_state, tuple)
        and len(initial_state) == 2
        and isinstance(initial_state[0], str)
    ):
        initial_state = [initial_state]  # type: ignore[list-item]
    stm = StateTransitionManager(
        initial_state=initial_state,  # type: ignore[arg-type]
        final_state=final_state,
        particle_db=particle_db,
        allowed_intermediate_particles=allowed_intermediate_particles,
        formalism=formalism,
        mass_conservation_factor=mass_conservation_factor,
        max_angular_momentum=max_angular_momentum,
        max_spin_magnitude=max_spin_magnitude,
        topology_building=topology_building,
        number_of_threads=number_of_threads,
    )
    if allowed_interaction_types is not None:
        if isinstance(allowed_interaction_types, str):
            interaction_types = [InteractionType.from_str(allowed_interaction_types)]
        else:
            interaction_types = [
                InteractionType.from_str(description)
                for description in allowed_interaction_types
            ]
        stm.set_allowed_interaction_types(list(interaction_types))
    problem_sets = stm.create_problem_sets()
    return stm.find_solutions(problem_sets)


def load_default_particles() -> ParticleCollection:
    """Load the default particle list that comes with `qrules`.

    Runs `.load_pdg` and supplements its output definitions from the file
    :download:`additional_definitions.yml
    </../src/qrules/additional_definitions.yml>`.
    """
    particle_db = load_pdg()
    additional_particles = io.load(ADDITIONAL_PARTICLES_DEFINITIONS_PATH)
    if not isinstance(additional_particles, ParticleCollection):
        msg = (
            f"Object loaded from {ADDITIONAL_PARTICLES_DEFINITIONS_PATH} is not a"
            f" {ParticleCollection.__name__}, but a"
            f" {type(additional_particles).__name__}"
        )
        raise TypeError(msg)
    particle_db.update(additional_particles)
    return particle_db
