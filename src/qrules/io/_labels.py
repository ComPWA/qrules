from __future__ import annotations

import logging
import re
from fractions import Fraction
from functools import singledispatch
from inspect import isfunction
from typing import TYPE_CHECKING, Any, cast

import attrs

from qrules.particle import Particle, ParticleWithSpin, Spin, _render_fraction
from qrules.quantum_numbers import InteractionProperties
from qrules.solving import EdgeSettings, NodeSettings, QNProblemSet
from qrules.topology import FrozenTransition, MutableTransition, Topology, Transition
from qrules.transition import ProblemSet, State

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qrules.argument_handling import Rule

_LOGGER = logging.getLogger(__name__)


def create_edge_label(
    graph: ProblemSet | QNProblemSet | Topology | Transition,
    edge_id: int,
    render_edge_id: bool,
) -> str:
    if isinstance(graph, Topology):
        if render_edge_id:
            return str(edge_id)
        return ""
    if isinstance(graph, (ProblemSet, QNProblemSet)):
        edge_setting = graph.solving_settings.states.get(edge_id)
        initial_fact = graph.initial_facts.states.get(edge_id)
        edge_property: EdgeSettings | ParticleWithSpin | None = None
        if edge_setting:
            edge_property = edge_setting
        if initial_fact:
            edge_property = initial_fact  # type: ignore[assignment]
        return __render_edge_with_id(edge_id, edge_property, render_edge_id)
    edge_prop = graph.states.get(edge_id)
    return __render_edge_with_id(edge_id, edge_prop, render_edge_id)


def __render_edge_with_id(edge_id: int, edge_prop: Any, render_edge_id: bool) -> str:
    if edge_prop is None or not edge_prop:
        return str(edge_id)
    edge_label = as_string(edge_prop)
    if not render_edge_id:
        return edge_label
    if "\n" in edge_label:
        return f"{edge_id}:\n{edge_label}"
    return f"{edge_id}: {edge_label}"


@singledispatch
def as_string(obj: Any) -> str:
    """Render an edge or node property on a `.Transition` as a `str`.

    This function is decorated with :func:`functools.singledispatch`, which means that
    you can easily register other converter functions. An example:

    >>> from qrules.io._labels import as_string
    >>> as_string(10)
    '10'
    >>> _ = as_string.register(int, lambda _: "new int rendering")
    >>> as_string(10)
    'new int rendering'
    """
    if obj is not None:
        _LOGGER.warning(f"No label renderer implemented type {type(obj).__name__}")
    return str(obj)


@as_string.register(int)
@as_string.register(float)
@as_string.register(str)
def _(obj: Any) -> str:
    return str(obj)


@as_string.register(dict)
def _(obj: dict) -> str:
    lines = []
    for key, value in obj.items():
        if isinstance(key, type) or callable(key):
            key_repr = key.__name__
        else:
            key_repr = key
        if not value and not key_repr.endswith(("magnitude", "projection")):
            continue
        value_repr = __render_key_and_value(key_repr, value)
        lines.append(f"{key_repr} = {value_repr}")
    return "\n".join(lines)


def __render_key_and_value(key: str, value: Any) -> str:
    if isinstance(value, (Fraction, int)):
        fraction = Fraction(value)
        no_pm = key.endswith("magnitude") or key == "pid"
        return _render_fraction(fraction, plusminus=not no_pm)
    return as_string(value)


@as_string.register(InteractionProperties)
def _(obj: InteractionProperties) -> str:
    lines = []
    if obj.l_magnitude is not None:
        if obj.l_projection is None:
            l_label = _render_fraction(Fraction(obj.l_magnitude))
        else:
            l_label = _spin_to_str(Spin(obj.l_magnitude, obj.l_projection))
        lines.append(f"L={l_label}")
    if obj.s_magnitude is not None:
        if obj.s_projection is None:
            s_label = _render_fraction(Fraction(obj.s_magnitude))
        else:
            s_label = _spin_to_str(Spin(obj.s_magnitude, obj.s_projection))
        lines.append(f"S={s_label}")
    if obj.parity_prefactor is not None:
        label = _render_fraction(Fraction(obj.parity_prefactor), plusminus=True)
        lines.append(f"P={label}")
    return "\n".join(lines)


@as_string.register(EdgeSettings)
@as_string.register(NodeSettings)
def _(settings: EdgeSettings | NodeSettings) -> str:
    output = ""
    if settings.rule_priorities:
        output += "RULES\n"
        rule_descriptions = (
            f"{__render_rule(rule)} - {__get_priority(rule, settings.rule_priorities)}"
            for rule in settings.conservation_rules
        )
        sorted_names = sorted(rule_descriptions, key=__extract_priority, reverse=True)
        output += "\n".join(sorted_names)
    if settings.qn_domains:
        if output:
            output += "\n"
        domains = sorted(
            f"{qn.__name__} ∊ {__render_domain(domain, key=qn.__name__)}"
            for qn, domain in settings.qn_domains.items()
        )
        output += "DOMAINS\n"
        output += "\n".join(domains)
    return output


def __get_priority(rule: Any, rule_priorities: dict[Any, int]) -> int | str:
    rule_type = __get_type(rule)
    return rule_priorities.get(rule_type, "NA")


def __render_rule(rule: Rule) -> str:
    return __get_type(rule).__name__


def __get_type(rule: Rule) -> type[Rule]:
    if isfunction(rule):
        return rule  # type: ignore[return-value]
    return type(rule)


def __extract_priority(description: str) -> int | float:
    """Get the priority from a rule description, as rendered by `.as_string`.

    Rules without a priority (``"NA"``) rank below any numeric priority.

    >>> descriptions = ["a - 9", "b - 10", "c - NA", "d - -1"]
    >>> sorted(descriptions, key=__extract_priority, reverse=True)
    ['b - 10', 'a - 9', 'd - -1', 'c - NA']
    """
    matches = re.match(r".* - (-?[0-9]+|NA)$", description)
    if matches is None:
        msg = f"{description} does not contain a priority number"
        raise ValueError(msg)
    priority = matches[1]
    if priority == "NA":
        return float("-inf")
    return int(priority)


def __render_domain(domain: list[Any], key: str) -> str:
    """Render a domain as a `str`.

    >>> half = Fraction(0.5)
    >>> __render_domain([-half, +half], key="spin_projection")
    '[-1/2, +1/2]'
    >>> __render_domain([0, 1], key="l_magnitude")
    '[0, 1]'
    >>> __render_domain([None, +1, -1], key="parity")
    '[-1, +1, None]'
    """
    domain = sorted(domain, key=lambda x: +9999 if x is None else x)
    domain_str = [__render_key_and_value(key, x) for x in domain]
    return "[" + ", ".join(domain_str) + "]"


@as_string.register(Particle)
def _(particle: Particle) -> str:
    return particle.name


@as_string.register(Spin)
def _spin_to_str(spin: Spin) -> str:
    spin_magnitude = _render_fraction(spin.magnitude)
    spin_projection = _render_fraction(spin.projection, plusminus=True)
    return f"|{spin_magnitude},{spin_projection}⟩"


@as_string.register(State)
def _state_to_str(state: State) -> str:
    particle = state.particle.name
    spin_projection = _render_fraction(state.spin_projection, plusminus=True)
    return f"{particle}[{spin_projection}]"


@as_string.register(tuple)
def _(obj: tuple) -> str:
    if len(obj) == 2:
        if isinstance(obj[0], Particle) and isinstance(obj[1], (Fraction, float, int)):
            state = State(*obj)
            return _state_to_str(state)
        if all(isinstance(o, (Fraction, float, int)) for o in obj):
            spin = Spin(*obj)
            return _spin_to_str(spin)
    return "\n".join(map(as_string, obj))


def get_particle_graphs(
    graphs: Iterable[Transition[ParticleWithSpin, InteractionProperties]],
) -> list[FrozenTransition[Particle, None]]:
    """Strip `list` of `.Transition` s of the spin projections.

    Extract a `list` of `.Transition` instances with only `.Particle` instances on the
    edges.

    .. seealso:: :doc:`/usage/visualize`
    """
    inventory = set()
    for transition in graphs:
        if isinstance(transition, FrozenTransition):
            transition = transition.convert(lambda s: (s.particle, s.spin_projection))
        stripped_transition = strip_projections(transition)
        topology = stripped_transition.topology
        particle_transition: FrozenTransition[Particle, None] = FrozenTransition(
            stripped_transition.topology,
            states=stripped_transition.states,
            interactions=dict.fromkeys(topology.nodes),
        )
        inventory.add(particle_transition)
    return sorted(
        inventory,
        key=lambda g: [g.states[i].mass for i in g.topology.intermediate_edge_ids],
    )


def strip_projections(
    graph: Transition[Any, InteractionProperties],
) -> FrozenTransition[Particle, InteractionProperties]:
    transition = cast("FrozenTransition[Any, InteractionProperties]", graph)
    return transition.convert(
        state_converter=__to_particle,
        interaction_converter=lambda i: attrs.evolve(
            i, l_projection=None, s_projection=None
        ),
    )


def __to_particle(state: Any) -> Particle:
    if isinstance(state, State):
        return state.particle
    if isinstance(state, tuple) and len(state) == 2:
        return state[0]
    msg = f"Cannot extract a particle from type {type(state).__name__}"
    raise NotImplementedError(msg)


def collapse_graphs(
    graphs: Iterable[Transition[Any, Any]],
) -> list[FrozenTransition[tuple, None]]:
    graphs = list(graphs)
    transition_groups: dict[Topology, MutableTransition[set[Particle], None]] = {
        g.topology: MutableTransition(
            g.topology,
            states={i: set() for i in g.topology.edges},
            interactions=dict.fromkeys(g.topology.nodes),
        )
        for g in graphs
    }
    for transition in graphs:
        topology = transition.topology
        group = transition_groups[topology]
        for state_id, state in transition.states.items():
            group.states[state_id].add(_strip_properties(state))
    collected_graphs: list[FrozenTransition[tuple[Particle, ...], None]] = []
    for topology in sorted(transition_groups):
        group = transition_groups[topology]
        collected_graphs.append(
            FrozenTransition(
                topology,
                states={
                    i: tuple(sorted(particles, key=_sorting_key))
                    for i, particles in group.states.items()
                },
                interactions=group.interactions,
            )
        )
    return collected_graphs


def _strip_properties(state: Any) -> Any:
    if isinstance(state, State):
        return state.particle
    return state


def _sorting_key(obj: Any) -> Any:
    if isinstance(obj, State):
        return obj.particle.name
    if isinstance(obj, str):
        return obj.lower()
    return obj
