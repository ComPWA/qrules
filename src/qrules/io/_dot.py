"""Generate dot sources.

See :doc:`/usage/visualize` for more info.
"""

from __future__ import annotations

import logging
import re
import string
from collections import abc
from functools import singledispatch
from inspect import isfunction
from numbers import Number
from typing import TYPE_CHECKING, Any, Iterable, cast

import attrs
from attrs import Attribute, define, field
from attrs.converters import default_if_none

from qrules.particle import Particle, ParticleWithSpin, Spin
from qrules.quantum_numbers import InteractionProperties, _to_fraction
from qrules.solving import EdgeSettings, NodeSettings, QNProblemSet, QNResult
from qrules.topology import FrozenTransition, MutableTransition, Topology, Transition
from qrules.transition import ProblemSet, ReactionInfo, State

if TYPE_CHECKING:
    from qrules.argument_handling import Rule

_LOGGER = logging.getLogger(__name__)


def _check_booleans(instance: GraphPrinter, attribute: Attribute, value: bool) -> None:
    if instance.strip_spin and instance.collapse_graphs:
        msg = "Cannot both strip spin and collapse graphs"
        raise ValueError(msg)
    if instance.collapse_graphs and instance.render_node:
        msg = "Collapsed graphs cannot be rendered with node properties"
        raise ValueError(msg)


def _create_default_figure_style(style: dict[str, Any] | None) -> dict[str, Any]:
    figure_style = {"bgcolor": None}
    if style is None:
        return figure_style
    figure_style.update(style)
    return figure_style


@define(on_setattr=_check_booleans)
class GraphPrinter:
    render_node: bool | None = None
    render_final_state_id: bool = True
    render_resonance_id: bool = False
    render_initial_state_id: bool = False
    strip_spin: bool = False
    collapse_graphs: bool = False

    figure_style: dict[str, Any] = field(
        converter=_create_default_figure_style, default=None
    )
    edge_style: dict[str, Any] = field(
        converter=default_if_none(factory=dict),  # type: ignore[misc]
        default=None,
    )
    node_style: dict[str, Any] = field(
        converter=default_if_none(factory=dict),  # type: ignore[misc]
        default=None,
    )
    indent: int = 4

    def __call__(self, obj: Any) -> str:
        lines = self._create_preface()
        lines += self._render(obj)
        indented_lines = [self.indent * " " + s for s in lines]
        dot = "digraph {\n"
        dot += "\n".join(indented_lines)
        dot += "\n}\n"
        return dot

    def _create_preface(self) -> list[str]:
        return [
            "rankdir=LR",
            "node [shape=none, width=0]",
            "edge [arrowhead=none]",
            *_create_graphviz_assignments(self.figure_style),
        ]

    def _render(self, obj: Any) -> list[str]:
        if isinstance(obj, QNResult):
            obj = obj.solutions
        if isinstance(obj, ReactionInfo):
            obj = obj.transitions
        if isinstance(obj, abc.Iterable):
            return self._render_multiple_transitions(obj)
        if isinstance(obj, (ProblemSet, QNProblemSet, Topology, Transition)):
            return self._render_transition(obj)
        msg = f"No DOT rendering for type {type(obj).__name__}"
        raise NotImplementedError(msg)

    def _render_multiple_transitions(self, obj: Iterable) -> list[str]:
        if self.collapse_graphs:
            transitions: list = _collapse_graphs(obj)
        elif self.strip_spin:
            if self.render_node:
                transitions = sorted({_strip_projections(t) for t in obj})
            else:
                transitions = _get_particle_graphs(obj)
        else:
            transitions = list(obj)
        lines = []
        for i, graph in enumerate(reversed(list(transitions))):
            lines += self._render_transition(graph, prefix=f"T{i}_")
        return lines

    def _render_transition(  # noqa: C901, PLR0912, PLR0915
        self,
        obj: ProblemSet | QNProblemSet | Topology | Transition,
        prefix: str = "",
    ) -> list[str]:
        lines: list[str] = []
        if isinstance(obj, tuple) and len(obj) == 2:
            topology: Topology = obj[0]
            rendered_graph: ProblemSet | QNProblemSet | Topology | Transition = obj[1]
        elif isinstance(obj, (ProblemSet, QNProblemSet, Transition)):
            rendered_graph = obj
            topology = obj.topology
        elif isinstance(obj, Topology):
            rendered_graph = obj
            topology = obj
        else:
            msg = f"Cannot render {type(obj).__name__} as dot"
            raise NotImplementedError(msg)
        for edge_id in topology.incoming_edge_ids | topology.outgoing_edge_ids:
            if edge_id in topology.incoming_edge_ids:
                render = self.render_initial_state_id
            else:
                render = self.render_final_state_id
            label = _create_edge_label(rendered_graph, edge_id, render)
            graphviz_node = prefix + _get_graphviz_node(edge_id)
            lines += [self._create_graphviz_node(graphviz_node, label, self.edge_style)]
        lines += [_create_same_rank_line(topology.incoming_edge_ids, prefix)]
        lines += [_create_same_rank_line(topology.outgoing_edge_ids, prefix)]
        for i, edge in topology.edges.items():
            j, k = edge.ending_node_id, edge.originating_node_id
            from_node = prefix + _get_graphviz_node(i, k)
            to_node = prefix + _get_graphviz_node(i, j)
            if j is None or k is None:
                lines += [self._create_graphviz_edge(from_node, to_node)]
            else:
                label = _create_edge_label(rendered_graph, i, self.render_resonance_id)
                lines += [self._create_graphviz_edge(from_node, to_node, label)]
        if isinstance(obj, (ProblemSet, QNProblemSet)):
            node_settings = obj.solving_settings.interactions
            for node_id, settings in node_settings.items():
                label = ""
                if self.render_node:
                    label = as_string(settings)
                node = f"{prefix}N{node_id}"
                lines += [self._create_graphviz_node(node, label, self.node_style)]
        if isinstance(obj, Transition):
            for node_id, node_prop in obj.interactions.items():
                label = ""
                if self.render_node:
                    label = as_string(node_prop)
                node = f"{prefix}N{node_id}"
                lines += [self._create_graphviz_node(node, label, self.node_style)]
        if isinstance(obj, Topology):
            render_node = self.render_node
            if render_node is None and len(topology.nodes) > 1:
                render_node = True
            for node_id in topology.nodes:
                label = ""
                if render_node:
                    label = f"({node_id})"
                node = f"{prefix}N{node_id}"
                lines += [self._create_graphviz_node(node, label, self.node_style)]
        return lines

    def _create_graphviz_edge(
        self, from_node: str, to_node: str, label: str = ""
    ) -> str:
        style = dict(self.edge_style)  # copy
        if "label" in style:
            del style["label"]
        if label:
            style["label"] = label
        styling = _create_graphviz_styling(style)
        return f"{from_node} -> {to_node}{styling}"

    @staticmethod
    def _create_graphviz_node(node: str, label: str, style: dict[str, Any]) -> str:
        style = dict(style)  # copy
        style["label"] = label
        styling = _create_graphviz_styling(style)
        return f"{node}{styling}"


def _create_graphviz_styling(graphviz_attrs: dict[str, Any]) -> str:
    """Create a `str` of Graphviz attribute assignments for a node or edge.

    See `Graphviz attributes <https://graphviz.org/doc/info/attrs.html>`_ for the
    assignment syntax.

    >>> _create_graphviz_styling({"size": 12})
    ' [size=12]'
    >>> _create_graphviz_styling({"color": "red", "size": 8})
    ' [color="red", size=8]'
    """
    if not graphviz_attrs:
        return ""
    assignments = _create_graphviz_assignments(graphviz_attrs)
    return f" [{', '.join(assignments)}]"


def _create_graphviz_assignments(graphviz_attrs: dict[str, Any]) -> list[str]:
    """Create a `list` of graphviz attribute assignments.

    See `Graphviz attributes <https://graphviz.org/doc/info/attrs.html>`_ for the
    assignment syntax.

    >>> _create_graphviz_assignments({"size": 12})
    ['size=12']
    >>> _create_graphviz_assignments({"color": "red", "size": 8})
    ['color="red"', 'size=8']
    >>> _create_graphviz_assignments({"shape": None})
    ['shape=none']
    """
    items = []
    for key, value in graphviz_attrs.items():
        if value is None:
            value = "none"
        elif isinstance(value, str):
            value = f'"{value}"'
        items.append(f"{key}={value}")
    return items


def _get_graphviz_node(edge_id: int, node_id: int | None = None) -> str:
    if node_id is None:
        if edge_id < 0:  # initial state
            return string.ascii_uppercase[-edge_id - 1]
        return str(edge_id)
    return f"N{node_id}"


def _create_same_rank_line(node_edge_ids: Iterable[int], prefix: str = "") -> str:
    name_list = [f"{prefix}{_get_graphviz_node(i)}" for i in node_edge_ids]
    name_string = ", ".join(name_list)
    return f"{{ rank=same {name_string} }}"


def _create_edge_label(
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

    >>> from qrules.io._dot import as_string
    >>> as_string(10)
    '10'
    >>> _ = as_string.register(int, lambda _: "new int rendering")
    >>> as_string(10)
    'new int rendering'
    """
    _LOGGER.warning(f"No DOT renderer implemented type {type(obj).__name__}")
    return str(obj)


as_string.register(str, lambda _: _)  # avoid warning for str type


@as_string.register(dict)
def _(obj: dict) -> str:
    lines = []
    for key, value in obj.items():
        if isinstance(key, type) or callable(key):
            key_repr = key.__name__
        else:
            key_repr = key
        if value != 0 or any(s in key_repr for s in ["magnitude", "projection"]):
            pm = not any(s in key_repr for s in ["pid", "mass", "width", "magnitude"])
            value_repr = __render_fraction(value, pm)
            lines.append(f"{key_repr} = {value_repr}")
    return "\n".join(lines)


def __render_fraction(value: Any, plusminus: bool) -> str:
    plusminus &= isinstance(value, Number) and bool(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        nom, denom = value.as_integer_ratio()
        if denom == 2:
            if plusminus:
                return f"{nom:+}/{denom}"
            return f"{nom}/{denom}"
    if plusminus:
        return f"{value:+}"
    return str(value)


@as_string.register(InteractionProperties)
def _(obj: InteractionProperties) -> str:
    lines = []
    if obj.l_magnitude is not None:
        if obj.l_projection is None:
            l_label = _to_fraction(obj.l_magnitude)
        else:
            l_label = _spin_to_str(Spin(obj.l_magnitude, obj.l_projection))
        lines.append(f"L={l_label}")
    if obj.s_magnitude is not None:
        if obj.s_projection is None:
            s_label = _to_fraction(obj.s_magnitude)
        else:
            s_label = _spin_to_str(Spin(obj.s_magnitude, obj.s_projection))
        lines.append(f"S={s_label}")
    if obj.parity_prefactor is not None:
        label = _to_fraction(obj.parity_prefactor, render_plus=True)
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
            f"{qn.__name__} ∊ {domain}" for qn, domain in settings.qn_domains.items()
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


def __extract_priority(description: str) -> str:
    matches = re.match(r".* \- ([0-9]+|NA)$", description)
    if matches is None:
        msg = f"{description} does not contain a priority number"
        raise ValueError(msg)
    return matches[1]


@as_string.register(Particle)
def _(particle: Particle) -> str:
    return particle.name


@as_string.register(Spin)
def _spin_to_str(spin: Spin) -> str:
    spin_magnitude = _to_fraction(spin.magnitude)
    spin_projection = _to_fraction(spin.projection, render_plus=True)
    return f"|{spin_magnitude},{spin_projection}⟩"


@as_string.register(State)
def _state_to_str(state: State) -> str:
    particle = state.particle.name
    spin_projection = _to_fraction(state.spin_projection, render_plus=True)
    return f"{particle}[{spin_projection}]"


@as_string.register(tuple)
def _(obj: tuple) -> str:
    if len(obj) == 2:
        if isinstance(obj[0], Particle) and isinstance(obj[1], (float, int)):
            state = State(*obj)
            return _state_to_str(state)
        if all(isinstance(o, (float, int)) for o in obj):
            spin = Spin(*obj)
            return _spin_to_str(spin)
    if all(isinstance(o, Particle) for o in obj):
        return "\n".join(map(as_string, obj))
    _LOGGER.warning(f"No DOT render implemented for tuple of size {len(obj)}")
    return str(obj)


def _get_particle_graphs(
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
        stripped_transition = _strip_projections(transition)
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


def _strip_projections(
    graph: Transition[Any, InteractionProperties],
) -> FrozenTransition[Particle, InteractionProperties]:
    if isinstance(graph, MutableTransition):
        transition = graph.freeze()
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


def _collapse_graphs(
    graphs: Iterable[Transition[ParticleWithSpin, InteractionProperties]],
) -> list[FrozenTransition[tuple[Particle, ...], None]]:
    transition_groups: dict[Topology, MutableTransition[set[Particle], None]] = {
        g.topology: MutableTransition(
            g.topology,
            states={i: set() for i in g.topology.edges},  # type: ignore[misc]
            interactions=dict.fromkeys(g.topology.nodes),  # type: ignore[misc]
        )
        for g in graphs
    }
    for transition in graphs:
        topology = transition.topology
        group = transition_groups[topology]
        for state_id, state in transition.states.items():
            if isinstance(state, State):
                particle = state.particle
            else:
                particle, _ = state
            group.states[state_id].add(particle)
    collected_graphs: list[FrozenTransition[tuple[Particle, ...], None]] = []
    for topology in sorted(transition_groups):
        group = transition_groups[topology]
        collected_graphs.append(
            FrozenTransition(
                topology,
                states={
                    i: tuple(sorted(particles, key=lambda p: p.name))
                    for i, particles in group.states.items()
                },
                interactions=group.interactions,
            )
        )
    return collected_graphs
