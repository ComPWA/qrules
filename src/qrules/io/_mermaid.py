"""Generate Mermaid flowchart sources.

This module provides a lightweight Mermaid renderer for graph-like objects in qrules.
"""

from __future__ import annotations

import logging
import re
import string
from collections import abc
from fractions import Fraction
from functools import singledispatch
from inspect import isfunction
from typing import TYPE_CHECKING, Any, cast

import attrs

from qrules.particle import Particle, ParticleWithSpin, Spin, _render_fraction
from qrules.quantum_numbers import InteractionProperties
from qrules.solving import EdgeSettings, NodeSettings, QNProblemSet, QNResult
from qrules.topology import FrozenTransition, MutableTransition, Topology, Transition
from qrules.transition import ProblemSet, ReactionInfo, State

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qrules.argument_handling import Rule

_LOGGER = logging.getLogger(__name__)


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
    if obj is not None:
        _LOGGER.warning(f"No Mermaid renderer implemented type {type(obj).__name__}")
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
    return __escape_mermaid_node_text(output)


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


def __render_domain(domain: list[Any], key: str) -> str:
    domain = sorted(domain, key=lambda x: +9999 if x is None else x)
    domain_str = [__render_key_and_value(key, x) for x in domain]
    return "[" + ", ".join(domain_str) + "]"


def __escape_mermaid_node_text(text: str) -> str:
    # Mermaid can misinterpret ASCII square brackets inside rich node/edge labels.
    # Use visually equivalent Unicode brackets to keep labels readable and parse-safe.
    return text.replace("[", " [").replace("]", " ]")


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


def _get_mermaid_node(edge_id: int, node_id: int | None = None) -> str:
    if node_id is None:
        if edge_id < 0:
            return string.ascii_uppercase[-edge_id - 1]
        return str(edge_id)
    return f"N{node_id}"


def _get_particle_graphs(
    graphs: Iterable[Transition[ParticleWithSpin, InteractionProperties]],
) -> list[FrozenTransition[Particle, None]]:
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
    graphs: Iterable[Transition[Any, Any]],
) -> list[FrozenTransition[tuple, None]]:
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
    if isinstance(state, str):
        return state
    return state


def _sorting_key(obj: Any) -> Any:
    if isinstance(obj, State):
        return obj.particle.name
    if isinstance(obj, str):
        return obj.lower()
    return obj


class MermaidPrinter:
    """Render qrules graph objects as Mermaid flowcharts."""

    def __init__(
        self,
        *,
        render_node: bool | None = None,
        render_final_state_id: bool = True,
        render_resonance_id: bool = False,
        render_initial_state_id: bool = False,
        strip_spin: bool = False,
        collapse_graphs: bool = False,
        figure_style: dict[str, Any] | None = None,
        edge_style: dict[str, Any] | None = None,
        node_style: dict[str, Any] | None = None,
    ) -> None:
        self.render_node = render_node
        self.render_final_state_id = render_final_state_id
        self.render_resonance_id = render_resonance_id
        self.render_initial_state_id = render_initial_state_id
        self.strip_spin = strip_spin
        self.collapse_graphs = collapse_graphs
        self.figure_style = dict(figure_style) if figure_style else {}
        self.edge_style = dict(edge_style) if edge_style else {}
        self.node_style = dict(node_style) if node_style else {}

    def __call__(self, obj: Any) -> str:
        lines = ["flowchart LR"]
        lines.extend(self._create_figure_style_lines())
        lines.extend(self._render(obj))
        return "\n".join(lines) + "\n"

    def _render(self, obj: Any) -> list[str]:
        if isinstance(obj, QNResult):
            obj = obj.solutions
        if isinstance(obj, ReactionInfo):
            obj = obj.transitions
        if isinstance(obj, abc.Iterable):
            return self._render_multiple_transitions(obj)
        if isinstance(obj, (ProblemSet, QNProblemSet, Topology, Transition)):
            return self._render_transition(obj)
        msg = f"No Mermaid rendering for type {type(obj).__name__}"
        raise NotImplementedError(msg)

    def _render_multiple_transitions(self, obj: Iterable) -> list[str]:
        transitions: Iterable[Transition[Any, Any]]
        if self.collapse_graphs:
            transitions = _collapse_graphs(obj)
        elif self.strip_spin:
            if self.render_node:
                transitions = sorted({_strip_projections(t) for t in obj})
            else:
                transitions = _get_particle_graphs(obj)
        else:
            transitions = list(obj)

        lines: list[str] = []
        for i, graph in enumerate(reversed(list(transitions))):
            lines.extend(self._render_transition(graph, prefix=f"T{i}_"))  # type:ignore[arg-type]
        return lines

    def _render_transition(  # ruff: ignore[complex-structure, too-many-branches, too-many-locals, too-many-statements]
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
            msg = f"Cannot render {type(obj).__name__} as Mermaid"
            raise NotImplementedError(msg)

        node_lines: list[str] = []
        edge_lines: list[str] = []
        node_labels: dict[str, str] = {}
        node_order: list[str] = []

        def add_node(raw_id: str, label: str = "") -> str:
            node_id = self._normalize_node_id(raw_id)
            if node_id not in node_labels:
                node_order.append(node_id)
                node_labels[node_id] = label
            elif label:
                node_labels[node_id] = label
            return node_id

        for edge_id in topology.incoming_edge_ids | topology.outgoing_edge_ids:
            if edge_id in topology.incoming_edge_ids:
                render = self.render_initial_state_id
            else:
                render = self.render_final_state_id
            label = _create_edge_label(rendered_graph, edge_id, render)
            add_node(prefix + _get_mermaid_node(edge_id), label)

        if self.render_node is None:
            render_node = (
                isinstance(rendered_graph, Topology) and len(topology.nodes) > 1
            )
        else:
            render_node = self.render_node

        for node_id in topology.nodes:
            label = ""
            if isinstance(rendered_graph, Topology) and render_node:
                label = f"({node_id})"
            add_node(f"{prefix}N{node_id}", label)

        if isinstance(rendered_graph, (ProblemSet, QNProblemSet)) and render_node:
            for (
                node_id,
                settings,
            ) in rendered_graph.solving_settings.interactions.items():
                add_node(f"{prefix}N{node_id}", as_string(settings))

        if isinstance(rendered_graph, Transition) and render_node:
            for node_id, node_prop in rendered_graph.interactions.items():
                add_node(f"{prefix}N{node_id}", as_string(node_prop))

        edge_style_lines: list[str] = []
        for edge_index, (edge_id, edge) in enumerate(topology.edges.items()):
            j, k = edge.ending_node_id, edge.originating_node_id
            from_node = add_node(prefix + _get_mermaid_node(edge_id, k))
            to_node = add_node(prefix + _get_mermaid_node(edge_id, j))
            if j is None or k is None:
                edge_lines.append(self._create_mermaid_edge(from_node, to_node))
            else:
                label = _create_edge_label(
                    rendered_graph, edge_id, self.render_resonance_id
                )
                edge_lines.append(self._create_mermaid_edge(from_node, to_node, label))
            if self.edge_style:
                edge_style_lines.append(
                    self._create_mermaid_link_style(edge_index, self.edge_style)
                )

        node_lines.extend(
            self._create_mermaid_node(node_id, node_labels[node_id])
            for node_id in node_order
        )
        style_lines: list[str] = []
        if self.node_style:
            style_lines.extend(
                self._create_mermaid_node_style(node_id, self.node_style)
                for node_id in node_order
            )
        if self.edge_style:
            style_lines.extend(edge_style_lines)

        lines.extend(node_lines)
        lines.extend(edge_lines)
        lines.extend(style_lines)
        return lines

    def _create_figure_style_lines(self) -> list[str]:
        if not self.figure_style:
            return []
        style = self._format_style_dict(self.figure_style)
        if not style:
            return []
        return [f"classDef default {style}"]

    def _create_mermaid_node_style(self, node_id: str, style: dict[str, Any]) -> str:
        style_definition = self._format_style_dict(style, target="node")
        if not style_definition:
            return ""
        return f"    style {node_id} {style_definition}"

    def _create_mermaid_link_style(self, edge_index: int, style: dict[str, Any]) -> str:
        style_definition = self._format_style_dict(style, target="edge")
        if not style_definition:
            return ""
        return f"    linkStyle {edge_index} {style_definition}"

    def _format_style_dict(
        self, style: dict[str, Any], *, target: str = "figure"
    ) -> str:
        parts: list[str] = []
        for key, value in style.items():
            if value is None:
                continue
            style_key = self._normalize_style_key(key, target)
            if style_key is None:
                continue
            parts.append(f"{style_key}:{value}")
        return ",".join(parts)

    @staticmethod
    def _normalize_style_key(key: str, target: str) -> str | None:
        normalized = str(key).strip().lower().replace("_", "")
        if target == "node":
            mapping = {
                "bgcolor": "fill",
                "backgroundcolor": "fill",
                "fill": "fill",
                "fillcolor": "fill",
                "color": "color",
                "fontcolor": "color",
                "font": "font-family",
                "fontfamily": "font-family",
                "fontsize": "font-size",
                "stroke": "stroke",
                "strokecolor": "stroke",
                "fontweight": "font-weight",
            }
        elif target == "edge":
            mapping = {
                "bgcolor": "stroke",
                "backgroundcolor": "stroke",
                "fill": "stroke",
                "fillcolor": "stroke",
                "color": "stroke",
                "fontcolor": "color",
                "font": "font-family",
                "fontfamily": "font-family",
                "fontsize": "font-size",
                "stroke": "stroke",
                "strokecolor": "stroke",
                "fontweight": "font-weight",
            }
        else:
            mapping = {
                "bgcolor": "fill",
                "backgroundcolor": "fill",
                "fill": "fill",
                "fillcolor": "fill",
                "color": "color",
                "fontcolor": "color",
                "font": "font-family",
                "fontfamily": "font-family",
                "fontsize": "font-size",
            }
        return mapping.get(normalized)

    def _create_mermaid_node(self, node_id: str, label: str = "") -> str:
        if label:
            escaped_label = self._escape_label(label)
            return f'    {node_id}["{escaped_label}"]'
        return f"    {node_id}"

    def _create_mermaid_edge(
        self, from_node: str, to_node: str, label: str = ""
    ) -> str:
        if label:
            if "|" in label:
                escaped_label = self._escape_label(label)
                return f'    {from_node} --"{escaped_label}"--> {to_node}'
            escaped_label = self._escape_label(label, for_edge=True)
            if any(char in escaped_label for char in "[]()"):
                escaped_label = f'"{escaped_label}"'
            return f"    {from_node} -->|{escaped_label}| {to_node}"
        return f"    {from_node} --> {to_node}"

    @staticmethod
    def _normalize_node_id(node_id: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_]", "_", node_id)
        if not normalized:
            normalized = "node"
        if normalized[0].isdigit():
            normalized = f"n_{normalized}"
        return normalized

    @staticmethod
    def _escape_label(label: str, *, for_edge: bool = False) -> str:
        text = str(label).strip()
        if not text:
            return ""
        escaped = text.replace("\\", "\\\\")
        escaped = escaped.replace('"', '\\"')
        escaped = escaped.replace("\n", "<br/>")
        if for_edge:
            escaped = escaped.replace("|", "\\|")
        return escaped
