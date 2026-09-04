"""Generate Mermaid flowchart sources.

This module provides a lightweight Mermaid renderer for graph-like objects in qrules.
"""

from __future__ import annotations

import logging
import re
import string
from collections import abc
from typing import TYPE_CHECKING, Any, Literal

from attrs import define, field

from qrules.io import _labels
from qrules.solving import QNProblemSet, QNResult
from qrules.topology import Topology, Transition
from qrules.transition import ProblemSet, ReactionInfo

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

_LOGGER = logging.getLogger(__name__)

_NodeShape = Literal["circle", "rectangle", "rounded"]

_LABEL_ESCAPES: dict[str, str] = {
    "\\": r"\\",
    '"': r"\"",
    "\n": "<br/>",
}
_NODE_LABEL_TABLE = str.maketrans(_LABEL_ESCAPES)
_EDGE_LABEL_TABLE = str.maketrans({**_LABEL_ESCAPES, "|": r"\|"})

_STYLE_KEYS = {
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
_NODE_STYLE_KEYS = {
    **_STYLE_KEYS,
    "stroke": "stroke",
    "strokecolor": "stroke",
    "fontweight": "font-weight",
}
_EDGE_STYLE_KEYS = {
    **_NODE_STYLE_KEYS,
    "bgcolor": "stroke",
    "backgroundcolor": "stroke",
    "fill": "stroke",
    "fillcolor": "stroke",
    "color": "stroke",
}


def _get_mermaid_node(edge_id: int, node_id: int | None = None) -> str:
    if node_id is None:
        if edge_id < 0:
            return string.ascii_uppercase[-edge_id - 1]
        return str(edge_id)
    return f"N{node_id}"


def _to_style_dict(style: dict[str, Any] | None) -> dict[str, Any]:
    return dict(style) if style else {}


_NodeRole = Literal["state", "interaction", "intermediate"]
"""What a node represents in a transition, which decides its shape and its styling."""
_NODE_SHAPES: dict[_NodeRole, _NodeShape] = {
    "state": "rectangle",
    "interaction": "circle",
    "intermediate": "rounded",
}


@define
class _NodeRegistry:
    """Nodes of a single flowchart, in the order in which they were declared.

    Nodes are declared several times while a transition is rendered, first for their
    label and later as the endpoint of an edge. A repeated declaration only refines what
    is already known: an empty label or role leaves the existing one in place.
    """

    _labels: dict[str, str] = field(factory=dict)
    _roles: dict[str, _NodeRole] = field(factory=dict)

    def add(
        self, raw_id: str, label: str = "", *, role: _NodeRole | None = None
    ) -> str:
        node_id = _normalize_node_id(raw_id)
        if node_id not in self._labels:
            self._labels[node_id] = label
            self._roles[node_id] = role or "state"
        else:
            if label:
                self._labels[node_id] = label
            if role is not None:
                self._roles[node_id] = role
        return node_id

    def __iter__(self) -> Iterator[tuple[str, str, _NodeRole]]:
        for node_id, label in self._labels.items():
            yield node_id, label, self._roles[node_id]


def _normalize_node_id(node_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", node_id)
    if not normalized:
        normalized = "node"
    if normalized[0].isdigit():
        normalized = f"n_{normalized}"
    return normalized


@define
class _RenderContext:
    """Everything the render helpers need while building one flowchart."""

    topology: Topology
    graph: _labels.RenderedGraph
    prefix: str
    render_node: bool
    render_label: Callable[[Any], str]
    folded_initial_edge_id: int | None = None
    """Initial state edge that is rendered as the label of the node it points to."""
    nodes: _NodeRegistry = field(factory=_NodeRegistry)

    def create_edge_label(self, edge_id: int, render_edge_id: bool) -> str:
        return _labels.create_edge_label(
            self.graph, edge_id, render_edge_id, render_label=self.render_label
        )

    def add_state_node(
        self,
        edge_id: int,
        node_id: int | None = None,
        *,
        label: str = "",
        role: _NodeRole | None = None,
    ) -> str:
        """Declare the node that an edge starts or ends on.

        Without a ``node_id``, this is the node that represents the state on the edge
        itself. With one, it is the interaction node that the edge is attached to.
        """
        raw_id = self.prefix + _get_mermaid_node(edge_id, node_id)
        return self.nodes.add(raw_id, label, role=role)

    def add_interaction_node(
        self, node_id: int | None, label: str = "", *, role: _NodeRole | None = None
    ) -> str:
        return self.nodes.add(f"{self.prefix}N{node_id}", label, role=role)

    def add_interaction_nodes(self) -> None:
        """Declare a node for each interaction node in the topology.

        Topologies have no interaction properties to render, so their nodes are labeled
        with their node ID instead.
        """
        for node_id in self.topology.nodes:
            if isinstance(self.graph, Topology) and self.render_node:
                self.add_interaction_node(node_id, f"({node_id})", role="interaction")
            else:
                self.add_interaction_node(node_id)
        if not self.render_node:
            return
        if isinstance(self.graph, (ProblemSet, QNProblemSet)):
            interactions = self.graph.solving_settings.interactions
            for node_id, settings in interactions.items():
                self.add_interaction_node(node_id, self.render_label(settings))
        elif isinstance(self.graph, Transition):
            for node_id, node_prop in self.graph.interactions.items():
                self.add_interaction_node(
                    node_id, self.render_label(node_prop), role="interaction"
                )


@define(kw_only=True)
class MermaidPrinter:
    """Render qrules graph objects as Mermaid flowcharts."""

    render_node: bool | None = None
    render_final_state_id: bool = True
    render_resonance_id: bool = False
    render_initial_state_id: bool = False
    strip_spin: bool = False
    collapse_graphs: bool = False
    figure_style: dict[str, Any] = field(converter=_to_style_dict, default=None)
    edge_style: dict[str, Any] = field(converter=_to_style_dict, default=None)
    node_style: dict[str, Any] = field(converter=_to_style_dict, default=None)
    latex: bool = True

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
        transitions = _labels.select_transitions(
            obj,
            collapse=self.collapse_graphs,
            render_node=self.render_node,
            strip_spin=self.strip_spin,
        )
        lines: list[str] = []
        for i, graph in enumerate(reversed(transitions)):
            lines.extend(self._render_transition(graph, prefix=f"T{i}_"))
        return lines

    def _render_transition(
        self,
        obj: _labels.RenderInput,
        prefix: str = "",
    ) -> list[str]:
        context = self._create_render_context(obj, prefix)
        self._add_state_nodes(context)
        context.add_interaction_nodes()
        edge_lines = self._create_edge_lines(context)
        node_lines = [
            self._create_mermaid_node(node_id, label, shape=_NODE_SHAPES[role])
            for node_id, label, role in context.nodes
        ]
        style_lines = self._create_style_lines(context.nodes, len(edge_lines))
        return [*node_lines, *edge_lines, *style_lines]

    def _create_render_context(
        self, obj: _labels.RenderInput, prefix: str
    ) -> _RenderContext:
        topology, graph = _labels.unpack_render_input(obj)
        if self.render_node is None:
            render_node = isinstance(graph, Topology) and len(topology.nodes) > 1
        else:
            render_node = self.render_node
        context = _RenderContext(
            graph=graph,
            prefix=prefix,
            render_label=_labels.as_latex if self.latex else _labels.as_string,
            render_node=render_node,
            topology=topology,
        )
        context.folded_initial_edge_id = self._find_folded_initial_edge(context)
        return context

    def _find_folded_initial_edge(self, context: _RenderContext) -> int | None:
        """Find the initial state edge whose label can be merged into its node.

        A decay with a single initial state has no incoming edge to draw, so its label
        is written onto the interaction node that the edge points to. This is only
        possible if that node does not carry a label of its own.
        """
        topology = context.topology
        if context.render_node or len(topology.incoming_edge_ids) != 1:
            return None
        edge_id = next(iter(topology.incoming_edge_ids))
        if topology.edges[edge_id].ending_node_id is None:
            return None
        if not context.create_edge_label(edge_id, self.render_initial_state_id):
            return None
        return edge_id

    def _add_state_nodes(self, context: _RenderContext) -> None:
        topology = context.topology
        for edge_id in topology.incoming_edge_ids | topology.outgoing_edge_ids:
            render_edge_id = (
                self.render_initial_state_id
                if edge_id in topology.incoming_edge_ids
                else self.render_final_state_id
            )
            label = context.create_edge_label(edge_id, render_edge_id)
            if edge_id == context.folded_initial_edge_id:
                context.add_interaction_node(
                    topology.edges[edge_id].ending_node_id, label
                )
            else:
                context.add_state_node(edge_id, label=label)

    def _create_edge_lines(self, context: _RenderContext) -> list[str]:
        lines: list[str] = []
        for edge_id, edge in context.topology.edges.items():
            if edge_id == context.folded_initial_edge_id:
                continue
            from_node = context.add_state_node(edge_id, edge.originating_node_id)
            to_node = context.add_state_node(edge_id, edge.ending_node_id)
            label = ""
            if edge.originating_node_id is not None and edge.ending_node_id is not None:
                label = context.create_edge_label(edge_id, self.render_resonance_id)
            if label:
                state_node = context.add_state_node(
                    edge_id, label=label, role="intermediate"
                )
                lines.extend([
                    self._create_mermaid_edge(from_node, state_node),
                    self._create_mermaid_edge(state_node, to_node),
                ])
            else:
                lines.append(self._create_mermaid_edge(from_node, to_node))
        return lines

    def _create_style_lines(
        self, nodes: _NodeRegistry, number_of_links: int
    ) -> list[str]:
        lines: list[str] = []
        for node_id, _, role in nodes:
            if role == "intermediate":
                if self.edge_style:
                    lines.append(
                        self._create_mermaid_node_style(
                            node_id, self.edge_style, target="edge"
                        )
                    )
            elif self.node_style:
                lines.append(self._create_mermaid_node_style(node_id, self.node_style))
        if self.edge_style:
            lines.extend(
                self._create_mermaid_link_style(i, self.edge_style)
                for i in range(number_of_links)
            )
        return lines

    def _create_figure_style_lines(self) -> list[str]:
        if not self.figure_style:
            return []
        style = self._format_style_dict(self.figure_style)
        if not style:
            return []
        return [f"classDef default {style}"]

    def _create_mermaid_node_style(
        self, node_id: str, style: dict[str, Any], *, target: str = "node"
    ) -> str:
        style_definition = self._format_style_dict(style, target=target)
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
            if style_key == "font-size" and isinstance(value, (int, float)):
                value = f"{value}px"
            parts.append(f"{style_key}:{value}")
        return ",".join(parts)

    @staticmethod
    def _normalize_style_key(key: str, target: str) -> str | None:
        normalized = str(key).strip().lower().replace("_", "")
        if target == "node":
            mapping = _NODE_STYLE_KEYS
        elif target == "edge":
            mapping = _EDGE_STYLE_KEYS
        else:
            mapping = _STYLE_KEYS
        return mapping.get(normalized)

    def _create_mermaid_node(
        self, node_id: str, label: str = "", *, shape: _NodeShape = "rectangle"
    ) -> str:
        if not label:
            return f'    {node_id}@{{ shape: text, label: " " }}'
        if self.latex:
            if shape == "rounded":
                label = self._apply_latex_color(label, self.edge_style, target="edge")
            else:
                style = {**self.figure_style, **self.node_style}
                label = self._apply_latex_color(label, style, target="node")
        escaped_label = self._escape_label(label)
        if shape == "circle":
            return f'    {node_id}(("{escaped_label}"))'
        if shape == "rounded":
            return f'    {node_id}("{escaped_label}")'
        return f'    {node_id}["{escaped_label}"]'

    def _create_mermaid_edge(
        self, from_node: str, to_node: str, label: str = ""
    ) -> str:
        if label:
            if self.latex:
                label = self._apply_latex_color(label, self.edge_style, target="edge")
            if "|" in label:
                escaped_label = self._escape_label(label)
                return f'    {from_node} --"{escaped_label}"--- {to_node}'
            escaped_label = self._escape_label(label, for_edge=True)
            if self.latex:
                return f'    {from_node} ---|"{escaped_label}"| {to_node}'
            if any(char in escaped_label for char in "[]()"):
                escaped_label = f'"{escaped_label}"'
            return f"    {from_node} ---|{escaped_label}| {to_node}"
        return f"    {from_node} --- {to_node}"

    @classmethod
    def _apply_latex_color(
        cls, label: str, style: dict[str, Any], *, target: str
    ) -> str:
        color = next(
            (
                value
                for key, value in reversed(style.items())
                if value is not None
                and cls._normalize_style_key(key, target) == "color"
            ),
            None,
        )
        if color is None:
            return label
        return Rf"\textcolor{{{color}}}{{{label}}}"

    def _escape_label(self, label: str, *, for_edge: bool = False) -> str:
        if self.latex:
            escaped_label = _escape_latex_for_mermaid(label)
            return f"$${escaped_label}$$"
        table = _EDGE_LABEL_TABLE if for_edge else _NODE_LABEL_TABLE
        return str(label).strip().translate(table)


def _escape_latex_for_mermaid(label: str) -> str:
    R"""Escape a KaTeX label so that it survives Mermaid's string lexer.

    Within a quoted label, Mermaid reads ``\\`` and ``\"`` as escape sequences and
    passes any other backslash sequence through untouched. A backslash therefore has to
    be doubled only where it precedes another backslash or a quote. Escaping the quotes
    first and rewriting each remaining ``\\`` pair with three backslashes produces
    exactly that: the KaTeX row separator ``\\`` arrives as two backslashes, and an
    accent such as ``\"`` arrives with its quote intact.

    >>> print(_escape_latex_for_mermaid(R"L = 0 \\ S = 1"))
    L = 0 \\\ S = 1
    >>> print(_escape_latex_for_mermaid(R"\"o"))
    \\\"o
    """
    escaped_label = str(label).strip().replace("\n", " ").replace('"', R"\"")
    return escaped_label.replace(2 * "\\", 3 * "\\")
