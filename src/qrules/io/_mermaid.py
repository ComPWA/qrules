"""Generate Mermaid flowchart sources.

This module provides a lightweight Mermaid renderer for graph-like objects in qrules.
"""

from __future__ import annotations

import logging
import re
import string
from collections import abc
from typing import TYPE_CHECKING, Any

import attrs

from qrules.io import _labels
from qrules.solving import QNProblemSet, QNResult
from qrules.topology import Topology, Transition
from qrules.transition import ProblemSet, ReactionInfo

if TYPE_CHECKING:
    from collections.abc import Iterable

_LOGGER = logging.getLogger(__name__)

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


@attrs.define(kw_only=True)
class MermaidPrinter:
    """Render qrules graph objects as Mermaid flowcharts."""

    render_node: bool | None = None
    render_final_state_id: bool = True
    render_resonance_id: bool = False
    render_initial_state_id: bool = False
    strip_spin: bool = False
    collapse_graphs: bool = False
    figure_style: dict[str, Any] = attrs.field(converter=_to_style_dict, default=None)
    edge_style: dict[str, Any] = attrs.field(converter=_to_style_dict, default=None)
    node_style: dict[str, Any] = attrs.field(converter=_to_style_dict, default=None)
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
        transitions: Iterable[Transition[Any, Any]]
        if self.collapse_graphs:
            transitions = _labels.collapse_graphs(obj)
        elif self.strip_spin:
            if self.render_node:
                transitions = sorted({_labels.strip_projections(t) for t in obj})
            else:
                transitions = _labels.get_particle_graphs(obj)
        else:
            transitions = list(obj)

        lines: list[str] = []
        for i, graph in enumerate(reversed(list(transitions))):
            lines.extend(self._render_transition(graph, prefix=f"T{i}_"))
        return lines

    def _render_transition(  # ruff: ignore[complex-structure, too-many-branches, too-many-locals, too-many-statements]
        self,
        obj: _labels.RenderInput,
        prefix: str = "",
    ) -> list[str]:
        lines: list[str] = []
        if _labels.is_render_pair(obj):
            topology, rendered_graph = obj
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

        if self.render_node is None:
            render_node = (
                isinstance(rendered_graph, Topology) and len(topology.nodes) > 1
            )
        else:
            render_node = self.render_node
        render_label = _labels.as_latex if self.latex else _labels.as_string

        folded_initial_edge_id: int | None = None
        if len(topology.incoming_edge_ids) == 1 and not render_node:
            initial_edge_id = next(iter(topology.incoming_edge_ids))
            initial_edge = topology.edges[initial_edge_id]
            initial_edge_label = _labels.create_edge_label(
                rendered_graph,
                initial_edge_id,
                self.render_initial_state_id,
                render_label=render_label,
            )
            if initial_edge.ending_node_id is not None and initial_edge_label:
                folded_initial_edge_id = initial_edge_id

        for edge_id in topology.incoming_edge_ids | topology.outgoing_edge_ids:
            if edge_id in topology.incoming_edge_ids:
                render = self.render_initial_state_id
            else:
                render = self.render_final_state_id
            label = _labels.create_edge_label(
                rendered_graph, edge_id, render, render_label=render_label
            )
            if edge_id == folded_initial_edge_id:
                node_id = topology.edges[edge_id].ending_node_id
                add_node(f"{prefix}N{node_id}", label)
            else:
                add_node(prefix + _get_mermaid_node(edge_id), label)

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
                add_node(f"{prefix}N{node_id}", render_label(settings))

        if isinstance(rendered_graph, Transition) and render_node:
            for node_id, node_prop in rendered_graph.interactions.items():
                add_node(f"{prefix}N{node_id}", render_label(node_prop))

        edge_style_lines: list[str] = []
        edge_index = 0
        for edge_id, edge in topology.edges.items():
            if edge_id == folded_initial_edge_id:
                continue
            j, k = edge.ending_node_id, edge.originating_node_id
            from_node = add_node(prefix + _get_mermaid_node(edge_id, k))
            to_node = add_node(prefix + _get_mermaid_node(edge_id, j))
            if j is None or k is None:
                edge_lines.append(self._create_mermaid_edge(from_node, to_node))
            else:
                label = _labels.create_edge_label(
                    rendered_graph,
                    edge_id,
                    self.render_resonance_id,
                    render_label=render_label,
                )
                edge_lines.append(self._create_mermaid_edge(from_node, to_node, label))
            if self.edge_style:
                edge_style_lines.append(
                    self._create_mermaid_link_style(edge_index, self.edge_style)
                )
            edge_index += 1

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

    def _create_mermaid_node(self, node_id: str, label: str = "") -> str:
        if label:
            if self.latex:
                style = {**self.figure_style, **self.node_style}
                label = self._apply_latex_color(label, style, target="node")
            escaped_label = self._escape_label(label)
        else:
            escaped_label = " "
        return f'    {node_id}@{{ shape: text, label: "{escaped_label}" }}'

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

    @staticmethod
    def _normalize_node_id(node_id: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_]", "_", node_id)
        if not normalized:
            normalized = "node"
        if normalized[0].isdigit():
            normalized = f"n_{normalized}"
        return normalized

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
