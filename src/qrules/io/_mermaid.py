"""Generate Mermaid flowchart sources.

This module provides a lightweight Mermaid renderer for graph-like objects in qrules.
"""

from __future__ import annotations

import logging
import re
import string
from collections import abc
from typing import TYPE_CHECKING, Any

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


def _get_mermaid_node(edge_id: int, node_id: int | None = None) -> str:
    if node_id is None:
        if edge_id < 0:
            return string.ascii_uppercase[-edge_id - 1]
        return str(edge_id)
    return f"N{node_id}"


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
            label = _labels.create_edge_label(rendered_graph, edge_id, render)
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
                add_node(f"{prefix}N{node_id}", _labels.as_string(settings))

        if isinstance(rendered_graph, Transition) and render_node:
            for node_id, node_prop in rendered_graph.interactions.items():
                add_node(f"{prefix}N{node_id}", _labels.as_string(node_prop))

        edge_style_lines: list[str] = []
        for edge_index, (edge_id, edge) in enumerate(topology.edges.items()):
            j, k = edge.ending_node_id, edge.originating_node_id
            from_node = add_node(prefix + _get_mermaid_node(edge_id, k))
            to_node = add_node(prefix + _get_mermaid_node(edge_id, j))
            if j is None or k is None:
                edge_lines.append(self._create_mermaid_edge(from_node, to_node))
            else:
                label = _labels.create_edge_label(
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
        return f'    {node_id}@{{ shape: text, label: " " }}'

    def _create_mermaid_edge(
        self, from_node: str, to_node: str, label: str = ""
    ) -> str:
        if label:
            if "|" in label:
                escaped_label = self._escape_label(label)
                return f'    {from_node} --"{escaped_label}"--- {to_node}'
            escaped_label = self._escape_label(label, for_edge=True)
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

    @staticmethod
    def _escape_label(label: str, *, for_edge: bool = False) -> str:
        table = _EDGE_LABEL_TABLE if for_edge else _NODE_LABEL_TABLE
        return str(label).strip().translate(table)
