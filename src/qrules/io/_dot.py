"""Generate dot sources.

See :doc:`/usage/visualize` for more info.
"""

from __future__ import annotations

import logging
import string
from collections import abc
from typing import TYPE_CHECKING, Any

from attrs import define, field
from attrs.converters import default_if_none

from qrules.io import _labels
from qrules.solving import QNProblemSet, QNResult
from qrules.topology import Topology, Transition
from qrules.transition import ProblemSet, ReactionInfo

if TYPE_CHECKING:
    from collections.abc import Iterable

_LOGGER = logging.getLogger(__name__)


def _create_default_figure_style(style: dict[str, Any] | None) -> dict[str, Any]:
    figure_style = {"bgcolor": None}
    if style is None:
        return figure_style
    figure_style.update(style)
    return figure_style


@define(on_setattr=_check_booleans)
class GraphvizPrinter:
    render_node: bool | None = None
    render_final_state_id: bool = True
    render_resonance_id: bool = False
    render_initial_state_id: bool = False
    collapse: _labels.CollapseMode | None = None

    figure_style: dict[str, Any] = field(
        converter=_create_default_figure_style, default=None
    )
    edge_style: dict[str, Any] = field(
        converter=default_if_none(factory=dict),
        default=None,
    )
    node_style: dict[str, Any] = field(
        converter=default_if_none(factory=dict),
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
        transitions = _labels.prepare_transitions(
            obj,
            collapse=self.collapse,
            render_node=self.render_node,
        )
        lines = []
        for i, graph in enumerate(reversed(list(transitions))):
            lines += self._render_transition(graph, prefix=f"T{i}_")
        return lines

    def _render_transition(  # ruff: ignore[complex-structure, too-many-branches, too-many-statements]
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
            msg = f"Cannot render {type(obj).__name__} as dot"
            raise NotImplementedError(msg)
        for edge_id in topology.incoming_edge_ids | topology.outgoing_edge_ids:
            if edge_id in topology.incoming_edge_ids:
                render = self.render_initial_state_id
            else:
                render = self.render_final_state_id
            label = _labels.create_edge_label(rendered_graph, edge_id, render)
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
                label = _labels.create_edge_label(
                    rendered_graph,
                    edge_id=i,
                    render_edge_id=self.render_resonance_id,
                )
                lines += [self._create_graphviz_edge(from_node, to_node, label)]
        if isinstance(obj, (ProblemSet, QNProblemSet)):
            node_settings = obj.solving_settings.interactions
            for node_id, settings in node_settings.items():
                label = ""
                if self.render_node:
                    label = _labels.as_string(settings)
                node = f"{prefix}N{node_id}"
                lines += [self._create_graphviz_node(node, label, self.node_style)]
        if isinstance(obj, Transition):
            for node_id, node_prop in obj.interactions.items():
                label = ""
                if self.render_node:
                    label = _labels.as_string(node_prop)
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
        style.pop("label", None)
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
    name_string = " ".join(name_list)
    return f"{{ rank=same; {name_string} }}"
