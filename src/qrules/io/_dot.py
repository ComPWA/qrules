"""Generate dot sources.

See :doc:`/usage/visualize` for more info.
"""

import logging
import re
import string
from collections import abc
from functools import singledispatch
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, cast

import attrs
from attrs import Attribute, define, field
from attrs.converters import default_if_none

from qrules.particle import Particle, ParticleWithSpin, Spin
from qrules.quantum_numbers import InteractionProperties, _to_fraction
from qrules.solving import EdgeSettings, NodeSettings
from qrules.topology import (
    FrozenTransition,
    MutableTransition,
    Topology,
    Transition,
)
from qrules.transition import ProblemSet, ReactionInfo, State


def _check_booleans(
    instance: "GraphPrinter", attribute: Attribute, value: bool
) -> None:
    # pylint: disable=unused-argument
    if instance.strip_spin and instance.collapse_graphs:
        raise ValueError("Cannot both strip spin and collapse graphs")
    if instance.collapse_graphs and instance.render_node:
        raise ValueError(
            "Collapsed graphs cannot be rendered with node properties"
        )


def _create_default_figure_style(
    style: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    figure_style = {"bgcolor": None}
    if style is None:
        return figure_style
    figure_style.update(style)
    return figure_style


@define(on_setattr=_check_booleans)
class GraphPrinter:
    render_node: Optional[bool] = None
    render_final_state_id: bool = True
    render_resonance_id: bool = False
    render_initial_state_id: bool = False
    strip_spin: bool = False
    collapse_graphs: bool = False

    figure_style: Dict[str, Any] = field(
        converter=_create_default_figure_style, default=None
    )
    edge_style: Dict[str, Any] = field(
        converter=default_if_none(factory=dict), default=None  # type: ignore[misc]
    )
    node_style: Dict[str, Any] = field(
        converter=default_if_none(factory=dict), default=None  # type: ignore[misc]
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

    def _create_preface(self) -> List[str]:
        return [
            "rankdir=LR",
            "node [shape=none, width=0]",
            "edge [arrowhead=none]",
            *_create_graphviz_assignments(self.figure_style),
        ]

    def _render(self, obj: Any) -> List[str]:
        if isinstance(obj, ReactionInfo):
            obj = obj.transitions
        if isinstance(obj, abc.Iterable):
            return self._render_multiple_transitions(obj)
        if isinstance(obj, (ProblemSet, Topology, Transition)):
            return self._render_transition(obj)
        raise NotImplementedError

    def _render_multiple_transitions(self, obj: Iterable) -> List[str]:
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

    def _render_transition(
        self,
        obj: Union[ProblemSet, Topology, Transition],
        prefix: str = "",
    ) -> List[str]:
        # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        lines: List[str] = []
        if isinstance(obj, tuple) and len(obj) == 2:
            topology: Topology = obj[0]
            rendered_graph: Union[ProblemSet, Topology, Transition] = obj[1]
        elif isinstance(obj, (ProblemSet, Transition)):
            rendered_graph = obj
            topology = obj.topology
        elif isinstance(obj, Topology):
            rendered_graph = obj
            topology = obj
        else:
            raise NotImplementedError(
                f"Cannot render {type(obj).__name__} as dot"
            )
        for edge_id in topology.incoming_edge_ids | topology.outgoing_edge_ids:
            if edge_id in topology.incoming_edge_ids:
                render = self.render_initial_state_id
            else:
                render = self.render_final_state_id
            label = _create_edge_label(rendered_graph, edge_id, render)
            graphviz_node = prefix + _get_graphviz_node(edge_id)
            lines += [
                self._create_graphviz_node(
                    graphviz_node, label, self.edge_style
                )
            ]
        lines += [_create_same_rank_line(topology.incoming_edge_ids, prefix)]
        lines += [_create_same_rank_line(topology.outgoing_edge_ids, prefix)]
        for i, edge in topology.edges.items():
            j, k = edge.ending_node_id, edge.originating_node_id
            from_node = prefix + _get_graphviz_node(i, k)
            to_node = prefix + _get_graphviz_node(i, j)
            if j is None or k is None:
                lines += [self._create_graphviz_edge(from_node, to_node)]
            else:
                label = _create_edge_label(
                    rendered_graph, i, self.render_resonance_id
                )
                lines += [
                    self._create_graphviz_edge(from_node, to_node, label)
                ]
        if isinstance(obj, ProblemSet):
            node_settings = obj.solving_settings.interactions
            for node_id, settings in node_settings.items():
                label = ""
                if self.render_node:
                    label = as_string(settings)
                node = f"{prefix}N{node_id}"
                lines += [
                    self._create_graphviz_node(node, label, self.node_style)
                ]
        if isinstance(obj, Transition):
            for node_id, node_prop in obj.interactions.items():
                label = ""
                if self.render_node:
                    label = as_string(node_prop)
                node = f"{prefix}N{node_id}"
                lines += [
                    self._create_graphviz_node(node, label, self.node_style)
                ]
        if isinstance(obj, Topology):
            render_node = self.render_node
            if render_node is None and len(topology.nodes) > 1:
                render_node = True
            for node_id in topology.nodes:
                label = ""
                if render_node:
                    label = f"({node_id})"
                node = f"{prefix}N{node_id}"
                lines += [
                    self._create_graphviz_node(node, label, self.node_style)
                ]
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
    def _create_graphviz_node(
        node: str, label: str, style: Dict[str, Any]
    ) -> str:
        style = dict(style)  # copy
        style["label"] = label
        styling = _create_graphviz_styling(style)
        return f"{node}{styling}"


def _create_graphviz_styling(graphviz_attrs: Dict[str, Any]) -> str:
    """Create a `str` of Graphviz attribute assignments for a node or edge.

    See `Graphviz attributes <https://graphviz.org/doc/info/attrs.html>`_ for
    the assignment syntax.

    >>> _create_graphviz_styling({"size": 12})
    ' [size=12]'
    >>> _create_graphviz_styling({"color": "red", "size": 8})
    ' [color="red", size=8]'
    """
    if not graphviz_attrs:
        return ""
    assignments = _create_graphviz_assignments(graphviz_attrs)
    return f" [{', '.join(assignments)}]"


def _create_graphviz_assignments(graphviz_attrs: Dict[str, Any]) -> List[str]:
    """Create a `list` of graphviz attribute assignments.

    See `Graphviz attributes <https://graphviz.org/doc/info/attrs.html>`_ for
    the assignment syntax.

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


def _get_graphviz_node(edge_id: int, node_id: Optional[int] = None) -> str:
    if node_id is None:
        if edge_id < 0:  # initial state
            return string.ascii_uppercase[-edge_id - 1]
        return str(edge_id)
    return f"N{node_id}"


def _create_same_rank_line(
    node_edge_ids: Iterable[int], prefix: str = ""
) -> str:
    name_list = [f"{prefix}{_get_graphviz_node(i)}" for i in node_edge_ids]
    name_string = ", ".join(name_list)
    return f"{{ rank=same {name_string} }}"


def _create_edge_label(
    graph: Union[ProblemSet, Topology, Transition],
    edge_id: int,
    render_edge_id: bool,
) -> str:
    if isinstance(graph, Topology):
        if render_edge_id:
            return str(edge_id)
        return ""
    if isinstance(graph, ProblemSet):
        edge_setting = graph.solving_settings.states.get(edge_id)
        initial_fact = graph.initial_facts.states.get(edge_id)
        edge_property: Optional[Union[EdgeSettings, ParticleWithSpin]] = None
        if edge_setting:
            edge_property = edge_setting
        if initial_fact:
            edge_property = initial_fact
        return __render_edge_with_id(edge_id, edge_property, render_edge_id)
    edge_prop = graph.states.get(edge_id)
    return __render_edge_with_id(edge_id, edge_prop, render_edge_id)


def __render_edge_with_id(
    edge_id: int, edge_prop: Any, render_edge_id: bool
) -> str:
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

    This function is decorated with :func:`functools.singledispatch`, which
    means that you can easily register other converter functions. An example:

    >>> from qrules.io._dot import as_string
    >>> as_string(10)
    '10'
    >>> _ = as_string.register(int, lambda _: "new int rendering")
    >>> as_string(10)
    'new int rendering'
    """
    logging.warning(f"No DOT renderer implemented type {type(obj).__name__}")
    return str(obj)


as_string.register(str, lambda _: _)  # avoid warning for str type


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
def _(settings: Union[EdgeSettings, NodeSettings]) -> str:
    output = ""
    if settings.rule_priorities:
        output += "RULE PRIORITIES\n"
        rule_names = (
            f"{item[0].__name__} - {item[1]}"  # type: ignore[union-attr]
            for item in settings.rule_priorities.items()
        )
        sorted_names = sorted(rule_names, key=__extract_priority, reverse=True)
        output += "\n".join(sorted_names)
    if settings.qn_domains:
        if output:
            output += "\n"
        domains = sorted(
            f"{item[0].__name__} ∊ {item[1]}"
            for item in settings.qn_domains.items()
        )
        output += "DOMAINS\n"
        output += "\n".join(domains)
    return output


def __extract_priority(description: str) -> int:
    matches = re.match(r".* \- ([0-9]+)$", description)
    if matches is None:
        raise ValueError(f"{description} does not contain a priority number")
    priority = matches[1]
    return int(priority)


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
    logging.warning(f"No DOT render implemented for tuple of size {len(obj)}")
    return str(obj)


def _get_particle_graphs(
    graphs: Iterable[Transition[ParticleWithSpin, InteractionProperties]],
) -> "List[FrozenTransition[Particle, None]]":
    """Strip `list` of `.Transition` s of the spin projections.

    Extract a `list` of `.Transition` instances with only `.Particle` instances
    on the edges.

    .. seealso:: :doc:`/usage/visualize`
    """
    inventory = set()
    for transition in graphs:
        if isinstance(transition, FrozenTransition):
            transition = transition.convert(
                lambda s: (s.particle, s.spin_projection)
            )
        stripped_transition = _strip_projections(transition)
        topology = stripped_transition.topology
        particle_transition: FrozenTransition[
            Particle, None
        ] = FrozenTransition(
            stripped_transition.topology,
            states=stripped_transition.states,
            interactions={i: None for i in topology.nodes},
        )
        inventory.add(particle_transition)
    return sorted(
        inventory,
        key=lambda g: [
            g.states[i].mass for i in g.topology.intermediate_edge_ids
        ],
    )


def _strip_projections(
    graph: Transition[Any, InteractionProperties],
) -> "FrozenTransition[Particle, InteractionProperties]":
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
    raise NotImplementedError(
        f"Cannot extract a particle from type {type(state).__name__}"
    )


def _collapse_graphs(
    graphs: Iterable[Transition[ParticleWithSpin, InteractionProperties]],
) -> "List[FrozenTransition[Tuple[Particle, ...], None]]":
    transition_groups: "Dict[Topology, MutableTransition[Set[Particle], None]]" = {
        g.topology: MutableTransition(
            g.topology,
            states={i: set() for i in g.topology.edges},  # type: ignore[misc]
            interactions={i: None for i in g.topology.nodes},  # type: ignore[misc]
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
    collected_graphs: "List[FrozenTransition[Tuple[Particle, ...], None]]" = []
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
