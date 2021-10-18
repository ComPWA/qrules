"""Generate dot sources.

See :doc:`/usage/visualize` for more info.
"""

import functools
import re
from collections import abc
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from qrules.combinatorics import InitialFacts
from qrules.particle import Particle, ParticleCollection, ParticleWithSpin
from qrules.quantum_numbers import InteractionProperties, _to_fraction
from qrules.solving import EdgeSettings, GraphSettings, NodeSettings
from qrules.topology import StateTransitionGraph, Topology
from qrules.transition import ProblemSet, StateTransition

_DOT_HEAD = """digraph {
    rankdir=LR;
    node [shape=point, width=0];
    edge [arrowhead=none];
"""
_DOT_TAIL = "}\n"
_DOT_RANK_SAME = "    {{ rank=same {} }};\n"
_DOT_DEFAULT_NODE = '    "{}" [shape=none, label="{}"];\n'
_DOT_DEFAULT_EDGE = '    "{}" -> "{}";\n'
_DOT_LABEL_EDGE = '    "{}" -> "{}" [label="{}"];\n'


def embed_dot(func: Callable) -> Callable:
    """Add a DOT head and tail to some DOT content."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        dot = _DOT_HEAD
        dot += func(*args, **kwargs)
        dot += _DOT_TAIL
        return dot

    return wrapper


@embed_dot
def graph_list_to_dot(
    graphs: Iterable[StateTransitionGraph],
    *,
    render_node: bool,
    render_final_state_id: bool,
    render_resonance_id: bool,
    render_initial_state_id: bool,
    strip_spin: bool,
    collapse_graphs: bool,
) -> str:
    if strip_spin and collapse_graphs:
        raise ValueError("Cannot both strip spin and collapse graphs")
    if collapse_graphs:
        if render_node:
            raise ValueError(
                "Collapsed graphs cannot be rendered with node properties"
            )
        graphs = _collapse_graphs(graphs)
    elif strip_spin:
        if render_node:
            raise ValueError(
                "Graphs without spin projections cannot be rendered with node"
                " properties"
            )
        graphs = _get_particle_graphs(graphs)
    dot = ""
    if not isinstance(graphs, abc.Sequence):
        graphs = list(graphs)
    for i, graph in enumerate(reversed(graphs)):
        dot += __graph_to_dot_content(
            graph,
            prefix=f"g{i}_",
            render_node=render_node,
            render_final_state_id=render_final_state_id,
            render_resonance_id=render_resonance_id,
            render_initial_state_id=render_initial_state_id,
        )
    return dot


@embed_dot
def graph_to_dot(
    graph: StateTransitionGraph,
    *,
    render_node: bool,
    render_final_state_id: bool,
    render_resonance_id: bool,
    render_initial_state_id: bool,
) -> str:
    return __graph_to_dot_content(
        graph,
        render_node=render_node,
        render_final_state_id=render_final_state_id,
        render_resonance_id=render_resonance_id,
        render_initial_state_id=render_initial_state_id,
    )


def __graph_to_dot_content(  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    graph: Union[
        ProblemSet,
        StateTransition,
        StateTransitionGraph,
        Topology,
        Tuple[Topology, InitialFacts],
        Tuple[Topology, GraphSettings],
    ],
    prefix: str = "",
    *,
    render_node: bool,
    render_final_state_id: bool,
    render_resonance_id: bool,
    render_initial_state_id: bool,
) -> str:
    dot = ""
    if isinstance(graph, tuple) and len(graph) == 2:
        topology: Topology = graph[0]
        rendered_graph: Union[
            GraphSettings,
            InitialFacts,
            ProblemSet,
            StateTransition,
            StateTransitionGraph,
            Topology,
        ] = graph[1]
    elif isinstance(graph, ProblemSet):
        rendered_graph = graph
        topology = graph.topology
    elif isinstance(graph, (StateTransition, StateTransitionGraph)):
        rendered_graph = graph
        topology = graph.topology
    elif isinstance(graph, Topology):
        rendered_graph = graph
        topology = graph
    else:
        raise NotImplementedError(
            f"Cannot render {graph.__class__.__name__} as dot"
        )
    top = topology.incoming_edge_ids
    outs = topology.outgoing_edge_ids
    for edge_id in top | outs:
        if edge_id in top:
            render = render_initial_state_id
        else:
            render = render_final_state_id
        edge_label = __get_edge_label(rendered_graph, edge_id, render)
        dot += _DOT_DEFAULT_NODE.format(
            prefix + __node_name(edge_id),
            edge_label,
        )
    dot += __rank_string(top, prefix)
    dot += __rank_string(outs, prefix)
    for i, edge in topology.edges.items():
        j, k = edge.ending_node_id, edge.originating_node_id
        if j is None or k is None:
            dot += _DOT_DEFAULT_EDGE.format(
                prefix + __node_name(i, k), prefix + __node_name(i, j)
            )
        else:
            dot += _DOT_LABEL_EDGE.format(
                prefix + __node_name(i, k),
                prefix + __node_name(i, j),
                __get_edge_label(rendered_graph, i, render_resonance_id),
            )
    if isinstance(graph, ProblemSet):
        node_props = graph.solving_settings.node_settings
        for node_id, settings in node_props.items():
            node_label = ""
            if render_node:
                node_label = __node_label(settings)
            dot += _DOT_DEFAULT_NODE.format(
                f"{prefix}node{node_id}", node_label
            )
    if isinstance(graph, (StateTransition, StateTransitionGraph)):
        if isinstance(graph, StateTransition):
            interactions: Mapping[
                int, InteractionProperties
            ] = graph.interactions
        else:
            interactions = {i: graph.get_node_props(i) for i in topology.nodes}
        for node_id, node_prop in interactions.items():
            node_label = ""
            if render_node:
                node_label = __node_label(node_prop)
            dot += _DOT_DEFAULT_NODE.format(
                f"{prefix}node{node_id}", node_label
            )
    if isinstance(graph, Topology):
        if len(topology.nodes) > 1:
            for node_id in topology.nodes:
                node_label = ""
                if render_node:
                    node_label = f"({node_id})"
                dot += _DOT_DEFAULT_NODE.format(
                    f"{prefix}node{node_id}", node_label
                )
    return dot


def __node_name(edge_id: int, node_id: Optional[int] = None) -> str:
    if node_id is None:
        return f"edge{edge_id}"
    return f"node{node_id}"


def __rank_string(node_edge_ids: Iterable[int], prefix: str = "") -> str:
    name_list = [f'"{prefix}{__node_name(i)}"' for i in node_edge_ids]
    name_string = ", ".join(name_list)
    return _DOT_RANK_SAME.format(name_string)


def __get_edge_label(
    graph: Union[
        GraphSettings,
        InitialFacts,
        ProblemSet,
        StateTransition,
        StateTransitionGraph,
        Topology,
    ],
    edge_id: int,
    render_edge_id: bool,
) -> str:
    if isinstance(graph, GraphSettings):
        edge_setting = graph.edge_settings.get(edge_id)
        return ___render_edge_with_id(edge_id, edge_setting, render_edge_id)
    if isinstance(graph, InitialFacts):
        initial_fact = graph.edge_props.get(edge_id)
        return ___render_edge_with_id(edge_id, initial_fact, render_edge_id)
    if isinstance(graph, ProblemSet):
        edge_setting = graph.solving_settings.edge_settings.get(edge_id)
        initial_fact = graph.initial_facts.edge_props.get(edge_id)
        edge_property: Optional[Union[EdgeSettings, ParticleWithSpin]] = None
        if edge_setting:
            edge_property = edge_setting
        if initial_fact:
            edge_property = initial_fact
        return ___render_edge_with_id(edge_id, edge_property, render_edge_id)
    if isinstance(graph, StateTransition):
        graph = graph.to_graph()
    if isinstance(graph, StateTransitionGraph):
        edge_prop = graph.get_edge_props(edge_id)
        return ___render_edge_with_id(edge_id, edge_prop, render_edge_id)
    if isinstance(graph, Topology):
        if render_edge_id:
            return str(edge_id)
        return ""
    raise NotImplementedError(
        f"Cannot render {graph.__class__.__name__} as dot"
    )


def ___render_edge_with_id(
    edge_id: int,
    edge_prop: Optional[
        Union[EdgeSettings, ParticleCollection, Particle, ParticleWithSpin]
    ],
    render_edge_id: bool,
) -> str:
    if edge_prop is None or not edge_prop:
        return str(edge_id)
    edge_label = __render_edge_property(edge_prop)
    if not render_edge_id:
        return edge_label
    if "\n" in edge_label:
        return f"{edge_id}:\n{edge_label}"
    return f"{edge_id}: {edge_label}"


def __render_edge_property(
    edge_prop: Optional[
        Union[EdgeSettings, ParticleCollection, Particle, ParticleWithSpin]
    ]
) -> str:
    if isinstance(edge_prop, EdgeSettings):
        return __render_settings(edge_prop)
    if isinstance(edge_prop, Particle):
        return edge_prop.name
    if isinstance(edge_prop, tuple):
        particle, spin_projection = edge_prop
        projection_label = _to_fraction(spin_projection, render_plus=True)
        return f"{particle.name}[{projection_label}]"
    if isinstance(edge_prop, ParticleCollection):
        return "\n".join(sorted(edge_prop.names))
    raise NotImplementedError


def __node_label(node_prop: Union[InteractionProperties, NodeSettings]) -> str:
    if isinstance(node_prop, NodeSettings):
        return __render_settings(node_prop)
    if isinstance(node_prop, InteractionProperties):
        output = ""
        if node_prop.l_magnitude is not None:
            l_magnitude = _to_fraction(node_prop.l_magnitude)
            if node_prop.l_projection is None:
                l_label = l_magnitude
            else:
                l_projection = _to_fraction(node_prop.l_projection)
                l_label = f"({l_magnitude}, {l_projection})"
            output += f"l={l_label}\n"
        if node_prop.s_magnitude is not None:
            s_magnitude = _to_fraction(node_prop.s_magnitude)
            if node_prop.s_projection is None:
                s_label = s_magnitude
            else:
                s_projection = _to_fraction(node_prop.s_projection)
                s_label = f"({s_magnitude}, {s_projection})"
            output += f"s={s_label}\n"
        if node_prop.parity_prefactor is not None:
            label = _to_fraction(node_prop.parity_prefactor, render_plus=True)
            output += f"P={label}"
        return output
    raise NotImplementedError


def __render_settings(settings: Union[EdgeSettings, NodeSettings]) -> str:
    output = ""
    if settings.rule_priorities:
        output += "RULE PRIORITIES\n"
        rule_names = map(
            lambda item: f"{item[0].__name__} - {item[1]}",  # type: ignore[union-attr]
            settings.rule_priorities.items(),
        )
        sorted_names = sorted(rule_names, key=__extract_priority, reverse=True)
        output += "\n".join(sorted_names)
    if settings.qn_domains:
        if output:
            output += "\n"
        domains = map(
            lambda item: f"{item[0].__name__} ∊ {item[1]}",
            settings.qn_domains.items(),
        )
        output += "DOMAINS\n"
        output += "\n".join(sorted(domains))
    return output


def __extract_priority(description: str) -> int:
    matches = re.match(r".* \- ([0-9]+)$", description)
    if matches is None:
        raise ValueError(f"{description} does not contain a priority number")
    priority = matches[1]
    return int(priority)


def _get_particle_graphs(
    graphs: Iterable[StateTransitionGraph[ParticleWithSpin]],
) -> List[StateTransitionGraph[Particle]]:
    """Strip `list` of `.StateTransitionGraph` s of the spin projections.

    Extract a `list` of `.StateTransitionGraph` instances with only
    `.Particle` instances on the edges.

    .. seealso:: :doc:`/usage/visualize`
    """
    inventory: List[StateTransitionGraph[Particle]] = []
    for transition in graphs:
        if isinstance(transition, StateTransition):
            transition = transition.to_graph()
        if any(
            transition.compare(
                other, edge_comparator=lambda e1, e2: e1[0] == e2
            )
            for other in inventory
        ):
            continue
        stripped_graph = __strip_spin(transition)
        inventory.append(stripped_graph)
    inventory = sorted(
        inventory,
        key=lambda g: [
            g.get_edge_props(i).mass for i in g.topology.intermediate_edge_ids
        ],
    )
    return inventory


def __strip_spin(
    graph: StateTransitionGraph[ParticleWithSpin],
) -> StateTransitionGraph[Particle]:
    if isinstance(graph, StateTransition):
        graph = graph.to_graph()
    new_edge_props = {}
    for edge_id in graph.topology.edges:
        edge_props = graph.get_edge_props(edge_id)
        if edge_props:
            new_edge_props[edge_id] = edge_props[0]
    return StateTransitionGraph[Particle](
        topology=graph.topology,
        node_props={
            i: node_props
            for i, node_props in zip(
                graph.topology.nodes,
                map(
                    graph.get_node_props,
                    graph.topology.nodes,
                ),
            )
            if node_props
        },
        edge_props=new_edge_props,
    )


def _collapse_graphs(
    graphs: Iterable[StateTransitionGraph[ParticleWithSpin]],
) -> List[StateTransitionGraph[ParticleCollection]]:
    def merge_into(
        graph: StateTransitionGraph[Particle],
        merged_graph: StateTransitionGraph[ParticleCollection],
    ) -> None:
        if (
            graph.topology.intermediate_edge_ids
            != merged_graph.topology.intermediate_edge_ids
        ):
            raise ValueError(
                "Cannot merge graphs that don't have the same edge IDs"
            )
        for i in graph.topology.edges:
            particle = graph.get_edge_props(i)
            other_particles = merged_graph.get_edge_props(i)
            if particle not in other_particles:
                other_particles += particle

    def is_same_shape(
        graph: StateTransitionGraph[Particle],
        merged_graph: StateTransitionGraph[ParticleCollection],
    ) -> bool:
        if graph.topology.edges != merged_graph.topology.edges:
            return False
        for edge_id in (
            graph.topology.incoming_edge_ids | graph.topology.outgoing_edge_ids
        ):
            edge_prop = merged_graph.get_edge_props(edge_id)
            if len(edge_prop) != 1:
                return False
            other_particle = next(iter(edge_prop))
            if other_particle != graph.get_edge_props(edge_id):
                return False
        return True

    particle_graphs = _get_particle_graphs(graphs)
    inventory: List[StateTransitionGraph[ParticleCollection]] = []
    for graph in particle_graphs:
        append_to_inventory = True
        for merged_graph in inventory:
            if is_same_shape(graph, merged_graph):
                merge_into(graph, merged_graph)
                append_to_inventory = False
                break
        if append_to_inventory:
            new_edge_props = {
                edge_id: ParticleCollection({graph.get_edge_props(edge_id)})
                for edge_id in graph.topology.edges
            }
            inventory.append(
                StateTransitionGraph[ParticleCollection](
                    topology=graph.topology,
                    node_props={
                        i: graph.get_node_props(i)
                        for i in graph.topology.nodes
                    },
                    edge_props=new_edge_props,
                )
            )
    return inventory
