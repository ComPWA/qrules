# pylint: disable=too-many-return-statements
"""Serialization module for the `qrules`.

The `.io` module provides tools to export or import objects from `qrules` to
and from disk, so that they can be used by external packages, or just to store
(cache) the state of the system.
"""

import json
from collections import abc
from pathlib import Path
from typing import Any, Dict, Optional

import attrs
import yaml

from qrules.particle import Particle, ParticleCollection
from qrules.topology import StateTransitionGraph, Topology
from qrules.transition import (
    ProblemSet,
    ReactionInfo,
    State,
    StateTransition,
    StateTransitionCollection,
)

from . import _dict, _dot


def asdict(instance: object) -> dict:
    # pylint: disable=protected-access
    if isinstance(instance, Particle):
        return _dict.from_particle(instance)
    if isinstance(instance, ParticleCollection):
        return _dict.from_particle_collection(instance)
    if isinstance(
        instance,
        (ReactionInfo, State, StateTransition, StateTransitionCollection),
    ):
        return attrs.asdict(
            instance,
            recurse=True,
            filter=lambda a, _: a.init,
            value_serializer=_dict._value_serializer,
        )
    if isinstance(instance, StateTransitionGraph):
        return _dict.from_stg(instance)
    if isinstance(instance, Topology):
        return _dict.from_topology(instance)
    raise NotImplementedError(
        "No conversion for dict available for class"
        f" {instance.__class__.__name__}"
    )


def fromdict(definition: dict) -> object:
    keys = set(definition.keys())
    if __REQUIRED_PARTICLE_FIELDS <= keys:
        return _dict.build_particle(definition)
    if keys == {"particles"}:
        return _dict.build_particle_collection(definition)
    if keys == {"transition_groups", "formalism"}:
        return _dict.build_reaction_info(definition)
    if keys == {"topology", "states", "interactions"}:
        return _dict.build_state_transition(definition)
    if keys == {"transitions"}:
        return _dict.build_stc(definition)
    if keys == {"topology", "edge_props", "node_props"}:
        return _dict.build_stg(definition)
    if keys == __REQUIRED_TOPOLOGY_FIELDS:
        return _dict.build_topology(definition)
    raise NotImplementedError(f"Could not determine type from keys {keys}")


__REQUIRED_PARTICLE_FIELDS = {
    field.name
    for field in attrs.fields(Particle)
    if field.default == attrs.NOTHING
}
__REQUIRED_TOPOLOGY_FIELDS = {
    field.name for field in attrs.fields(Topology) if field.init
}


def asdot(
    instance: object,
    *,
    render_node: bool = False,
    render_final_state_id: bool = True,
    render_resonance_id: bool = False,
    render_initial_state_id: bool = False,
    strip_spin: bool = False,
    collapse_graphs: bool = False,
    edge_style: Optional[Dict[str, Any]] = None,
    node_style: Optional[Dict[str, Any]] = None,
    **figure_style: Any,
) -> str:
    """Convert a `object` to a DOT language `str`.

    Only works for objects that can be represented as a graph, particularly a
    `.StateTransitionGraph` or a `list` of `.StateTransitionGraph` instances.

    Args:
        instance: the input `object` that is to be rendered as DOT (graphviz)
            language.

        strip_spin: Normally, each `.StateTransitionGraph` has a `.Particle`
            with a spin projection on its edges. This option hides the
            projections, leaving only `.Particle` names on edges.

        collapse_graphs: Group all transitions by equivalent kinematic topology
            and combine all allowed particles on each edge.

        render_node: Whether or not to render node ID (in the case of a
            `.Topology`) and/or node properties (in the case of a
            `.StateTransitionGraph`). Meaning of the labels:

            - :math:`P`: parity prefactor
            - :math:`s`: tuple of **coupled spin** magnitude and its
              projection
            - :math:`l`: tuple of **angular momentum** and its projection

            See `.InteractionProperties` for more info.

        render_final_state_id: Add edge IDs for the final state edges.
        render_resonance_id: Add edge IDs for the intermediate state edges.
        render_initial_state_id: Add edge IDs for the initial state edges.
        edge_style: Styling of a Graphviz edge.
        node_style: Styling of a Graphviz node.
        figure_style: Styling of the whole figure.

    .. seealso::

        See `Graphviz attributes <https://graphviz.org/doc/info/attrs.html>`_
        for the available styling arguments.

    .. seealso:: :doc:`/usage/visualize`
    """
    if edge_style is None:
        edge_style = {}
    if node_style is None:
        node_style = {}
    if isinstance(instance, StateTransition):
        instance = instance.to_graph()
    if isinstance(instance, (ProblemSet, StateTransitionGraph, Topology)):
        dot = _dot.graph_to_dot(
            instance,
            render_node=render_node,
            render_final_state_id=render_final_state_id,
            render_resonance_id=render_resonance_id,
            render_initial_state_id=render_initial_state_id,
            edge_style=edge_style,
            node_style=node_style,
        )
        return _dot.insert_graphviz_styling(dot, graphviz_attrs=figure_style)
    if isinstance(instance, (ReactionInfo, StateTransitionCollection)):
        instance = instance.to_graphs()
    if isinstance(instance, abc.Iterable):
        dot = _dot.graph_list_to_dot(
            instance,
            render_node=render_node,
            render_final_state_id=render_final_state_id,
            render_resonance_id=render_resonance_id,
            render_initial_state_id=render_initial_state_id,
            strip_spin=strip_spin,
            collapse_graphs=collapse_graphs,
            edge_style=edge_style,
            node_style=node_style,
        )
        return _dot.insert_graphviz_styling(dot, graphviz_attrs=figure_style)
    raise NotImplementedError(
        f"Cannot convert a {instance.__class__.__name__} to DOT language"
    )


def load(filename: str) -> object:
    with open(filename) as stream:
        file_extension = _get_file_extension(filename)
        if file_extension == "json":
            definition = json.load(stream)
            return fromdict(definition)
        if file_extension in ["yaml", "yml"]:
            definition = yaml.load(stream, Loader=yaml.SafeLoader)
            return fromdict(definition)
    raise NotImplementedError(
        f'No loader defined for file type "{file_extension}"'
    )


class _IncreasedIndent(yaml.Dumper):
    # pylint: disable=too-many-ancestors
    def increase_indent(
        self, flow: bool = False, indentless: bool = False
    ) -> None:
        return super().increase_indent(flow, False)

    def write_line_break(self, data: Optional[str] = None) -> None:
        """See https://stackoverflow.com/a/44284819."""
        super().write_line_break(data)
        if len(self.indents) == 1:
            super().write_line_break()


def write(instance: object, filename: str) -> None:
    with open(filename, "w") as stream:
        file_extension = _get_file_extension(filename)
        if file_extension == "json":
            json.dump(asdict(instance), stream, indent=2, cls=JSONSetEncoder)
            return
        if file_extension in ["yaml", "yml"]:
            yaml.dump(
                asdict(instance),
                stream,
                sort_keys=False,
                Dumper=_IncreasedIndent,
                default_flow_style=False,
            )
            return
        if file_extension == "gv":
            if isinstance(instance, str):  # direct output of asdot
                output_str = instance
            else:
                output_str = asdot(instance)
            with open(filename, "w") as stream:
                stream.write(output_str)
            return
    raise NotImplementedError(
        f'No writer defined for file type "{file_extension}"'
    )


def _get_file_extension(filename: str) -> str:
    path = Path(filename)
    extension = path.suffix.lower()
    if not extension:
        raise ValueError(f'No file extension in file name "{filename}"')
    extension = extension[1:]
    return extension


class JSONSetEncoder(json.JSONEncoder):
    """`~json.JSONEncoder` that supports `set` and `frozenset`.

    >>> import json
    >>> instance = {"val1": {1, 2, 3}, "val2": frozenset({2, 3, 4, 5})}
    >>> json.dumps(instance, cls=JSONSetEncoder)
    '{"val1": [1, 2, 3], "val2": [2, 3, 4, 5]}'
    """

    # https://stackoverflow.com/a/8230505
    def default(self, o: Any) -> Any:
        if isinstance(o, (frozenset, set)):
            return list(o)
        return super().default(o)
