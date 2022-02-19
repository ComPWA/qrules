# pylint: disable=import-outside-toplevel
"""Serialization from and to a `dict`."""

import json
from collections import abc
from os.path import dirname, realpath
from typing import Any, Dict

import attrs

from qrules.particle import (
    Parity,
    Particle,
    ParticleCollection,
    ParticleWithSpin,
    Spin,
)
from qrules.quantum_numbers import InteractionProperties
from qrules.topology import Edge, StateTransitionGraph, Topology
from qrules.transition import (
    ReactionInfo,
    State,
    StateTransition,
    StateTransitionCollection,
)


def from_particle_collection(particles: ParticleCollection) -> dict:
    return {"particles": [from_particle(p) for p in particles]}


def from_particle(particle: Particle) -> dict:
    return attrs.asdict(
        particle,
        recurse=True,
        value_serializer=_value_serializer,
        filter=lambda attribute, value: attribute.default != value,
    )


def from_stg(graph: StateTransitionGraph[ParticleWithSpin]) -> dict:
    topology = graph.topology
    edge_props_def = {}
    for i in topology.edges:
        particle, spin_projection = graph.get_edge_props(i)
        if isinstance(spin_projection, float) and spin_projection.is_integer():
            spin_projection = int(spin_projection)
        edge_props_def[i] = {
            "particle": from_particle(particle),
            "spin_projection": spin_projection,
        }
    node_props_def = {}
    for i in topology.nodes:
        node_prop = graph.get_node_props(i)
        node_props_def[i] = attrs.asdict(
            node_prop, filter=lambda a, v: a.init and a.default != v
        )
    return {
        "topology": from_topology(topology),
        "edge_props": edge_props_def,
        "node_props": node_props_def,
    }


def from_topology(topology: Topology) -> dict:
    return attrs.asdict(
        topology,
        recurse=True,
        value_serializer=_value_serializer,
        filter=lambda a, v: a.init and a.default != v,
    )


def _value_serializer(  # pylint: disable=unused-argument
    inst: type, field: attrs.Attribute, value: Any
) -> Any:
    if isinstance(value, abc.Mapping):
        if all(map(lambda p: isinstance(p, Particle), value.values())):
            return {k: v.name for k, v in value.items()}
        return dict(value)
    if not isinstance(
        inst, (ReactionInfo, State, StateTransition, StateTransitionCollection)
    ) and isinstance(value, Particle):
        return value.name
    if isinstance(value, Parity):
        return {"value": value.value}
    if isinstance(value, Spin):
        return {
            "magnitude": value.magnitude,
            "projection": value.projection,
        }
    return value


def build_particle_collection(
    definition: dict, do_validate: bool = True
) -> ParticleCollection:
    if do_validate:
        validate_particle_collection(definition)
    return ParticleCollection(
        build_particle(p) for p in definition["particles"]
    )


def build_particle(definition: dict) -> Particle:
    isospin_def = definition.get("isospin", None)
    if isospin_def is not None:
        definition["isospin"] = Spin(**isospin_def)
    for parity in ["parity", "c_parity", "g_parity"]:
        parity_def = definition.get(parity, None)
        if parity_def is not None:
            definition[parity] = Parity(**parity_def)
    return Particle(**definition)


def build_reaction_info(definition: dict) -> ReactionInfo:
    transition_groups = [
        build_stc(graph_def) for graph_def in definition["transition_groups"]
    ]
    return ReactionInfo(
        transition_groups=transition_groups,
        formalism=definition["formalism"],
    )


def build_stg(definition: dict) -> StateTransitionGraph[ParticleWithSpin]:
    topology = build_topology(definition["topology"])
    edge_props_def: Dict[int, dict] = definition["edge_props"]
    edge_props: Dict[int, ParticleWithSpin] = {}
    for i, edge_def in edge_props_def.items():
        particle = build_particle(edge_def["particle"])
        spin_projection = float(edge_def["spin_projection"])
        if spin_projection.is_integer():
            spin_projection = int(spin_projection)
        edge_props[int(i)] = (particle, spin_projection)
    node_props_def: Dict[int, dict] = definition["node_props"]
    node_props = {
        int(i): InteractionProperties(**node_def)
        for i, node_def in node_props_def.items()
    }
    return StateTransitionGraph(
        topology=topology,
        edge_props=edge_props,
        node_props=node_props,
    )


def build_stc(definition: dict) -> StateTransitionCollection:
    transitions = [
        build_state_transition(graph_def)
        for graph_def in definition["transitions"]
    ]
    return StateTransitionCollection(transitions=transitions)


def build_state_transition(definition: dict) -> StateTransition:
    topology = build_topology(definition["topology"])
    states = {
        int(i): State(
            particle=build_particle(state_def["particle"]),
            spin_projection=float(state_def["spin_projection"]),
        )
        for i, state_def in definition["states"].items()
    }
    interactions = {
        int(i): InteractionProperties(**interaction_def)
        for i, interaction_def in definition["interactions"].items()
    }
    return StateTransition(
        topology=topology,
        states=states,
        interactions=interactions,
    )


def build_topology(definition: dict) -> Topology:
    nodes = definition["nodes"]
    edges_def: Dict[int, dict] = definition["edges"]
    edges = {int(i): Edge(**edge_def) for i, edge_def in edges_def.items()}
    return Topology(
        edges=edges,
        nodes=nodes,
    )


def validate_particle_collection(instance: dict) -> None:
    import jsonschema

    jsonschema.validate(instance=instance, schema=__SCHEMA_PARTICLES)


__QRULES_PATH = dirname(dirname(realpath(__file__)))
with open(f"{__QRULES_PATH}/particle-validation.json") as __STREAM:
    __SCHEMA_PARTICLES = json.load(__STREAM)
