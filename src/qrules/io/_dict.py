"""Serialization from and to a `dict`."""

from __future__ import annotations

import json
from collections import abc
from os.path import dirname, realpath
from typing import Any

import attrs

from qrules.particle import Parity, Particle, ParticleCollection, Spin
from qrules.quantum_numbers import InteractionProperties
from qrules.topology import Edge, FrozenTransition, Topology
from qrules.transition import ReactionInfo, State


def from_particle_collection(particles: ParticleCollection) -> dict:
    return {"particles": [from_attrs_decorated(p) for p in particles]}


def from_attrs_decorated(inst: Any) -> dict:
    return attrs.asdict(
        inst,
        recurse=True,
        value_serializer=_value_serializer,
        filter=lambda a, v: a.init and a.default != v,
    )


def _value_serializer(inst: type, field: attrs.Attribute, value: Any) -> Any:
    if isinstance(value, abc.Mapping):
        if all(isinstance(p, Particle) for p in value.values()):
            return {k: v.name for k, v in value.items()}
        return dict(value)
    if not isinstance(inst, (ReactionInfo, State, FrozenTransition)):  # noqa: SIM102
        if isinstance(value, Particle):
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
    return ParticleCollection(build_particle(p) for p in definition["particles"])


def build_particle(definition: dict) -> Particle:
    isospin_def = definition.get("isospin")
    if isospin_def is not None:
        definition["isospin"] = Spin(**isospin_def)
    for parity in ["parity", "c_parity", "g_parity"]:
        parity_def = definition.get(parity)
        if parity_def is not None:
            definition[parity] = Parity(**parity_def)
    return Particle(**definition)


def build_reaction_info(definition: dict) -> ReactionInfo:
    transitions = [
        build_transition(transition_def) for transition_def in definition["transitions"]
    ]
    return ReactionInfo(transitions, formalism=definition["formalism"])


def build_transition(
    definition: dict,
) -> FrozenTransition[State, InteractionProperties]:
    topology = build_topology(definition["topology"])
    states_def: dict[int, dict] = definition["states"]
    states: dict[int, State] = {}
    for i, edge_def in states_def.items():
        states[int(i)] = build_state(edge_def)
    interactions_def: dict[int, dict] = definition["interactions"]
    interactions = {
        int(i): InteractionProperties(**node_def)
        for i, node_def in interactions_def.items()
    }
    return FrozenTransition(topology, states, interactions)


def build_state(definition: Any) -> State:
    if isinstance(definition, (list, tuple)) and len(definition) == 2:
        particle = build_particle(definition[0])
        spin_projection = float(definition[1])
        return State(particle, spin_projection)
    if isinstance(definition, dict):
        particle = build_particle(definition["particle"])
        spin_projection = float(definition["spin_projection"])
        return State(particle, spin_projection)
    raise NotImplementedError


def build_topology(definition: dict) -> Topology:
    nodes = definition["nodes"]
    edges_def: dict[int, dict] = definition["edges"]
    edges = {int(i): Edge(**edge_def) for i, edge_def in edges_def.items()}
    return Topology(nodes, edges)


def validate_particle_collection(instance: dict) -> None:
    import jsonschema  # noqa: PLC0415

    jsonschema.validate(instance=instance, schema=__SCHEMA_PARTICLES)


__QRULES_PATH = dirname(dirname(realpath(__file__)))
with open(f"{__QRULES_PATH}/particle-validation.json") as __STREAM:
    __SCHEMA_PARTICLES = json.load(__STREAM)
