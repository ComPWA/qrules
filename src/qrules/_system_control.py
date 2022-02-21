"""Functions that steer operations of `qrules`."""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Type

import attrs

from .particle import Particle, ParticleCollection, ParticleWithSpin
from .quantum_numbers import (
    EdgeQuantumNumber,
    EdgeQuantumNumbers,
    InteractionProperties,
    NodeQuantumNumber,
    NodeQuantumNumbers,
    Parity,
)
from .settings import InteractionType
from .solving import GraphEdgePropertyMap, GraphNodePropertyMap, GraphSettings
from .topology import MutableTransition

Strength = float

GraphSettingsGroups = Dict[
    Strength, List[Tuple[MutableTransition, GraphSettings]]
]


def create_edge_properties(
    particle: Particle,
    spin_projection: Optional[float] = None,
) -> GraphEdgePropertyMap:
    edge_qn_mapping: Dict[str, Type[EdgeQuantumNumber]] = {
        qn_name: qn_type
        for qn_name, qn_type in EdgeQuantumNumbers.__dict__.items()
        if not qn_name.startswith("__")
    }  # Note using attrs.fields does not work here because init=False
    property_map: GraphEdgePropertyMap = {}
    isospin = None
    for qn_name, value in attrs.asdict(particle, recurse=False).items():
        if isinstance(value, Parity):
            value = value.value
        if qn_name in edge_qn_mapping:
            property_map[edge_qn_mapping[qn_name]] = value
        else:
            if "isospin" in qn_name:
                isospin = value
            elif "spin" in qn_name:
                property_map[EdgeQuantumNumbers.spin_magnitude] = value

    if spin_projection is not None:
        property_map[EdgeQuantumNumbers.spin_projection] = spin_projection
    if isospin is not None:
        property_map[EdgeQuantumNumbers.isospin_magnitude] = isospin.magnitude
        property_map[
            EdgeQuantumNumbers.isospin_projection
        ] = isospin.projection
    return property_map


def create_node_properties(
    interactions: InteractionProperties,
) -> GraphNodePropertyMap:
    node_qn_mapping: Dict[str, Type[NodeQuantumNumber]] = {
        qn_name: qn_type
        for qn_name, qn_type in NodeQuantumNumbers.__dict__.items()
        if not qn_name.startswith("__")
    }  # Note using attrs.fields does not work here because init=False
    property_map: GraphNodePropertyMap = {}
    for qn_name, value in attrs.asdict(interactions).items():
        if value is None:
            continue
        if qn_name in node_qn_mapping:
            property_map[node_qn_mapping[qn_name]] = value
        else:
            raise TypeError(
                "Missmatch between InteractionProperties and"
                " NodeQuantumNumbers. NodeQuantumNumbers does not define"
                f" {qn_name}"
            )
    return property_map


def create_particle(
    states: GraphEdgePropertyMap, particle_db: ParticleCollection
) -> ParticleWithSpin:
    """Create a Particle with spin projection from a qn dictionary.

    The implementation assumes the edge properties match the attributes of a
    particle inside the `.ParticleCollection`.

    Args:
        states: The quantum number dictionary.
        particle_db: A `.ParticleCollection` which is used to retrieve a
          reference `.particle` to lower the memory footprint.

    Raises:
        KeyError: If the edge properties do not contain the pid information or
          no particle with the same pid is found in the `.ParticleCollection`.

        ValueError: If the edge properties do not contain spin projection info.
    """
    particle = particle_db.find(int(states[EdgeQuantumNumbers.pid]))
    if EdgeQuantumNumbers.spin_projection not in states:
        raise ValueError(
            f"{GraphEdgePropertyMap.__name__} does not contain a spin"
            " projection"
        )
    spin_projection = states[EdgeQuantumNumbers.spin_projection]

    return (particle, spin_projection)


def create_interaction_properties(
    qn_solution: GraphNodePropertyMap,
) -> InteractionProperties:
    converted_solution = {k.__name__: v for k, v in qn_solution.items()}
    kw_args = {
        x.name: converted_solution[x.name]
        for x in attrs.fields(InteractionProperties)
        if x.name in converted_solution
    }

    return attrs.evolve(InteractionProperties(), **kw_args)


def filter_interaction_types(
    valid_determined_interaction_types: List[InteractionType],
    allowed_interaction_types: List[InteractionType],
) -> List[InteractionType]:
    int_type_intersection = list(
        set(allowed_interaction_types)
        & set(valid_determined_interaction_types)
    )
    if int_type_intersection:
        return int_type_intersection
    logging.warning(
        "The specified list of interaction types %s"
        " does not intersect with the valid list of interaction types %s"
        ".\nUsing valid list instead.",
        allowed_interaction_types,
        valid_determined_interaction_types,
    )
    return valid_determined_interaction_types


class InteractionDeterminator(ABC):
    """Interface for interaction determination."""

    @abstractmethod
    def check(
        self,
        in_states: List[ParticleWithSpin],
        out_states: List[ParticleWithSpin],
        interactions: InteractionProperties,
    ) -> List[InteractionType]:
        pass


class GammaCheck(InteractionDeterminator):
    """Conservation check for photons."""

    def check(
        self,
        in_states: List[ParticleWithSpin],
        out_states: List[ParticleWithSpin],
        interactions: InteractionProperties,
    ) -> List[InteractionType]:
        int_types = list(InteractionType)
        for particle, _ in in_states + out_states:
            if "gamma" in particle.name:
                int_types = [InteractionType.EM]
                break
        return int_types


class LeptonCheck(InteractionDeterminator):
    """Conservation check lepton numbers."""

    def check(
        self,
        in_states: List[ParticleWithSpin],
        out_states: List[ParticleWithSpin],
        interactions: InteractionProperties,
    ) -> List[InteractionType]:
        node_interaction_types = list(InteractionType)
        for particle, _ in in_states + out_states:
            if particle.is_lepton():
                if particle.name.startswith("nu("):
                    node_interaction_types = [InteractionType.WEAK]
                    break
                node_interaction_types = [
                    InteractionType.EM,
                    InteractionType.WEAK,
                ]
        return node_interaction_types


def remove_duplicate_solutions(
    solutions: List[
        "MutableTransition[ParticleWithSpin, InteractionProperties]"
    ],
    remove_qns_list: Optional[Set[Type[NodeQuantumNumber]]] = None,
    ignore_qns_list: Optional[Set[Type[NodeQuantumNumber]]] = None,
) -> "List[MutableTransition[ParticleWithSpin, InteractionProperties]]":
    if remove_qns_list is None:
        remove_qns_list = set()
    if ignore_qns_list is None:
        ignore_qns_list = set()
    logging.info("removing duplicate solutions...")
    logging.info(f"removing these qns from graphs: {remove_qns_list}")
    logging.info(f"ignoring qns in graph comparison: {ignore_qns_list}")

    filtered_solutions: List[
        MutableTransition[ParticleWithSpin, InteractionProperties]
    ] = []
    remove_counter = 0
    for sol_graph in solutions:
        sol_graph = _remove_qns_from_graph(sol_graph, remove_qns_list)
        found_graph = _check_equal_ignoring_qns(
            sol_graph, filtered_solutions, ignore_qns_list
        )
        if found_graph is None:
            filtered_solutions.append(sol_graph)
        else:
            # check if found solution also has the prefactors
            # if not overwrite them
            remove_counter += 1

    logging.info(f"removed {remove_counter} solutions")
    return filtered_solutions


def _remove_qns_from_graph(  # pylint: disable=too-many-branches
    graph: "MutableTransition[ParticleWithSpin, InteractionProperties]",
    qn_list: Set[Type[NodeQuantumNumber]],
) -> "MutableTransition[ParticleWithSpin, InteractionProperties]":
    new_interactions = {}
    for node_id in graph.topology.nodes:
        interactions = graph.interactions[node_id]
        new_interactions[node_id] = attrs.evolve(
            interactions, **{x.__name__: None for x in qn_list}
        )

    return attrs.evolve(graph, interactions=new_interactions)


def _check_equal_ignoring_qns(
    ref_graph: MutableTransition,
    solutions: List[MutableTransition],
    ignored_qn_list: Set[Type[NodeQuantumNumber]],
) -> Optional[MutableTransition]:
    """Define equal operator for graphs, ignoring certain quantum numbers."""
    if not isinstance(ref_graph, MutableTransition):
        raise TypeError("Reference graph has to be of type MutableTransition")
    found_graph = None
    interaction_comparator = NodePropertyComparator(ignored_qn_list)
    for graph in solutions:
        if isinstance(graph, MutableTransition):
            if graph.compare(
                ref_graph,
                state_comparator=lambda e1, e2: e1 == e2,
                interaction_comparator=interaction_comparator,
            ):
                found_graph = graph
                break
    return found_graph


class NodePropertyComparator:
    """Functor for comparing node properties in two graphs."""

    def __init__(
        self,
        ignored_qn_list: Optional[Set[Type[NodeQuantumNumber]]] = None,
    ) -> None:
        self.__ignored_qn_list = ignored_qn_list if ignored_qn_list else set()

    def __call__(
        self,
        interactions1: InteractionProperties,
        interactions2: InteractionProperties,
    ) -> bool:
        return attrs.evolve(
            interactions1,
            **{x.__name__: None for x in self.__ignored_qn_list},
        ) == attrs.evolve(
            interactions2,
            **{x.__name__: None for x in self.__ignored_qn_list},
        )


def filter_graphs(
    graphs: List[MutableTransition],
    filters: Iterable[Callable[[MutableTransition], bool]],
) -> List[MutableTransition]:
    r"""Implement filtering of a list of `.MutableTransition` 's.

    This function can be used to select a subset of
    `.MutableTransition` 's from a list. Only the graphs passing
    all supplied filters will be returned.

    Note:
        For the more advanced user, lambda functions can be used as filters.

    Example:
        Selecting only the solutions, in which the :math:`\rho` decays via
        p-wave:

        .. code-block:: python

            my_filter = require_interaction_property(
                "rho",
                InteractionQuantumNumberNames.L,
                create_spin_domain([1], True),
            )
            filtered_solutions = filter_graphs(solutions, [my_filter])
    """
    filtered_graphs = graphs
    for filter_ in filters:
        if not filtered_graphs:
            break
        filtered_graphs = list(filter(filter_, filtered_graphs))
    return filtered_graphs


def require_interaction_property(
    ingoing_particle_name: str,
    interaction_qn: Type[NodeQuantumNumber],
    allowed_values: List,
) -> "Callable[[MutableTransition[ParticleWithSpin, InteractionProperties]], bool]":
    """Filter function.

    Closure, which can be used as a filter function in :func:`.filter_graphs`.

    It selects graphs based on a requirement on the property of specific
    interaction nodes.

    Args:
        ingoing_particle_name (str): name of particle, used to find nodes which
            have a particle with this name as "ingoing"

        interaction_qn ([Type[NodeQuantumNumber]]): interaction quantum number

        allowed_values (list): list of allowed values, that the interaction
            quantum number may take

    Return:
        Callable[Any, bool]:
            - *True* if the graph has nodes with an ingoing particle of the
              given name, and the graph fullfills the quantum number
              requirement
            - *False* otherwise
    """

    def check(
        graph: "MutableTransition[ParticleWithSpin, InteractionProperties]",
    ) -> bool:
        node_ids = _find_node_ids_with_ingoing_particle_name(
            graph, ingoing_particle_name
        )
        if not node_ids:
            return False
        for i in node_ids:
            if (
                getattr(graph.interactions[i], interaction_qn.__name__)
                not in allowed_values
            ):
                return False
        return True

    return check


def _find_node_ids_with_ingoing_particle_name(
    graph: "MutableTransition[ParticleWithSpin, InteractionProperties]",
    ingoing_particle_name: str,
) -> List[int]:
    topology = graph.topology
    found_node_ids = []
    for node_id in topology.nodes:
        for edge_id in topology.get_edge_ids_ingoing_to_node(node_id):
            states = graph.states[edge_id]
            edge_particle_name = states[0].name
            if str(ingoing_particle_name) in str(edge_particle_name):
                found_node_ids.append(node_id)
                break
    return found_node_ids
