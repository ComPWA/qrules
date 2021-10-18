"""Default configuration for `qrules`.

It is possible to change some settings from the outside, for instance:

>>> import qrules
>>> qrules.settings.MAX_ANGULAR_MOMENTUM = 4
>>> qrules.settings.MAX_SPIN_MAGNITUDE = 3
"""

from copy import deepcopy
from enum import Enum, auto
from os.path import dirname, join, realpath
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from qrules.conservation_rules import (
    BaryonNumberConservation,
    BottomnessConservation,
    ChargeConservation,
    CharmConservation,
    ConservationRule,
    EdgeQNConservationRule,
    ElectronLNConservation,
    GraphElementRule,
    MassConservation,
    MuonLNConservation,
    StrangenessConservation,
    TauLNConservation,
    c_parity_conservation,
    clebsch_gordan_helicity_to_canonical,
    g_parity_conservation,
    gellmann_nishijima,
    helicity_conservation,
    identical_particle_symmetrization,
    isospin_conservation,
    isospin_validity,
    ls_spin_validity,
    parity_conservation,
    parity_conservation_helicity,
    spin_conservation,
    spin_magnitude_conservation,
    spin_validity,
)
from qrules.particle import Particle, ParticleCollection
from qrules.quantum_numbers import EdgeQuantumNumbers as EdgeQN
from qrules.quantum_numbers import NodeQuantumNumbers as NodeQN
from qrules.quantum_numbers import arange
from qrules.solving import EdgeSettings, NodeSettings

__QRULES_PATH = dirname(realpath(__file__))
ADDITIONAL_PARTICLES_DEFINITIONS_PATH: str = join(
    __QRULES_PATH, "additional_definitions.yml"
)

CONSERVATION_LAW_PRIORITIES: Dict[
    Union[GraphElementRule, EdgeQNConservationRule, ConservationRule], int
] = {
    MassConservation: 10,
    ElectronLNConservation: 45,
    MuonLNConservation: 44,
    TauLNConservation: 43,
    BaryonNumberConservation: 90,
    StrangenessConservation: 69,
    CharmConservation: 70,
    BottomnessConservation: 68,
    ChargeConservation: 100,
    spin_conservation: 8,
    spin_magnitude_conservation: 8,
    parity_conservation: 6,
    c_parity_conservation: 5,
    g_parity_conservation: 3,
    isospin_conservation: 60,
    ls_spin_validity: 89,
    helicity_conservation: 7,
    parity_conservation_helicity: 4,
    identical_particle_symmetrization: 2,
}
"""Determines the order with which to verify conservation rules."""


EDGE_RULE_PRIORITIES: Dict[GraphElementRule, int] = {
    gellmann_nishijima: 50,
    isospin_validity: 61,
    spin_validity: 62,
}
"""Determines the order with which to verify `.Edge` conservation rules."""


class InteractionType(Enum):
    """Types of interactions in the form of an enumerate."""

    STRONG = auto()
    EM = auto()
    WEAK = auto()

    @staticmethod
    def from_str(description: str) -> "InteractionType":
        description_lower = description.lower()
        if description_lower.startswith("e"):
            return InteractionType.EM
        if description_lower.startswith("s"):
            return InteractionType.STRONG
        if description_lower.startswith("w"):
            return InteractionType.WEAK
        raise ValueError(
            f'Could not determine interaction type from "{description}"'
        )


def create_interaction_settings(  # pylint: disable=too-many-locals,too-many-arguments
    formalism: str,
    particle_db: ParticleCollection,
    nbody_topology: bool = False,
    mass_conservation_factor: Optional[float] = 3.0,
    max_angular_momentum: int = 2,
    max_spin_magnitude: float = 2.0,
) -> Dict[InteractionType, Tuple[EdgeSettings, NodeSettings]]:
    """Create a container that holds the settings for `.InteractionType`."""
    formalism_edge_settings = EdgeSettings(
        conservation_rules={
            isospin_validity,
            gellmann_nishijima,
            spin_validity,
        },
        rule_priorities=EDGE_RULE_PRIORITIES,
        qn_domains=_create_domains(particle_db),
    )
    formalism_node_settings = NodeSettings(
        rule_priorities=CONSERVATION_LAW_PRIORITIES
    )

    angular_momentum_domain = __get_ang_mom_magnitudes(
        nbody_topology, max_angular_momentum
    )
    spin_magnitude_domain = __get_spin_magnitudes(
        nbody_topology, max_spin_magnitude
    )
    if "helicity" in formalism:
        formalism_node_settings.conservation_rules = {
            spin_magnitude_conservation,
            helicity_conservation,
        }
        formalism_node_settings.qn_domains = {
            NodeQN.l_magnitude: angular_momentum_domain,
            NodeQN.s_magnitude: spin_magnitude_domain,
        }
    elif formalism == "canonical":
        formalism_node_settings.conservation_rules = {
            spin_magnitude_conservation
        }
        if nbody_topology:
            formalism_node_settings.conservation_rules = {
                spin_conservation,
                ls_spin_validity,
            }
        formalism_node_settings.qn_domains = {
            NodeQN.l_magnitude: angular_momentum_domain,
            NodeQN.l_projection: __extend_negative(angular_momentum_domain),
            NodeQN.s_magnitude: spin_magnitude_domain,
            NodeQN.s_projection: __extend_negative(spin_magnitude_domain),
        }
    if formalism == "canonical-helicity":
        formalism_node_settings.conservation_rules.update(
            {
                clebsch_gordan_helicity_to_canonical,
                ls_spin_validity,
            }
        )
        formalism_node_settings.qn_domains.update(
            {
                NodeQN.l_projection: [0],
                NodeQN.s_projection: __extend_negative(spin_magnitude_domain),
            }
        )
    if mass_conservation_factor is not None:
        formalism_node_settings.conservation_rules.add(
            MassConservation(mass_conservation_factor)
        )

    interaction_type_settings = {}
    weak_node_settings = deepcopy(formalism_node_settings)
    weak_node_settings.conservation_rules.update(
        [
            ChargeConservation(),
            ElectronLNConservation(),
            MuonLNConservation(),
            TauLNConservation(),
            BaryonNumberConservation(),
            identical_particle_symmetrization,
        ]
    )
    weak_node_settings.interaction_strength = 10 ** (-4)
    weak_edge_settings = deepcopy(formalism_edge_settings)

    interaction_type_settings[InteractionType.WEAK] = (
        weak_edge_settings,
        weak_node_settings,
    )

    em_node_settings = deepcopy(weak_node_settings)
    em_node_settings.conservation_rules.update(
        {
            CharmConservation(),
            StrangenessConservation(),
            BottomnessConservation(),
            parity_conservation,
            c_parity_conservation,
        }
    )
    if "helicity" in formalism:
        em_node_settings.conservation_rules.add(parity_conservation_helicity)
        em_node_settings.qn_domains.update({NodeQN.parity_prefactor: [-1, 1]})

    em_node_settings.interaction_strength = 1
    em_edge_settings = deepcopy(weak_edge_settings)
    interaction_type_settings[InteractionType.EM] = (
        em_edge_settings,
        em_node_settings,
    )

    strong_node_settings = deepcopy(em_node_settings)
    strong_node_settings.conservation_rules.update(
        {isospin_conservation, g_parity_conservation}
    )

    strong_node_settings.interaction_strength = 60
    strong_edge_settings = deepcopy(em_edge_settings)
    interaction_type_settings[InteractionType.STRONG] = (
        strong_edge_settings,
        strong_node_settings,
    )

    return interaction_type_settings


def __get_ang_mom_magnitudes(
    is_nbody: bool, max_angular_momentum: int
) -> List[float]:
    if is_nbody:
        return [0]
    return _int_domain(0, max_angular_momentum)  # type: ignore[return-value]


def __get_spin_magnitudes(
    is_nbody: bool, max_spin_magnitude: float
) -> List[float]:
    if is_nbody:
        return [0]
    return _halves_domain(0, max_spin_magnitude)


def _create_domains(particle_db: ParticleCollection) -> Dict[Any, list]:
    domains: Dict[Any, list] = {
        EdgeQN.electron_lepton_number: [-1, 0, +1],
        EdgeQN.muon_lepton_number: [-1, 0, +1],
        EdgeQN.tau_lepton_number: [-1, 0, +1],
        EdgeQN.parity: [-1, +1],
        EdgeQN.c_parity: [-1, +1, None],
        EdgeQN.g_parity: [-1, +1, None],
    }

    for edge_qn, getter in {
        EdgeQN.charge: lambda p: p.charge,
        EdgeQN.baryon_number: lambda p: p.baryon_number,
        EdgeQN.strangeness: lambda p: p.strangeness,
        EdgeQN.charmness: lambda p: p.charmness,
        EdgeQN.bottomness: lambda p: p.bottomness,
    }.items():
        domains[edge_qn] = __extend_negative(
            __positive_int_domain(particle_db, getter)
        )

    domains[EdgeQN.spin_magnitude] = __positive_halves_domain(
        particle_db, lambda p: p.spin
    )
    domains[EdgeQN.spin_projection] = __extend_negative(
        domains[EdgeQN.spin_magnitude]
    )
    domains[EdgeQN.isospin_magnitude] = __positive_halves_domain(
        particle_db,
        lambda p: 0 if p.isospin is None else p.isospin.magnitude,
    )
    domains[EdgeQN.isospin_projection] = __extend_negative(
        domains[EdgeQN.isospin_magnitude]
    )
    return domains


def __positive_halves_domain(
    particle_db: ParticleCollection, attr_getter: Callable[[Particle], Any]
) -> List[float]:
    values = set(map(attr_getter, particle_db))
    return _halves_domain(0, max(values))


def __positive_int_domain(
    particle_db: ParticleCollection, attr_getter: Callable[[Particle], Any]
) -> List[int]:
    values = set(map(attr_getter, particle_db))
    return _int_domain(0, max(values))


def _halves_domain(start: float, stop: float) -> List[float]:
    if start % 0.5 != 0.0:
        raise ValueError(f"Start value {start} needs to be multiple of 0.5")
    if stop % 0.5 != 0.0:
        raise ValueError(f"Stop value {stop} needs to be multiple of 0.5")
    return [
        int(v) if v.is_integer() else v
        for v in arange(start, stop + 0.25, delta=0.5)
    ]


def _int_domain(start: int, stop: int) -> List[int]:
    return list(range(start, stop + 1))


def __extend_negative(
    magnitudes: Iterable[Union[int, float]]
) -> List[Union[int, float]]:
    return sorted(list(magnitudes) + [-x for x in magnitudes if x > 0])
