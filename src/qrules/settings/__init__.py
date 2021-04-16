"""Default configuration for the `expertsystem`."""

from copy import deepcopy
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from qrules.conservation_rules import (
    BaryonNumberConservation,
    BottomnessConservation,
    ChargeConservation,
    CharmConservation,
    ElectronLNConservation,
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

from .defaults import CONSERVATION_LAW_PRIORITIES, EDGE_RULE_PRIORITIES


class InteractionType(Enum):
    """Types of interactions in the form of an enumerate."""

    STRONG = auto()
    EM = auto()
    WEAK = auto()


def create_interaction_settings(  # pylint: disable=too-many-locals,too-many-arguments
    formalism_type: str,
    particles: ParticleCollection,
    nbody_topology: bool = False,
    mass_conservation_factor: Optional[float] = 3.0,
    max_angular_momentum: int = 2,
    max_spin_magnitude: int = 2,
) -> Dict[InteractionType, Tuple[EdgeSettings, NodeSettings]]:
    """Create a container that holds the settings for `.InteractionType`."""
    formalism_edge_settings = EdgeSettings(
        conservation_rules={
            isospin_validity,
            gellmann_nishijima,
            spin_validity,
        },
        rule_priorities=EDGE_RULE_PRIORITIES,
        qn_domains=_create_domains(particles),
    )
    formalism_node_settings = NodeSettings(
        rule_priorities=CONSERVATION_LAW_PRIORITIES
    )

    angular_momentum_domain = _get_ang_mom_magnitudes(
        nbody_topology, max_angular_momentum
    )
    spin_magnitude_domain = _get_spin_magnitudes(
        nbody_topology, max_spin_magnitude
    )
    if "helicity" in formalism_type:
        formalism_node_settings.conservation_rules = {
            spin_magnitude_conservation,
            helicity_conservation,
        }
        formalism_node_settings.qn_domains = {
            NodeQN.l_magnitude: angular_momentum_domain,
            NodeQN.s_magnitude: spin_magnitude_domain,
        }
    elif formalism_type == "canonical":
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
    if formalism_type == "canonical-helicity":
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
    if "helicity" in formalism_type:
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


def _get_ang_mom_magnitudes(
    is_nbody: bool, max_angular_momentum: int
) -> List[float]:
    if is_nbody:
        return [0]
    return _int_domain(0, max_angular_momentum)  # type: ignore


def _get_spin_magnitudes(
    is_nbody: bool, max_spin_magnitude: int
) -> List[float]:
    if is_nbody:
        return [0]
    return _halves_domain(0, max_spin_magnitude)


def _create_domains(particles: ParticleCollection) -> Dict[Any, list]:
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
            __positive_int_domain(particles, getter)
        )

    domains[EdgeQN.spin_magnitude] = __positive_halves_domain(
        particles, lambda p: p.spin
    )
    domains[EdgeQN.spin_projection] = __extend_negative(
        domains[EdgeQN.spin_magnitude]
    )
    domains[EdgeQN.isospin_magnitude] = __positive_halves_domain(
        particles,
        lambda p: 0 if p.isospin is None else p.isospin.magnitude,
    )
    domains[EdgeQN.isospin_projection] = __extend_negative(
        domains[EdgeQN.isospin_magnitude]
    )
    return domains


def __positive_halves_domain(
    particles: ParticleCollection, attr_getter: Callable[[Particle], Any]
) -> List[float]:
    values = set(map(attr_getter, particles))
    return _halves_domain(0, max(values))


def __positive_int_domain(
    particles: ParticleCollection, attr_getter: Callable[[Particle], Any]
) -> List[int]:
    values = set(map(attr_getter, particles))
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
