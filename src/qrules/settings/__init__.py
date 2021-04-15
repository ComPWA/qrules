"""Default configuration for the `expertsystem`."""

from copy import deepcopy
from enum import Enum, auto
from typing import Dict, Iterable, List, Optional, Tuple, Union

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
from qrules.quantum_numbers import EdgeQuantumNumbers as EdgeQN
from qrules.quantum_numbers import NodeQuantumNumbers as NodeQN
from qrules.quantum_numbers import arange
from qrules.solving import EdgeSettings, NodeSettings

from .defaults import (
    CONSERVATION_LAW_PRIORITIES,
    EDGE_RULE_PRIORITIES,
    MAX_ANGULAR_MOMENTUM,
)


class InteractionTypes(Enum):
    """Types of interactions in the form of an enumerate."""

    STRONG = auto()
    EM = auto()
    WEAK = auto()


def create_interaction_settings(
    formalism_type: str,
    nbody_topology: bool = False,
    mass_conservation_factor: Optional[float] = 3.0,
) -> Dict[InteractionTypes, Tuple[EdgeSettings, NodeSettings]]:
    """Create a container that holds the settings for `.InteractionTypes`."""
    interaction_type_settings = {}
    formalism_edge_settings = EdgeSettings(
        conservation_rules={
            isospin_validity,
            gellmann_nishijima,
            spin_validity,
        },
        rule_priorities=EDGE_RULE_PRIORITIES,
        qn_domains={
            EdgeQN.charge: [-2, -1, 0, 1, 2],
            EdgeQN.baryon_number: [-1, 0, 1],
            EdgeQN.electron_lepton_number: [-1, 0, 1],
            EdgeQN.muon_lepton_number: [-1, 0, 1],
            EdgeQN.tau_lepton_number: [-1, 0, 1],
            EdgeQN.parity: [-1, 1],
            EdgeQN.c_parity: [-1, 1, None],
            EdgeQN.g_parity: [-1, 1, None],
            EdgeQN.spin_magnitude: _halves_domain(0, 2),
            EdgeQN.spin_projection: __extend_negative(_halves_domain(0, 2)),
            EdgeQN.isospin_magnitude: _halves_domain(0, 1.5),
            EdgeQN.isospin_projection: __extend_negative(
                _halves_domain(0, 1.5)
            ),
            EdgeQN.charmness: [-1, 0, 1],
            EdgeQN.strangeness: [-1, 0, 1],
            EdgeQN.bottomness: [-1, 0, 1],
        },
    )
    formalism_node_settings = NodeSettings(
        rule_priorities=CONSERVATION_LAW_PRIORITIES
    )

    if "helicity" in formalism_type:
        formalism_node_settings.conservation_rules = {
            spin_magnitude_conservation,
            helicity_conservation,
        }
        formalism_node_settings.qn_domains = {
            NodeQN.l_magnitude: _get_ang_mom_magnitudes(nbody_topology),
            NodeQN.s_magnitude: _get_spin_magnitudes(nbody_topology),
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
            NodeQN.l_magnitude: _get_ang_mom_magnitudes(nbody_topology),
            NodeQN.l_projection: __extend_negative(
                _get_ang_mom_magnitudes(nbody_topology)
            ),
            NodeQN.s_magnitude: _get_spin_magnitudes(nbody_topology),
            NodeQN.s_projection: __extend_negative(
                _get_spin_magnitudes(nbody_topology)
            ),
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
                NodeQN.s_projection: __extend_negative(
                    _get_spin_magnitudes(nbody_topology)
                ),
            }
        )
    if mass_conservation_factor is not None:
        formalism_node_settings.conservation_rules.add(
            MassConservation(mass_conservation_factor)
        )

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

    interaction_type_settings[InteractionTypes.WEAK] = (
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

    interaction_type_settings[InteractionTypes.EM] = (
        em_edge_settings,
        em_node_settings,
    )

    strong_node_settings = deepcopy(em_node_settings)
    strong_node_settings.conservation_rules.update(
        {isospin_conservation, g_parity_conservation}
    )
    strong_node_settings.interaction_strength = 60

    strong_edge_settings = deepcopy(em_edge_settings)

    interaction_type_settings[InteractionTypes.STRONG] = (
        strong_edge_settings,
        strong_node_settings,
    )

    return interaction_type_settings


def _get_ang_mom_magnitudes(is_nbody: bool) -> List[float]:
    if is_nbody:
        return [0]
    return _int_domain(0, MAX_ANGULAR_MOMENTUM)  # type: ignore


def _get_spin_magnitudes(is_nbody: bool) -> List[float]:
    if is_nbody:
        return [0]
    return _halves_domain(0, 2)


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
