"""Default settings for the framework.

It is possible to change these settings from the outside, like:

>>> from qrules.settings import defaults
>>> defaults.MAX_ANGULAR_MOMENTUM = 4
>>> defaults.MAX_SPIN_MAGNITUDE = 3
"""

from os.path import dirname, join, realpath
from typing import Dict, Union

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

__QRULES_PATH = dirname(dirname(realpath(__file__)))
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

MAX_ANGULAR_MOMENTUM: int = 2
"""Maximum angular momentum over which to generate :math:`LS`-couplings."""

MAX_SPIN_MAGNITUDE: int = 2
"""Maximum spin magnitude over which to generate :math:`LS`-couplings."""
