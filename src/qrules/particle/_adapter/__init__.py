"""Shared helpers for converting external particle databases to QRules particles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs

from qrules.quantum_numbers import Parity

if TYPE_CHECKING:
    from qrules.particle import Particle


def mark_unknown_parities(particle: Particle) -> Particle:
    """Distinguish parities that are not defined from parities that are unknown.

    Particle databases do not consistently distinguish a parity that does not exist for
    a particle from a parity that exists, but has not been measured. This function
    decides that based on the other quantum numbers of the particle:

    - The (spatial) parity exists for all particles, so a missing parity is unknown.
    - :math:`C`-parity exists only for particles that are their own antiparticle.
    - :math:`G`-parity exists only for non-strange, flavorless mesons with isospin.

    Missing parities that exist are set to `.Parity.UNKNOWN`, the others stay `None`.
    """
    changes: dict[str, Parity] = {}
    if particle.parity is None:
        changes["parity"] = Parity.UNKNOWN
    if particle.c_parity is None and _has_c_parity(particle):
        changes["c_parity"] = Parity.UNKNOWN
    if particle.g_parity is None and _has_g_parity(particle):
        changes["g_parity"] = Parity.UNKNOWN
    if not changes:
        return particle
    return attrs.evolve(particle, **changes)


def _has_c_parity(particle: Particle) -> bool:
    return particle.charge == 0 and _is_flavorless_boson(particle)


def _has_g_parity(particle: Particle) -> bool:
    return particle.isospin is not None and _is_flavorless_boson(particle)


def _is_flavorless_boson(particle: Particle) -> bool:
    return not any((
        particle.baryon_number,
        particle.strangeness,
        particle.charmness,
        particle.bottomness,
        particle.topness,
        particle.electron_lepton_number,
        particle.muon_lepton_number,
        particle.tau_lepton_number,
    ))
