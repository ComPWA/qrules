"""Convert records from the official PDG API to QRules particles."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import TYPE_CHECKING

import pdg
from pdg.units import convert

from qrules._pdg_latex import create_latex_name
from qrules.particle import Particle, ParticleCollection, Spin
from qrules.quantum_numbers import Parity

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pdg.api import PdgApi
    from pdg.particle import PdgParticle


# These states are also excluded by the existing Scikit-HEP based loader. The neutral
# kaon mass eigenstates do not have a definite flavor, while the B(s2) entries have
# inconsistent isospin information for QRules' particle model.
_SKIPPED_MC_IDS = {-535, 130, 310, 535}


class _UnsupportedParticleError(ValueError):
    """Raised when a PDG entry cannot be represented by QRules."""


def load_pdg() -> ParticleCollection:
    """Load particle definitions from the official PDG database."""
    api = pdg.connect()
    particles = ParticleCollection()
    for source_particle in _iter_particles(api):
        try:
            particle = _convert_particle(source_particle)
        except _UnsupportedParticleError:
            continue
        particles.add(particle)
    return particles


def _iter_particles(api: PdgApi) -> Iterator[PdgParticle]:
    """Iterate over unique, charge-specific particles with a Monte Carlo ID."""
    mc_ids: set[int] = set()
    for particle_group in api.get_particles():
        source_particles = (
            particle_group if isinstance(particle_group, list) else [particle_group]
        )
        for source_particle in source_particles:
            mcid = source_particle.mcid
            if mcid is not None:
                mc_ids.add(mcid)
    for mcid in sorted(mc_ids):
        yield api.get_particle_by_mcid(mcid)


def _convert_particle(source: PdgParticle) -> Particle:
    mcid = source.mcid
    if mcid is None:
        msg = f"Particle {source.name} has no Monte Carlo ID"
        raise _UnsupportedParticleError(msg)
    if mcid in _SKIPPED_MC_IDS or abs(mcid) >= 1_000_000_000:
        msg = f"Particle {source.name} is not supported"
        raise _UnsupportedParticleError(msg)

    charge = _to_integer_charge(source.charge)
    spin = _to_spin(
        source.quantum_J,
        mcid,
        is_hadron=source.is_baryon or source.is_meson,
    )
    mass = _to_mass(source)
    width = _to_width(source)

    strangeness, charmness, bottomness, topness = _flavor_quantum_numbers(
        mcid,
        is_baryon=source.is_baryon,
        is_meson=source.is_meson,
    )
    baryon_number = _baryon_number(mcid, is_baryon=source.is_baryon)
    lepton_numbers = _lepton_numbers(mcid, is_lepton=source.is_lepton)
    isospin = _to_isospin(
        source.quantum_I,
        charge=charge,
        baryon_number=baryon_number,
        flavor_numbers=(strangeness, charmness, bottomness, topness),
    )
    parity = _to_parity(source.quantum_P)
    if source.is_lepton:
        # QRules convention: fermions and antifermions have opposite intrinsic parity.
        parity = Parity(+1 if mcid > 0 else -1)
    c_parity = _to_parity(source.quantum_C) if source.self_conjugate else None

    return Particle(
        name=source.name,
        latex=create_latex_name(
            source.name,
            isospin=source.quantum_I,
            self_conjugate=source.self_conjugate,
        ),
        pid=mcid,
        spin=spin,
        mass=mass,
        width=width,
        charge=charge,
        isospin=isospin,
        strangeness=strangeness,
        charmness=charmness,
        bottomness=bottomness,
        topness=topness,
        baryon_number=baryon_number,
        electron_lepton_number=lepton_numbers[0],
        muon_lepton_number=lepton_numbers[1],
        tau_lepton_number=lepton_numbers[2],
        parity=parity,
        c_parity=c_parity,
        g_parity=_to_parity(source.quantum_G),
    )


def _to_integer_charge(value: float) -> int:
    if not float(value).is_integer():
        msg = f"QRules does not support fractional charge {value}"
        raise _UnsupportedParticleError(msg)
    return int(value)


def _to_spin(value: str | None, mcid: int, *, is_hadron: bool) -> Fraction:
    spin = _to_fraction(value)
    if spin is not None:
        return spin

    # For hadrons, the final digit of an MC ID is 2J+1. This preserves definite
    # spins encoded in IDs where the current RPP text reports a range or "?".
    spin_code = abs(mcid) % 10
    if is_hadron and spin_code > 0:
        return Fraction(spin_code - 1, 2)
    msg = f"Cannot determine spin for MC ID {mcid} from {value!r}"
    raise _UnsupportedParticleError(msg)


def _to_mass(source: PdgParticle) -> float:
    for candidate in _particle_and_antiparticle(source):
        if candidate.has_mass_entry:
            mass = candidate.mass
            if mass is not None:
                return mass
            mass = _range_central_value(candidate, quantity="mass")
            if mass is not None:
                return mass
    if abs(source.mcid) in {12, 14, 16, 21, 22}:
        return 0.0
    msg = f"Particle {source.name} has no supported mass value"
    raise _UnsupportedParticleError(msg)


def _to_width(source: PdgParticle) -> float:
    for candidate in _particle_and_antiparticle(source):
        if candidate.has_width_entry or candidate.has_lifetime_entry:
            width = candidate.width
            if width is not None:
                return width
            width = _range_central_value(candidate, quantity="width")
            if width is not None:
                return width
    return 0.0


def _particle_and_antiparticle(source: PdgParticle) -> tuple[PdgParticle, ...]:
    if source.self_conjugate:
        return (source,)
    return source, source.antiparticle


def _range_central_value(
    source: PdgParticle,
    *,
    quantity: str,
) -> float | None:
    properties = source.masses() if quantity == "mass" else source.widths()
    prop = source.best(properties, f"{source.name} {quantity}")
    summary = prop.best_summary()
    if summary is None or summary.is_lower_limit or summary.is_upper_limit:
        return None
    value = summary.get_value("GeV")
    if value is not None:
        return value
    range_value = _central_value_from_range(summary.value_text)
    if range_value is None:
        return None
    return convert(range_value, summary.units, "GeV")


def _central_value_from_range(value: str) -> float | None:
    """Select the preferred value, or midpoint, from a PDG range."""
    components = re.split(r"\s+to\s+", value.strip(), flags=re.IGNORECASE)
    if len(components) not in {2, 3}:
        return None
    try:
        numbers = [float(component) for component in components]
    except ValueError:
        return None
    if len(numbers) == 3:
        return numbers[1]
    return sum(numbers) / 2


def _to_fraction(value: str | None) -> Fraction | None:
    if value is None:
        return None
    try:
        return Fraction(value)
    except ValueError:
        return None


def _to_parity(value: str | None) -> Parity | None:
    if value == "+":
        return Parity(+1)
    if value == "-":
        return Parity(-1)
    return None


def _baryon_number(mcid: int, *, is_baryon: bool) -> int:
    if not is_baryon:
        return 0
    return +1 if mcid > 0 else -1


def _lepton_numbers(mcid: int, *, is_lepton: bool) -> tuple[int, int, int]:
    if not is_lepton:
        return 0, 0, 0
    lepton_number = +1 if mcid > 0 else -1
    generation = (abs(mcid) - 11) // 2
    values = [0, 0, 0]
    if generation not in range(len(values)):
        return 0, 0, 0
    values[generation] = lepton_number
    return values[0], values[1], values[2]


def _flavor_quantum_numbers(
    mcid: int,
    *,
    is_baryon: bool,
    is_meson: bool,
) -> tuple[int, int, int, int]:
    """Derive S, C, B' and T from the quark digits in a standard MC ID."""
    abs_mcid = abs(mcid)
    quark3 = (abs_mcid // 10) % 10
    quark2 = (abs_mcid // 100) % 10
    quark1 = (abs_mcid // 1_000) % 10
    net_quarks = dict.fromkeys((3, 4, 5, 6), 0)
    particle_sign = +1 if mcid > 0 else -1

    if is_baryon:
        for flavor in (quark1, quark2, quark3):
            if flavor in net_quarks:
                net_quarks[flavor] += particle_sign
    elif is_meson and quark2 != quark3:
        # For a positive meson ID, the heavier flavor is a quark when it is
        # up-type and an antiquark when it is down-type. A negative ID reverses
        # the assignment.
        heavier_sign = particle_sign if quark2 % 2 == 0 else -particle_sign
        lighter_sign = -heavier_sign
        if quark2 in net_quarks:
            net_quarks[quark2] += heavier_sign
        if quark3 in net_quarks:
            net_quarks[quark3] += lighter_sign

    return (
        -net_quarks[3],
        +net_quarks[4],
        -net_quarks[5],
        +net_quarks[6],
    )


def _to_isospin(
    value: str | None,
    *,
    charge: int,
    baryon_number: int,
    flavor_numbers: tuple[int, int, int, int],
) -> Spin | None:
    magnitude = _to_fraction(value)
    if magnitude is None:
        return None
    projection = Fraction(
        2 * charge - baryon_number - sum(flavor_numbers),
        2,
    )
    return Spin(magnitude, projection)
