from fractions import Fraction
from typing import Any, cast
from unittest.mock import MagicMock, PropertyMock

import pytest
from pdg.errors import PdgNoDataError

from qrules._pdg_adapter import _load_pdg_particles, _to_mass, _to_width
from qrules.particle import ParticleCollection, load_pdg
from qrules.quantum_numbers import Parity


@pytest.fixture(scope="module")
def official_particles() -> ParticleCollection:
    return load_pdg(source="pdg")


def test_uses_scikit_hep_source_by_default(
    official_particles: ParticleCollection,
):
    default_particles = load_pdg()
    scikit_hep_particles = load_pdg(source="particle")

    assert default_particles == scikit_hep_particles
    assert default_particles.find(-2212).name == "p~"
    assert official_particles.find(-2212).name == "pbar"


def test_rejects_unknown_source():
    with pytest.raises(ValueError, match="Unknown particle source"):
        load_pdg(source=cast("Any", "unknown"))


def test_caches_particle_definitions_and_returns_independent_collections(
    official_particles: ParticleCollection,
):
    cache_info_before = _load_pdg_particles.cache_info()

    second_collection = load_pdg(source="pdg")

    cache_info_after = _load_pdg_particles.cache_info()
    assert cache_info_after.hits == cache_info_before.hits + 1
    assert cache_info_after.misses == cache_info_before.misses
    assert second_collection == official_particles
    assert second_collection is not official_particles

    second_collection.discard("gamma")
    assert "gamma" not in second_collection
    assert "gamma" in official_particles


@pytest.mark.parametrize(
    ("mcid", "name"),
    [
        (12, "nu_e"),
        (-2212, "pbar"),
        (443, "J/psi(1S)"),
        (9010221, "f_0(980)0"),
        (5122, "Lambda_b()0"),
    ],
)
def test_uses_official_names(
    official_particles: ParticleCollection,
    mcid: int,
    name: str,
):
    assert official_particles.find(mcid).name == name


def test_all_particles_have_latex(official_particles: ParticleCollection):
    assert all(particle.latex is not None for particle in official_particles)


@pytest.mark.parametrize(
    ("mcid", "latex"),
    [
        (12, R"\nu_{e}"),
        (211, R"\pi^{+}"),
        (-2212, R"\overline{p}"),
        (443, R"J/\psi(1S)"),
        (9010221, R"f_{0}(980)"),
        (5122, R"\Lambda_{b}^{0}"),
    ],
)
def test_uses_generated_latex(
    official_particles: ParticleCollection,
    mcid: int,
    latex: str,
):
    assert official_particles.find(mcid).latex == latex


def test_pion_quantum_numbers(official_particles: ParticleCollection):
    pion = official_particles.find(211)
    assert pion.spin == 0
    assert pion.charge == +1
    assert pion.isospin is not None
    assert pion.isospin.magnitude == 1
    assert pion.isospin.projection == +1
    assert pion.parity == Parity(-1)
    assert pion.c_parity is None
    assert pion.g_parity == Parity(-1)


def test_antiproton_quantum_numbers(official_particles: ParticleCollection):
    antiproton = official_particles.find(-2212)
    assert antiproton.spin == Fraction(1, 2)
    assert antiproton.charge == -1
    assert antiproton.baryon_number == -1
    assert antiproton.isospin is not None
    assert antiproton.isospin.projection == Fraction(-1, 2)
    assert antiproton.parity == Parity(-1)


@pytest.mark.parametrize(
    ("mcid", "lepton_numbers"),
    [
        (11, (+1, 0, 0)),
        (-12, (-1, 0, 0)),
        (13, (0, +1, 0)),
        (-14, (0, -1, 0)),
        (15, (0, 0, +1)),
        (-16, (0, 0, -1)),
    ],
)
def test_lepton_numbers(
    official_particles: ParticleCollection,
    mcid: int,
    lepton_numbers: tuple[int, int, int],
):
    particle = official_particles.find(mcid)
    assert (
        particle.electron_lepton_number,
        particle.muon_lepton_number,
        particle.tau_lepton_number,
    ) == lepton_numbers


@pytest.mark.parametrize(
    ("mcid", "flavor_numbers"),
    [
        (+321, (+1, 0, 0, 0)),
        (-321, (-1, 0, 0, 0)),
        (+411, (0, +1, 0, 0)),
        (-411, (0, -1, 0, 0)),
        (+521, (0, 0, +1, 0)),
        (-521, (0, 0, -1, 0)),
    ],
)
def test_flavor_numbers(
    official_particles: ParticleCollection,
    mcid: int,
    flavor_numbers: tuple[int, int, int, int],
):
    particle = official_particles.find(mcid)
    assert (
        particle.strangeness,
        particle.charmness,
        particle.bottomness,
        particle.topness,
    ) == flavor_numbers


def test_prefers_official_spin(official_particles: ParticleCollection):
    assert official_particles.find(104122).spin == Fraction(3, 2)


def test_uses_measured_mass_and_width(official_particles: ParticleCollection):
    rho = official_particles.find(113)
    assert rho.mass == pytest.approx(0.7752611563582926)
    assert rho.width == pytest.approx(0.14739133387028722)


def test_derives_width_from_lifetime(official_particles: ParticleCollection):
    muon = official_particles.find(13)
    assert muon.width == pytest.approx(2.9959292110062035e-19)


def test_uses_zero_width_for_stable_particle(
    official_particles: ParticleCollection,
):
    assert official_particles.find(22).width == 0.0


def test_uses_antiparticle_mass_if_particle_has_no_mass():
    source = MagicMock(self_conjugate=False, mcid=1, name="particle")
    source.has_mass_entry = False
    type(source).mass = PropertyMock(side_effect=PdgNoDataError("no mass"))
    source.antiparticle.has_mass_entry = True
    source.antiparticle.mass = 0.5

    assert _to_mass(source) == 0.5


def test_uses_antiparticle_width_if_particle_has_no_decay_data():
    source = MagicMock(self_conjugate=False)
    source.has_width_entry = False
    source.has_lifetime_entry = False
    source.width = 0.0
    source.antiparticle.has_width_entry = True
    source.antiparticle.has_lifetime_entry = False
    source.antiparticle.width = 0.25

    assert _to_width(source) == 0.25


@pytest.mark.parametrize(
    ("mcid", "width"),
    [
        (9010221, 0.055),  # 10 to 100 MeV
        (2224, 0.117),  # 114 to 117 to 120 MeV
    ],
)
def test_uses_central_value_for_width_ranges(
    official_particles: ParticleCollection,
    mcid: int,
    width: float,
):
    assert official_particles.find(mcid).width == pytest.approx(width)
