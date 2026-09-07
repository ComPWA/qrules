from fractions import Fraction

import pytest

from qrules._pdg import load_pdg as load_official_pdg
from qrules.particle import ParticleCollection
from qrules.particle import load_pdg as load_scikit_hep_pdg
from qrules.quantum_numbers import Parity


@pytest.fixture(scope="module")
def official_particles() -> ParticleCollection:
    return load_official_pdg()


@pytest.fixture(scope="module")
def scikit_hep_particles() -> ParticleCollection:
    return load_scikit_hep_pdg()


def test_current_mcids_are_covered(
    official_particles: ParticleCollection,
    scikit_hep_particles: ParticleCollection,
):
    official_mcids = {particle.pid for particle in official_particles}
    current_mcids = {particle.pid for particle in scikit_hep_particles}
    assert current_mcids <= official_mcids


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


def test_flavor_numbers_match_current_loader(
    official_particles: ParticleCollection,
    scikit_hep_particles: ParticleCollection,
):
    for current in scikit_hep_particles:
        official = official_particles.find(current.pid)
        assert (
            official.strangeness,
            official.charmness,
            official.bottomness,
            official.topness,
        ) == (
            current.strangeness,
            current.charmness,
            current.bottomness,
            current.topness,
        )


def test_prefers_official_spin(official_particles: ParticleCollection):
    assert official_particles.find(104122).spin == Fraction(3, 2)


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
