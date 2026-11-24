import pytest

from qrules.particle import Particle, Spin
from qrules.particle._adapter import mark_unknown_parities
from qrules.quantum_numbers import Parity


def describe_mark_unknown_parities():
    def it_keeps_known_parities():
        particle = Particle(
            name="X0", pid=1, mass=1.0, spin=0, parity=-1, c_parity=+1, g_parity=-1
        )
        assert mark_unknown_parities(particle) is particle

    def it_marks_c_and_g_parity_of_flavorless_neutral_meson_as_unknown():
        particle = Particle(
            name="X0", pid=1, mass=1.0, spin=0, parity=-1, isospin=Spin(1, 0)
        )
        marked = mark_unknown_parities(particle)
        assert marked.c_parity == Parity.UNKNOWN
        assert marked.g_parity == Parity.UNKNOWN

    def it_marks_only_g_parity_of_charged_meson_as_unknown():
        particle = Particle(
            name="X+",
            pid=1,
            mass=1.0,
            spin=0,
            charge=+1,
            parity=-1,
            isospin=Spin(1, +1),
        )
        marked = mark_unknown_parities(particle)
        assert marked.c_parity is None
        assert marked.g_parity == Parity.UNKNOWN

    def it_does_not_mark_g_parity_without_isospin():
        particle = Particle(name="X0", pid=1, mass=1.0, spin=1, parity=-1)
        marked = mark_unknown_parities(particle)
        assert marked.c_parity == Parity.UNKNOWN
        assert marked.g_parity is None

    @pytest.mark.parametrize(
        "flavor",
        [
            {"baryon_number": 1},
            {"strangeness": 1},
            {"charmness": 1},
            {"bottomness": 1},
            {"topness": 1},
            {"electron_lepton_number": 1},
            {"muon_lepton_number": 1},
            {"tau_lepton_number": 1},
        ],
    )
    def it_does_not_mark_c_and_g_parity_of_flavored_particles(flavor: dict):
        particle = Particle(name="X", pid=1, mass=1.0, spin=0, parity=-1, **flavor)
        marked = mark_unknown_parities(particle)
        assert marked.c_parity is None
        assert marked.g_parity is None
