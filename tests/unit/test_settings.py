# pylint: disable=no-self-use
import pytest

from qrules.particle import ParticleCollection
from qrules.quantum_numbers import EdgeQuantumNumbers as EdgeQN
from qrules.settings import (
    InteractionType,
    _create_domains,
    _halves_domain,
    _int_domain,
    create_interaction_settings,
)


class TestInteractionType:
    @pytest.mark.parametrize(
        ("description", "expected"),
        [
            ("EM", InteractionType.EM),
            ("e", InteractionType.EM),
            ("electromagnetic", InteractionType.EM),
            ("w", InteractionType.WEAK),
            ("weak", InteractionType.WEAK),
            ("strong", InteractionType.STRONG),
            ("S", InteractionType.STRONG),
            ("", ValueError),
            ("non-existing", ValueError),
        ],
    )
    def test_from_str(self, description: str, expected: InteractionType):
        if expected is ValueError:
            with pytest.raises(ValueError, match=r"interaction type"):
                assert InteractionType.from_str(description)
        else:
            assert InteractionType.from_str(description) == expected


def test_create_domains(particle_database: ParticleCollection):
    pdg = particle_database
    pions = pdg.filter(lambda p: p.name.startswith("pi"))
    domains = _create_domains(pions)
    assert len(domains) == 15
    assert domains[EdgeQN.baryon_number] == [0]
    assert domains[EdgeQN.strangeness] == [0]
    assert domains[EdgeQN.charmness] == [0]
    assert domains[EdgeQN.bottomness] == [0]
    assert domains[EdgeQN.charge] == [-1, 0, +1]
    assert domains[EdgeQN.spin_magnitude] == [0, 0.5, 1, 1.5, 2]
    assert (
        domains[EdgeQN.spin_projection]
        == [-2, -1.5, -1, -0.5] + domains[EdgeQN.spin_magnitude]
    )
    assert domains[EdgeQN.isospin_magnitude] == [0, 0.5, 1]
    assert domains[EdgeQN.isospin_projection] == [-1, -0.5, 0, 0.5, 1]


@pytest.mark.parametrize("interaction_type", list(InteractionType))
@pytest.mark.parametrize("nbody_topology", [False, True])
@pytest.mark.parametrize(
    "formalism", ["canonical", "canonical-helicity", "helicity"]
)
def test_create_interaction_settings(
    particle_database: ParticleCollection,
    interaction_type: InteractionType,
    nbody_topology: bool,
    formalism: str,
):
    settings = create_interaction_settings(
        formalism,
        particle_db=particle_database,
        nbody_topology=nbody_topology,
    )
    assert set(settings) == set(InteractionType)

    edge_settings, node_settings = settings[interaction_type]
    edge_qn_domains_str = {  # strings are easier to compare with pytest
        qn_type.__name__: domain
        for qn_type, domain in edge_settings.qn_domains.items()
    }
    assert edge_qn_domains_str == {
        "baryon_number": [-1, 0, +1],
        "electron_lepton_number": [-1, 0, +1],
        "muon_lepton_number": [-1, 0, +1],
        "tau_lepton_number": [-1, 0, +1],
        "parity": [-1, +1],
        "c_parity": [-1, +1, None],
        "g_parity": [-1, +1, None],
        "spin_magnitude": _halves_domain(0, 4),
        "spin_projection": _halves_domain(-4, +4),
        "charge": _int_domain(-2, 2),
        "isospin_magnitude": _halves_domain(0, 1.5),
        "isospin_projection": _halves_domain(-1.5, +1.5),
        "strangeness": _int_domain(-3, +3),
        "charmness": _int_domain(-1, 1),
        "bottomness": _int_domain(-1, 1),
    }

    expected = {
        "l_magnitude": _int_domain(0, 2),
        "s_magnitude": _halves_domain(0, 2),
    }
    if "canonical" in formalism:
        expected["l_projection"] = [-2, -1, 0, 1, 2]
        expected["s_projection"] = _halves_domain(-2, 2)
    if formalism == "canonical-helicity":
        expected["l_projection"] = [0]
    if "helicity" in formalism and interaction_type != InteractionType.WEAK:
        expected["parity_prefactor"] = [-1, 1]
    if nbody_topology:
        expected["l_magnitude"] = [0]
        expected["s_magnitude"] = [0]
    if nbody_topology and formalism != "helicity":
        expected["l_projection"] = [0]
        expected["s_projection"] = [0]

    node_qn_domains_str = {  # strings are easier to compare with pytest
        qn_type.__name__: domain
        for qn_type, domain in node_settings.qn_domains.items()
    }
    assert node_qn_domains_str == expected


@pytest.mark.parametrize(
    ("start", "stop", "expected"),
    [
        (-0.3, 0.5, None),
        (-2.0, 0.5, [-2, -1.5, -1, -0.5, 0, 0.5]),
        (-1, +1, [-1, -0.5, 0, 0.5, +1]),
    ],
)
def test_halves_range(start: float, stop: float, expected: list):
    if expected is None:
        with pytest.raises(ValueError, match=r"needs to be multiple of 0.5"):
            _halves_domain(start, stop)
    else:
        assert _halves_domain(start, stop) == expected
