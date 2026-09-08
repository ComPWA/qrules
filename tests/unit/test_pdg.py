import pytest

from qrules.particle import ParticleCollection, load_pdg


@pytest.fixture(scope="session")
def pdg() -> ParticleCollection:
    return load_pdg()


def test_contains_representative_particles(pdg: ParticleCollection):
    expected_names = {
        "J/psi(1S)",
        "f_0(500)0",
        "f_0(980)0",
        "gamma",
        "nu_e",
        "pbar",
        "pi+",
    }
    assert expected_names <= set(pdg.names)


def test_has_unique_names_and_mcids(pdg: ParticleCollection):
    assert len(pdg.names) == len(set(pdg.names))
    assert len(pdg) == len({particle.pid for particle in pdg})


def test_default_particle_definitions_extend_pdg(
    pdg: ParticleCollection,
    particle_database: ParticleCollection,
):
    pdg_names = set(pdg.names)
    default_names = set(particle_database.names)
    assert pdg_names <= default_names
    assert default_names - pdg_names == {"Y(4260)"}
