import pytest

from qrules import io
from qrules.particle import ParticleCollection


def test_not_implemented_errors(
    output_dir: str, particle_selection: ParticleCollection
):
    with pytest.raises(NotImplementedError):
        io.load(__file__)
    with pytest.raises(NotImplementedError):
        io.write(particle_selection, output_dir + "test.py")
    with pytest.raises(ValueError, match=r"No file extension in file name"):
        io.write(particle_selection, output_dir + "no_file_extension")
    with pytest.raises(NotImplementedError):
        io.write(666, output_dir + "wont_work_anyway.yml")


def test_serialization(
    output_dir: str, particle_selection: ParticleCollection, skh_particle_version: str
):
    io.write(particle_selection, output_dir + "particle_selection.yml")
    n_particles = len(particle_selection)
    if skh_particle_version < "0.16":
        assert n_particles == 181
    else:
        assert n_particles == 193
    asdict = io.asdict(particle_selection)
    imported_collection = io.fromdict(asdict)
    assert isinstance(imported_collection, ParticleCollection)
    assert n_particles == len(imported_collection)
    for particle in particle_selection:
        exported = particle_selection[particle.name]
        imported = imported_collection[particle.name]
        assert imported == exported
