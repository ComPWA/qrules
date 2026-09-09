import pytest

from qrules import io
from qrules.particle import ParticleCollection


def describe_serialization():
    def it_sorts_particles_by_name(particle_selection: ParticleCollection):
        names = [p["name"] for p in io.asdict(particle_selection)["particles"]]
        assert names == sorted(names)

    def it_ignores_insertion_order(particle_selection: ParticleCollection):
        reversed_collection = ParticleCollection()
        reversed_collection.update(reversed(list(particle_selection)))
        assert io.asdict(reversed_collection) == io.asdict(particle_selection)

    def it_not_implemented_errors(
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

    def it_serialization(
        output_dir: str,
        particle_selection: ParticleCollection,
    ):
        io.write(particle_selection, output_dir + "particle_selection.yml")
        n_particles = len(particle_selection)
        assert n_particles > 0
        asdict = io.asdict(particle_selection)
        imported_collection = io.fromdict(asdict)
        assert isinstance(imported_collection, ParticleCollection)
        assert n_particles == len(imported_collection)
        for particle in particle_selection:
            exported = particle_selection[particle.name]
            imported = imported_collection[particle.name]
            assert imported == exported
