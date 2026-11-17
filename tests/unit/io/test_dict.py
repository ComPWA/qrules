import pytest
from jsonschema import ValidationError

from qrules import io
from qrules.io._dict import validate_particle_collection
from qrules.particle import Particle, ParticleCollection
from qrules.quantum_numbers import Parity


def describe_serialization():
    def it_sorts_particles_by_name(particle_selection: ParticleCollection):
        names = [p["name"] for p in io.asdict(particle_selection)["particles"]]
        assert names == sorted(names)

    def it_ignores_insertion_order(particle_selection: ParticleCollection):
        reversed_collection = ParticleCollection()
        reversed_collection.update(reversed(list(particle_selection)))
        assert io.asdict(reversed_collection) == io.asdict(particle_selection)

    def it_round_trips_unknown_parities():
        particle = Particle(
            name="X0",
            pid=1,
            mass=1.0,
            spin=0,
            parity=Parity.UNKNOWN,
            c_parity=Parity.UNKNOWN,
            g_parity=-1,
        )
        definition = io.asdict(particle)
        assert definition["parity"] == {"value": "?"}
        assert definition["c_parity"] == {"value": "?"}
        assert definition["g_parity"] == {"value": -1}
        collection_definition = {"particles": [definition]}
        validate_particle_collection(collection_definition)
        imported = io.fromdict(collection_definition)
        assert isinstance(imported, ParticleCollection)
        assert imported["X0"] == particle

    def it_rejects_invalid_parity_values():
        definition = {
            "particles": [
                {"name": "X0", "pid": 1, "mass": 1.0, "spin": 0, "parity": {"value": 0}}
            ]
        }
        with pytest.raises(ValidationError):
            validate_particle_collection(definition)

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
        skh_particle_version: str,
    ):
        io.write(particle_selection, output_dir + "particle_selection.yml")
        n_particles = len(particle_selection)
        if skh_particle_version < "0.16":
            assert n_particles == 181
        elif skh_particle_version < "0.25":
            assert n_particles == 193
        else:
            assert n_particles == 194
        asdict = io.asdict(particle_selection)
        imported_collection = io.fromdict(asdict)
        assert isinstance(imported_collection, ParticleCollection)
        assert n_particles == len(imported_collection)
        for particle in particle_selection:
            exported = particle_selection[particle.name]
            imported = imported_collection[particle.name]
            assert imported == exported
