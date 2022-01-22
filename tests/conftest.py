import pytest

from qrules import load_default_particles
from qrules.particle import ParticleCollection
from qrules.settings import NumberOfThreads

# Ensure consistent test coverage when running pytest multithreaded
# https://github.com/ComPWA/qrules/issues/11
NumberOfThreads.set(1)


@pytest.fixture(scope="session")
def particle_database() -> ParticleCollection:
    return load_default_particles()


@pytest.fixture(scope="session")
def output_dir(pytestconfig) -> str:
    return f"{pytestconfig.rootpath}/tests/output/"
