import sys

import pytest

from qrules import load_default_particles
from qrules.particle import ParticleCollection
from qrules.settings import NumberOfThreads

if sys.version_info < (3, 8):
    from importlib_metadata import version
else:
    from importlib.metadata import version

# Ensure consistent test coverage when running pytest multithreaded
# https://github.com/ComPWA/qrules/issues/11
NumberOfThreads.set(1)


@pytest.fixture(scope="session")
def particle_database() -> ParticleCollection:
    return load_default_particles()


@pytest.fixture(scope="session")
def output_dir(pytestconfig) -> str:
    return f"{pytestconfig.rootpath}/tests/output/"


@pytest.fixture(scope="session")
def skh_particle_version() -> str:
    major, minor, *_ = (int(i) for i in version("particle").split("."))
    particle_version = f"{major}.{minor}"
    if (major, minor) < (0, 11):
        pytest.skip(f"Version {particle_version} is not supported in the tests")
    return particle_version
