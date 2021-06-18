# pylint: disable=no-self-use
from qrules import Result


class TestResult:
    def test_get_intermediate_state_names(self, result: Result):
        intermediate_particles = result.get_intermediate_particles()
        assert intermediate_particles.names == ["f(0)(980)", "f(0)(1500)"]
