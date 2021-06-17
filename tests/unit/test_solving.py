# pylint: disable=no-self-use
from typing import Callable

import pytest

from qrules import Result


class TestResult:
    @pytest.mark.parametrize("formalism", ["canonical", "helicity"])
    def test_get_intermediate_state_names(
        self,
        formalism: str,
        get_reaction: Callable[[str], Result],
    ):
        result = get_reaction(formalism)
        intermediate_particles = result.get_intermediate_particles()
        assert intermediate_particles.names == {"f(0)(1500)", "f(0)(980)"}
