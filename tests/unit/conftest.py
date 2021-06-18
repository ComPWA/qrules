# pylint: disable=redefined-outer-name
import logging

import pytest
from _pytest.fixtures import SubRequest

import qrules
from qrules import ReactionInfo

logging.basicConfig(level=logging.ERROR)


@pytest.fixture(scope="session", params=["canonical-helicity", "helicity"])
def reaction(request: SubRequest) -> ReactionInfo:
    formalism: str = request.param
    return qrules.generate_transitions(
        initial_state=[("J/psi(1S)", [-1, 1])],
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        allowed_interaction_types="strong",
        formalism=formalism,
    )
