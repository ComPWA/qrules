# pylint: disable=redefined-outer-name
import logging

import pytest
from _pytest.fixtures import SubRequest

import qrules
from qrules import Result

logging.basicConfig(level=logging.ERROR)


@pytest.fixture(scope="session")
def jpsi_to_gamma_pi_pi_canonical_solutions() -> Result:
    return qrules.generate_transitions(
        initial_state=[("J/psi(1S)", [-1, 1])],
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        allowed_interaction_types="strong only",
        formalism_type="canonical-helicity",
    )


@pytest.fixture(scope="session")
def jpsi_to_gamma_pi_pi_helicity_solutions() -> Result:
    return qrules.generate_transitions(
        initial_state=[("J/psi(1S)", [-1, 1])],
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        allowed_interaction_types="strong only",
        formalism_type="helicity",
    )


@pytest.fixture(scope="session", params=["canonical", "helicity"])
def result(
    request: SubRequest,
    jpsi_to_gamma_pi_pi_canonical_solutions: Result,
    jpsi_to_gamma_pi_pi_helicity_solutions: Result,
) -> Result:
    formalism: str = request.param
    if formalism == "canonical":
        return jpsi_to_gamma_pi_pi_canonical_solutions
    if formalism == "helicity":
        return jpsi_to_gamma_pi_pi_helicity_solutions
    raise NotImplementedError(
        f'No {Result.__name__} for formalism type "{formalism}"'
    )
