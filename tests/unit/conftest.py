import logging
from typing import TYPE_CHECKING

import pytest
from _pytest.fixtures import SubRequest

import qrules
from qrules import ReactionInfo
from qrules.topology import Edge, Topology

if TYPE_CHECKING:
    from qrules.transition import SpinFormalism

logging.basicConfig(level=logging.ERROR)


@pytest.fixture(scope="session", params=["canonical-helicity", "helicity"])
def reaction(request: SubRequest) -> ReactionInfo:
    formalism: SpinFormalism = request.param
    return qrules.generate_transitions(
        initial_state=[("J/psi(1S)", [-1, 1])],
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)"],
        allowed_interaction_types="strong",
        formalism=formalism,
    )


@pytest.fixture(scope="session")
def two_to_three_decay() -> Topology:
    r"""Create a dummy `Topology`.

    Has the following shape:

    .. code-block::

        e-1 -- (N0) -- e3 -- (N1) -- e4 -- (N2) -- e2
              /               \             \
            e-2                e0            e1
    """
    return Topology(
        nodes={0, 1, 2},
        edges={
            -2: Edge(None, 0),
            -1: Edge(None, 0),
            0: Edge(1, None),
            1: Edge(2, None),
            2: Edge(2, None),
            3: Edge(0, 1),
            4: Edge(1, 2),
        },
    )
