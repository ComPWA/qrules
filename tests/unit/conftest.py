import logging

import pytest
from _pytest.fixtures import SubRequest

import qrules
from qrules import ReactionInfo
from qrules.topology import Edge, Topology

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


@pytest.fixture(scope="session")
def two_to_three_decay() -> Topology:
    r"""Create a dummy `Topology`.

    Has the following shape:

    .. code-block::

        e0 -- (N0) -- e2 -- (N1) -- e3 -- (N2) -- e6
              /               \             \
            e1                 e4            e5
    """
    return Topology(
        nodes={0, 1, 2},
        edges={
            0: Edge(None, 0),
            1: Edge(None, 0),
            2: Edge(0, 1),
            3: Edge(1, 2),
            4: Edge(1, None),
            5: Edge(2, None),
            6: Edge(2, None),
        },
    )
