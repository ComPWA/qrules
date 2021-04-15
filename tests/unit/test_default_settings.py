import pytest

from qrules.default_settings import (
    InteractionTypes,
    _halves_range,
    create_default_interaction_settings,
)
from qrules.quantum_numbers import EdgeQuantumNumbers, NodeQuantumNumbers


@pytest.mark.parametrize("interaction_type", list(InteractionTypes))
@pytest.mark.parametrize("nbody_topology", [False, True])
@pytest.mark.parametrize(
    "formalism_type", ["canonical", "canonical-helicity", "helicity"]
)
def test_create_default_interaction_settings(
    interaction_type: InteractionTypes,
    nbody_topology: bool,
    formalism_type: str,
):
    settings = create_default_interaction_settings(
        formalism_type, nbody_topology
    )
    assert set(settings) == set(InteractionTypes)

    edge_settings, node_settings = settings[interaction_type]
    assert edge_settings.qn_domains == {
        EdgeQuantumNumbers.baryon_number: [-1, 0, 1],
        EdgeQuantumNumbers.electron_lepton_number: [-1, 0, 1],
        EdgeQuantumNumbers.muon_lepton_number: [-1, 0, 1],
        EdgeQuantumNumbers.tau_lepton_number: [-1, 0, 1],
        EdgeQuantumNumbers.parity: [-1, 1],
        EdgeQuantumNumbers.c_parity: [-1, 1, None],
        EdgeQuantumNumbers.g_parity: [-1, 1, None],
        EdgeQuantumNumbers.spin_magnitude: _halves_range(0, 2),
        EdgeQuantumNumbers.spin_projection: _halves_range(-2, +2),
        EdgeQuantumNumbers.charge: [-2, -1, 0, 1, 2],
        EdgeQuantumNumbers.isospin_magnitude: _halves_range(0, 1.5),
        EdgeQuantumNumbers.isospin_projection: _halves_range(-1.5, +1.5),
        EdgeQuantumNumbers.strangeness: [-1, 0, 1],
        EdgeQuantumNumbers.charmness: [-1, 0, 1],
        EdgeQuantumNumbers.bottomness: [-1, 0, 1],
    }
    expected = {
        NodeQuantumNumbers.l_magnitude: [0, 1, 2],
        NodeQuantumNumbers.s_magnitude: _halves_range(0, 2),
    }
    if "canonical" in formalism_type:
        expected[NodeQuantumNumbers.l_projection] = [-2, -1, 0, 1, 2]
        expected[NodeQuantumNumbers.s_projection] = _halves_range(-2, 2)
    if formalism_type == "canonical-helicity":
        expected[NodeQuantumNumbers.l_projection] = [0]
    if (
        "helicity" in formalism_type
        and interaction_type != InteractionTypes.WEAK
    ):
        expected[NodeQuantumNumbers.parity_prefactor] = [-1, 1]
    if nbody_topology:
        expected[NodeQuantumNumbers.l_magnitude] = [0]
        expected[NodeQuantumNumbers.s_magnitude] = [0]
    if nbody_topology and formalism_type != "helicity":
        expected[NodeQuantumNumbers.l_projection] = [0]
        expected[NodeQuantumNumbers.s_projection] = [0]
    assert node_settings.qn_domains == expected


@pytest.mark.parametrize(
    "start, stop, expected",
    [
        (-0.3, 0.5, None),
        (-2.0, 0.5, [-2, -1.5, -1, -0.5, 0, 0.5]),
        (-1, +1, [-1, -0.5, 0, 0.5, +1]),
    ],
)
def test_halves_range(start: float, stop: float, expected: list):
    if expected is None:
        with pytest.raises(ValueError):
            _halves_range(start, stop)
    else:
        assert _halves_range(start, stop) == expected
