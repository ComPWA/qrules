import pytest

from qrules.conservation_rules import ChargeConservation, ChargeFacts


@pytest.mark.parametrize(
    ("graph_input", "expected_value"),
    [
        (
            (
                [ChargeFacts(charge=0)],
                [ChargeFacts(charge=-1), ChargeFacts(charge=1)],
            ),
            True,
        ),
        (
            (
                [ChargeFacts(charge=0)],
                [ChargeFacts(charge=1), ChargeFacts(charge=1)],
            ),
            False,
        ),
    ],
)
def test_charge_conservation(graph_input, expected_value):
    assert ChargeConservation()(*graph_input) == expected_value
