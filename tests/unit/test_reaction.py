import pytest

from qrules import _determine_interaction_types
from qrules.settings import InteractionType as IT  # noqa: N817


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("all", {IT.STRONG, IT.WEAK, IT.EM}),
        ("EM", {IT.EM}),
        ("electromagnetic", {IT.EM}),
        ("electro-weak", {IT.EM, IT.WEAK}),
        ("ew", {IT.EM, IT.WEAK}),
        ("w", {IT.WEAK}),
        ("strong", {IT.STRONG}),
        ("only strong", {IT.STRONG}),
        ("S", {IT.STRONG}),
        (["e", "s", "w"], {IT.STRONG, IT.WEAK, IT.EM}),
        ("strong and EM", {IT.STRONG, IT.EM}),
        ("", ValueError),
        ("non-existing", ValueError),
    ],
)
def test_determine_interaction_types(description, expected):
    if expected is ValueError:
        with pytest.raises(ValueError, match="interaction type"):
            assert _determine_interaction_types(description)
    else:
        assert _determine_interaction_types(description) == expected
