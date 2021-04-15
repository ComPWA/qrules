import pytest

from qrules.settings import (
    InteractionTypes,
    _halves_domain,
    _int_domain,
    create_interaction_settings,
)


@pytest.mark.parametrize("interaction_type", list(InteractionTypes))
@pytest.mark.parametrize("nbody_topology", [False, True])
@pytest.mark.parametrize(
    "formalism_type", ["canonical", "canonical-helicity", "helicity"]
)
def test_create_interaction_settings(
    interaction_type: InteractionTypes,
    nbody_topology: bool,
    formalism_type: str,
):
    settings = create_interaction_settings(formalism_type, nbody_topology)
    assert set(settings) == set(InteractionTypes)

    edge_settings, node_settings = settings[interaction_type]
    edge_qn_domains_str = {  # strings are easier to compare with pytest
        qn_type.__name__: domain
        for qn_type, domain in edge_settings.qn_domains.items()
    }
    assert edge_qn_domains_str == {
        "baryon_number": [-1, 0, +1],
        "electron_lepton_number": [-1, 0, +1],
        "muon_lepton_number": [-1, 0, +1],
        "tau_lepton_number": [-1, 0, +1],
        "parity": [-1, +1],
        "c_parity": [-1, +1, None],
        "g_parity": [-1, +1, None],
        "spin_magnitude": _halves_domain(0, 2),
        "spin_projection": _halves_domain(-2, +2),
        "charge": _int_domain(-2, 2),
        "isospin_magnitude": _halves_domain(0, 1.5),
        "isospin_projection": _halves_domain(-1.5, +1.5),
        "strangeness": _int_domain(-1, 1),
        "charmness": _int_domain(-1, 1),
        "bottomness": _int_domain(-1, 1),
    }

    expected = {
        "l_magnitude": _int_domain(0, 2),
        "s_magnitude": _halves_domain(0, 2),
    }
    if "canonical" in formalism_type:
        expected["l_projection"] = [-2, -1, 0, 1, 2]
        expected["s_projection"] = _halves_domain(-2, 2)
    if formalism_type == "canonical-helicity":
        expected["l_projection"] = [0]
    if (
        "helicity" in formalism_type
        and interaction_type != InteractionTypes.WEAK
    ):
        expected["parity_prefactor"] = [-1, 1]
    if nbody_topology:
        expected["l_magnitude"] = [0]
        expected["s_magnitude"] = [0]
    if nbody_topology and formalism_type != "helicity":
        expected["l_projection"] = [0]
        expected["s_projection"] = [0]

    node_qn_domains_str = {  # strings are easier to compare with pytest
        qn_type.__name__: domain
        for qn_type, domain in node_settings.qn_domains.items()
    }
    assert node_qn_domains_str == expected


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
            _halves_domain(start, stop)
    else:
        assert _halves_domain(start, stop) == expected