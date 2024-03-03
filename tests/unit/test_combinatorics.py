from __future__ import annotations

import pytest

from qrules.combinatorics import (
    _generate_kinematic_permutations,
    _get_kinematic_representation,
    _KinematicRepresentation,
    _permutate_outer_edges,
    create_initial_facts,
    permutate_topology_kinematically,
)
from qrules.topology import Topology, create_isobar_topologies


@pytest.fixture(scope="session")
def three_body_decay() -> Topology:
    topologies = create_isobar_topologies(3)
    return next(iter(topologies))


def test_create_initial_facts(three_body_decay, particle_database):
    initial_facts = create_initial_facts(
        three_body_decay,
        initial_state=[("J/psi(1S)", [-1, +1])],
        final_state=["gamma", "pi0", "pi0"],
        particle_db=particle_database,
    )
    assert len(initial_facts) == 4

    for fact in initial_facts:
        edge_ids = sorted(fact.states)
        assert edge_ids == [-1, 0, 1, 2]
        particle_names = [fact.states[i][0].name for i in edge_ids]
        assert particle_names == ["J/psi(1S)", "gamma", "pi0", "pi0"]
        _, initial_polarization = fact.states[-1]
        assert initial_polarization in {-1, +1}


def test_generate_kinematic_permutations_groupings(three_body_decay: Topology):
    topology = three_body_decay
    particle_names = {
        -1: "J/psi(1S)",
        0: "gamma",
        1: "pi0",
        2: "pi0",
    }
    allowed_kinematic_groupings = [_KinematicRepresentation(["pi0", "pi0"])]
    permutations = _generate_kinematic_permutations(
        topology, particle_names, allowed_kinematic_groupings
    )
    assert len(permutations) == 1

    permutations = _generate_kinematic_permutations(topology, particle_names)
    assert len(permutations) == 2
    assert permutations[0].get_originating_final_state_edge_ids(1) == {1, 2}
    assert permutations[1].get_originating_final_state_edge_ids(1) == {0, 2}


@pytest.mark.parametrize(
    ("n_permutations", "decay_type"),
    [
        (3, "two_to_three_decay"),
        (3, "three_body_decay"),
    ],
)
def test_permutate_outer_edges(
    n_permutations: int,
    decay_type: str,
    three_body_decay: Topology,
    two_to_three_decay: Topology,
):
    if decay_type == "two_to_three_decay":
        topology = two_to_three_decay
    elif decay_type == "three_body_decay":
        topology = three_body_decay
    else:
        raise NotImplementedError(decay_type)
    permutations = _permutate_outer_edges(topology)
    assert len(permutations) == n_permutations


@pytest.mark.parametrize(
    "final_state_groupings",
    [
        ["pi0", "pi0"],
        [["pi0", "pi0"]],
        [[["pi0", "pi0"]]],
        ["gamma", "pi0"],
        [["gamma", "pi0"]],
        [[["gamma", "pi0"]]],
        None,
    ],
)
def test_permutate_topology_kinematically(
    final_state_groupings,
    three_body_decay: Topology,
):
    permutations = permutate_topology_kinematically(
        topology=three_body_decay,
        initial_state=[("J/psi(1S)", [-1, +1])],
        final_state=["gamma", "pi0", "pi0"],
        final_state_groupings=final_state_groupings,
    )
    if final_state_groupings is None:
        assert len(permutations) == 2
    else:
        assert len(permutations) == 1


@pytest.mark.parametrize(
    ("n_permutations", "initial_state", "final_state"),
    [
        (2, ["J/psi(1S)"], ["gamma", "pi0", "pi0"]),
        (3, ["J/psi(1S)"], ["gamma", "pi-", "pi+"]),
        (2, ["e+", "e-"], ["gamma", "pi0", "pi0"]),
        (3, ["e+", "e-"], ["gamma", "pi-", "pi+"]),
    ],
)
def test_generate_kinematic_permutations(
    n_permutations: int,
    initial_state: list[str],
    final_state: list[str],
    three_body_decay: Topology,
    two_to_three_decay: Topology,
):
    if len(initial_state) == 1:
        topology = three_body_decay
    elif len(initial_state) == 2:
        topology = two_to_three_decay
    else:
        raise NotImplementedError
    particle_names = dict(
        zip(
            sorted(topology.incoming_edge_ids) + sorted(topology.outgoing_edge_ids),
            list(initial_state) + list(final_state),
        )
    )
    permutations = _generate_kinematic_permutations(topology, particle_names)
    assert len(permutations) == n_permutations


class TestKinematicRepresentation:
    def test_constructor(self):
        representation = _KinematicRepresentation(
            initial_state=["J/psi"],
            final_state=["gamma", "pi0"],
        )
        assert representation.initial_state == [["J/psi"]]
        assert representation.final_state == [["gamma", "pi0"]]
        representation = _KinematicRepresentation([["gamma", "pi0"]])
        assert representation.initial_state is None
        assert representation.final_state == [["gamma", "pi0"]]

    def test_from_topology(self, three_body_decay: Topology):
        states = {
            -1: "J/psi",
            0: "pi0",
            1: "pi0",
            2: "gamma",
        }
        kinematic_representation1 = _get_kinematic_representation(
            three_body_decay, states
        )
        assert kinematic_representation1.initial_state == [
            ["J/psi"],
            ["J/psi"],
        ]
        assert kinematic_representation1.final_state == [
            ["gamma", "pi0"],
            ["gamma", "pi0", "pi0"],
        ]

        kinematic_representation2 = _get_kinematic_representation(
            topology=three_body_decay,
            particle_names={
                -1: "J/psi",
                0: "pi0",
                1: "gamma",
                2: "pi0",
            },
        )
        assert kinematic_representation1 == kinematic_representation2

        kinematic_representation3 = _get_kinematic_representation(
            topology=three_body_decay,
            particle_names={
                -1: "J/psi",
                0: "pi0",
                1: "gamma",
                2: "gamma",
            },
        )
        assert kinematic_representation2 != kinematic_representation3

    def test_repr_and_equality(self):
        kinematic_representation = _KinematicRepresentation(
            initial_state=[["J/psi"]],
            final_state=[["gamma", "pi0"], ["gamma", "pi0", "pi0"]],
        )
        constructed_from_repr = eval(str(kinematic_representation))
        assert constructed_from_repr == kinematic_representation

    def test_in_operator(self):
        kinematic_representation = _KinematicRepresentation(
            [["gamma", "pi0"], ["gamma", "pi0", "pi0"]],
        )
        subset_representation = _KinematicRepresentation(
            [["gamma", "pi0", "pi0"]],
        )
        assert subset_representation in kinematic_representation
        assert [["J/psi"]] not in kinematic_representation
        assert [["gamma", "pi0"]] in kinematic_representation
        with pytest.raises(ValueError, match=r"Cannot compare "):
            assert 0.0 in kinematic_representation
        with pytest.raises(
            TypeError,
            match=r"Comparison representation needs to be a list of lists",
        ):
            assert ["should be nested list"] in kinematic_representation
