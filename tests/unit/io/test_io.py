import json
from typing import Callable, Optional

import pytest

from qrules import io
from qrules.particle import Particle, ParticleCollection
from qrules.topology import (
    StateTransitionGraph,
    Topology,
    create_isobar_topologies,
    create_n_body_topology,
)
from qrules.transition import Result


def through_dict(instance):
    asdict = io.asdict(instance)
    asdict = json.loads(json.dumps(asdict))  # check JSON serialization
    return io.fromdict(asdict)


@pytest.mark.parametrize("formalism", ["canonical", "helicity"])
def test_asdict_fromdict(
    particle_selection: ParticleCollection,
    formalism: Optional[str],
    get_reaction: Callable[[str], Result],
):
    if formalism is None:
        # ParticleCollection
        fromdict = through_dict(particle_selection)
        assert isinstance(fromdict, ParticleCollection)
        assert particle_selection == fromdict
        # Particle
        for particle in particle_selection:
            fromdict = through_dict(particle)
            assert isinstance(fromdict, Particle)
            assert particle == fromdict
        # Topology
        for n_final_states in range(2, 6):
            for topology in create_isobar_topologies(n_final_states):
                fromdict = through_dict(topology)
                assert isinstance(fromdict, Topology)
                assert topology == fromdict
            for n_initial_states in range(1, 3):
                topology = create_n_body_topology(
                    n_initial_states, n_final_states
                )
                fromdict = through_dict(topology)
                assert isinstance(fromdict, Topology)
                assert topology == fromdict
    else:
        # StateTransitionGraph
        result = get_reaction(formalism)
        for graph in result.transitions:
            fromdict = through_dict(graph)
            assert isinstance(fromdict, StateTransitionGraph)
            assert graph == fromdict
        # Result
        fromdict = through_dict(result)
        assert isinstance(fromdict, Result)
        assert result == fromdict


def test_fromdict_exceptions():
    with pytest.raises(NotImplementedError):
        io.fromdict({"non-sense": 1})
