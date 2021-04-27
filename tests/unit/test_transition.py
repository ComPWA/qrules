import pytest

from qrules.transition import StateTransitionManager


class TestStateTransitionManager:
    @staticmethod
    def test_allowed_intermediate_particles():
        stm = StateTransitionManager(
            initial_state=[("J/psi(1S)", [-1, +1])],
            final_state=["p", "p~", "eta"],
            number_of_threads=1,
        )
        particle_name = "N(753)"
        with pytest.raises(
            LookupError,
            match="Could not find any matches for allowed intermediate particle",
        ):
            stm.set_allowed_intermediate_particles([particle_name])
