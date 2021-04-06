import qrules as q


def test_script():
    result = q.generate_transitions(
        initial_state="D0",
        final_state=["K~0", "K+", "K-"],
        allowed_intermediate_particles=[
            "a(0)(980)",
            "a(2)(1320)-",
            "phi(1020)",
        ],
        number_of_threads=1,
    )
    assert len(result.transitions) == 5
    assert result.get_intermediate_particles().names == {
        "a(0)(980)+",
        "a(0)(980)-",
        "a(0)(980)0",
        "a(2)(1320)-",
        "phi(1020)",
    }
