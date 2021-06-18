import qrules


def test_script():
    reaction = qrules.generate_transitions(
        initial_state="D0",
        final_state=["K~0", "K+", "K-"],
        allowed_intermediate_particles=[
            "a(0)(980)",
            "a(2)(1320)-",
            "phi(1020)",
        ],
        number_of_threads=1,
    )
    assert len(reaction.transition_groups) == 3
    assert len(reaction.transition_groups[0]) == 2
    assert len(reaction.transition_groups[1]) == 1
    assert len(reaction.transition_groups[2]) == 2
    assert reaction.get_intermediate_particles().names == [
        "a(0)(980)-",
        "a(0)(980)0",
        "a(0)(980)+",
        "a(2)(1320)-",
        "phi(1020)",
    ]
