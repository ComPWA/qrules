import qrules


def test_script():
    reaction = qrules.generate_transitions(
        initial_state="D0",
        final_state=["Kbar0", "K+", "K-"],
        allowed_intermediate_particles=[
            "a_0(980)",
            "a_2(1320)-",
            "phi(1020)0",
        ],
    )
    groupings = sorted(reaction.group_by_topology().values())
    assert len(groupings) == 3
    assert len(groupings[0]) == 2
    assert len(groupings[1]) == 2
    assert len(groupings[2]) == 1
    assert reaction.get_intermediate_particles().names == [
        "a_0(980)-",
        "a_0(980)0",
        "a_0(980)+",
        "a_2(1320)-",
        "phi(1020)0",
    ]
