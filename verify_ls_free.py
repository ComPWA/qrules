"""Handoff helper: verify + benchmark the `ls_couplings=False` option (task #19).

Starting point for the in-progress LS-free work described in HANDOFF.md. Run with:

    uv run python verify_ls_free.py

The equivalence check compares the *intermediate-state (spin, parity) signatures*,
NOT the raw transition counts: with ``ls_couplings=False`` the solver does not
enumerate (L, S) combinations, so it produces fewer transitions but must allow the
same set of intermediate quantum numbers. If the signature sets differ, the diff is
printed -- that is the thing to debug.

This file is gitignored (local handoff artifact, not part of the branch).
"""

from __future__ import annotations

import time
from typing import Any

import qrules
from qrules.quantum_numbers import EdgeQuantumNumbers as E
from qrules.workflow import create_qn_problem_sets, find_qn_transitions

PDG = qrules.load_pdg()


def signatures(transitions: Any) -> set[tuple[Any, Any]]:
    return {
        (state[E.spin_magnitude], state[E.parity])
        for transition in transitions
        for state in transition.intermediate_states.values()
    }


def run(label: str, **reaction_kwargs: Any) -> None:
    print(f"=== {label} ===")
    sig_by_mode: dict[bool, set[tuple[Any, Any]]] = {}
    for ls_couplings in (True, False):
        collection = create_qn_problem_sets(
            particle_db=PDG, ls_couplings=ls_couplings, **reaction_kwargs
        )
        n_problem_sets = sum(map(len, collection.problem_sets.values()))
        t0 = time.time()
        transitions = find_qn_transitions(collection, PDG)
        dt = time.time() - t0
        sig_by_mode[ls_couplings] = signatures(transitions)
        print(
            f"  ls_couplings={ls_couplings!s:5}: {n_problem_sets:4d} problem sets,"
            f" {len(transitions):5d} transitions,"
            f" {len(sig_by_mode[ls_couplings]):3d} (J,P) signatures, {dt:6.1f}s"
        )
    default, ls_free = sig_by_mode[True], sig_by_mode[False]
    print(f"  intermediate (J,P) signatures identical: {default == ls_free}")
    if default != ls_free:
        print("    only with LS on: ", sorted(map(str, default - ls_free)))
        print("    only with LS off:", sorted(map(str, ls_free - default)))
    print()


if __name__ == "__main__":
    run(
        "J/psi -> gamma pi0 pi0 (strong)",
        initial_state=["J/psi(1S)"],
        final_state=["gamma", "pi0", "pi0"],
        allowed_intermediate_particles=["f(0)(980)", "f(0)(1500)", "f(2)(1270)"],
        allowed_interaction_types="strong",
        max_angular_momentum=2,
    )
    run(
        "gamma p -> pi0 p (strong, EM)",
        initial_state=["gamma", "p"],
        final_state=["pi0", "p"],
        allowed_intermediate_particles=[
            "Delta(1232)",
            "N(1440)",
            "rho(770)",
            "omega(782)",
        ],
        allowed_interaction_types=["strong", "em"],
        max_angular_momentum=2,
    )
