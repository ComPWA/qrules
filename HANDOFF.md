<!-- cspell:ignore gpgsign MAINT Physik procs punyaa -->

# Handoff — qrules spin-projection removal & 2-to-n performance work

> For a Claude instance picking up this work. Written 2026-07-14.
> Full prior transcript (if you need exact snippets/errors):
> `/home/tau/redeboer/.claude/projects/-data-local-redeboer-work-ComPWA-qrules/520b1f91-f1c2-477a-9a2f-58f2863cbd5a.jsonl`

## Standing constraints (do not violate)

- **Everything local. Never push or post to GitHub.** Reading via `gh issue view` is fine.
- **All commits unsigned:** `git -c commit.gpgsign=false commit --no-verify` (GPG unavailable here; the user re-signs before pushing).
- Commit-message trailer on **every** commit:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01TWmXKcjJMmRVJCUi58JDLH
  ```
- Project rules (CLAUDE.md): `uv run` everything; `poe style` before committing; `pytest -m 'not slow'` for fast suite; UPPER-CASE conventional commits (BREAK/BEHAVIOR/ENH/FEAT/DOC/MAINT/DX); global doctests are on (`--doctest-modules` over docs/src/tests); comments only for non-obvious rationale; mypy `# type: ignore` suppressions belong in `pyproject.toml`, not inline.

## Where we are: the stacked branches

This is **epic ComPWA/qrules#336**, built as a stack of ~10 local branches, each on top of the previous. State recorded in memory file:
`/home/tau/redeboer/.claude/projects/-data-local-redeboer-work-ComPWA-qrules/memory/epic-336-branch-stack.md`

- Branch 9 `two-to-n-topologies`: issue #29 (2-to-n production reactions, s/t/u Mandelstam channels, `group_by_channel`, `allowed_channels`). Tip commit `90eaa95`.
- **Branch 10 `remove-spin-projections` = current working branch**, built on `90eaa95`. This session's work lives here.

The user reviews diffs between stacked branches and opens PRs themselves.

## Commits already made on `remove-spin-projections` (newest last)

1. `bb003d3` **BREAK: remove spin projections from states and workflow** — `StateTransition = FrozenTransition[Particle, InteractionProperties]`; deleted `State` class and `ParticleWithSpin`; initial/final states are plain particle-name sequences; projection rules/domains removed from default `create_interaction_settings`; `SpinFormalism` now only selects output QN filters. Projection facts/rules KEPT for extensibility.
2. `0afca3c` **DOC: add notebook on re-enabling spin projections** — `docs/usage/spin-projections.ipynb`.
3. `38d7c68` **DOC: add three-body double-exchange examples to production notebook** — γp→pπ⁰η section in `docs/usage/production.ipynb`; `.cspell.json` words "regge", "mystnb".
4. `c428ee4` **ENH: solve QN problem sets in parallel in find_qn_transitions** — `number_of_threads` param + `_solve_qn_problem` helper using `Pool`. Benchmark: γp→pπ⁰η, 3 intermediate families (240 problem sets), 205s → 31s (6.7×), identical transitions.
5. `12c2744` **ENH: drop dominated QN problem sets before solving** — `remove_dominated_qn_problem_sets` in `solving.py`, called at top of `find_qn_transitions`. Removes problem sets whose per-element conservation-rule sets are supersets of another's (same facts+domains) → subset solutions. Helpers: `_create_domains_key`, `_create_rule_map` (rule id = `type(rule).__name__` + sorted `__dict__` items, so deep-copies compare equal but parametrized instances differ), `_is_sub_rule_map`; refactored `_create_merge_key` to share new `_create_facts_key`. Added `test_remove_dominated_qn_problem_sets`. Benchmark: 240→30 problem sets, 205s→12.7s serial (16×), 31s→1.9s over 8 procs, identical transitions.

## DONE — task #19: LS-free solving option (committed 2026-07-14 as `034c55c`)

**Completed on a different machine** (`/home/punyaa/Uni/Physik/Job_Fritsch/qrules`, session bd312cfc): the `TEMP: stash result of Claude Fable` commit (`5bb7fdc`, still on `origin/remove-spin-projections`) was reset and replaced by **`034c55c` ENH: make enumeration of LS couplings optional** — the local branch therefore diverges from origin by design; HANDOFF.md and `verify_ls_free.py` are untracked local artifacts again.

Beyond the edits listed below (all kept), finishing the task required four fixes:

1. `_iter_ls_couplings` used the triangle inequality `|L-S| ≤ J ≤ L+S`, which wrongly admits e.g. integer `J` from half-integer `(L ± 1/2)` series. Fixed to coupling-series membership `J ∈ couple(L, S)`; this was the source of a spurious `(J,P) = (2,+)` intermediate signature in γp→pπ⁰η (a baryon edge with integer spin). Regression doctest added to `SpinCoupling`.
2. `CParityNodeInput`/`GParityNodeInput` had **mandatory** `l_magnitude`/`s_magnitude` fields, so in LS-free problem sets `c_parity_conservation`/`g_parity_conservation` were silently "not executed" (QN path) and made `validate_full_solution` discard **all** completed particle-level solutions (STM path → "No solutions were found"). Fields are now optional; only the particle-antiparticle branch (which genuinely needs L,S) returns undetermined.
3. `_build_all_arguments` (argument_handling.py) dropped falsy args (`if arg`), so an **empty node property map** was silently omitted → `c_parity_conservation() missing 1 required positional argument`. Filter removed.
4. Solutions lacked node entries when nothing was solved on a node: `CSPSolver.__convert_solution_keys` and `collect_qn_transitions` now initialize every topology node (and edge) with an empty property map (fixes `FrozenTransition._assert_all_defined` and `_remove_qns_from_graph` KeyError).

`ls_couplings` is threaded through `generate_transitions` (`__init__.py`), `StateTransitionManager` (`transition.py`), `create_qn_problem_sets`/`generate_qn_transitions` (`workflow.py`). New test `test_generate_qn_transitions_without_ls_couplings`. Equivalence of intermediate (J,P) signatures verified for J/ψ→γπ⁰π⁰, γp→pπ⁰, γp→pπ⁰η. Benchmark (γp→pπ⁰η, 240 problem sets): 6232→3832 transitions, 25.3s→2.9s serial (8.7×), 3.8s→0.6s over 8 procs. `poe style`, fast suite (880) and slow suite (43) all pass.

**Original rationale (user's, correct):** make LS-coupling optional the way spin projections were, because L,S can be reconstructed afterward from the particles' spins and parities. When off, the solver stops enumerating (L,S) domains; two "existence" rules enforce the physics instead.

### Edits from the previous session (committed in `034c55c`)

**`src/qrules/conservation_rules.py`**

- `from collections.abc import Callable, Iterator` (added `Iterator`).
- New `SpinParityFacts` TypedDict (`spin_magnitude`, `parity`).
- New module helpers before `clebsch_gordan_helicity_to_canonical`: `_couple_spins`, `_split_isobar_node`, `_iter_ls_couplings`.
- New class `SpinCoupling(max_angular_momentum)` — LS-free counterpart of `spin_magnitude_conservation`; checks some (L,S) up to max couples the magnitudes; has doctests.
- New class `SpinParityCoupling(max_angular_momentum)` — LS-free counterpart of spin_magnitude + parity_conservation; also enforces `P_in = P_out·(-1)^L` for some valid coupling; has doctests.

**`src/qrules/settings.py`**

- Imported `SpinCoupling`, `SpinParityCoupling`.
- Added to `CONSERVATION_LAW_PRIORITIES`: `SpinCoupling: 8`, `SpinParityCoupling: 6` (both need `# type: ignore[dict-item]`).
- `create_interaction_settings` gained `ls_couplings: bool = True` (+ `# noqa: PLR0917`). When `True`: unchanged behavior (spin_magnitude_conservation + l/s_magnitude node domains; `parity_rules = {parity_conservation}`). When `False`: node settings use `{SpinCoupling(max_angular_momentum)}` with **no** node qn_domains, and `parity_rules = {SpinParityCoupling(max_angular_momentum)}`. The `em_node_settings` block now spreads `*parity_rules` where it previously listed `parity_conservation`.

**`src/qrules/workflow.py`**

- `create_qn_problem_sets` gained `ls_couplings: bool = True`, passed to `create_interaction_settings`; docstring updated.
- `generate_qn_transitions` gained `ls_couplings: bool = True`; docstring updated.

### Next steps (all done 2026-07-14)

All nine steps of the original plan were completed (see the DONE section above for the extra fixes that turned out to be necessary). Known remaining limitation, documented in the code: in LS-free mode, the particle-antiparticle branch of C-/G-parity conservation (composite `C = (-1)^{L+S}` for pairs with undefined individual C/G) cannot be evaluated and is skipped — an existence-style `CParityCoupling`/`GParityCoupling` could close that gap if it ever matters.

## Useful commands

```shell
uv run pytest -m 'not slow' -q
uv run pytest -m slow -q
uv run pytest --doctest-modules src/qrules/conservation_rules.py
poe style
git -c commit.gpgsign=false commit --no-verify -m "..."
```

## Scratchpad

The 2026-07-14 session's scratchpad (benchmark + debug scripts for the LS-free work):
`/tmp/claude-1000/-home-punyaa-Uni-Physik-Job-Fritsch-qrules/bd312cfc-412f-4b86-bd7d-b0de51f34839/scratchpad`
(session-isolated; a different session won't look here automatically).
