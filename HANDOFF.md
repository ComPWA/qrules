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

## IN PROGRESS — task #19: LS-free solving option (NOT tested, NOT committed)
**Rationale (user's, correct):** make LS-coupling optional the way spin projections were, because L,S can be reconstructed afterward from the particles' spins and parities. When off, the solver stops enumerating (L,S) domains; two "existence" rules enforce the physics instead.

### Uncommitted edits already made
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

### Next steps to finish task #19
1. **Still TODO:** thread `ls_couplings` through `generate_transitions` in `src/qrules/__init__.py` (particle-level façade) — not done yet.
2. Run doctests: `uv run pytest --doctest-modules src/qrules/conservation_rules.py`.
3. **Equivalence check:** LS-free solving must yield the same set of intermediate-state (spin, parity) signatures as default solving for a known reaction (J/ψ→γπ⁰π⁰ and γp→pπ⁰). Compare *intermediate-state signatures*, NOT raw transition counts — LS-free deliberately produces fewer/no LS variants.
4. **Benchmark** LS-free vs default on γp→pπ⁰η to quantify the combinatorics reduction; put numbers in the commit message.
5. Verify `_restrict_domains_to_rules` (workflow.py) handles nodes with no qn_domains cleanly (SpinCoupling/SpinParityCoupling consume edge QNs only — should be fine, confirm no empty-domain surprise).
6. Confirm the dedup (`_create_rule_map`) treats `SpinCoupling(2)` instances as equal across deep-copies (they store a name-mangled `_SpinCoupling__max_angular_momentum` key; sorted `__dict__.items()` handles it).
7. `poe style` (watch cspell/ruff/mypy), then `pytest -m 'not slow'` and `pytest -m slow`.
8. Commit as **ENH** (new option) with benchmark, unsigned, with the standard trailer.
9. Update `epic-336-branch-stack.md` to record parallelization, dedup, and LS-free additions on branch 10.

## Useful commands
```shell
uv run pytest -m 'not slow' -q
uv run pytest -m slow -q
uv run pytest --doctest-modules src/qrules/conservation_rules.py
poe style
git -c commit.gpgsign=false commit --no-verify -m "..."
```

## Scratchpad
This session's scratchpad (prototypes, benchmark scripts, rendered PNGs):
`/data/local/redeboer/tmp/claude-735/-data-local-redeboer-work-ComPWA-qrules/520b1f91-f1c2-477a-9a2f-58f2863cbd5a/scratchpad`
(session-isolated; a different session won't look here automatically).
