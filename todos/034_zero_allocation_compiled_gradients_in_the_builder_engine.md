# 034 — Zero-allocation compiled gradients in the builder engine

| Field | Value |
|---|---|
| ID | 034 |
| Title | Zero-allocation compiled gradients in the builder engine |
| Status | IN_PROGRESS |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-27 |
| Updated | 2026-09-27 |
| Related Files / Commits / PRs | branch `fix/zero-alloc-engine`; `experiments/claude_zero_alloc_engine_prompt.md` (+ addendum, rev2); `docs/architecture/zero_alloc_engine_report.md`; `src/models/pregame/tape_scalars.jl`; `src/harness/{tape_profile,perf_monitor}.jl`; `test/tape_allocation_tests.jl` |

## Context & Problem Statement

The W2 grid (TODO 031) ran `td_base` at 16 threads with sampler threads at 39–49 % CPU and ~8 GC
threads each burning ~⅔ of a sampler thread. The manager measured 432 KB allocated per compiled
ReverseDiff gradient (0.24 ms each): four `tracker_∇broadcast` instructions — `clamp` with scalar
bounds and the fused Gamma pxG arm — i.e. scalars inside fused broadcasts. The fix existed only in
prototypes (`l10`, `l11`); nothing in `src/`, `test/` or the harness gated on allocation. Blocks
TODO 031 W2. Work package: `experiments/claude_zero_alloc_engine_prompt.md`, with the smoke
performance-monitor addendum (§2.6) and its rev2 measurement tools.

## Acceptance Criteria

- [ ] Before/after audit, 15 W2 + 12 W1 recipes × {largest, season-opening} fold: after = 0 B.
- [ ] Log-density and gradient equal to pre-fix ≤ 1e-12 relative (prior + ±0.8 points), same set.
- [ ] 27 `recipe_hash` values unchanged; persisted W1 fit `decb01be-…` still loads.
- [ ] `test/tape_allocation_tests.jl` passes on the fix and fails on `8effca72`.
- [ ] Hard smoke check `tape_allocation` (> 1,024 B) exists, is required, is tested.
- [ ] Named suites pass on the beast; `run_parallel_tests.jl` no worse than known T007.
- [ ] §2.4 performance table (bytes, single-chain 150+150, 16-thread 16-chain) before/after.
- [ ] AD guide §10.5, `AGENTS.md` digest, this TODO; `./scripts/todo.sh check` passes.
- [ ] (addendum) smoke performance probe, metrics, review check and report exist and are tested;
      pre-fix `td_base` flagged `review`, fixed `td_base` passes, reports attached; `[SUMMARY]`
      carries the performance fields.

## Ideas & Candidate Solutions

- Per site: reduce first and multiply after (Gamma arm), regroup into unfused binary broadcasts
  (pyramid tiers, league deltas, hierarchical κ/HA), lift a sampled scalar with `tape_scalar`
  (NegBin `r`, lineup weights), `Ref` constants (clamp bounds), `tape_fill` (intercept/HA),
  matrix-product centring (GRW `mean(dims=1)`). Rejected: lifting everywhere — the six-input
  pyramid dual kernel was slower than the allocating original at one thread.
- Deferred follow-ups: GRW tapes still carry ~14k scalar instructions from another source
  (0 B, but 0.6 ms/gradient); HTML flame graphs and `JET.@report_opt` (new dependencies).

## Work Log & Progress

No work recorded yet. Append chronological entries using `[YYYY-MM-DD @agent]`
(optionally add UTC time), with `@human`, `@antigravity`, `@claude`, or `@pi`.
Include session/worktree when claiming. Example (replace with actual facts):

<!-- - [2026-09-27 @pi] Claimed in session <id>, worktree <path>; next action: ... -->

## Verification & Findings

Not run yet. Record commands, pass/fail or skips, wall time, benchmark metrics,
R-hat / bulk and tail ESS / divergences / predictive scores as applicable, and
links to artifacts and immutable run IDs. Distinguish hypotheses from findings.
