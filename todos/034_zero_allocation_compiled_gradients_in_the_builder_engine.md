# 034 — Zero-allocation compiled gradients in the builder engine

| Field | Value |
|---|---|
| ID | 034 |
| Title | Zero-allocation compiled gradients in the builder engine |
| Status | BLOCKED |
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

- [x] Before/after audit, 15 W2 + 12 W1 recipes × {largest, season-opening} fold: after = 0 B.
- [x] Log-density and gradient equal to pre-fix ≤ 1e-12 relative (prior + ±0.8 points), same set.
- [x] 27 `recipe_hash` values unchanged; persisted W1 fit `decb01be-…` still loads.
- [x] `test/tape_allocation_tests.jl` passes on the fix and fails on `8effca72`.
- [x] Hard smoke check `tape_allocation` (> 1,024 B) exists, is required, is tested.
- [x] Named suites pass on the beast; `run_parallel_tests.jl` no worse than known T007.
- [x] §2.4 performance table (bytes, single-chain 150+150, 16-thread 16-chain) before/after.
- [x] AD guide §10.5, `AGENTS.md` digest, this TODO; `./scripts/todo.sh check` passes.
- [x] (addendum) smoke performance probe, metrics, review check and report exist and are tested;
      `[SUMMARY]` carries the performance fields; pre-fix `td_base` flagged `review`, reports attached.
- [ ] (addendum) fixed `td_base` → `pass`: **not met** — still `review` (GC ~32 %) from AdvancedHMC /
      DynamicPPL allocation outside the tape; human decision (report § Blocked).

## Ideas & Candidate Solutions

- Per site: reduce first and multiply after (Gamma arm), regroup into unfused binary broadcasts
  (pyramid tiers, league deltas, hierarchical κ/HA), lift a sampled scalar with `tape_scalar`
  (NegBin `r`, lineup weights), `Ref` constants (clamp bounds), `tape_fill` (intercept/HA),
  matrix-product centring (GRW `mean(dims=1)`). Rejected: lifting everywhere — the six-input
  pyramid dual kernel was slower than the allocating original at one thread.
- Deferred follow-ups: GRW tapes still carry ~14k scalar instructions from another source
  (0 B, but 0.6 ms/gradient); HTML flame graphs and `JET.@report_opt` (new dependencies).

## Work Log & Progress

- [2026-09-27 @claude] Claimed; worktree `.worktrees/BayesianFootball-zero-alloc`, branch
  `fix/zero-alloc-engine` from `8effca72`. Audit script + per-instruction profiler (`dd0472c6`);
  before-table on all 27 recipes × 2 folds: 35–923 KB per gradient, 7 distinct offender kinds.
- [2026-09-27 @claude] Engine fix (`edc48474`): `tape_scalar`/`tape_fill`, `Ref` clamp bounds,
  reduce-first Gamma arm, lifted NegBin `r` and lineup weights, split league-delta/κ/HA kernels,
  matrix-product GRW centring. Pyramid term regrouped after lifting proved slower (`cfd71867`).
- [2026-09-27 @claude] Hard `tape_allocation` smoke check + `SMOKE_REQUIRED_CHECKS` (2-line
  `store.jl` change, flagged), `test/tape_allocation_tests.jl` (147 recipes) (`cfd71867`).
- [2026-09-27 @claude] Addendum §2.6 smoke performance probe (`08f2f309`), rev2 tools (GC_Diff,
  profiler per thread, Profile.Allocs) (`9472300c`), profiler cap (`49eba057`); docs (`4fef9697`).
- [2026-09-27 @claude] Evidence on the beast (`6f146ae2`, `49eba057`); report
  `docs/architecture/zero_alloc_engine_report.md`. BLOCKED on one human decision: fixed `td_base`
  still `review` (sampler-side allocation), see report § Blocked. W2 grid is otherwise unblocked:
  `td_base` and `grw_step_a2_carry_jump_ldelta` smokes pass all six required hard checks.

## Verification & Findings

- Audit after (`/root/BF_runs/logs/6f146ae2/audit_after.tsv`): **0 B** for all 54 recipe×fold rows
  (was 35,248–923,456 B); parity vs pre-fix at prior + δ∈{0.003, ±0.8}: worst log-density 8.5e-16,
  gradient 3.7e-15; 27/27 recipe hashes unchanged; W1 fit `decb01be-…` loads (40 folds, 710 fixtures).
- `test/tape_allocation_tests.jl`: fix 150/150; on `8effca72` 3 pass / **147 fail**.
- Suites on `6f146ae2`: tape_allocation, pyramid_tier, multiscale_grw, builder, harness_runner (136),
  harness_scoring, harness_portfolio, scoped_splitter all pass; `run_parallel_tests.jl` 24/25, only
  known T007. `harness_runner_tests.jl` 136/136 again on `49eba057`.
- §2.4, 16 threads × 16 chains × (100+100): `td_base` 34.3 s → 11.6 s (**2.97×**), GC 41 % → 24 %;
  `grw_step_a2_carry_jump_ldelta` 1,181 s → 656 s (**1.80×**), GC 15 % → 7 %. Single chain 150+150:
  `td_base` 457 → 33 KB and 0.244 → 0.162 ms per leapfrog; GRW 1,167 → 260 KB, 1.007 → 0.822 ms.
- Monitor: pre-fix `td_base` review (GC 44.5 %, 59 % GC-stall samples, eff 7.9); fixed review (GC
  31.6 %, eff 3.1; remaining 28 KB/leapfrog is AdvancedHMC/DynamicPPL); profiler overhead +2.5 %.
- Findings for follow-up: GRW tapes still ~14.8k instructions (0 B) and 7.3× in-situ/bare at 16
  threads with 7 % GC; all 28 legacy engines clamp with scalar bounds (not fixed, out of scope).
