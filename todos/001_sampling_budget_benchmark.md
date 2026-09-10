# 001 — Benchmark MCMC Sampling Budget and Target Acceptance on Fold 1

| Field | Value |
|---|---|
| ID | 001 |
| Title | Benchmark MCMC Sampling Budget and Target Acceptance on Fold 1 |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-09 |
| Updated | 2026-09-10 |
| Related Files / Commits / PRs | [Experiment 08 README](../experiments/scottish_lower/08_goal_decomposition/README.md); [Fold 1 smoke evidence](../experiments/scottish_lower/08_goal_decomposition/results/smoke_fold1_2026-09-09.md); [smoke runner](../experiments/scottish_lower/08_goal_decomposition/r08_smoke.jl); [benchmark runner](../experiments/scottish_lower/08_goal_decomposition/r08_sampling_budget_benchmark.jl); [benchmark evidence](../experiments/scottish_lower/08_goal_decomposition/results/sampling_budget_fold1_2026-09-10.md) |

## Context & Problem Statement

Benchmark a defensible Fold 1 NUTS sampling budget and target acceptance for
Experiment 08's four goal-decomposition arms without changing their model, data,
or filtration contract. The discussion-reported production duration is 36–40
hours, so the working hypothesis is that B may cost roughly half of A while still
passing the required thresholds: maximum R-hat ≤ 1.05, minimum ESS ≥ 400, and
zero divergences.

The Experiment 08 Fold 1 smoke evidence reports four chains × 1,000 warmup ×
1,000 retained draws, zero divergences, and minimum ESS above 1,470 (reported
range 1,479–2,104). Those timing and ESS values are **discussion-reported, not
newly verified by this TODO**; the linked Experiment 08 README and evidence file
remain the authoritative record.

The matched configurations are all four chains: **A** = 1,000 warmup / 1,000
retained / `delta = 0.95`; **B** = 500 / 500 / `delta = 0.95`; **C** = 500 / 500
/ `delta = 0.90`. `max_depth = 8` is an optional follow-up only, not an A/B/C
confound.

## Acceptance Criteria

- [x] A, B, and C use the declared four-chain settings on identical Fold 1 data,
  model arms, and filtration, with no undeclared sampler differences.
- [x] Each arm/configuration records wall time, divergences, maximum R-hat,
  minimum bulk/tail ESS, ESS/sec, and explicit threshold pass/fail.
- [x] The comparison defines its primary metric before sampling and retains raw
  diagnostics sufficient to distinguish measured findings from discussion context.
- [x] A recommendation identifies the lowest safe setting, the trade-off against
  A, and every unsafe or disproportionately expensive option.
- [x] Findings are linked into Experiment 08 without overwriting established
  smoke evidence or representing this task's context as newly verified.

## Ideas & Candidate Solutions

- Hold the mass-matrix choice fixed in A/B/C and make ESS/sec the primary
  efficiency metric, with convergence thresholds as hard gates. Record whether
  the current metric is diagonal or dense; test the alternative only as a
  separately labeled follow-up because mass-matrix geometry is a confound.
- Compare the `delta = 0.95` baseline against C's `delta = 0.90` through dual
  averaging adaptation: lower target acceptance can shorten trajectories but may
  add divergences or depress ESS/sec.
- Inspect tree-depth saturation separately. Repeated hits at the current maximum
  tree depth can bias an apparent saving or hide poor exploration; test
  `max_depth = 8` only after A/B/C if saturation evidence warrants it.
- If B clears gates and approaches half of A's cost, prefer it; otherwise retain
  A. Do not infer production-grid savings from one arm without reporting the
  four-arm spread.

## Work Log & Progress

- [2026-09-09 @pi] Created as ACTIVE/P1 and unassigned. Defined matched A/B/C,
  remote safety/preflight requirements, and evidence boundaries; no sampling was
  run by this task.
- [2026-09-10 @claude] Claimed in session `claude_todo`. Preparing Fold 1 benchmark runner across configurations A (1000/1000, 0.95), B (500/500, 0.95), and C (500/500, 0.90) to execute on mcmc-beast.
- [2026-09-10 @claude] Wrote `r08_sampling_budget_benchmark.jl`: the production
  `sample_fold` path, A asserted identical to `L08_SAMPLER`, and four independent
  4-chain fits per (arm, config) in one flat interleaved 16-slot queue with common
  seeds. Primary metric and gates were written to the manifest before sampling.
  Preflight passed at 10:57 on the beast. Authorised run `abc_20260910` in a new tmux
  session `sampling_budget_bench`, leaving `goal_decomposition_grid` untouched: 96/96
  chains, 178.5 min, no failures. Also audited the 40 m00 production-grid checkpoints
  read-only. Declined the `max_depth = 8` follow-up because depth never exceeded 7
  (grid: 8) against a cap of 10. Closed as COMPLETED.

## Verification & Findings

**Measured on 2026-09-10.** Full tables are in
[`sampling_budget_fold1_2026-09-10.md`](../experiments/scottish_lower/08_goal_decomposition/results/sampling_budget_fold1_2026-09-10.md);
raw CSVs, manifest and log are in `results/sampling_budget_fold1/abc_20260910/`.
The run used mcmc-beast, Julia 1.12.6, 16 pinned threads, BLAS 1 and default GC.
Arms: m00 and m01. Values below are medians of 4 fits.

| Arm | Cfg | Wall s (minutes) | Core-s × A | max R-hat (worst) | min bulk / tail ESS (worst fit) | ESS/wall-s (× A) | Divergences | Strict gate (ESS ≥ 400, 0 div) | ESS ≥ 200 + 0 div |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m00 | A | 2,964 (49.4) | 1.000 | 1.0073 | 1,622 / 2,007 | 0.586 (1.00) | 1 / 16,000 | 3/4 | 3/4 |
| m00 | B | 1,775 (29.6) | 0.588 | 1.0129 | 841 / 849 | 0.501 (0.86) | 0 / 8,000 | 4/4 | 4/4 |
| m00 | C | 1,562 (26.0) | 0.480 | 1.0099 | 721 / 557 | 0.483 (0.82) | 1 / 8,000 | 3/4 | 3/4 |
| m01 | A | 2,559 (42.6) | 1.000 | 1.0068 | 1,598 / 1,469 | 0.677 (1.00) | 0 / 16,000 | 4/4 | 4/4 |
| m01 | B | 1,509 (25.1) | 0.606 | 1.0102 | 737 / 583 | 0.483 (0.71) | 0 / 8,000 | 4/4 | 4/4 |
| m01 | C | 1,357 (22.6) | 0.568 | 1.0123 | 859 / 665 | 0.587 (0.87) | 0 / 8,000 | 4/4 | 4/4 |

Across all 24 fits, BFMI was at least 0.667 and the maximum tree depth was 7 with zero capped
transitions. The 40-fold m00 grid under A (read-only checkpoint audit) has fold 1 min
ESS 2,003 and a worst fold of fold 6 at 680, **0.339×** fold 1, with one divergence
(fold 3).

**Findings**

- Half the budget costs 0.59–0.61× A's core time, not 0.5×. A linear fit through the
  two budget points gives an implied fixed per-chain cost of about 430–500 s. ESS per
  draw is unchanged or lower, so the primary metric (min ESS per wall-second) is
  highest under A in both arms: B is 0.71–0.86× A and C is 0.82–0.87× A.
- On fold 1 every cell clears ESS ≥ 400. Scaled by the observed worst-fold ratio
  (0.34–0.39), B projects to 285–340 (m00) and 198–275 (m01) at the worst fold, and C
  to 189–325. A projects to 590–680 and was observed at 680.
- The discussion-reported hypothesis that B costs about half of A while holding every
  threshold is **not supported**. Its cost is 0.6×, and the ESS ≥ 400 threshold is
  projected to fail on the harder folds.
- Divergences are rare, and they occur under the control: 1 in 4,000 in A m00 rep 3
  and in C m00 rep 4. The zero-divergence gate fails A on 1 of 4 Fold 1 fits, so it is
  not a budget-sensitive criterion.
- δ 0.90 (C) cuts leapfrog steps (17–23 vs 21–30) and core time by 6–18% against B.
  ESS per draw moves in opposite directions for the two arms.
- Chains in the lightly loaded queue tail (about 12 concurrent) ran up to 2× faster
  than under full 16-chain load. At full load, sampling threads used about 48% of wall
  time and GC threads about 39% of process CPU. This is followed up in
  [TODO 003](003_benchmark_julia_gc_tuning_and_heap_size_hint_on_multi_core_sampling.md).

**Recommendation**

- Keep **A** (4 × 1,000 / 1,000, δ 0.95) for production grids. It is the lowest
  setting projected safe under ESS ≥ 400 and the most efficient per second.
- **B is unsafe under ESS ≥ 400.** It saves about 40% core time, roughly 37.5 h → 22.5 h
  for four arms if fold 1's ratio holds on the larger folds.
- **C is unsafe on both ESS gates** at the projected worst fold, and it also produced a
  divergence.
- If a saving is required, test about 750/750 at δ 0.95 on the weakest fold (fold 6),
  not fold 1: projected worst-fold ESS about 510 at about 0.8× core time. This
  projection is untested.
- The divergence tolerance is TODO 002's policy decision.

Boundaries: fold 1 only; two of the four arms (m00, m01); one host and runtime
configuration. The worst-fold projections rest on one 40-fold m00 fit, and m01 borrows
m00's fold ratio. The earlier discussion-reported smoke timings were not re-verified
here: the smoke evidence was measured on archpc with 8 threads and is not directly
comparable.
