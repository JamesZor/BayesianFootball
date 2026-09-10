# Experiment 08 Fold 1 NUTS sampling-budget benchmark — 2026-09-10

TODO [001](../../../../todos/001_sampling_budget_benchmark.md). This file records
measured values only. The Fold 1 smoke evidence in
[`smoke_fold1_2026-09-09.md`](smoke_fold1_2026-09-09.md) is unchanged.

## Command and provenance

```bash
# preflight (no sampling), then the authorised run, tmux session `sampling_budget_bench`
env JULIA_PKG_PRECOMPILE_AUTO=0 L08_RUN_BENCH=true L08_BENCH_TAG=abc_20260910 \
  /root/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia --project -t 16 \
  experiments/scottish_lower/08_goal_decomposition/r08_sampling_budget_benchmark.jl
```

- Host `mcmc-beast` (AMD Ryzen 9 5950X, 16 cores / 32 threads), idle before launch.
  Julia 1.12.6, 16 default + 1 interactive thread, `pinthreads(:cores)`, BLAS 1 thread.
  GC runtime defaults (16 mark threads, no heap-size hint).
- Worktree `/root/BF_goal_decomposition` at `af2c65ea`. `l08_workflow.jl` held the
  uncommitted TODO 002 gate relaxation (`32acd30f…` in the manifest). The benchmark
  does not call the changed code (`L08_THRESHOLDS`, `l08_assert_promotion`): it audits
  every chain with `audit_fold` and applies its own declared gates.
- Datastore snapshot `8e4aae64…`; incident registry `7571c875…`. Fold 1: 720 training
  fixtures, 20 OOS fixtures, 25 teams, 39 referees, 97 parameters per arm, matching
  the structural contract on a real chain.
- Raw outputs in [`sampling_budget_fold1/abc_20260910/`](sampling_budget_fold1/abc_20260910/):
  pre-declaration manifest, `chains.csv` (96 per-chain timings and internals),
  `fits.csv` (24 pooled-fit audits with every gate), `summary.csv`,
  `production_grid_m00_A_fold_audit.csv`, and the run log. `chains.jls` (all 96 raw
  chains, 57 MB) is git-ignored; copies are on the beast and archpc.

## Design (declared before sampling)

| Config | Chains | Warmup | Retained | δ | max_depth |
|---|---:|---:|---:|---:|---:|
| A (control, asserted identical to `L08_SAMPLER`) | 4 | 1,000 | 1,000 | 0.95 | 10 |
| B | 4 | 500 | 500 | 0.95 | 10 |
| C | 4 | 500 | 500 | 0.90 | 10 |

- All three configs share the diagonal metric (`AdvancedHMC.DiagEuclideanMetric`), uniform
  initialisation, `AutoReverseDiff(compile = true)`, and the production path
  `sample_fold → run_sampler(::QueuedNUTSConfig, chain_id)`.
- Arms `m00_recombined_control` and `m01_decomposed_baseline`. There are **four
  independent 4-chain fits per (arm, config)**: 96 chains in one flat 16-slot FIFO
  queue, interleaved rep-major in a seeded shuffle.
- Seeds are common across cells (`20260910 + 1000·rep + chain`), so initial values are
  paired across configs.
- **Primary metric:** min ESS per wall-second. min ESS is the minimum over all 97
  parameters of min(bulk, tail) for the pooled fit; wall-seconds is the slowest of its
  4 chains. **Secondary:** the same min ESS per core-second (sum of the 4 chains).
- **Hard gates, every replicate:** max R-hat ≤ 1.05, bulk and tail ESS ≥ 400, zero
  divergences, BFMI ≥ 0.30, depth-capped rate < 5%. The committed production gate (the
  same with ESS ≥ 200) is reported alongside.

Queue: 11:00:42–13:59:14 beast clock, 10,712 s (178.5 min) wall, zero failed chains.

## Per-fit results (4 replicates per cell)

| Arm | Cfg | Rep | Wall s | Core s | max R-hat | min bulk | min tail | ESS/wall-s | Div | Capped | min BFMI | Load | Strict gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| m00 | A | 1 | 2,962 | 11,266 | 1.0053 | 1,710 | 2,015 | 0.577 | 0 | 0 | 0.739 | 16.0 | pass |
| m00 | A | 2 | 3,000 | 11,365 | 1.0066 | 1,622 | 2,007 | 0.541 | 0 | 0 | 0.835 | 16.0 | pass |
| m00 | A | 3 | 2,965 | 11,233 | 1.0073 | 1,761 | 2,160 | 0.594 | **1** | 0 | 0.753 | 16.0 | **fail (1 div)** |
| m00 | A | 4 | 2,691 | 8,496 | 1.0057 | 2,082 | 2,137 | 0.774 | 0 | 0 | 0.747 | 14.2 | pass |
| m01 | A | 1 | 2,603 | 8,766 | 1.0063 | 1,984 | 1,881 | 0.723 | 0 | 0 | 0.749 | 16.0 | pass |
| m01 | A | 2 | 2,586 | 9,749 | 1.0062 | 2,041 | 2,253 | 0.789 | 0 | 0 | 0.759 | 16.0 | pass |
| m01 | A | 3 | 2,409 | 8,702 | 1.0044 | 1,858 | 1,469 | 0.610 | 0 | 0 | 0.755 | 16.0 | pass |
| m01 | A | 4 | 2,532 | 8,333 | 1.0068 | 1,598 | 2,060 | 0.631 | 0 | 0 | 0.726 | 14.3 | pass |
| m00 | B | 1 | 1,778 | 6,664 | 1.0087 | 941 | 890 | 0.500 | 0 | 0 | 0.789 | 16.0 | pass |
| m00 | B | 2 | 1,815 | 6,822 | 1.0085 | 841 | 1,027 | 0.464 | 0 | 0 | 0.687 | 16.0 | pass |
| m00 | B | 3 | 1,772 | 6,560 | 1.0077 | 1,000 | 889 | 0.502 | 0 | 0 | 0.786 | 16.0 | pass |
| m00 | B | 4 | 1,628 | 5,810 | 1.0129 | 1,005 | 849 | 0.522 | 0 | 0 | 0.721 | 15.2 | pass |
| m01 | B | 1 | 1,494 | 5,503 | 1.0082 | 752 | 583 | 0.390 | 0 | 0 | 0.783 | 16.0 | pass |
| m01 | B | 2 | 1,346 | 4,987 | 1.0097 | 737 | 615 | 0.457 | 0 | 0 | 0.667 | 16.0 | pass |
| m01 | B | 3 | 1,549 | 5,725 | 1.0102 | 789 | 935 | 0.509 | 0 | 0 | 0.758 | 16.0 | pass |
| m01 | B | 4 | 1,524 | 5,074 | 1.0080 | 986 | 1,084 | 0.647 | 0 | 0 | 0.733 | 15.0 | pass |
| m00 | C | 1 | 1,388 | 5,099 | 1.0075 | 721 | 557 | 0.401 | 0 | 0 | 0.748 | 16.0 | pass |
| m00 | C | 2 | 1,569 | 5,578 | 1.0099 | 869 | 897 | 0.554 | 0 | 0 | 0.836 | 16.0 | pass |
| m00 | C | 3 | 1,554 | 5,603 | 1.0078 | 757 | 924 | 0.487 | 0 | 0 | 0.732 | 16.0 | pass |
| m00 | C | 4 | 1,585 | 5,219 | 1.0097 | 758 | 1,030 | 0.478 | **1** | 0 | 0.781 | 14.8 | **fail (1 div)** |
| m01 | C | 1 | 1,310 | 4,880 | 1.0123 | 1,128 | 1,014 | 0.774 | 0 | 0 | 0.814 | 16.0 | pass |
| m01 | C | 2 | 1,385 | 4,991 | 1.0078 | 859 | 665 | 0.480 | 0 | 0 | 0.824 | 16.0 | pass |
| m01 | C | 3 | 1,330 | 5,014 | 1.0092 | 955 | 764 | 0.574 | 0 | 0 | 0.771 | 16.0 | pass |
| m01 | C | 4 | 1,488 | 4,928 | 1.0107 | 893 | 1,113 | 0.600 | 0 | 0 | 0.778 | 14.8 | pass |

"Load" is the mean number of chains running concurrently over the fit's chains. Rep 4
of every cell ran partly in the queue tail.

## Cell summary

Medians over replicates. Ratios compare against A for the same arm.

| Arm | Cfg | Strict pass | ESS ≥ 200 + 0 div | Wall s | Core-s × A | Worst R-hat | min ESS median (worst) | ESS / draw | ESS/wall-s | ESS/wall-s × A | Mean leapfrog | Step size | Divergences |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| m00 | A | 3/4 | 3/4 | 2,964 | 1.000 | 1.0073 | 1,735 (1,622) | 0.434 | **0.586** | 1.000 | 30.4 | 0.130–0.205 | 1 / 16,000 |
| m00 | B | 4/4 | 4/4 | 1,775 | 0.588 | 1.0129 | 869 (841) | 0.435 | 0.501 | 0.856 | 30.4 | 0.121–0.204 | 0 / 8,000 |
| m00 | C | 3/4 | 3/4 | 1,562 | 0.480 | 1.0099 | 758 (557) | 0.379 | 0.483 | 0.824 | 22.7 | 0.134–0.290 | 1 / 8,000 |
| m01 | A | 4/4 | 4/4 | 2,559 | 1.000 | 1.0068 | 1,739 (1,469) | 0.435 | **0.677** | 1.000 | 24.3 | 0.118–0.263 | 0 / 16,000 |
| m01 | B | 4/4 | 4/4 | 1,509 | 0.606 | 1.0102 | 702 (583) | 0.351 | 0.483 | 0.714 | 21.1 | 0.110–0.249 | 0 / 8,000 |
| m01 | C | 4/4 | 4/4 | 1,357 | 0.568 | 1.0123 | 828 (665) | 0.414 | 0.587 | 0.868 | 17.4 | 0.206–0.299 | 0 / 8,000 |

All 24 fits pass R-hat (≤ 1.013), BFMI (≥ 0.667) and depth: the maximum tree depth
reached was 7 and no transition hit the cap of 10. Medians restricted to full-load
replicates 1–3 lead to the same conclusions.

## Production-grid context: m00 under config A, all 40 folds

Read-only audit of the checkpoints left by the 2026-09-09 grid, which failed its gate
([TODO 002](../../../../todos/002_complete_experiment_08_goal_decomposition_production_grid.md)).

- Fold 1 min ESS is 2,003. The worst fold is **fold 6 at 680** (`sigma_regular_raw[1]`),
  **0.339×** fold 1. Eight of 40 folds fall below 1,000 (6, 38, 8, 3, 36, 4, …).
- Max R-hat 1.0101 (fold 21); one divergence (fold 3); max tree depth 8; zero capped.
- Mean leapfrog steps per transition: 25.4–34.1. Fold 1 has the highest value (34.1).

## Findings

1. **Half the budget costs 0.59–0.61× A, not 0.5×.** Full-load chain medians are
   A 2,818 s / B 1,626 s (m00) and A 2,225 s / B 1,364 s (m01). A linear fit through the
   two budget points implies a fixed per-chain cost of about 430–500 s, 27–37% of a B
   chain: tape compilation, step-size search, and the early warmup windows. This is an
   inference from two points, not a direct measurement.
2. **ESS scales with retained draws, so the primary metric gets worse.** ESS per draw
   is unchanged or lower in B/C (0.35–0.44 vs 0.43). min ESS per wall-second relative
   to A is B 0.86×/0.71× and C 0.82×/0.87× (m00/m01). A is the most efficient
   configuration per second in both arms.
3. **δ = 0.90 trades per-iteration cost for per-draw quality.** C takes larger steps
   (up to 0.30) with 17–23 leapfrog steps instead of 21–30, saving 6–18% of core time
   against B. ESS per draw falls for m00 (0.379 vs 0.435) and rises for m01 (0.414 vs
   0.351). Net ESS/s versus B is about equal for m00 and better for m01.
4. **On fold 1 alone, every cell clears ESS ≥ 400, including the worst replicate**
   (557 for C, 583 for B).
5. **At the projected worst fold, B and C fall below 400.** Scaling the fold-1 medians
   by the observed worst-fold ratio (0.34–0.39: 680 against fold 1's 2,003 in the grid,
   or against this benchmark's A median of 1,735):

   | Arm | Cfg | Fold-1 min ESS median (worst rep) | Projected worst-fold min ESS |
   |---|---|---:|---:|
   | m00 | A | 1,735 (1,622) | 590–680 (observed: 680) |
   | m00 | B | 869 (841) | 295–340 (285–330) |
   | m00 | C | 758 (557) | 257–297 (189–218) |
   | m01 | B | 702 (583) | 238–275 (198–229) |
   | m01 | C | 828 (665) | 281–325 (226–260) |

   The m01 row applies m00's fold-to-fold ratio because no 40-fold m01 fit exists.
6. **Zero divergences is not a reliable gate for any configuration.** Divergences
   occurred under A (m00 rep 3: 1 in 4,000) and under C (m00 rep 4: 1 in 4,000), and the
   grid saw one under A (fold 3, 1 in 160,000). The worst per-fit rate, 1 in 4,000
   (0.025%), is 4× below the 0.1% rule. With 1–2 events per cell the configs cannot be
   told apart on divergence rate.
7. **Tree depth never saturated.** The maximum was 7 here and 8 in the grid, against a cap of 10.
   The optional `max_depth = 8` follow-up is unwarranted: a cap of 8 would only
   truncate the deepest grid trajectories.
8. **Concurrency dominates per-chain speed.** Chains that ran in the queue tail with
   about 12 concurrent chains finished up to 2× faster than the full-load median
   (m00 A: 1,374 s vs 2,818 s; m01 A: 1,411 s vs 2,225 s). During the run the 16
   sampling threads used about 48% of wall time and the GC threads about 39% of process
   CPU (TODO [003](../../../../todos/003_benchmark_julia_gc_tuning_and_heap_size_hint_on_multi_core_sampling.md)).
   This is the largest efficiency lever observed. It is not caused by any A/B/C difference.

## Recommendation

- **Keep A (4 × 1,000 / 1,000, δ = 0.95) for the production grid.** It is the only
  config projected to keep worst-fold min ESS above 400 (observed 680 across 40 m00
  folds), and it has the best min ESS per wall-second in both arms.
- **B (500/500, δ 0.95) is unsafe under the ESS ≥ 400 gate.** It saves about 40% of
  core time (projected m00 grid sampling 9.4 h → about 5.5 h), but its projected
  worst-fold min ESS is 285–340 (m00) and 198–275 (m01). Under the committed ESS ≥ 200
  gate it is marginal for m01.
- **C (500/500, δ 0.90) is unsafe on both gates.** It is the cheapest option (0.48–0.57×
  A core time), but it has the lowest worst-replicate ESS (557 on fold 1, projected
  189–218 at the worst fold) and a divergence.
- **Untested intermediate.** If a saving is needed under the 400 gate, the arithmetic
  above points to about 750/750 at δ = 0.95: projected worst-fold ESS 0.75× A's 680,
  about 510; core time about 0.8× A. This is a hypothesis and needs its own run on the
  weakest fold (fold 6), not fold 1.
- **The divergence rule is a policy decision for TODO 002, not a budget question.** A
  zero-tolerance gate fails the control itself on 1 of 4 Fold 1 fits.
- **Pursue runtime/GC throughput (TODO 003) before cutting draws.** The measured
  concurrency penalty is larger than any saving available from the draw budget.
