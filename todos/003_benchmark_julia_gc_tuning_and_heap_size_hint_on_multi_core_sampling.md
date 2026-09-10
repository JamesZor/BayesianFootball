# 003 — Benchmark Julia GC Tuning and Heap Size Hint on Multi-Core Sampling

| Field | Value |
|---|---|
| ID | 003 |
| Title | Benchmark Julia GC Tuning and Heap Size Hint on Multi-Core Sampling |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-10 |
| Updated | 2026-09-10 |
| Related Files / Commits / PRs | [Task 001](001_sampling_budget_benchmark.md); [r08_sampling_budget_benchmark.jl](../experiments/scottish_lower/08_goal_decomposition/r08_sampling_budget_benchmark.jl); [evidence](../experiments/scottish_lower/08_goal_decomposition/results/gc_runtime_fold1_2026-09-10.md); [r08_gradient_path_probe.jl](../experiments/scottish_lower/08_goal_decomposition/r08_gradient_path_probe.jl); [r08_ad_fix_equivalence.jl](../experiments/scottish_lower/08_goal_decomposition/r08_ad_fix_equivalence.jl); [src/samplers/engines/nuts.jl](../src/samplers/engines/nuts.jl) |

## Context & Problem Statement

During the 16-core NUTS sampling benchmark on `mcmc-beast` (PID 2621390), thread-level OS profiling (`ps -T -p 2621390`) revealed that:
1. The 16 sampling worker threads are only utilized at ~47.7% CPU capacity.
2. The 15 spawned GC threads consume **~38.5% of total CPU time** in frequent stop-the-world collections.
3. `mcmc-beast` has **125 GB of RAM** (116 GB completely available), yet Julia's conservative default heap thresholds keep memory constrained to ~2.5 GB by triggering collections constantly.

Because NUTS generates high volumes of short-lived trajectory tree objects (`AdvancedHMC.BinaryTree`, `PhasePoint`), tuning Julia's runtime flags (`--heap-size-hint` and `--gcthreads`) could dramatically reduce collection frequency and thread contention, unlocking an estimated 1.5× to 1.8× speedup.

## Acceptance Criteria

- [x] Execute an empirical benchmark on Fold 1 across three GC runtime configurations:
  - **GC-A (Default baseline)**: `julia -t 16`
  - **GC-B (Heap Hint)**: `julia -t 16 --heap-size-hint=48G`
  - **GC-C (Heap Hint + GC Thread Clamping)**: `julia -t 16 --heap-size-hint=48G --gcthreads=2`
- [x] Measure and record for each configuration:
  - Wallclock runtime per chain / batch
  - Total GC CPU time and pause time (via `Base.gc_num()` and OS thread metrics)
  - Peak resident memory (RSS)
  - Statistical convergence verification (confirm max R-hat, min ESS, and posterior draws are unaffected)
- [x] Calculate the empirical wallclock speedup factor vs the default baseline.
- [x] Synthesize findings, provide concrete recommendations, and update runner invocation standards in `AGENTS.md` if positive. The flags were not positive, so `AGENTS.md` is unchanged.

## Ideas & Candidate Solutions

- **Heap Expansion**: Setting `--heap-size-hint=48G` lets generational allocation pools grow before triggering major collections. Since NUTS objects die rapidly, minor generations will be reclaimed efficiently.
- **Thread Starvation Prevention**: `-t 16` automatically spawns 15 GC threads in Julia 1.10+. 16 workers + 15 GC threads = 31 active threads on 16 physical cores. Setting `--gcthreads=2` eliminates scheduler thrashing and preserves L1/L2/L3 CPU cache state.
- **Safety**: 48 GB is well within the 116 GB available RAM on `mcmc-beast`.
- **Pre-measurement notes (@claude, 2026-09-10, written before any GC run).**
  Julia 1.12.6 `--help` on the beast says `--gcthreads` defaults to N = compute
  threads (16) mark threads and 0 sweep threads. It says `--heap-size-hint` *forces*
  collection above the given value: a ceiling, not a floor that defers collection. At
  a live heap of about 2.5 GB, a 48G hint may therefore change little. Mark threads
  run while mutators are stopped, and the host has 32 logical CPUs, so
  `--gcthreads=2` could lengthen each pause rather than relieve contention. These are
  hypotheses; the runs decide.
- **Design (@claude).** Reuse `r08_sampling_budget_benchmark.jl` with GC telemetry
  added. Each GC configuration is its own Julia process running an identical
  homogeneous 16-chain batch: m00, config B (500/500, δ 0.95), 4 reps × 4 chains, all
  16 slots busy. Seeds match TODO 001 run `abc_20260910`. GC flags cannot change the
  arithmetic, so every chain should be **bit-identical** to its 001 counterpart and to
  the other GC runs. That checks "posterior draws unaffected" exactly, beyond R-hat/ESS.
  Order: GC-A, GC-B, GC-C, then GC-A again as a drift check. Measured per run: batch
  wall and per-chain elapsed; `Base.gc_num()` deltas (GC time, pause count, full
  sweeps, time-to-safepoint, allocated bytes); OS per-thread CPU split into Julia
  mutator threads vs all others (GC); peak RSS (2 s sampler and `Sys.maxrss`); and
  `Threads.ngcthreads()` and heap hint as read back from the runtime.

## Work Log & Progress

- [2026-09-10 @antigravity] Created task following OS thread profiling discovery during Task 001. Assigned to Claude as an immediate follow-up benchmark on Fold 1.
- [2026-09-10 @claude] Claimed after TODO 001 closed (`a79fddc7`), same session.
  Recorded the pre-measurement notes and design above before running anything.
- [2026-09-10 @claude] Added GC telemetry to `r08_sampling_budget_benchmark.jl`
  (`Base.gc_num()` deltas, per-OS-thread CPU split, RSS sampler, bit-for-bit reference
  comparison, and a harness-only `T` config of 50/50). Added the
  `results/sampling_budget_fold1/run_gc_bench.sh` driver (one process per GC
  configuration). Harness check on m00 × T × 16 chains:
  `gcT1_default` 578 s wall, GC 52% of wall (299 s), 26,191 pauses (1,738 full), peak
  RSS 2.7 GiB. `gcT2_hint48_gc2` 746 s (+29%), GC 459 s (mark 324 s vs 133 s), peak RSS
  3.0 GiB. **Chains 16/16 bit-identical** across the two runtimes, so the determinism
  premise holds. Both runs allocated 4,247 GiB. T is warmup-dominated and not
  evidence for production configs.
- [2026-09-10 @claude] Launched the config-B sequence in beast tmux
  `sampling_budget_bench` at 14:32 CEST: GC-A default, GC-B `--heap-size-hint=48G`,
  GC-C `48G --gcthreads=2`, **GC-D `--gcthreads=16,1`**, then GC-A′ default (drift).
  Every run is compared bit-for-bit with TODO 001's `abc_20260910`. GC-D goes beyond the
  three requested configurations and is labelled exploratory: the harness showed sweep
  at 56% of GC time, and `,1` enables the concurrent sweep thread.
- [2026-09-10 @claude] Config-B results: GC-A 1,645 s; GC-B (48G) 1,634 s; GC-C
  (48G, gc 2) 2,029 s; GC-A′ 1,663 s (drift 1.1%). Every run allocated 11,873 GiB and
  every run was 16/16 bit-identical to TODO 001. GC-D (`--gcthreads=16,1`)
  **livelocked**: 15 mark threads plus 1 mutator spinning at 100% CPU, 15 mutators in
  futex wait, RSS frozen, 0 of 16 chains after 87 min. Recorded
  `/proc` evidence, then killed it (SIGTERM ignored; SIGKILL).
- [2026-09-10 @claude] Root-caused the allocation with `r08_gradient_path_probe.jl`
  (1 thread). The compiled gradient costs 0.20 ms and about 5 KB, but `Turing.sample`
  as `run_sampler` calls it costs 3.40 ms and 18.5 MB per leapfrog. Turing 0.41.4 ignores
  the `adtype` keyword on `sample` and uses the `NUTS` default, AutoForwardDiff. With
  `adtype` in the constructor it costs 0.233 ms and 33 KB. Ran GC-E, a 16-chain batch
  with a benchmark-only, env-gated override of queued `run_sampler`
  (`L08_BENCH_AD_FIX`; `src/` untouched): 25.2 s, **65.6×**, 22 GiB allocated, GC 17% of
  wall. Posterior equivalence with `r08_ad_fix_equivalence.jl`: z sd 0.90, max |z| 3.12
  of 97, sd ratio 0.955–1.049. Closed as COMPLETED; the `src/` fix is left for an owner
  decision.

## Verification & Findings

**Measured on 2026-09-10.** Full tables are in
[`gc_runtime_fold1_2026-09-10.md`](../experiments/scottish_lower/08_goal_decomposition/results/gc_runtime_fold1_2026-09-10.md);
raw outputs are in `results/sampling_budget_fold1/gc*_20260910/` and the matching `.log`
files. Setup: mcmc-beast, Julia 1.12.6, `-t 16` pinned, BLAS 1. Workload: m00 × config B
(500/500, δ 0.95) × 16 chains, seeds shared with TODO 001. Each configuration ran as a
separate process with the machine to itself.

| Config | Flags | Batch wall s | Speedup vs default | Chain median s | GC time s (% of wall) | Pauses / full | Max pause ms | Non-mutator (GC) CPU share | Peak RSS GiB | Draws |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GC-A | default | 1,645.0 | 1.006 | 1,613 | 862 (52.4%) | 69,367 / 4,787 | 94 | 37.8% | 3.03 | 16/16 identical to 001 |
| GC-B | `--heap-size-hint=48G` | 1,633.5 | 1.013 | 1,597 | 845 (51.7%) | 69,284 / 4,989 | 91 | 37.8% | 2.95 | 16/16 identical |
| GC-C | `--heap-size-hint=48G --gcthreads=2` | 2,028.6 | **0.815** | 1,977 | 1,231 (60.7%) | 53,186 / 3,478 | **312** | 6.9% | 3.02 | 16/16 identical |
| GC-D | `--gcthreads=16,1` (exploratory) | **livelock**: 0/16 chains in 87 min | — | — | — | — | — | — | 2.59 (frozen) | — |
| GC-A′ | default (drift check) | 1,663.2 | 0.995 | 1,635 | 879 (52.8%) | 68,741 / 5,196 | 93 | 38.9% | 2.98 | 16/16 identical |
| **GC-E** | default GC + **AD fix** (benchmark-only override) | **25.2** | **65.6** | 24.1 | 4.2 (16.8%) | 399 / 5 | 88 | 5.8% | 2.29 | equivalent (see below) |

Speedups are against the mean of GC-A and GC-A′ (1,654.1 s). Run-to-run drift was 1.1%.

**Findings**

- `--heap-size-hint=48G` is **neutral**: 1.3% faster, inside the drift band, with the same
  pause count and the same ~3 GiB peak RSS. As `julia --help` states, the hint is a
  collection ceiling, not a floor that defers collection. The pre-measurement prediction held.
- `--gcthreads=2` is **harmful**: 18.5% slower. With fewer mark threads, mark time rose
  from 373 s to 866 s and the maximum pause from 94 ms to 312 ms, and mutators wait
  through each stop-the-world pause. The thread-starvation hypothesis is refuted: mark
  threads run while mutators are stopped, on a 32-logical-CPU host.
- `--gcthreads=16,1` (concurrent sweep) **livelocked** in one run. This was not
  reproduced; do not use it.
- The GC pressure has one cause: **the production sampler runs ForwardDiff.**
  `Samplers.run_sampler` passes `adtype = AutoReverseDiff(compile = true)` to `sample`,
  and Turing 0.41.4 silently ignores that keyword. The backend comes from `spl.adtype`,
  and `NUTS` defaults to `AutoForwardDiff()`. Measured on 1 thread: 3.40 ms and 18.5 MB
  per leapfrog as called, against 0.233 ms and 33 KB with `adtype` in the `NUTS`
  constructor. At 16 cores the batch is **65.6× faster** (1,654 s → 25.2 s), allocation
  drops from 11,873 GiB to 22 GiB, and mutator utilisation rises from 48% to 78%.
- The fix leaves the posterior unchanged within Monte Carlo error. GC-E vs GC-A: max R-hat
  1.0140 vs 1.0128, min bulk/tail ESS 841/789 vs 841/849, 1 vs 0 divergences in 8,000.
  Per-parameter mean differences have z sd 0.90 with max |z| 3.12 across 97 parameters,
  and sd ratios fall in 0.955–1.049.

**Recommendations**

- Adopt none of the three GC flags. `AGENTS.md` runner standards are unchanged.
- Move `adtype = AutoReverseDiff(compile = true)` into the `NUTS(...)` constructor at
  both sites in `src/samplers/engines/nuts.jl` (lines 103/108 and 128/131), with a
  regression test asserting the Hamiltonian's `LogDensityFunction` adtype. It changes
  every NUTS experiment's sampler, so it needs an owner decision and its own TODO.
- TODO 002 should decide the AD backend before resuming the grid. m00's 40 checkpoints
  are ForwardDiff. m01–m03 would take about 30 h as-is, and the fold-1 ratio suggests
  under an hour per arm with the fix. Large folds are unmeasured, and mixed backends
  must be recorded in provenance.
- TODO 001's cost ratios were measured on the ForwardDiff path and should be re-measured
  after the fix. Its ESS and worst-fold conclusions are unaffected.

Boundaries: one arm (m00), one fold, one budget (B), one host. Each GC setting has one
run, plus the drift repeat for the default. The livelock is a single unreproduced
observation. The AD-fix speedup on larger folds is unmeasured.
