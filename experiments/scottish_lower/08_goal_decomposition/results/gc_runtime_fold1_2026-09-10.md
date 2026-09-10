# Experiment 08 Fold 1 GC runtime benchmark and sampler-AD root cause — 2026-09-10

TODO [003](../../../../todos/003_benchmark_julia_gc_tuning_and_heap_size_hint_on_multi_core_sampling.md).
Measured values only. Raw outputs are in [`sampling_budget_fold1/`](sampling_budget_fold1/):
one directory and log per run tag, the driver `run_gc_bench.sh`, the GC-D livelock
evidence, the gradient-path probe, and the AD equivalence CSV. The `chains.jls` files
are git-ignored and kept on the beast and archpc.

## Headline

- **The GC flags do not help.** `--heap-size-hint=48G` has no measurable effect (1.013×,
  inside the ±1.1% drift band). Adding `--gcthreads=2` makes the batch **18.5% slower**.
  `--gcthreads=16,1` **livelocked**.
- **The GC pressure comes from a sampler bug, not GC tuning.**
  `Samplers.run_sampler` passes `adtype = AutoReverseDiff(compile = true)` to
  `sample`. Turing 0.41.4 reads the AD backend only from the sampler object (`spl.adtype`),
  so the keyword is silently ignored and every production NUTS chain runs the `NUTS`
  default, **AutoForwardDiff**. Moving `adtype` into the `NUTS` constructor makes the same
  16-chain batch **65.6× faster** (1,654 s → 25.2 s) and cuts allocation from 11,873 GiB to
  22 GiB. The posterior is equivalent within Monte Carlo error.

## Setup

- Host `mcmc-beast` (Ryzen 9 5950X, 16 cores / 32 threads, 125.7 GiB), Julia 1.12.6,
  `-t 16`, `pinthreads(:cores)`, BLAS 1. Turing 0.41.4, DynamicPPL 0.38.10,
  AdvancedHMC 0.8.6, ReverseDiff 1.17.0.
- Workload per run: one Julia process running `r08_sampling_budget_benchmark.jl` on
  m00 × config B (4 chains × 500 warmup × 500 retained, δ 0.95) × 4 replicates = 16
  chains, all 16 slots busy. Seeds match TODO 001 run `abc_20260910`. Runs were
  sequential, each process owning the machine, driven by `run_gc_bench.sh TAG B abc_20260910 [flags]`.
- Telemetry covers the queue only: `Base.gc_num()` deltas; per-OS-thread CPU from
  `/proc/self/task`, split into the 17 Julia mutator threads and all others (GC); a 2 s
  RSS sampler plus `Sys.maxrss()`. GC thread count and heap hint are read back from the
  runtime.

## GC configurations (production AD path)

Speedup is measured against the mean of GC-A and GC-A′ (1,654.1 s).

| Run | Flags | Queue wall s | Speedup | Chain median s | GC time s (share of wall) | Pauses / full | Mark / sweep s | Max pause ms | Mutator util. | Non-mutator CPU s (threads) | Peak RSS GiB (maxrss) | Draws vs TODO 001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GC-A | default (16 mark, 0 sweep) | 1,645.0 | 1.006 | 1,613 | 862 (52.4%) | 69,367 / 4,787 | 373 / 489 | 94 | 0.481 | 7,708 (15) | 3.03 (3.06) | 16/16 bit-identical |
| GC-B | `--heap-size-hint=48G` | 1,633.5 | 1.013 | 1,597 | 845 (51.7%) | 69,284 / 4,989 | 385 / 460 | 91 | 0.486 | 7,718 (15) | 2.95 (2.98) | 16/16 bit-identical |
| GC-C | `--heap-size-hint=48G --gcthreads=2` | 2,028.6 | **0.815** | 1,977 | 1,231 (60.7%) | 53,186 / 3,478 | 866 / 365 | **312** | 0.410 | 993 (1) | 3.02 (3.29) | 16/16 bit-identical |
| GC-D | `--gcthreads=16,1` (exploratory) | **livelock** | — | 0 chains finished in 87 min | — | — | — | — | — | — | 2.59 (frozen) | — |
| GC-A′ | default (drift check) | 1,663.2 | 0.995 | 1,635 | 879 (52.8%) | 68,741 / 5,196 | 404 / 474 | 93 | 0.479 | 8,118 (15) | 2.98 (2.98) | 16/16 bit-identical |

Every run allocated exactly 11,873 GiB, about 7.2 GiB/s, or 742 GiB per 1,000-iteration
chain. The heap never grew past about 3 GiB under any flag. `--heap-size-hint` is a ceiling
that forces collection, not a floor that defers it (`julia --help`). TODO 001's
mixed-queue m00-B chains (median 1,615 s) agree with GC-A's 1,613 s.

**GC-D livelock** ([evidence](sampling_budget_fold1/gcD_gc16_sweep1_20260910_livelock_evidence.txt)).
Over a 60 s window, 15 GC mark threads and the mutator that triggered the collection
each burned 59.9 CPU-s in state `R`. The other 15 mutators waited in `futex_do_wait`
with zero CPU. RSS stayed at exactly 2,592,592 kB, and no chain completed in 87 minutes,
against 27 minutes for the whole default batch. SIGTERM was ignored for 15 s, so the
process was killed with SIGKILL. This is one observation, not reproduced. No matching
upstream report was found; the nearest are JuliaLang/julia#51044 (high-gcthreads
slowdown) and #42364 (GC hang waiting for threads).

The TODO 003 harness check on the tiny `T` config (50/50) gave the same pattern:
default 578 s against 746 s for `48G --gcthreads=2`, with 16/16 bit-identical draws.

## Root cause: the sampler runs ForwardDiff

`r08_gradient_path_probe.jl`, 1 thread, m00, Fold 1, same seed:

| Path | Bytes | ms |
|---|---:|---:|
| Experiment 08 compiled tape, per gradient | 2,688 | 0.201 |
| `logdensity_and_gradient`, DynamicPPL LDF with `AutoReverseDiff(compile=true)` | 5,472 | 0.201 |
| **`Turing.sample(m, NUTS(0, 0.95), 200; adtype = …)`**, as `run_sampler` calls it, per leapfrog | **18,551,650** | **3.403** |
| `Turing.sample(m, NUTS(0, 0.95; adtype = …), 200)`, per leapfrog | 32,952 | 0.233 |
| `AbstractMCMC.step` by hand, adtype in constructor, per leapfrog | 31,538 | — |
| bare `AdvancedHMC.transition` with Turing's Hamiltonian and kernel, per leapfrog | 27,431 | 0.222 |
| Turing's per-iteration `Transition` re-evaluation / `deepcopy(varinfo)` | 229,776 / 30,512 | 0.180 / 0.040 |

`Turing.NUTS(0, 0.95).adtype` is `ADTypes.AutoForwardDiff()`. In Turing 0.41.4,
`hmc.jl:200` and `hmc.jl:282` build the `LogDensityFunction` with `adtype=spl.adtype`, and
no `sample`/`step` method reads an `adtype` keyword. Both sites in
`src/samplers/engines/nuts.jl` are affected: `NUTSConfig` (lines 103/108) and
`QueuedNUTSConfig` (lines 128/131). For 97 parameters, ForwardDiff evaluates the model on
dual numbers in about 9 chunks per gradient. That is consistent with the 15× time and
18.5 MB per leapfrog, though the chunk count was inferred, not measured. The same
200-iteration run takes 42.7 s with 32% GC one way and 2.9 s with 0.0 s GC the other.

## The fix at 16 cores: GC-E

`L08_BENCH_AD_FIX=true` redefines only the queued `run_sampler` method, in the benchmark
process, with `adtype` in the `NUTS` constructor. `src/` is untouched. GC flags are default.

| Run | Queue wall s | Speedup vs GC-A/A′ | Chain median s | GC time s (share) | Allocated GiB | Mutator util. | Peak RSS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| GC-E | **25.2** | **65.6×** | 24.1 | 4.2 (16.8%) | 22.1 | 0.777 | 2.29 |

The gain is larger than the single-thread 14.6×. Removing 99.8% of the allocation also
removes the stop-the-world contention between 16 threads, so mutator utilisation rises
from 48% to 78%.

**Convergence and equivalence** (4 fits each; draws are not bit-identical, 0/16, because
the AD arithmetic differs):

| Run | max R-hat | min bulk ESS | min tail ESS | divergences | min BFMI | mean leapfrog |
|---|---:|---:|---:|---:|---:|---:|
| GC-A (ForwardDiff) | 1.0128 | 841 | 849 | 0 / 8,000 | 0.687 | 29.6–30.8 |
| GC-E (ReverseDiff) | 1.0140 | 841 | 789 | 1 / 8,000 | 0.765 | 25.4–31.0 |

`r08_ad_fix_equivalence.jl` pools each run's 16 chains. Across the 97 parameters,
z = Δmean / √(MCSE²₁ + MCSE²₂) has mean +0.06 and sd 0.90, with 2 values above |2|
(N(0,1) expects 4.4) and one above |3| (`raw_attack[14]`, 3.12; about a 16% chance
across 97 parameters). The ratio of posterior sds has median 0.992 (range 0.955–1.049).

## Fix applied and verified (same day)

`src/samplers/engines/nuts.jl` now builds the sampler through `Samplers.nuts_algorithm`
with `adtype = AutoReverseDiff(compile = true)` in the `NUTS` constructor, and the ignored
`sample` keyword is removed. `test/sampler_adtype_tests.jl` failed on the unfixed code:
the model was evaluated with `ForwardDiff.Dual`. After the fix it passes 18/18
([log](sampling_budget_fold1/sampler_adtype_tests_green.log)). The unmodified runner on
the fixed production path (`prodfix_default_20260910`) completed the 16-chain batch in
**24.8 s**, with 22.1 GiB allocated, and its draws are **16/16 bit-identical to GC-E**.

## Recommendations

1. **Do not adopt `--heap-size-hint=48G`, `--gcthreads=2` or `--gcthreads=N,1`** as runner
   standards. They are neutral, harmful and unsafe respectively. `AGENTS.md` is
   unchanged because the TODO's condition ("if positive") was not met.
2. **Fix `run_sampler`**: pass `adtype = AutoReverseDiff(compile = true)` to the `NUTS`
   constructor in both `NUTSConfig` and `QueuedNUTSConfig`, and drop the ignored
   `sample` keyword. It is a two-line change, but it changes every NUTS experiment's
   sampler, so it needs an owner's decision and its own task. It should include a
   regression test asserting the Hamiltonian's `LogDensityFunction` carries the
   intended adtype.
3. **Before resuming the TODO 002 production grid, decide the AD backend.** m00's 40
   checkpoints were sampled with ForwardDiff. Under the current code, m01–m03 would take
   about 30 h. With the fix, the fold-1 ratio suggests minutes to under an hour per arm.
   Larger folds must be measured, not assumed. Mixing backends across arms is
   statistically benign (equivalent posteriors above) but is a provenance difference to
   record.
4. **TODO 001's timings were all measured on the ForwardDiff path.** Its ESS-per-draw
   and worst-fold conclusions stand. Its cost ratios, especially the ~430–500 s implied
   fixed cost per chain, should be re-measured after the fix. At about 25 s per B batch,
   the draw budget stops being the grid's bottleneck.
