# Sampling performance — td_base

- **candidate**: td_base
- **git_sha**: 8effca72
- **fold**: 60 (largest)
- **probe**: 16 chains × (50 warm-up + 50 draws) on 16 threads; 2390 rows, 92 parameters

## Verdict

**review** — tape_allocation, gc_share, efficiency_ratio.

## Metrics

| metric | value | review if |
|---|---|---|
| bytes per compiled gradient (tape audit) | 422992 | > 1024 (hard check) |
| bare gradient latency (warmed min) | 0.1759 ms |  |
| tape instructions | 248 |  |
| tape record + compile | 0.002453 s |  |
| wall time | 18.91 s |  |
| full-concurrency window | 17.44 s |  |
| leapfrogs (warm-up included) | 211059 |  |
| in-situ ms per leapfrog | 1.388 |  |
| efficiency ratio | 7.891 | > 2.0 |
| bytes per leapfrog | 457500.0 |  |
| GC share (window, GC_Diff) | 44.5% | > 15.0% |
| GC share (whole run) | 43.9% |  |
| GC pauses / full sweeps (window) | 1018 / 1 |  |
| GC allocated (window) | 82.74 GiB |  |
| time to safepoint (window, summed) | 0.5907 s |  |
| max GC pause / max time to safepoint (process) | 96.48 ms / 19.46 ms |  |
| sampler-thread utilisation, mean (profile) | 100.0% | < 75.0% |
| sampler-thread utilisation, min | 100.0% |  |
| sampler samples in GC-stall frames | 59.4% |  |
| sampler work share = utilisation × (1 − GC-stall share) | 40.6% | (reported; spinning at a safepoint counts as awake) |
| sampler samples in JIT/LLVM frames | 2.2% |  |
| profiled sub-window | 5.041 s |  |
| /proc sampler-thread CPU ÷ (window × threads) (fallback) | 53.5% |  |
| /proc process CPU ÷ (window × threads) | 60.7% |  |
| /proc non-sampler CPU (cores) | 1.152 |  |

## Threads (statistical profiler)

| thread | pool | samples | utilisation |
|---|---|---|---|
| 1 | interactive | 243 | 0.0% |
| 2 | default | 243 | 100.0% |
| 3 | default | 243 | 100.0% |
| 4 | default | 243 | 100.0% |
| 5 | default | 242 | 100.0% |
| 6 | default | 243 | 100.0% |
| 7 | default | 243 | 100.0% |
| 8 | default | 243 | 100.0% |
| 9 | default | 243 | 100.0% |
| 10 | default | 243 | 100.0% |
| 11 | default | 243 | 100.0% |
| 12 | default | 243 | 100.0% |
| 13 | default | 243 | 100.0% |
| 14 | default | 243 | 100.0% |
| 15 | default | 243 | 100.0% |
| 16 | default | 243 | 100.0% |
| 17 | default | 243 | 100.0% |
| 18 | foreign: gc | 243 | 100.0% |
| 19 | foreign: gc | 243 | 100.0% |
| 20 | foreign: gc | 243 | 100.0% |
| 21 | foreign: gc | 243 | 100.0% |
| 22 | foreign: gc | 243 | 100.0% |
| 23 | foreign: gc | 243 | 100.0% |
| 24 | foreign: gc | 243 | 100.0% |
| 25 | foreign: gc | 243 | 100.0% |
| 26 | foreign: gc | 243 | 100.0% |
| 27 | foreign: gc | 243 | 100.0% |
| 28 | foreign: gc | 243 | 100.0% |
| 29 | foreign: gc | 243 | 100.0% |
| 30 | foreign: gc | 243 | 100.0% |
| 31 | foreign: gc | 243 | 100.0% |
| 32 | foreign: gc | 243 | 100.0% |

### Top frames by self samples (sampler threads, awake)

| frame | self | share |
|---|---|---|
| ` libc.so.6:-1` | 2242 | 57.7% |
| `_setindex! array.jl:991` | 260 | 6.7% |
| `getindex essentials.jl:920` | 146 | 3.8% |
| `+ float.jl:495` | 103 | 2.6% |
| `* float.jl:497` | 102 | 2.6% |
| `+ int.jl:87` | 101 | 2.6% |
| `== promotion.jl:637` | 81 | 2.1% |
| `diffresult_increment_deriv! propagation.jl:86` | 66 | 1.7% |
| `getindex essentials.jl:919` | 58 | 1.5% |
| `- float.jl:496` | 52 | 1.3% |
| ` :-1` | 43 | 1.1% |
| `macro expansion simdloop.jl:75` | 43 | 1.1% |
| `getproperty Base_compiler.jl:54` | 42 | 1.1% |
| `reinterpret essentials.jl:733` | 39 | 1.0% |
| `gc_sweep_pool gc-stock.c:1421` | 31 | 0.8% |

## Allocating tape instructions (tape audit)

| # | bytes per replay | instruction (T = tracked array, TReal = tracked scalar, Real = plain scalar) |
|---|---|---|
| 15 | 48 | `fill(TReal, Real) -> T3` |
| 26 | 368 | `fill(TReal, Real) -> T43` |
| 91 | 57624 | `tracker_∇broadcast[clamp](T2390, Real, Real) -> T2390` |
| 93 | 57624 | `tracker_∇broadcast[clamp](T2390, Real, Real) -> T2390` |
| 140 | 153664 | `tracker_∇broadcast[fused](TReal, Arr2390, TReal, Arr2390, T2390, TReal, T2390, TReal) -> T2390` |
| 142 | 153664 | `tracker_∇broadcast[fused](TReal, Arr2390, TReal, Arr2390, T2390, TReal, T2390, TReal) -> T2390` |

## Allocation outside the tape (Profile.Allocs)

Estimated bytes per gradient (`gradient`: `logdensity_and_gradient` through Turing's `LogDensityFunction`) or per leapfrog (`nuts_step`: NUTS transitions after setup). Tape allocation has no source frame here — see the tape audit.

| source | first non-Base frame | type | samples | est. bytes per unit |
|---|---|---|---|---|
| gradient | `ReverseDiff: macro expansion broadcast.jl:-1` | `Vector{Float64}` | 156 | 429400.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Memory{Float64}` | 6 | 940.8 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:290` | `Tuple{Float64, Vector{Float64}}` | 12 | 838.4 |
| gradient | `ReverseDiff: #fill#277#23 arrays.jl:12` | `Memory{Float64}` | 6 | 608.0 |
| gradient | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Vector{Float64}` | 4 | 339.2 |
| gradient | `DiffResults: MutableDiffResult DiffResults.jl:23` | `DiffResults.MutableDiffResult{1, Float64, Tuple{Vector{Float` | 3 | 38.4 |
| nuts_step | `ReverseDiff: macro expansion broadcast.jl:-1` | `Vector{Float64}` | 9498 | 421700.0 |
| nuts_step | `AdvancedHMC: ∂H∂r hamiltonian.jl:58` | `Vector{Float64}` | 1154 | 3012.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:243` | `Vector{Float64}` | 878 | 1606.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:237` | `Float64` | 847 | 1600.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:240` | `Vector{Float64}` | 817 | 1579.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:239` | `AdvancedHMC.Hamiltonian{AdvancedHMC.DiagEuclideanMetric{Floa` | 161 | 1503.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:247` | `AdvancedHMC.Hamiltonian{AdvancedHMC.DiagEuclideanMetric{Floa` | 1035 | 1449.0 |
| nuts_step | `AdvancedHMC: build_tree trajectory.jl:668` | `AdvancedHMC.GeneralisedNoUTurn{Float64}` | 407 | 1333.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:242` | `Float64` | 286 | 1278.0 |
| nuts_step | `AdvancedHMC: build_tree trajectory.jl:661` | `AdvancedHMC.MultinomialTS{Float64, AdvancedHMC.PhasePoint{Ve` | 777 | 1110.0 |

## Diagnosis

- **tape_allocation** (423000.0 vs 1024.0): The compiled tape allocates 422992 B per gradient. A tracked scalar (or a plain Real) inside a fused broadcast sends ReverseDiff to tracker_∇broadcast: lift it with tape_scalar or reduce first and multiply after (sum(x .* w) * ν); clamp bounds go in as Refs. See the allocating-instruction table and docs/turing_ad_performance_guide.md §10.5.
- **gc_share** (0.4452 vs 0.15): GC takes 44.5% of wall time and the tape allocates: fix the allocating instructions first (AD guide §10.5). The profiler put 59.4% of awake sampler-thread samples in GC-stall frames.
- **efficiency_ratio** (7.891 vs 2.0): A leapfrog in situ costs 7.89× the bare compiled gradient. With high GC this is collection pauses; with low GC suspect per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead — see the Profile.Allocs sites and hot frames) or memory-bandwidth contention from a large tape across all threads — compare tape_instructions and AD guide §10.1.
