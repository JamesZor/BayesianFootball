# Sampling performance — td_base

- **candidate**: td_base
- **git_sha**: 49eba057
- **fold**: 60 (largest)
- **probe**: 16 chains × (50 warm-up + 50 draws) on 16 threads; 2390 rows, 92 parameters

## Verdict

**review** — gc_share, efficiency_ratio.

## Metrics

| metric | value | review if |
|---|---|---|
| bytes per compiled gradient (tape audit) | 0 | > 1024 (hard check) |
| bare gradient latency (warmed min) | 0.135 ms |  |
| tape instructions | 255 |  |
| tape record + compile | 0.001845 s |  |
| wall time | 5.944 s |  |
| full-concurrency window | 5.145 s |  |
| leapfrogs (warm-up included) | 209970 |  |
| in-situ ms per leapfrog | 0.4246 |  |
| efficiency ratio | 3.145 | > 2.0 |
| bytes per leapfrog | 28300.0 |  |
| GC share (window, GC_Diff) | 31.6% | > 15.0% |
| GC share (whole run) | 29.9% |  |
| GC pauses / full sweeps (window) | 51 / 0 |  |
| GC allocated (window) | 4.941 GiB |  |
| time to safepoint (window, summed) | 0.1294 s |  |
| max GC pause / max time to safepoint (process) | 100.2 ms / 1846.0 ms |  |
| sampler-thread utilisation, mean (profile) | 100.0% | < 75.0% |
| sampler-thread utilisation, min | 100.0% |  |
| sampler samples in GC-stall frames | 28.5% |  |
| sampler work share = utilisation × (1 − GC-stall share) | 71.5% | (reported; spinning at a safepoint counts as awake) |
| sampler samples in JIT/LLVM frames | 0.6% |  |
| profiled sub-window | 5.033 s |  |
| /proc sampler-thread CPU ÷ (window × threads) (fallback) | 65.6% |  |
| /proc process CPU ÷ (window × threads) | 74.1% |  |
| /proc non-sampler CPU (cores) | 1.363 |  |

## Threads (statistical profiler)

| thread | pool | samples | utilisation |
|---|---|---|---|
| 1 | interactive | 354 | 0.0% |
| 2 | default | 354 | 100.0% |
| 3 | default | 354 | 100.0% |
| 4 | default | 354 | 100.0% |
| 5 | default | 354 | 100.0% |
| 6 | default | 354 | 100.0% |
| 7 | default | 354 | 100.0% |
| 8 | default | 354 | 100.0% |
| 9 | default | 354 | 100.0% |
| 10 | default | 354 | 100.0% |
| 11 | default | 354 | 100.0% |
| 12 | default | 354 | 100.0% |
| 13 | default | 354 | 100.0% |
| 14 | default | 354 | 100.0% |
| 15 | default | 354 | 100.0% |
| 16 | default | 354 | 100.0% |
| 17 | default | 354 | 100.0% |
| 18 | foreign: gc | 354 | 100.0% |
| 19 | foreign: gc | 354 | 100.0% |
| 20 | foreign: gc | 354 | 100.0% |
| 21 | foreign: gc | 354 | 100.0% |
| 22 | foreign: gc | 354 | 100.0% |
| 23 | foreign: gc | 354 | 100.0% |
| 24 | foreign: gc | 354 | 100.0% |
| 25 | foreign: gc | 354 | 100.0% |
| 26 | foreign: gc | 354 | 100.0% |
| 27 | foreign: gc | 354 | 100.0% |
| 28 | foreign: gc | 354 | 100.0% |
| 29 | foreign: gc | 354 | 100.0% |
| 30 | foreign: gc | 354 | 100.0% |
| 31 | foreign: gc | 354 | 100.0% |
| 32 | foreign: gc | 354 | 100.0% |

### Top frames by self samples (sampler threads, awake)

| frame | self | share |
|---|---|---|
| ` libc.so.6:-1` | 1601 | 28.3% |
| `_setindex! array.jl:991` | 513 | 9.1% |
| `* float.jl:497` | 470 | 8.3% |
| `+ float.jl:495` | 349 | 6.2% |
| `+ int.jl:87` | 288 | 5.1% |
| `diffresult_increment_deriv! propagation.jl:86` | 276 | 4.9% |
| `== promotion.jl:637` | 267 | 4.7% |
| `getindex essentials.jl:919` | 180 | 3.2% |
| `getindex essentials.jl:920` | 162 | 2.9% |
| `- float.jl:496` | 117 | 2.1% |
| `exp exp.jl:329` | 117 | 2.1% |
| ` :-1` | 99 | 1.7% |
| `reinterpret essentials.jl:733` | 90 | 1.6% |
| `< int.jl:519` | 78 | 1.4% |
| `getproperty Base_compiler.jl:54` | 72 | 1.3% |

## Allocating tape instructions (tape audit)

None — the compiled tape replays without allocating.

## Allocation outside the tape (Profile.Allocs)

Estimated bytes per gradient (`gradient`: `logdensity_and_gradient` through Turing's `LogDensityFunction`) or per leapfrog (`nuts_step`: NUTS transitions after setup). Tape allocation has no source frame here — see the tape audit.

| source | first non-Base frame | type | samples | est. bytes per unit |
|---|---|---|---|---|
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Memory{Float64}` | 8 | 1254.0 |
| gradient | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Vector{Float64}` | 5 | 640.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:290` | `ADTypes.AutoReverseDiff{true}` | 9 | 582.4 |
| gradient | `DiffResults: MutableDiffResult DiffResults.jl:23` | `DiffResults.MutableDiffResult{1, Float64, Tuple{Vector{Float` | 4 | 51.2 |
| nuts_step | `AdvancedHMC: ∂H∂r hamiltonian.jl:58` | `Memory{Float64}` | 1151 | 2957.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:237` | `Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{` | 871 | 1700.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:243` | `Memory{Float64}` | 885 | 1670.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:247` | `AdvancedHMC.DualValue{Float64, Vector{Float64}}` | 1051 | 1653.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:240` | `Float64` | 853 | 1520.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:242` | `Float64` | 301 | 1492.0 |
| nuts_step | `AdvancedHMC: build_tree trajectory.jl:668` | `AdvancedHMC.Hamiltonian{AdvancedHMC.DiagEuclideanMetric{Floa` | 442 | 1376.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:239` | `AdvancedHMC.Hamiltonian{AdvancedHMC.DiagEuclideanMetric{Floa` | 138 | 1289.0 |
| nuts_step | `AdvancedHMC: build_tree trajectory.jl:661` | `AdvancedHMC.MultinomialTS{Float64, AdvancedHMC.PhasePoint{Ve` | 706 | 1048.0 |
| nuts_step | `AdvancedHMC: ∂H∂θ hamiltonian.jl:47` | `Core.SimpleVector` | 892 | 1014.0 |

## Diagnosis

- **gc_share** (0.3162 vs 0.15): GC takes 31.6% of wall time with a zero-allocation tape: the garbage comes from the non-tape path — see the Profile.Allocs sites (DynamicPPL `LogDensityFunction` wrappers, the NUTS tree, chain storage). The profiler put 28.5% of awake sampler-thread samples in GC-stall frames.
- **efficiency_ratio** (3.145 vs 2.0): A leapfrog in situ costs 3.14× the bare compiled gradient. With high GC this is collection pauses; with low GC suspect per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead — see the Profile.Allocs sites and hot frames) or memory-bandwidth contention from a large tape across all threads — compare tape_instructions and AD guide §10.1.
