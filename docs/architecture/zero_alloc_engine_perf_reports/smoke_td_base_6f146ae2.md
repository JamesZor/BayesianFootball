# Sampling performance — td_base

- **experiment**: scottish_pyramid_tier_transition_w2
- **candidate**: td_base
- **recipe_hash**: 84a3541715dc09d9b2df1c46eaddbdff3c01416d5bbc3e1e71d2d8147a572629
- **git_sha**: 6f146ae2
- **fold**: 60 (the smoke's largest)
- **probe**: 16 chains × (50 warm-up + 50 draws) on 16 threads; 2390 rows, 92 parameters

## Verdict

**review** — gc_share, efficiency_ratio.

## Metrics

| metric | value | review if |
|---|---|---|
| bytes per compiled gradient (tape audit) | 0 | > 1024 (hard check) |
| bare gradient latency (warmed min) | 0.1387 ms |  |
| tape instructions | 255 |  |
| tape record + compile | 0.001581 s |  |
| wall time | 5.376 s |  |
| full-concurrency window | 4.74 s |  |
| leapfrogs (warm-up included) | 209970 |  |
| in-situ ms per leapfrog | 0.3829 |  |
| efficiency ratio | 2.762 | > 2.0 |
| bytes per leapfrog | 28350.0 |  |
| GC share (window, GC_Diff) | 32.7% | > 15.0% |
| GC share (whole run) | 30.8% |  |
| GC pauses / full sweeps (window) | 43 / 0 |  |
| GC allocated (window) | 5.031 GiB |  |
| time to safepoint (window, summed) | 0.1025 s |  |
| max GC pause / max time to safepoint (process) | 112.9 ms / 19.12 ms |  |
| sampler-thread utilisation, mean (profile) | 99.9% | < 75.0% |
| sampler-thread utilisation, min | 98.8% |  |
| sampler samples in GC-stall frames | 41.8% |  |
| sampler samples in JIT/LLVM frames | 0.9% |  |
| profiled sub-window | 4.788 s |  |
| /proc sampler-thread CPU ÷ (window × threads) (fallback) | 64.7% |  |
| /proc process CPU ÷ (window × threads) | 74.8% |  |
| /proc non-sampler CPU (cores) | 1.61 |  |

## Threads (statistical profiler)

| thread | pool | samples | utilisation |
|---|---|---|---|
| 1 | interactive | 333 | 0.0% |
| 2 | default | 332 | 100.0% |
| 3 | default | 333 | 100.0% |
| 4 | default | 332 | 100.0% |
| 5 | default | 332 | 100.0% |
| 6 | default | 333 | 100.0% |
| 7 | default | 333 | 100.0% |
| 8 | default | 333 | 100.0% |
| 9 | default | 333 | 100.0% |
| 10 | default | 333 | 100.0% |
| 11 | default | 333 | 100.0% |
| 12 | default | 332 | 100.0% |
| 13 | default | 332 | 100.0% |
| 14 | default | 333 | 100.0% |
| 15 | default | 332 | 100.0% |
| 16 | default | 333 | 98.8% |
| 17 | default | 332 | 99.4% |
| 18 | foreign: gc | 332 | 100.0% |
| 19 | foreign: gc | 332 | 100.0% |
| 20 | foreign: gc | 332 | 100.0% |
| 21 | foreign: gc | 333 | 100.0% |
| 22 | foreign: gc | 333 | 100.0% |
| 23 | foreign: gc | 332 | 100.0% |
| 24 | foreign: gc | 333 | 100.0% |
| 25 | foreign: gc | 333 | 100.0% |
| 26 | foreign: gc | 332 | 100.0% |
| 27 | foreign: gc | 332 | 100.0% |
| 28 | foreign: gc | 333 | 100.0% |
| 29 | foreign: gc | 333 | 100.0% |
| 30 | foreign: gc | 333 | 100.0% |
| 31 | foreign: gc | 333 | 100.0% |
| 32 | foreign: gc | 332 | 100.0% |

### Top frames by self samples (sampler threads, awake)

| frame | self | share |
|---|---|---|
| ` libc.so.6:-1` | 2149 | 40.4% |
| `_setindex! array.jl:991` | 398 | 7.5% |
| `* float.jl:497` | 345 | 6.5% |
| `+ float.jl:495` | 282 | 5.3% |
| `== promotion.jl:637` | 244 | 4.6% |
| `+ int.jl:87` | 205 | 3.9% |
| `diffresult_increment_deriv! propagation.jl:86` | 199 | 3.7% |
| `getindex essentials.jl:919` | 129 | 2.4% |
| `__madvise libc.so.6:-1` | 98 | 1.8% |
| `getindex essentials.jl:920` | 93 | 1.7% |
| `exp exp.jl:329` | 93 | 1.7% |
| ` :-1` | 84 | 1.6% |
| `- float.jl:496` | 80 | 1.5% |
| `reinterpret essentials.jl:733` | 71 | 1.3% |
| `getproperty Base_compiler.jl:54` | 62 | 1.2% |

## Allocating tape instructions (tape audit)

None — the compiled tape replays without allocating.

## Allocation outside the tape (Profile.Allocs)

Estimated bytes per gradient (`gradient`: `logdensity_and_gradient` through Turing's `LogDensityFunction`) or per leapfrog (`nuts_step`: NUTS transitions after setup). Tape allocation has no source frame here — see the tape audit.

| source | first non-Base frame | type | samples | est. bytes per unit |
|---|---|---|---|---|
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Vector{Float64}` | 4 | 915.2 |
| gradient | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Vector{Float64}` | 6 | 652.8 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:290` | `DynamicPPL.LogDensityAt{DynamicPPL.Model{typeof(BayesianFoot` | 8 | 556.8 |
| gradient | `DiffResults: MutableDiffResult DiffResults.jl:23` | `DiffResults.MutableDiffResult{1, Float64, Tuple{Vector{Float` | 4 | 51.2 |
| nuts_step | `AdvancedHMC: ∂H∂r hamiltonian.jl:58` | `Vector{Float64}` | 1134 | 3092.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:243` | `Memory{Float64}` | 889 | 1682.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:240` | `Vector{Float64}` | 892 | 1602.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:247` | `Core.SimpleVector` | 1006 | 1561.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:237` | `Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{` | 882 | 1552.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:242` | `AdvancedHMC.Hamiltonian{AdvancedHMC.DiagEuclideanMetric{Floa` | 326 | 1531.0 |
| nuts_step | `AdvancedHMC: build_tree trajectory.jl:668` | `AdvancedHMC.Termination` | 404 | 1278.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:239` | `AdvancedHMC.Hamiltonian{AdvancedHMC.DiagEuclideanMetric{Floa` | 136 | 1270.0 |
| nuts_step | `AdvancedHMC: build_tree trajectory.jl:661` | `AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS, AdvancedHM` | 737 | 1197.0 |
| nuts_step | `AdvancedHMC: ∂H∂θ hamiltonian.jl:47` | `DataType` | 861 | 938.7 |

## Diagnosis

- **gc_share** (0.3266 vs 0.15): GC takes 32.7% of wall time with a zero-allocation tape: the garbage comes from the non-tape path — see the Profile.Allocs sites (DynamicPPL `LogDensityFunction` wrappers, the NUTS tree, chain storage). The profiler put 41.8% of awake sampler-thread samples in GC-stall frames.
- **efficiency_ratio** (2.762 vs 2.0): A leapfrog in situ costs 2.76× the bare compiled gradient. With high GC this is collection pauses; with low GC suspect per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead — see the Profile.Allocs sites and hot frames) or memory-bandwidth contention from a large tape across all threads — compare tape_instructions and AD guide §10.1.
