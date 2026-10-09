# Sampling performance — grw_joint

- **experiment**: scottish_lower_qs_wave2_2426
- **candidate**: grw_joint
- **recipe_hash**: d373b3393d5079129fa7bed773b28e23cf2c5b67a2904b9ea189f9d37a55f8a6
- **git_sha**: ac30c4d7
- **fold**: 40 (the smoke's largest)
- **probe**: 16 chains × (50 warm-up + 50 draws) on 16 threads; 1060 rows, 976 parameters

## Verdict

**review** — gc_share, efficiency_ratio.

## Metrics

| metric | value | review if |
|---|---|---|
| bytes per compiled gradient (tape audit) | 0 | > 1024 (hard check) |
| bare gradient latency (warmed min) | 0.3124 ms |  |
| tape instructions | 7699 |  |
| tape record + compile | 0.004992 s |  |
| wall time | 66.57 s |  |
| full-concurrency window | 46.5 s |  |
| leapfrogs (warm-up included) | 714926 |  |
| in-situ ms per leapfrog | 1.364 |  |
| efficiency ratio | 4.366 | > 2.0 |
| bytes per leapfrog | 132000.0 |  |
| GC share (window, GC_Diff) | 30.6% | > 15.0% |
| GC share (whole run) | 29.7% |  |
| GC pauses / full sweeps (window) | 888 / 4 |  |
| GC allocated (window) | 59.93 GiB |  |
| time to safepoint (window, summed) | 3.547 s |  |
| max GC pause / max time to safepoint (process) | 114.2 ms / 2997.0 ms |  |
| sampler-thread utilisation, mean (profile) | 100.0% | < 75.0% |
| sampler-thread utilisation, min | 100.0% |  |
| sampler samples in GC-stall frames | 60.4% |  |
| sampler work share = utilisation × (1 − GC-stall share) | 39.6% | (reported; spinning at a safepoint counts as awake) |
| sampler samples in JIT/LLVM frames | 4.1% |  |
| profiled sub-window | 5.072 s |  |
| /proc sampler-thread CPU ÷ (window × threads) (fallback) | 62.7% |  |
| /proc process CPU ÷ (window × threads) | 72.9% |  |
| /proc non-sampler CPU (cores) | 1.624 |  |

## Threads (statistical profiler)

| thread | pool | samples | utilisation |
|---|---|---|---|
| 1 | interactive | 72 | 1.4% |
| 2 | default | 73 | 100.0% |
| 3 | default | 74 | 100.0% |
| 4 | default | 74 | 100.0% |
| 5 | default | 74 | 100.0% |
| 6 | default | 73 | 100.0% |
| 7 | default | 73 | 100.0% |
| 8 | default | 72 | 100.0% |
| 9 | default | 73 | 100.0% |
| 10 | default | 74 | 100.0% |
| 11 | default | 73 | 100.0% |
| 12 | default | 73 | 100.0% |
| 13 | default | 74 | 100.0% |
| 14 | default | 73 | 100.0% |
| 15 | default | 73 | 100.0% |
| 16 | default | 73 | 100.0% |
| 17 | default | 75 | 100.0% |
| 18 | foreign: gc | 74 | 100.0% |
| 19 | foreign: gc | 74 | 100.0% |
| 20 | foreign: gc | 73 | 100.0% |
| 21 | foreign: gc | 74 | 100.0% |
| 22 | foreign: gc | 75 | 100.0% |
| 23 | foreign: gc | 72 | 100.0% |
| 24 | foreign: gc | 73 | 100.0% |
| 25 | foreign: gc | 74 | 100.0% |
| 26 | foreign: gc | 73 | 100.0% |
| 27 | foreign: gc | 73 | 100.0% |
| 28 | foreign: gc | 75 | 100.0% |
| 29 | foreign: gc | 73 | 100.0% |
| 30 | foreign: gc | 75 | 100.0% |
| 31 | foreign: gc | 74 | 100.0% |
| 32 | foreign: gc | 75 | 100.0% |

### Top frames by self samples (sampler threads, awake)

| frame | self | share |
|---|---|---|
| ` libc.so.6:-1` | 722 | 61.5% |
| ` :-1` | 63 | 5.4% |
| `getproperty Base_compiler.jl:54` | 51 | 4.3% |
| `_setindex! array.jl:991` | 31 | 2.6% |
| `CallWrapper FunctionWrappers.jl:65` | 28 | 2.4% |
| `pull_deriv! tracked.jl:188` | 24 | 2.0% |
| `* float.jl:497` | 20 | 1.7% |
| `getindex essentials.jl:920` | 15 | 1.3% |
| `+ float.jl:495` | 15 | 1.3% |
| `reverse_exec! tape.jl:92` | 13 | 1.1% |
| `pull_value! tracked.jl:180` | 11 | 0.9% |
| `forward_exec! tape.jl:82` | 9 | 0.8% |
| `scalar_reverse_exec! scalars.jl:52` | 8 | 0.7% |
| `gc_sweep_sysimg staticdata.c:2347` | 8 | 0.7% |
| `ReverseExecutor tape.jl:93` | 7 | 0.6% |

## Allocating tape instructions (tape audit)

None — the compiled tape replays without allocating.

## Allocation outside the tape (Profile.Allocs)

Estimated bytes per gradient (`gradient`: `logdensity_and_gradient` through Turing's `LogDensityFunction`) or per leapfrog (`nuts_step`: NUTS transitions after setup). Tape allocation has no source frame here — see the tape audit.

| source | first non-Base frame | type | samples | est. bytes per unit |
|---|---|---|---|---|
| gradient | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Memory{Float64}` | 8 | 12540.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Profile.Allocs.BufferType` | 5 | 3168.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:290` | `DynamicPPL.LogDensityAt{DynamicPPL.Model{typeof(BayesianFoot` | 11 | 947.2 |
| gradient | `DiffResults: MutableDiffResult DiffResults.jl:23` | `DiffResults.MutableDiffResult{1, Float64, Tuple{Vector{Float` | 3 | 38.4 |
| nuts_step | `AdvancedHMC: ∂H∂r hamiltonian.jl:58` | `Vector{Float64}` | 5675 | 31200.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:240` | `Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{` | 3806 | 15060.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:243` | `Memory{Float64}` | 3674 | 14770.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:237` | `Vector{Float64}` | 3684 | 14700.0 |
| nuts_step | `AdvancedHMC: ∂H∂θ hamiltonian.jl:47` | `DataType` | 3272 | 8104.0 |
| nuts_step | `AdvancedHMC: neg_energy hamiltonian.jl:170` | `Profile.Allocs.BufferType` | 1439 | 8063.0 |
| nuts_step | `AdvancedHMC: combine trajectory.jl:467` | `Vector{Float64}` | 1420 | 7946.0 |
| nuts_step | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Memory{Float64}` | 1419 | 7816.0 |
| nuts_step | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Profile.Allocs.BufferType` | 1408 | 7552.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:247` | `AdvancedHMC.Hamiltonian{AdvancedHMC.DiagEuclideanMetric{Floa` | 3347 | 1751.0 |

## Diagnosis

- **gc_share** (0.3064 vs 0.15): GC takes 30.6% of wall time with a zero-allocation tape: the garbage comes from the non-tape path — see the Profile.Allocs sites (DynamicPPL `LogDensityFunction` wrappers, the NUTS tree, chain storage). The profiler put 60.4% of awake sampler-thread samples in GC-stall frames.
- **efficiency_ratio** (4.366 vs 2.0): A leapfrog in situ costs 4.37× the bare compiled gradient. With high GC this is collection pauses; with low GC suspect per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead — see the Profile.Allocs sites and hot frames) or memory-bandwidth contention from a large tape across all threads — compare tape_instructions and AD guide §10.1.
