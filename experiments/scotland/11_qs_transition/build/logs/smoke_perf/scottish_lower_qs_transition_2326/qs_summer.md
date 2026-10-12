# Sampling performance — qs_summer

- **experiment**: scottish_lower_qs_transition_2326
- **candidate**: qs_summer
- **recipe_hash**: bd0dc0ff0b9bad9d81185c173cc383baf9bc23dd0dbcdeb23840b79e87f81f75
- **git_sha**: 19d3ad4d-dirty
- **fold**: 60 (the smoke's largest)
- **probe**: 16 chains × (50 warm-up + 50 draws) on 16 threads; 1060 rows, 1021 parameters

## Verdict

**review** — gc_share, efficiency_ratio.

## Metrics

| metric | value | review if |
|---|---|---|
| bytes per compiled gradient (tape audit) | 0 | > 1024 (hard check) |
| bare gradient latency (warmed min) | 0.2973 ms |  |
| tape instructions | 8067 |  |
| tape record + compile | 0.00534 s |  |
| wall time | 105.5 s |  |
| full-concurrency window | 90.05 s |  |
| leapfrogs (warm-up included) | 1309112 |  |
| in-situ ms per leapfrog | 1.204 |  |
| efficiency ratio | 4.051 | > 2.0 |
| bytes per leapfrog | 135600.0 |  |
| GC share (window, GC_Diff) | 32.2% | > 15.0% |
| GC share (whole run) | 30.5% |  |
| GC pauses / full sweeps (window) | 1498 / 4 |  |
| GC allocated (window) | 144.3 GiB |  |
| time to safepoint (window, summed) | 1.635 s |  |
| max GC pause / max time to safepoint (process) | 125.3 ms / 981.8 ms |  |
| sampler-thread utilisation, mean (profile) | 100.0% | < 75.0% |
| sampler-thread utilisation, min | 100.0% |  |
| sampler samples in GC-stall frames | 35.2% |  |
| sampler work share = utilisation × (1 − GC-stall share) | 64.8% | (reported; spinning at a safepoint counts as awake) |
| sampler samples in JIT/LLVM frames | 1.8% |  |
| profiled sub-window | 5.02 s |  |
| /proc sampler-thread CPU ÷ (window × threads) (fallback) | 67.1% |  |
| /proc process CPU ÷ (window × threads) | 76.2% |  |
| /proc non-sampler CPU (cores) | 1.454 |  |

## Threads (statistical profiler)

| thread | pool | samples | utilisation |
|---|---|---|---|
| 1 | interactive | 109 | 0.9% |
| 2 | default | 110 | 100.0% |
| 3 | default | 109 | 100.0% |
| 4 | default | 109 | 100.0% |
| 5 | default | 110 | 100.0% |
| 6 | default | 109 | 100.0% |
| 7 | default | 110 | 100.0% |
| 8 | default | 110 | 100.0% |
| 9 | default | 110 | 100.0% |
| 10 | default | 110 | 100.0% |
| 11 | default | 110 | 100.0% |
| 12 | default | 109 | 100.0% |
| 13 | default | 110 | 100.0% |
| 14 | default | 109 | 100.0% |
| 15 | default | 110 | 100.0% |
| 16 | default | 110 | 100.0% |
| 17 | default | 110 | 100.0% |
| 18 | foreign: gc | 110 | 100.0% |
| 19 | foreign: gc | 110 | 100.0% |
| 20 | foreign: gc | 110 | 100.0% |
| 21 | foreign: gc | 110 | 100.0% |
| 22 | foreign: gc | 109 | 100.0% |
| 23 | foreign: gc | 109 | 100.0% |
| 24 | foreign: gc | 109 | 100.0% |
| 25 | foreign: gc | 109 | 100.0% |
| 26 | foreign: gc | 110 | 100.0% |
| 27 | foreign: gc | 109 | 100.0% |
| 28 | foreign: gc | 110 | 100.0% |
| 29 | foreign: gc | 109 | 100.0% |
| 30 | foreign: gc | 110 | 100.0% |
| 31 | foreign: gc | 110 | 100.0% |
| 32 | foreign: gc | 108 | 100.0% |

### Top frames by self samples (sampler threads, awake)

| frame | self | share |
|---|---|---|
| ` libc.so.6:-1` | 645 | 36.8% |
| ` :-1` | 163 | 9.3% |
| `getproperty Base_compiler.jl:54` | 147 | 8.4% |
| `CallWrapper FunctionWrappers.jl:65` | 108 | 6.2% |
| `pull_deriv! tracked.jl:188` | 59 | 3.4% |
| `_setindex! array.jl:991` | 49 | 2.8% |
| `setproperty! Base_compiler.jl:58` | 43 | 2.5% |
| `getindex essentials.jl:920` | 34 | 1.9% |
| `forward_exec! tape.jl:82` | 32 | 1.8% |
| `pull_value! tracked.jl:180` | 27 | 1.5% |
| `+ float.jl:495` | 26 | 1.5% |
| `reverse_exec! tape.jl:92` | 23 | 1.3% |
| `* float.jl:497` | 22 | 1.3% |
| `ReverseExecutor tape.jl:93` | 21 | 1.2% |
| `ForwardExecutor tape.jl:76` | 18 | 1.0% |

## Allocating tape instructions (tape audit)

None — the compiled tape replays without allocating.

## Allocation outside the tape (Profile.Allocs)

Estimated bytes per gradient (`gradient`: `logdensity_and_gradient` through Turing's `LogDensityFunction`) or per leapfrog (`nuts_step`: NUTS transitions after setup). Tape allocation has no source frame here — see the tape audit.

| source | first non-Base frame | type | samples | est. bytes per unit |
|---|---|---|---|---|
| gradient | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Vector{Float64}` | 8 | 3350.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Memory{Float64}` | 9 | 3347.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:290` | `DynamicPPL.LogDensityAt{DynamicPPL.Model{typeof(BayesianFoot` | 10 | 1133.0 |
| gradient | `DiffResults: MutableDiffResult DiffResults.jl:23` | `DiffResults.MutableDiffResult{1, Float64, Tuple{Vector{Float` | 2 | 25.6 |
| nuts_step | `AdvancedHMC: ∂H∂r hamiltonian.jl:58` | `Memory{Float64}` | 7660 | 32160.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:237` | `Vector{Float64}` | 6235 | 16820.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:240` | `Vector{Float64}` | 5192 | 16520.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:243` | `Float64` | 5172 | 16210.0 |
| nuts_step | `AdvancedHMC: ∂H∂θ hamiltonian.jl:47` | `AdvancedHMC.DualValue{Float64, Vector{Float64}}` | 4542 | 8796.0 |
| nuts_step | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Profile.Allocs.BufferType` | 2006 | 8680.0 |
| nuts_step | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Memory{Float64}` | 1942 | 8576.0 |
| nuts_step | `AdvancedHMC: combine trajectory.jl:467` | `Memory{Float64}` | 2022 | 8542.0 |
| nuts_step | `AdvancedHMC: neg_energy hamiltonian.jl:170` | `Profile.Allocs.BufferType` | 1934 | 8082.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:247` | `AdvancedHMC.DualValue{Float64, Vector{Float64}}` | 4486 | 1560.0 |

## Diagnosis

- **gc_share** (0.3223 vs 0.15): GC takes 32.2% of wall time with a zero-allocation tape: the garbage comes from the non-tape path — see the Profile.Allocs sites (DynamicPPL `LogDensityFunction` wrappers, the NUTS tree, chain storage). The profiler put 35.2% of awake sampler-thread samples in GC-stall frames.
- **efficiency_ratio** (4.051 vs 2.0): A leapfrog in situ costs 4.05× the bare compiled gradient. With high GC this is collection pauses; with low GC suspect per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead — see the Profile.Allocs sites and hot frames) or memory-bandwidth contention from a large tape across all threads — compare tape_instructions and AD guide §10.1.
