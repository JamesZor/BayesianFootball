# Sampling performance — grw_marketobs

- **experiment**: scottish_lower_qs_wave2_2426
- **candidate**: grw_marketobs
- **recipe_hash**: a599b0538367f7c3d634c975fa1da32cb8fcb3cb0f59977775fea69126f1c787
- **git_sha**: ac30c4d7
- **fold**: 40 (the smoke's largest)
- **probe**: 16 chains × (50 warm-up + 50 draws) on 16 threads; 1060 rows, 975 parameters

## Verdict

**review** — gc_share, efficiency_ratio.

## Metrics

| metric | value | review if |
|---|---|---|
| bytes per compiled gradient (tape audit) | 0 | > 1024 (hard check) |
| bare gradient latency (warmed min) | 0.2753 ms |  |
| tape instructions | 7670 |  |
| tape record + compile | 0.004909 s |  |
| wall time | 109.2 s |  |
| full-concurrency window | 101.2 s |  |
| leapfrogs (warm-up included) | 1288383 |  |
| in-situ ms per leapfrog | 1.305 |  |
| efficiency ratio | 4.741 | > 2.0 |
| bytes per leapfrog | 129800.0 |  |
| GC share (window, GC_Diff) | 36.2% | > 15.0% |
| GC share (whole run) | 35.0% |  |
| GC pauses / full sweeps (window) | 2087 / 9 |  |
| GC allocated (window) | 144.8 GiB |  |
| time to safepoint (window, summed) | 3.101 s |  |
| max GC pause / max time to safepoint (process) | 119.3 ms / 1992.0 ms |  |
| sampler-thread utilisation, mean (profile) | 100.0% | < 75.0% |
| sampler-thread utilisation, min | 100.0% |  |
| sampler samples in GC-stall frames | 56.4% |  |
| sampler work share = utilisation × (1 − GC-stall share) | 43.6% | (reported; spinning at a safepoint counts as awake) |
| sampler samples in JIT/LLVM frames | 2.3% |  |
| profiled sub-window | 5.024 s |  |
| /proc sampler-thread CPU ÷ (window × threads) (fallback) | 62.1% |  |
| /proc process CPU ÷ (window × threads) | 73.1% |  |
| /proc non-sampler CPU (cores) | 1.766 |  |

## Threads (statistical profiler)

| thread | pool | samples | utilisation |
|---|---|---|---|
| 1 | interactive | 88 | 2.3% |
| 2 | default | 87 | 100.0% |
| 3 | default | 87 | 100.0% |
| 4 | default | 90 | 100.0% |
| 5 | default | 87 | 100.0% |
| 6 | default | 88 | 100.0% |
| 7 | default | 90 | 100.0% |
| 8 | default | 88 | 100.0% |
| 9 | default | 88 | 100.0% |
| 10 | default | 89 | 100.0% |
| 11 | default | 89 | 100.0% |
| 12 | default | 88 | 100.0% |
| 13 | default | 89 | 100.0% |
| 14 | default | 89 | 100.0% |
| 15 | default | 88 | 100.0% |
| 16 | default | 89 | 100.0% |
| 17 | default | 88 | 100.0% |
| 18 | foreign: gc | 89 | 100.0% |
| 19 | foreign: gc | 90 | 100.0% |
| 20 | foreign: gc | 90 | 100.0% |
| 21 | foreign: gc | 89 | 100.0% |
| 22 | foreign: gc | 88 | 100.0% |
| 23 | foreign: gc | 88 | 100.0% |
| 24 | foreign: gc | 89 | 100.0% |
| 25 | foreign: gc | 89 | 100.0% |
| 26 | foreign: gc | 89 | 100.0% |
| 27 | foreign: gc | 89 | 100.0% |
| 28 | foreign: gc | 90 | 100.0% |
| 29 | foreign: gc | 88 | 100.0% |
| 30 | foreign: gc | 90 | 100.0% |
| 31 | foreign: gc | 89 | 100.0% |
| 32 | foreign: gc | 88 | 100.0% |

### Top frames by self samples (sampler threads, awake)

| frame | self | share |
|---|---|---|
| ` libc.so.6:-1` | 802 | 56.7% |
| ` :-1` | 103 | 7.3% |
| `getproperty Base_compiler.jl:54` | 65 | 4.6% |
| `CallWrapper FunctionWrappers.jl:65` | 60 | 4.2% |
| `_setindex! array.jl:991` | 27 | 1.9% |
| `pull_deriv! tracked.jl:188` | 23 | 1.6% |
| `* float.jl:497` | 18 | 1.3% |
| `forward_exec! tape.jl:82` | 17 | 1.2% |
| `reverse_exec! tape.jl:92` | 16 | 1.1% |
| `pull_value! tracked.jl:180` | 15 | 1.1% |
| `setproperty! Base_compiler.jl:58` | 14 | 1.0% |
| `getindex essentials.jl:920` | 12 | 0.8% |
| `scalar_forward_exec! scalars.jl:80` | 11 | 0.8% |
| `< int.jl:519` | 11 | 0.8% |
| `scalar_reverse_exec! scalars.jl:52` | 10 | 0.7% |

## Allocating tape instructions (tape audit)

None — the compiled tape replays without allocating.

## Allocation outside the tape (Profile.Allocs)

Estimated bytes per gradient (`gradient`: `logdensity_and_gradient` through Turing's `LogDensityFunction`) or per leapfrog (`nuts_step`: NUTS transitions after setup). Tape allocation has no source frame here — see the tape audit.

| source | first non-Base frame | type | samples | est. bytes per unit |
|---|---|---|---|---|
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Vector{Float64}` | 8 | 12530.0 |
| gradient | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Profile.Allocs.BufferType` | 8 | 9414.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:290` | `ADTypes.AutoReverseDiff{true}` | 9 | 556.8 |
| gradient | `DiffResults: MutableDiffResult DiffResults.jl:23` | `DiffResults.MutableDiffResult{1, Float64, Tuple{Vector{Float` | 1 | 12.8 |
| nuts_step | `AdvancedHMC: ∂H∂r hamiltonian.jl:58` | `Vector{Float64}` | 9573 | 31090.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:240` | `Memory{Float64}` | 5611 | 16200.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:237` | `Float64` | 5657 | 16120.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:243` | `Profile.Allocs.BufferType` | 5544 | 15120.0 |
| nuts_step | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Memory{Float64}` | 2109 | 7966.0 |
| nuts_step | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Profile.Allocs.BufferType` | 2115 | 7911.0 |
| nuts_step | `AdvancedHMC: combine trajectory.jl:467` | `Memory{Float64}` | 2133 | 7900.0 |
| nuts_step | `AdvancedHMC: ∂H∂θ hamiltonian.jl:47` | `Core.SimpleVector` | 4889 | 7782.0 |
| nuts_step | `AdvancedHMC: neg_energy hamiltonian.jl:170` | `Vector{Float64}` | 2075 | 7521.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:247` | `AdvancedHMC.PhasePoint{Vector{Float64}, AdvancedHMC.DualValu` | 4963 | 1635.0 |

## Diagnosis

- **gc_share** (0.362 vs 0.15): GC takes 36.2% of wall time with a zero-allocation tape: the garbage comes from the non-tape path — see the Profile.Allocs sites (DynamicPPL `LogDensityFunction` wrappers, the NUTS tree, chain storage). The profiler put 56.4% of awake sampler-thread samples in GC-stall frames.
- **efficiency_ratio** (4.741 vs 2.0): A leapfrog in situ costs 4.74× the bare compiled gradient. With high GC this is collection pauses; with low GC suspect per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead — see the Profile.Allocs sites and hot frames) or memory-bandwidth contention from a large tape across all threads — compare tape_instructions and AD guide §10.1.
