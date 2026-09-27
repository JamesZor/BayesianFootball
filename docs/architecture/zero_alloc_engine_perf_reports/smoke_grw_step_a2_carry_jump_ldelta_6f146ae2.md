# Sampling performance — grw_step_a2_carry_jump_ldelta

- **experiment**: scottish_pyramid_tier_transition_w2
- **candidate**: grw_step_a2_carry_jump_ldelta
- **recipe_hash**: ea2e2e545d649ff15ca7c071ddd94c8c143c365da722abea91f6d1e7019ef2e7
- **git_sha**: 6f146ae2
- **fold**: 60 (the smoke's largest)
- **probe**: 16 chains × (50 warm-up + 50 draws) on 16 threads; 2390 rows, 1955 parameters

## Verdict

**review** — efficiency_ratio.

## Metrics

| metric | value | review if |
|---|---|---|
| bytes per compiled gradient (tape audit) | 0 | > 1024 (hard check) |
| bare gradient latency (warmed min) | 0.7692 ms |  |
| tape instructions | 14940 |  |
| tape record + compile | 0.01011 s |  |
| wall time | 335.9 s |  |
| full-concurrency window | 236.0 s |  |
| leapfrogs (warm-up included) | 805123 |  |
| in-situ ms per leapfrog | 5.829 |  |
| efficiency ratio | 7.578 | > 2.0 |
| bytes per leapfrog | 259600.0 |  |
| GC share (window, GC_Diff) | 6.1% | > 15.0% |
| GC share (whole run) | 5.9% |  |
| GC pauses / full sweeps (window) | 1238 / 6 |  |
| GC allocated (window) | 134.0 GiB |  |
| time to safepoint (window, summed) | 2.478 s |  |
| max GC pause / max time to safepoint (process) | 147.3 ms / 19.12 ms |  |
| sampler-thread utilisation, mean (profile) | 100.0% | < 75.0% |
| sampler-thread utilisation, min | 100.0% |  |
| sampler samples in GC-stall frames | 26.3% |  |
| sampler samples in JIT/LLVM frames | 0.1% |  |
| profiled sub-window | 236.0 s |  |
| /proc sampler-thread CPU ÷ (window × threads) (fallback) | 89.7% |  |
| /proc process CPU ÷ (window × threads) | 96.8% |  |
| /proc non-sampler CPU (cores) | 1.13 |  |

## Threads (statistical profiler)

| thread | pool | samples | utilisation |
|---|---|---|---|
| 1 | interactive | 5784 | 0.2% |
| 2 | default | 5784 | 100.0% |
| 3 | default | 5784 | 100.0% |
| 4 | default | 5784 | 100.0% |
| 5 | default | 5784 | 100.0% |
| 6 | default | 5784 | 100.0% |
| 7 | default | 5784 | 100.0% |
| 8 | default | 5784 | 100.0% |
| 9 | default | 5784 | 100.0% |
| 10 | default | 5784 | 100.0% |
| 11 | default | 5784 | 100.0% |
| 12 | default | 5784 | 100.0% |
| 13 | default | 5784 | 100.0% |
| 14 | default | 5784 | 100.0% |
| 15 | default | 5784 | 100.0% |
| 16 | default | 5784 | 100.0% |
| 17 | default | 5784 | 100.0% |
| 18 | foreign: gc | 5784 | 100.0% |
| 19 | foreign: gc | 5784 | 100.0% |
| 20 | foreign: gc | 5784 | 100.0% |
| 21 | foreign: gc | 5784 | 100.0% |
| 22 | foreign: gc | 5784 | 100.0% |
| 23 | foreign: gc | 5784 | 100.0% |
| 24 | foreign: gc | 5784 | 100.0% |
| 25 | foreign: gc | 5784 | 100.0% |
| 26 | foreign: gc | 5784 | 100.0% |
| 27 | foreign: gc | 5784 | 100.0% |
| 28 | foreign: gc | 5784 | 100.0% |
| 29 | foreign: gc | 5784 | 100.0% |
| 30 | foreign: gc | 5784 | 100.0% |
| 31 | foreign: gc | 5784 | 100.0% |
| 32 | foreign: gc | 5784 | 100.0% |

### Top frames by self samples (sampler threads, awake)

| frame | self | share |
|---|---|---|
| ` libc.so.6:-1` | 23479 | 25.4% |
| `getproperty Base_compiler.jl:54` | 9278 | 10.0% |
| `dgemv_kernel_4x4 libopenblas64_.so:-1` | 8930 | 9.6% |
| `pull_deriv! tracked.jl:188` | 7468 | 8.1% |
| ` :-1` | 5310 | 5.7% |
| `_setindex! array.jl:991` | 5013 | 5.4% |
| `getindex essentials.jl:920` | 3229 | 3.5% |
| `+ float.jl:495` | 3012 | 3.3% |
| `diffresult_increment_deriv! propagation.jl:86` | 2695 | 2.9% |
| `setproperty! Base_compiler.jl:58` | 2681 | 2.9% |
| `CallWrapper FunctionWrappers.jl:65` | 2598 | 2.8% |
| `increment_deriv! propagation.jl:46` | 2417 | 2.6% |
| `* float.jl:497` | 2020 | 2.2% |
| `pull_value! tracked.jl:180` | 1392 | 1.5% |
| `getindex essentials.jl:919` | 1195 | 1.3% |

## Allocating tape instructions (tape audit)

None — the compiled tape replays without allocating.

## Allocation outside the tape (Profile.Allocs)

Estimated bytes per gradient (`gradient`: `logdensity_and_gradient` through Turing's `LogDensityFunction`) or per leapfrog (`nuts_step`: NUTS transitions after setup). Tape allocation has no source frame here — see the tape audit.

| source | first non-Base frame | type | samples | est. bytes per unit |
|---|---|---|---|---|
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Profile.Allocs.BufferType` | 13 | 43860.0 |
| gradient | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Vector{Float64}` | 12 | 25120.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:290` | `Tuple{Float64, Vector{Float64}}` | 6 | 908.8 |
| gradient | `DiffResults: MutableDiffResult DiffResults.jl:23` | `DiffResults.MutableDiffResult{1, Float64, Tuple{Vector{Float` | 3 | 38.4 |
| nuts_step | `AdvancedHMC: ∂H∂r hamiltonian.jl:58` | `Memory{Float64}` | 7662 | 62180.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:240` | `Vector{Float64}` | 5127 | 30920.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:237` | `Profile.Allocs.BufferType` | 5076 | 30180.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:243` | `Memory{Float64}` | 5042 | 29820.0 |
| nuts_step | `AdvancedHMC: ∂H∂θ hamiltonian.jl:47` | `Profile.Allocs.BufferType` | 4601 | 16410.0 |
| nuts_step | `AdvancedHMC: combine trajectory.jl:467` | `Memory{Float64}` | 1939 | 15670.0 |
| nuts_step | `AdvancedHMC: neg_energy hamiltonian.jl:170` | `Profile.Allocs.BufferType` | 1854 | 15100.0 |
| nuts_step | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Memory{Float64}` | 1875 | 15060.0 |
| nuts_step | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Memory{Float64}` | 1881 | 14420.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:247` | `Tuple{AdvancedHMC.DualValue{Float64, Vector{Float64}}}` | 4468 | 2447.0 |

## Diagnosis

- **efficiency_ratio** (7.578 vs 2.0): A leapfrog in situ costs 7.58× the bare compiled gradient. With high GC this is collection pauses; with low GC suspect per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead — see the Profile.Allocs sites and hot frames) or memory-bandwidth contention from a large tape across all threads — compare tape_instructions and AD guide §10.1.
