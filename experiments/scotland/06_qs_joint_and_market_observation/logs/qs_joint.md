# Sampling performance — qs_joint

- **experiment**: scottish_lower_qs_wave2_2426
- **candidate**: qs_joint
- **recipe_hash**: 604a4c3bb4385a4b73a21631a66d32bd8d26010d6460ed1b554beb18b379ed47
- **git_sha**: ac30c4d7
- **fold**: 40 (the smoke's largest)
- **probe**: 16 chains × (50 warm-up + 50 draws) on 16 threads; 1060 rows, 976 parameters

## Verdict

**review** — gc_share, efficiency_ratio.

## Metrics

| metric | value | review if |
|---|---|---|
| bytes per compiled gradient (tape audit) | 0 | > 1024 (hard check) |
| bare gradient latency (warmed min) | 0.3111 ms |  |
| tape instructions | 7728 |  |
| tape record + compile | 0.00489 s |  |
| wall time | 68.06 s |  |
| full-concurrency window | 46.92 s |  |
| leapfrogs (warm-up included) | 730160 |  |
| in-situ ms per leapfrog | 1.382 |  |
| efficiency ratio | 4.44 | > 2.0 |
| bytes per leapfrog | 132000.0 |  |
| GC share (window, GC_Diff) | 32.1% | > 15.0% |
| GC share (whole run) | 31.2% |  |
| GC pauses / full sweeps (window) | 910 / 5 |  |
| GC allocated (window) | 60.22 GiB |  |
| time to safepoint (window, summed) | 2.696 s |  |
| max GC pause / max time to safepoint (process) | 110.6 ms / 1992.0 ms |  |
| sampler-thread utilisation, mean (profile) | 100.0% | < 75.0% |
| sampler-thread utilisation, min | 100.0% |  |
| sampler samples in GC-stall frames | 57.7% |  |
| sampler work share = utilisation × (1 − GC-stall share) | 42.3% | (reported; spinning at a safepoint counts as awake) |
| sampler samples in JIT/LLVM frames | 2.2% |  |
| profiled sub-window | 5.004 s |  |
| /proc sampler-thread CPU ÷ (window × threads) (fallback) | 63.2% |  |
| /proc process CPU ÷ (window × threads) | 75.1% |  |
| /proc non-sampler CPU (cores) | 1.906 |  |

## Threads (statistical profiler)

| thread | pool | samples | utilisation |
|---|---|---|---|
| 1 | interactive | 90 | 1.1% |
| 2 | default | 91 | 100.0% |
| 3 | default | 92 | 100.0% |
| 4 | default | 92 | 100.0% |
| 5 | default | 92 | 100.0% |
| 6 | default | 92 | 100.0% |
| 7 | default | 91 | 100.0% |
| 8 | default | 92 | 100.0% |
| 9 | default | 92 | 100.0% |
| 10 | default | 91 | 100.0% |
| 11 | default | 91 | 100.0% |
| 12 | default | 90 | 100.0% |
| 13 | default | 91 | 100.0% |
| 14 | default | 91 | 100.0% |
| 15 | default | 90 | 100.0% |
| 16 | default | 91 | 100.0% |
| 17 | default | 92 | 100.0% |
| 18 | foreign: gc | 91 | 100.0% |
| 19 | foreign: gc | 91 | 100.0% |
| 20 | foreign: gc | 90 | 100.0% |
| 21 | foreign: gc | 91 | 100.0% |
| 22 | foreign: gc | 91 | 100.0% |
| 23 | foreign: gc | 92 | 100.0% |
| 24 | foreign: gc | 92 | 100.0% |
| 25 | foreign: gc | 91 | 100.0% |
| 26 | foreign: gc | 92 | 100.0% |
| 27 | foreign: gc | 91 | 100.0% |
| 28 | foreign: gc | 90 | 100.0% |
| 29 | foreign: gc | 92 | 100.0% |
| 30 | foreign: gc | 91 | 100.0% |
| 31 | foreign: gc | 91 | 100.0% |
| 32 | foreign: gc | 92 | 100.0% |

### Top frames by self samples (sampler threads, awake)

| frame | self | share |
|---|---|---|
| ` libc.so.6:-1` | 829 | 56.7% |
| ` :-1` | 89 | 6.1% |
| `getproperty Base_compiler.jl:54` | 64 | 4.4% |
| `CallWrapper FunctionWrappers.jl:65` | 40 | 2.7% |
| `pull_deriv! tracked.jl:188` | 39 | 2.7% |
| `* float.jl:497` | 31 | 2.1% |
| `_setindex! array.jl:991` | 31 | 2.1% |
| `forward_exec! tape.jl:82` | 23 | 1.6% |
| `getindex essentials.jl:920` | 20 | 1.4% |
| `+ float.jl:495` | 17 | 1.2% |
| `pull_value! tracked.jl:180` | 14 | 1.0% |
| `== promotion.jl:637` | 13 | 0.9% |
| `setproperty! Base_compiler.jl:58` | 12 | 0.8% |
| `reverse_exec! tape.jl:92` | 12 | 0.8% |
| `diffresult_increment_deriv! propagation.jl:86` | 10 | 0.7% |

## Allocating tape instructions (tape audit)

None — the compiled tape replays without allocating.

## Allocation outside the tape (Profile.Allocs)

Estimated bytes per gradient (`gradient`: `logdensity_and_gradient` through Turing's `LogDensityFunction`) or per leapfrog (`nuts_step`: NUTS transitions after setup). Tape allocation has no source frame here — see the tape audit.

| source | first non-Base frame | type | samples | est. bytes per unit |
|---|---|---|---|---|
| gradient | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Memory{Float64}` | 5 | 3168.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:290` | `DynamicPPL.LogDensityAt{DynamicPPL.Model{typeof(BayesianFoot` | 10 | 665.6 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Memory{Float64}` | 4 | 41.6 |
| gradient | `DiffResults: MutableDiffResult DiffResults.jl:23` | `DiffResults.MutableDiffResult{1, Float64, Tuple{Vector{Float` | 2 | 25.6 |
| nuts_step | `AdvancedHMC: ∂H∂r hamiltonian.jl:58` | `Profile.Allocs.BufferType` | 5819 | 31650.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:243` | `Memory{Float64}` | 3763 | 15490.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:237` | `Vector{Float64}` | 3625 | 15400.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:240` | `Memory{Float64}` | 3692 | 15260.0 |
| nuts_step | `AdvancedHMC: ∂H∂θ hamiltonian.jl:47` | `DataType` | 3375 | 8729.0 |
| nuts_step | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Vector{Float64}` | 1381 | 7793.0 |
| nuts_step | `AdvancedHMC: combine trajectory.jl:467` | `Profile.Allocs.BufferType` | 1385 | 7679.0 |
| nuts_step | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Vector{Float64}` | 1356 | 7530.0 |
| nuts_step | `AdvancedHMC: neg_energy hamiltonian.jl:170` | `Memory{Float64}` | 1416 | 7189.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:247` | `AdvancedHMC.DualValue{Float64, Vector{Float64}}` | 3372 | 1781.0 |

## Diagnosis

- **gc_share** (0.3211 vs 0.15): GC takes 32.1% of wall time with a zero-allocation tape: the garbage comes from the non-tape path — see the Profile.Allocs sites (DynamicPPL `LogDensityFunction` wrappers, the NUTS tree, chain storage). The profiler put 57.7% of awake sampler-thread samples in GC-stall frames.
- **efficiency_ratio** (4.44 vs 2.0): A leapfrog in situ costs 4.44× the bare compiled gradient. With high GC this is collection pauses; with low GC suspect per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead — see the Profile.Allocs sites and hot frames) or memory-bandwidth contention from a large tape across all threads — compare tape_instructions and AD guide §10.1.
