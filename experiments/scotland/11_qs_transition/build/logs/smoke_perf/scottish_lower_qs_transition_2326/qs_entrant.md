# Sampling performance — qs_entrant

- **experiment**: scottish_lower_qs_transition_2326
- **candidate**: qs_entrant
- **recipe_hash**: bf713bd121faad5d7bf63312bb438196444eb05199caf73d082a5cd08239a4be
- **git_sha**: 19d3ad4d-dirty
- **fold**: 60 (the smoke's largest)
- **probe**: 16 chains × (50 warm-up + 50 draws) on 16 threads; 1060 rows, 977 parameters

## Verdict

**review** — gc_share, efficiency_ratio.

## Metrics

| metric | value | review if |
|---|---|---|
| bytes per compiled gradient (tape audit) | 0 | > 1024 (hard check) |
| bare gradient latency (warmed min) | 0.3076 ms |  |
| tape instructions | 7733 |  |
| tape record + compile | 0.005674 s |  |
| wall time | 108.6 s |  |
| full-concurrency window | 86.58 s |  |
| leapfrogs (warm-up included) | 1302303 |  |
| in-situ ms per leapfrog | 1.221 |  |
| efficiency ratio | 3.97 | > 2.0 |
| bytes per leapfrog | 132300.0 |  |
| GC share (window, GC_Diff) | 29.7% | > 15.0% |
| GC share (whole run) | 27.8% |  |
| GC pauses / full sweeps (window) | 1288 / 4 |  |
| GC allocated (window) | 130.1 GiB |  |
| time to safepoint (window, summed) | 2.542 s |  |
| max GC pause / max time to safepoint (process) | 135.2 ms / 1974.0 ms |  |
| sampler-thread utilisation, mean (profile) | 100.0% | < 75.0% |
| sampler-thread utilisation, min | 100.0% |  |
| sampler samples in GC-stall frames | 46.4% |  |
| sampler work share = utilisation × (1 − GC-stall share) | 53.6% | (reported; spinning at a safepoint counts as awake) |
| sampler samples in JIT/LLVM frames | 1.4% |  |
| profiled sub-window | 5.031 s |  |
| /proc sampler-thread CPU ÷ (window × threads) (fallback) | 67.8% |  |
| /proc process CPU ÷ (window × threads) | 77.9% |  |
| /proc non-sampler CPU (cores) | 1.623 |  |

## Threads (statistical profiler)

| thread | pool | samples | utilisation |
|---|---|---|---|
| 1 | interactive | 70 | 1.4% |
| 2 | default | 72 | 100.0% |
| 3 | default | 71 | 100.0% |
| 4 | default | 70 | 100.0% |
| 5 | default | 71 | 100.0% |
| 6 | default | 71 | 100.0% |
| 7 | default | 71 | 100.0% |
| 8 | default | 71 | 100.0% |
| 9 | default | 70 | 100.0% |
| 10 | default | 70 | 100.0% |
| 11 | default | 71 | 100.0% |
| 12 | default | 72 | 100.0% |
| 13 | default | 71 | 100.0% |
| 14 | default | 72 | 100.0% |
| 15 | default | 71 | 100.0% |
| 16 | default | 71 | 100.0% |
| 17 | default | 69 | 100.0% |
| 18 | foreign: gc | 71 | 100.0% |
| 19 | foreign: gc | 71 | 100.0% |
| 20 | foreign: gc | 70 | 100.0% |
| 21 | foreign: gc | 71 | 100.0% |
| 22 | foreign: gc | 70 | 100.0% |
| 23 | foreign: gc | 71 | 100.0% |
| 24 | foreign: gc | 70 | 100.0% |
| 25 | foreign: gc | 72 | 100.0% |
| 26 | foreign: gc | 69 | 100.0% |
| 27 | foreign: gc | 70 | 100.0% |
| 28 | foreign: gc | 71 | 100.0% |
| 29 | foreign: gc | 71 | 100.0% |
| 30 | foreign: gc | 69 | 100.0% |
| 31 | foreign: gc | 70 | 100.0% |
| 32 | foreign: gc | 71 | 100.0% |

### Top frames by self samples (sampler threads, awake)

| frame | self | share |
|---|---|---|
| ` libc.so.6:-1` | 563 | 49.6% |
| ` :-1` | 89 | 7.8% |
| `getproperty Base_compiler.jl:54` | 81 | 7.1% |
| `CallWrapper FunctionWrappers.jl:65` | 46 | 4.1% |
| `_setindex! array.jl:991` | 33 | 2.9% |
| `pull_deriv! tracked.jl:188` | 32 | 2.8% |
| `+ float.jl:495` | 21 | 1.9% |
| `* float.jl:497` | 17 | 1.5% |
| `getindex essentials.jl:920` | 16 | 1.4% |
| `dgemm_kernel_ZEN libopenblas64_.so:-1` | 14 | 1.2% |
| `setproperty! Base_compiler.jl:58` | 12 | 1.1% |
| `== promotion.jl:637` | 10 | 0.9% |
| `ForwardExecutor tape.jl:76` | 9 | 0.8% |
| `pull_value! tracked.jl:180` | 9 | 0.8% |
| `forward_exec! tape.jl:82` | 8 | 0.7% |

## Allocating tape instructions (tape audit)

None — the compiled tape replays without allocating.

## Allocation outside the tape (Profile.Allocs)

Estimated bytes per gradient (`gradient`: `logdensity_and_gradient` through Turing's `LogDensityFunction`) or per leapfrog (`nuts_step`: NUTS transitions after setup). Tape allocation has no source frame here — see the tape audit.

| source | first non-Base frame | type | samples | est. bytes per unit |
|---|---|---|---|---|
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Vector{Float64}` | 8 | 12550.0 |
| gradient | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Profile.Allocs.BufferType` | 8 | 6317.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:290` | `DynamicPPL.LogDensityAt{DynamicPPL.Model{typeof(BayesianFoot` | 7 | 928.0 |
| gradient | `DiffResults: MutableDiffResult DiffResults.jl:23` | `DiffResults.MutableDiffResult{1, Float64, Tuple{Vector{Float` | 1 | 12.8 |
| nuts_step | `AdvancedHMC: ∂H∂r hamiltonian.jl:58` | `Vector{Float64}` | 7766 | 32360.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:237` | `Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{` | 5015 | 16240.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:240` | `Float64` | 5034 | 15740.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:243` | `Vector{Float64}` | 5005 | 15140.0 |
| nuts_step | `AdvancedHMC: ∂H∂θ hamiltonian.jl:47` | `Float64` | 4535 | 8509.0 |
| nuts_step | `AdvancedHMC: neg_energy hamiltonian.jl:170` | `Profile.Allocs.BufferType` | 1916 | 8056.0 |
| nuts_step | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Memory{Float64}` | 1879 | 8029.0 |
| nuts_step | `AdvancedHMC: combine trajectory.jl:467` | `Memory{Float64}` | 1846 | 7459.0 |
| nuts_step | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Profile.Allocs.BufferType` | 1810 | 7321.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:247` | `Tuple{AdvancedHMC.DualValue{Float64, Vector{Float64}}}` | 4463 | 1863.0 |

## Diagnosis

- **gc_share** (0.2973 vs 0.15): GC takes 29.7% of wall time with a zero-allocation tape: the garbage comes from the non-tape path — see the Profile.Allocs sites (DynamicPPL `LogDensityFunction` wrappers, the NUTS tree, chain storage). The profiler put 46.4% of awake sampler-thread samples in GC-stall frames.
- **efficiency_ratio** (3.97 vs 2.0): A leapfrog in situ costs 3.97× the bare compiled gradient. With high GC this is collection pauses; with low GC suspect per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead — see the Profile.Allocs sites and hot frames) or memory-bandwidth contention from a large tape across all threads — compare tape_instructions and AD guide §10.1.
