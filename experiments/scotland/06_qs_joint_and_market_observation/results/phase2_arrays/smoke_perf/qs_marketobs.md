# Sampling performance — qs_marketobs

- **experiment**: scottish_lower_qs_wave2_2426
- **candidate**: qs_marketobs
- **recipe_hash**: a8e460f6a1c995da5fa8a38468ee4af1afbf2f144911b955eb1c671e45c17503
- **git_sha**: d0d933ae
- **fold**: 40 (the smoke's largest)
- **probe**: 16 chains × (50 warm-up + 50 draws) on 16 threads; 1060 rows, 975 parameters

## Verdict

**review** — gc_share, efficiency_ratio.

## Metrics

| metric | value | review if |
|---|---|---|
| bytes per compiled gradient (tape audit) | 0 | > 1024 (hard check) |
| bare gradient latency (warmed min) | 0.284 ms |  |
| tape instructions | 7699 |  |
| tape record + compile | 0.005014 s |  |
| wall time | 110.1 s |  |
| full-concurrency window | 101.5 s |  |
| leapfrogs (warm-up included) | 1306249 |  |
| in-situ ms per leapfrog | 1.296 |  |
| efficiency ratio | 4.561 | > 2.0 |
| bytes per leapfrog | 129800.0 |  |
| GC share (window, GC_Diff) | 36.2% | > 15.0% |
| GC share (whole run) | 34.9% |  |
| GC pauses / full sweeps (window) | 2113 / 9 |  |
| GC allocated (window) | 146.2 GiB |  |
| time to safepoint (window, summed) | 3.031 s |  |
| max GC pause / max time to safepoint (process) | 112.1 ms / 1996.0 ms |  |
| sampler-thread utilisation, mean (profile) | 100.0% | < 75.0% |
| sampler-thread utilisation, min | 100.0% |  |
| sampler samples in GC-stall frames | 52.2% |  |
| sampler work share = utilisation × (1 − GC-stall share) | 47.8% | (reported; spinning at a safepoint counts as awake) |
| sampler samples in JIT/LLVM frames | 4.2% |  |
| profiled sub-window | 5.017 s |  |
| /proc sampler-thread CPU ÷ (window × threads) (fallback) | 62.1% |  |
| /proc process CPU ÷ (window × threads) | 73.2% |  |
| /proc non-sampler CPU (cores) | 1.769 |  |

## Threads (statistical profiler)

| thread | pool | samples | utilisation |
|---|---|---|---|
| 1 | interactive | 71 | 2.8% |
| 2 | default | 72 | 100.0% |
| 3 | default | 72 | 100.0% |
| 4 | default | 73 | 100.0% |
| 5 | default | 72 | 100.0% |
| 6 | default | 71 | 100.0% |
| 7 | default | 72 | 100.0% |
| 8 | default | 72 | 100.0% |
| 9 | default | 72 | 100.0% |
| 10 | default | 72 | 100.0% |
| 11 | default | 72 | 100.0% |
| 12 | default | 73 | 100.0% |
| 13 | default | 71 | 100.0% |
| 14 | default | 71 | 100.0% |
| 15 | default | 73 | 100.0% |
| 16 | default | 72 | 100.0% |
| 17 | default | 72 | 100.0% |
| 18 | foreign: gc | 72 | 100.0% |
| 19 | foreign: gc | 72 | 100.0% |
| 20 | foreign: gc | 73 | 100.0% |
| 21 | foreign: gc | 72 | 100.0% |
| 22 | foreign: gc | 72 | 100.0% |
| 23 | foreign: gc | 72 | 100.0% |
| 24 | foreign: gc | 72 | 100.0% |
| 25 | foreign: gc | 72 | 100.0% |
| 26 | foreign: gc | 72 | 100.0% |
| 27 | foreign: gc | 72 | 100.0% |
| 28 | foreign: gc | 72 | 100.0% |
| 29 | foreign: gc | 72 | 100.0% |
| 30 | foreign: gc | 73 | 100.0% |
| 31 | foreign: gc | 72 | 100.0% |
| 32 | foreign: gc | 72 | 100.0% |

### Top frames by self samples (sampler threads, awake)

| frame | self | share |
|---|---|---|
| ` libc.so.6:-1` | 627 | 54.4% |
| ` :-1` | 77 | 6.7% |
| `CallWrapper FunctionWrappers.jl:65` | 60 | 5.2% |
| `getproperty Base_compiler.jl:54` | 46 | 4.0% |
| `pull_deriv! tracked.jl:188` | 28 | 2.4% |
| `setproperty! Base_compiler.jl:58` | 24 | 2.1% |
| `_setindex! array.jl:991` | 20 | 1.7% |
| `getindex essentials.jl:920` | 16 | 1.4% |
| `+ float.jl:495` | 14 | 1.2% |
| `reverse_exec! tape.jl:92` | 14 | 1.2% |
| `scalar_forward_exec! scalars.jl:80` | 13 | 1.1% |
| `* float.jl:497` | 13 | 1.1% |
| `pull_value! tracked.jl:180` | 11 | 1.0% |
| `forward_exec! tape.jl:82` | 11 | 1.0% |
| `gc_sweep_pool gc-stock.c:1421` | 11 | 1.0% |

## Allocating tape instructions (tape audit)

None — the compiled tape replays without allocating.

## Allocation outside the tape (Profile.Allocs)

Estimated bytes per gradient (`gradient`: `logdensity_and_gradient` through Turing's `LogDensityFunction`) or per leapfrog (`nuts_step`: NUTS transitions after setup). Tape allocation has no source frame here — see the tape audit.

| source | first non-Base frame | type | samples | est. bytes per unit |
|---|---|---|---|---|
| gradient | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Profile.Allocs.BufferType` | 11 | 12570.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Memory{Float64}` | 8 | 9411.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:290` | `Tuple{Float64, Vector{Float64}}` | 6 | 326.4 |
| gradient | `DiffResults: MutableDiffResult DiffResults.jl:23` | `DiffResults.MutableDiffResult{1, Float64, Tuple{Vector{Float` | 2 | 25.6 |
| nuts_step | `AdvancedHMC: ∂H∂r hamiltonian.jl:58` | `Memory{Float64}` | 7600 | 31330.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:240` | `Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{` | 4345 | 15920.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:237` | `Memory{Float64}` | 4399 | 15890.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:243` | `Vector{Float64}` | 4453 | 15720.0 |
| nuts_step | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Memory{Float64}` | 1666 | 7994.0 |
| nuts_step | `AdvancedHMC: ∂H∂θ hamiltonian.jl:47` | `AdvancedHMC.DualValue{Float64, Vector{Float64}}` | 3816 | 7994.0 |
| nuts_step | `AdvancedHMC: combine trajectory.jl:467` | `Profile.Allocs.BufferType` | 1649 | 7923.0 |
| nuts_step | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Profile.Allocs.BufferType` | 1663 | 7882.0 |
| nuts_step | `AdvancedHMC: neg_energy hamiltonian.jl:170` | `Vector{Float64}` | 1579 | 7148.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:247` | `DataType` | 3900 | 1642.0 |

## Diagnosis

- **gc_share** (0.362 vs 0.15): GC takes 36.2% of wall time with a zero-allocation tape: the garbage comes from the non-tape path — see the Profile.Allocs sites (DynamicPPL `LogDensityFunction` wrappers, the NUTS tree, chain storage). The profiler put 52.2% of awake sampler-thread samples in GC-stall frames.
- **efficiency_ratio** (4.561 vs 2.0): A leapfrog in situ costs 4.56× the bare compiled gradient. With high GC this is collection pauses; with low GC suspect per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead — see the Profile.Allocs sites and hot frames) or memory-bandwidth contention from a large tape across all threads — compare tape_instructions and AD guide §10.1.
