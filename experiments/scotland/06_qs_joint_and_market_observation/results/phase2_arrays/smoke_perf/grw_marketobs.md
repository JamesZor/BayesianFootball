# Sampling performance — grw_marketobs

- **experiment**: scottish_lower_qs_wave2_2426
- **candidate**: grw_marketobs
- **recipe_hash**: 0a6fae3fb52278be2ef91b54ddaefd66fbaa96418b516b4b444b224a0e29b871
- **git_sha**: d0d933ae
- **fold**: 40 (the smoke's largest)
- **probe**: 16 chains × (50 warm-up + 50 draws) on 16 threads; 1060 rows, 975 parameters

## Verdict

**review** — gc_share, efficiency_ratio.

## Metrics

| metric | value | review if |
|---|---|---|
| bytes per compiled gradient (tape audit) | 0 | > 1024 (hard check) |
| bare gradient latency (warmed min) | 0.2777 ms |  |
| tape instructions | 7670 |  |
| tape record + compile | 0.005135 s |  |
| wall time | 110.8 s |  |
| full-concurrency window | 100.8 s |  |
| leapfrogs (warm-up included) | 1288383 |  |
| in-situ ms per leapfrog | 1.312 |  |
| efficiency ratio | 4.725 | > 2.0 |
| bytes per leapfrog | 129900.0 |  |
| GC share (window, GC_Diff) | 35.8% | > 15.0% |
| GC share (whole run) | 34.3% |  |
| GC pauses / full sweeps (window) | 2101 / 9 |  |
| GC allocated (window) | 143.0 GiB |  |
| time to safepoint (window, summed) | 2.806 s |  |
| max GC pause / max time to safepoint (process) | 114.9 ms / 2020.0 ms |  |
| sampler-thread utilisation, mean (profile) | 100.0% | < 75.0% |
| sampler-thread utilisation, min | 100.0% |  |
| sampler samples in GC-stall frames | 57.3% |  |
| sampler work share = utilisation × (1 − GC-stall share) | 42.7% | (reported; spinning at a safepoint counts as awake) |
| sampler samples in JIT/LLVM frames | 3.2% |  |
| profiled sub-window | 5.06 s |  |
| /proc sampler-thread CPU ÷ (window × threads) (fallback) | 62.7% |  |
| /proc process CPU ÷ (window × threads) | 73.1% |  |
| /proc non-sampler CPU (cores) | 1.658 |  |

## Threads (statistical profiler)

| thread | pool | samples | utilisation |
|---|---|---|---|
| 1 | interactive | 93 | 1.1% |
| 2 | default | 94 | 100.0% |
| 3 | default | 94 | 100.0% |
| 4 | default | 94 | 100.0% |
| 5 | default | 93 | 100.0% |
| 6 | default | 94 | 100.0% |
| 7 | default | 94 | 100.0% |
| 8 | default | 95 | 100.0% |
| 9 | default | 94 | 100.0% |
| 10 | default | 94 | 100.0% |
| 11 | default | 94 | 100.0% |
| 12 | default | 94 | 100.0% |
| 13 | default | 95 | 100.0% |
| 14 | default | 94 | 100.0% |
| 15 | default | 93 | 100.0% |
| 16 | default | 94 | 100.0% |
| 17 | default | 95 | 100.0% |
| 18 | foreign: gc | 95 | 100.0% |
| 19 | foreign: gc | 94 | 100.0% |
| 20 | foreign: gc | 94 | 100.0% |
| 21 | foreign: gc | 95 | 100.0% |
| 22 | foreign: gc | 94 | 100.0% |
| 23 | foreign: gc | 94 | 100.0% |
| 24 | foreign: gc | 93 | 100.0% |
| 25 | foreign: gc | 94 | 100.0% |
| 26 | foreign: gc | 95 | 100.0% |
| 27 | foreign: gc | 94 | 100.0% |
| 28 | foreign: gc | 94 | 100.0% |
| 29 | foreign: gc | 93 | 100.0% |
| 30 | foreign: gc | 92 | 100.0% |
| 31 | foreign: gc | 94 | 100.0% |
| 32 | foreign: gc | 94 | 100.0% |

### Top frames by self samples (sampler threads, awake)

| frame | self | share |
|---|---|---|
| ` libc.so.6:-1` | 870 | 57.8% |
| ` :-1` | 101 | 6.7% |
| `getproperty Base_compiler.jl:54` | 56 | 3.7% |
| `CallWrapper FunctionWrappers.jl:65` | 51 | 3.4% |
| `pull_deriv! tracked.jl:188` | 33 | 2.2% |
| `_setindex! array.jl:991` | 28 | 1.9% |
| `* float.jl:497` | 25 | 1.7% |
| `reverse_exec! tape.jl:92` | 20 | 1.3% |
| `setproperty! Base_compiler.jl:58` | 16 | 1.1% |
| `+ float.jl:495` | 14 | 0.9% |
| `pull_value! tracked.jl:180` | 13 | 0.9% |
| `getindex essentials.jl:920` | 12 | 0.8% |
| `gc_sweep_pool gc-stock.c:1421` | 11 | 0.7% |
| `forward_exec! tape.jl:82` | 10 | 0.7% |
| `unary_scalar_forward_exec! scalars.jl:93` | 10 | 0.7% |

## Allocating tape instructions (tape audit)

None — the compiled tape replays without allocating.

## Allocation outside the tape (Profile.Allocs)

Estimated bytes per gradient (`gradient`: `logdensity_and_gradient` through Turing's `LogDensityFunction`) or per leapfrog (`nuts_step`: NUTS transitions after setup). Tape allocation has no source frame here — see the tape audit.

| source | first non-Base frame | type | samples | est. bytes per unit |
|---|---|---|---|---|
| gradient | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Vector{Float64}` | 8 | 9418.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Profile.Allocs.BufferType` | 8 | 9414.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:290` | `Tuple{Float64, Vector{Float64}}` | 7 | 588.8 |
| gradient | `DiffResults: MutableDiffResult DiffResults.jl:23` | `DiffResults.MutableDiffResult{1, Float64, Tuple{Vector{Float` | 5 | 64.0 |
| nuts_step | `AdvancedHMC: ∂H∂r hamiltonian.jl:58` | `Memory{Float64}` | 9453 | 30700.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:243` | `Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{` | 5584 | 16090.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:237` | `Memory{Float64}` | 5665 | 15850.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:240` | `Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{` | 5585 | 15440.0 |
| nuts_step | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Profile.Allocs.BufferType` | 2122 | 8488.0 |
| nuts_step | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Memory{Float64}` | 2089 | 8221.0 |
| nuts_step | `AdvancedHMC: combine trajectory.jl:467` | `Profile.Allocs.BufferType` | 2073 | 8054.0 |
| nuts_step | `AdvancedHMC: ∂H∂θ hamiltonian.jl:47` | `Vector{Float64}` | 4935 | 7851.0 |
| nuts_step | `AdvancedHMC: neg_energy hamiltonian.jl:170` | `Profile.Allocs.BufferType` | 2060 | 7409.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:247` | `Core.SimpleVector` | 4963 | 1603.0 |

## Diagnosis

- **gc_share** (0.3577 vs 0.15): GC takes 35.8% of wall time with a zero-allocation tape: the garbage comes from the non-tape path — see the Profile.Allocs sites (DynamicPPL `LogDensityFunction` wrappers, the NUTS tree, chain storage). The profiler put 57.3% of awake sampler-thread samples in GC-stall frames.
- **efficiency_ratio** (4.725 vs 2.0): A leapfrog in situ costs 4.73× the bare compiled gradient. With high GC this is collection pauses; with low GC suspect per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead — see the Profile.Allocs sites and hot frames) or memory-bandwidth contention from a large tape across all threads — compare tape_instructions and AD guide §10.1.
