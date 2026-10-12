# Sampling performance — qs_a2

- **experiment**: scottish_lower_qs_transition_2326
- **candidate**: qs_a2
- **recipe_hash**: ecb6eab8243b0b7e27020e24498844667cb386555618d8222c91d8c1173cbf32
- **git_sha**: unknown
- **fold**: 60 (the smoke's largest)
- **probe**: 16 chains × (50 warm-up + 50 draws) on 16 threads; 1060 rows, 1019 parameters

## Verdict

**review** — gc_share, efficiency_ratio.

## Metrics

| metric | value | review if |
|---|---|---|
| bytes per compiled gradient (tape audit) | 0 | > 1024 (hard check) |
| bare gradient latency (warmed min) | 0.3267 ms |  |
| tape instructions | 7816 |  |
| tape record + compile | 0.005419 s |  |
| wall time | 147.0 s |  |
| full-concurrency window | 91.39 s |  |
| leapfrogs (warm-up included) | 1295461 |  |
| in-situ ms per leapfrog | 1.563 |  |
| efficiency ratio | 4.786 | > 2.0 |
| bytes per leapfrog | 140800.0 |  |
| GC share (window, GC_Diff) | 22.5% | > 15.0% |
| GC share (whole run) | 21.0% |  |
| GC pauses / full sweeps (window) | 1059 / 4 |  |
| GC allocated (window) | 103.4 GiB |  |
| time to safepoint (window, summed) | 2.558 s |  |
| max GC pause / max time to safepoint (process) | 132.2 ms / 1978.0 ms |  |
| sampler-thread utilisation, mean (profile) | 100.0% | < 75.0% |
| sampler-thread utilisation, min | 100.0% |  |
| sampler samples in GC-stall frames | 54.1% |  |
| sampler work share = utilisation × (1 − GC-stall share) | 45.9% | (reported; spinning at a safepoint counts as awake) |
| sampler samples in JIT/LLVM frames | 4.4% |  |
| profiled sub-window | 5.016 s |  |
| /proc sampler-thread CPU ÷ (window × threads) (fallback) | 75.2% |  |
| /proc process CPU ÷ (window × threads) | 82.3% |  |
| /proc non-sampler CPU (cores) | 1.144 |  |

## Threads (statistical profiler)

| thread | pool | samples | utilisation |
|---|---|---|---|
| 1 | interactive | 90 | 1.1% |
| 2 | default | 89 | 100.0% |
| 3 | default | 90 | 100.0% |
| 4 | default | 90 | 100.0% |
| 5 | default | 90 | 100.0% |
| 6 | default | 91 | 100.0% |
| 7 | default | 92 | 100.0% |
| 8 | default | 89 | 100.0% |
| 9 | default | 91 | 100.0% |
| 10 | default | 90 | 100.0% |
| 11 | default | 90 | 100.0% |
| 12 | default | 91 | 100.0% |
| 13 | default | 91 | 100.0% |
| 14 | default | 90 | 100.0% |
| 15 | default | 89 | 100.0% |
| 16 | default | 92 | 100.0% |
| 17 | default | 91 | 100.0% |
| 18 | foreign: gc | 91 | 100.0% |
| 19 | foreign: gc | 91 | 100.0% |
| 20 | foreign: gc | 90 | 100.0% |
| 21 | foreign: gc | 91 | 100.0% |
| 22 | foreign: gc | 90 | 100.0% |
| 23 | foreign: gc | 90 | 100.0% |
| 24 | foreign: gc | 90 | 100.0% |
| 25 | foreign: gc | 91 | 100.0% |
| 26 | foreign: gc | 91 | 100.0% |
| 27 | foreign: gc | 90 | 100.0% |
| 28 | foreign: gc | 90 | 100.0% |
| 29 | foreign: gc | 90 | 100.0% |
| 30 | foreign: gc | 91 | 100.0% |
| 31 | foreign: gc | 90 | 100.0% |
| 32 | foreign: gc | 91 | 100.0% |

### Top frames by self samples (sampler threads, awake)

| frame | self | share |
|---|---|---|
| ` libc.so.6:-1` | 809 | 55.9% |
| `getproperty Base_compiler.jl:54` | 99 | 6.8% |
| ` :-1` | 79 | 5.5% |
| `pull_deriv! tracked.jl:188` | 52 | 3.6% |
| `CallWrapper FunctionWrappers.jl:65` | 42 | 2.9% |
| `_setindex! array.jl:991` | 36 | 2.5% |
| `dgemv_kernel_4x4 libopenblas64_.so:-1` | 31 | 2.1% |
| `+ float.jl:495` | 21 | 1.5% |
| `setproperty! Base_compiler.jl:58` | 17 | 1.2% |
| `* float.jl:497` | 17 | 1.2% |
| `getindex essentials.jl:920` | 15 | 1.0% |
| `gc_sweep_pool gc-stock.c:1421` | 11 | 0.8% |
| `macro expansion simdloop.jl:75` | 9 | 0.6% |
| `diffresult_increment_deriv! propagation.jl:86` | 9 | 0.6% |
| `scalar_forward_exec! scalars.jl:80` | 9 | 0.6% |

## Allocating tape instructions (tape audit)

None — the compiled tape replays without allocating.

## Allocation outside the tape (Profile.Allocs)

Estimated bytes per gradient (`gradient`: `logdensity_and_gradient` through Turing's `LogDensityFunction`) or per leapfrog (`nuts_step`: NUTS transitions after setup). Tape allocation has no source frame here — see the tape audit.

| source | first non-Base frame | type | samples | est. bytes per unit |
|---|---|---|---|---|
| gradient | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Vector{Float64}` | 7 | 3331.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Memory{Float64}` | 7 | 3328.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:290` | `ADTypes.AutoReverseDiff{true}` | 5 | 435.2 |
| gradient | `DiffResults: MutableDiffResult DiffResults.jl:23` | `DiffResults.MutableDiffResult{1, Float64, Tuple{Vector{Float` | 5 | 64.0 |
| nuts_step | `AdvancedHMC: ∂H∂r hamiltonian.jl:58` | `Memory{Float64}` | 6125 | 32470.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:237` | `Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{` | 4163 | 16540.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:240` | `Vector{Float64}` | 4065 | 16250.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:243` | `Profile.Allocs.BufferType` | 4125 | 16080.0 |
| nuts_step | `AdvancedHMC: combine trajectory.jl:467` | `Vector{Float64}` | 1520 | 8527.0 |
| nuts_step | `AdvancedHMC: ∂H∂θ hamiltonian.jl:47` | `Vector{Float64}` | 3562 | 8384.0 |
| nuts_step | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Vector{Float64}` | 1560 | 8309.0 |
| nuts_step | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Memory{Float64}` | 1556 | 8072.0 |
| nuts_step | `AdvancedHMC: neg_energy hamiltonian.jl:170` | `Vector{Float64}` | 1510 | 7928.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:247` | `Tuple{AdvancedHMC.DualValue{Float64, Vector{Float64}}}` | 3510 | 2135.0 |

## Diagnosis

- **gc_share** (0.2245 vs 0.15): GC takes 22.5% of wall time with a zero-allocation tape: the garbage comes from the non-tape path — see the Profile.Allocs sites (DynamicPPL `LogDensityFunction` wrappers, the NUTS tree, chain storage). The profiler put 54.1% of awake sampler-thread samples in GC-stall frames.
- **efficiency_ratio** (4.786 vs 2.0): A leapfrog in situ costs 4.79× the bare compiled gradient. With high GC this is collection pauses; with low GC suspect per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead — see the Profile.Allocs sites and hot frames) or memory-bandwidth contention from a large tape across all threads — compare tape_instructions and AD guide §10.1.
