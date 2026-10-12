# Sampling performance — qs_control

- **experiment**: scottish_lower_qs_transition_2326
- **candidate**: qs_control
- **recipe_hash**: f75623499919dd9e2e045595bdbe3735c1b2e7748b0655602844ba68a0f46ba7
- **git_sha**: unknown
- **fold**: 60 (the smoke's largest)
- **probe**: 16 chains × (50 warm-up + 50 draws) on 16 threads; 1060 rows, 975 parameters

## Verdict

**review** — gc_share, efficiency_ratio.

## Metrics

| metric | value | review if |
|---|---|---|
| bytes per compiled gradient (tape audit) | 0 | > 1024 (hard check) |
| bare gradient latency (warmed min) | 0.2757 ms |  |
| tape instructions | 7699 |  |
| tape record + compile | 0.004946 s |  |
| wall time | 99.0 s |  |
| full-concurrency window | 89.92 s |  |
| leapfrogs (warm-up included) | 1306249 |  |
| in-situ ms per leapfrog | 1.163 |  |
| efficiency ratio | 4.219 | > 2.0 |
| bytes per leapfrog | 129800.0 |  |
| GC share (window, GC_Diff) | 32.4% | > 15.0% |
| GC share (whole run) | 31.2% |  |
| GC pauses / full sweeps (window) | 1552 / 5 |  |
| GC allocated (window) | 144.1 GiB |  |
| time to safepoint (window, summed) | 2.328 s |  |
| max GC pause / max time to safepoint (process) | 120.7 ms / 1978.0 ms |  |
| sampler-thread utilisation, mean (profile) | 100.0% | < 75.0% |
| sampler-thread utilisation, min | 100.0% |  |
| sampler samples in GC-stall frames | 34.2% |  |
| sampler work share = utilisation × (1 − GC-stall share) | 65.8% | (reported; spinning at a safepoint counts as awake) |
| sampler samples in JIT/LLVM frames | 2.9% |  |
| profiled sub-window | 5.014 s |  |
| /proc sampler-thread CPU ÷ (window × threads) (fallback) | 66.1% |  |
| /proc process CPU ÷ (window × threads) | 74.5% |  |
| /proc non-sampler CPU (cores) | 1.348 |  |

## Threads (statistical profiler)

| thread | pool | samples | utilisation |
|---|---|---|---|
| 1 | interactive | 135 | 0.0% |
| 2 | default | 137 | 100.0% |
| 3 | default | 135 | 100.0% |
| 4 | default | 136 | 100.0% |
| 5 | default | 136 | 100.0% |
| 6 | default | 136 | 100.0% |
| 7 | default | 136 | 100.0% |
| 8 | default | 137 | 100.0% |
| 9 | default | 136 | 100.0% |
| 10 | default | 137 | 100.0% |
| 11 | default | 135 | 100.0% |
| 12 | default | 136 | 100.0% |
| 13 | default | 136 | 100.0% |
| 14 | default | 136 | 100.0% |
| 15 | default | 137 | 100.0% |
| 16 | default | 135 | 100.0% |
| 17 | default | 136 | 100.0% |
| 18 | foreign: gc | 135 | 100.0% |
| 19 | foreign: gc | 136 | 100.0% |
| 20 | foreign: gc | 136 | 100.0% |
| 21 | foreign: gc | 136 | 100.0% |
| 22 | foreign: gc | 137 | 100.0% |
| 23 | foreign: gc | 136 | 100.0% |
| 24 | foreign: gc | 135 | 100.0% |
| 25 | foreign: gc | 137 | 100.0% |
| 26 | foreign: gc | 137 | 100.0% |
| 27 | foreign: gc | 137 | 100.0% |
| 28 | foreign: gc | 135 | 100.0% |
| 29 | foreign: gc | 136 | 100.0% |
| 30 | foreign: gc | 136 | 100.0% |
| 31 | foreign: gc | 135 | 100.0% |
| 32 | foreign: gc | 135 | 100.0% |

### Top frames by self samples (sampler threads, awake)

| frame | self | share |
|---|---|---|
| ` libc.so.6:-1` | 781 | 35.9% |
| ` :-1` | 225 | 10.3% |
| `getproperty Base_compiler.jl:54` | 154 | 7.1% |
| `CallWrapper FunctionWrappers.jl:65` | 134 | 6.2% |
| `_setindex! array.jl:991` | 69 | 3.2% |
| `pull_deriv! tracked.jl:188` | 61 | 2.8% |
| `* float.jl:497` | 57 | 2.6% |
| `getindex essentials.jl:920` | 38 | 1.7% |
| `setproperty! Base_compiler.jl:58` | 36 | 1.7% |
| `+ float.jl:495` | 36 | 1.7% |
| `pull_value! tracked.jl:180` | 28 | 1.3% |
| `reverse_exec! tape.jl:92` | 28 | 1.3% |
| `getindex essentials.jl:919` | 27 | 1.2% |
| `forward_exec! tape.jl:82` | 26 | 1.2% |
| `ForwardExecutor tape.jl:76` | 25 | 1.1% |

## Allocating tape instructions (tape audit)

None — the compiled tape replays without allocating.

## Allocation outside the tape (Profile.Allocs)

Estimated bytes per gradient (`gradient`: `logdensity_and_gradient` through Turing's `LogDensityFunction`) or per leapfrog (`nuts_step`: NUTS transitions after setup). Tape allocation has no source frame here — see the tape audit.

| source | first non-Base frame | type | samples | est. bytes per unit |
|---|---|---|---|---|
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Profile.Allocs.BufferType` | 8 | 9418.0 |
| gradient | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Memory{Float64}` | 7 | 6291.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:290` | `ADTypes.AutoReverseDiff{true}` | 7 | 576.0 |
| gradient | `DiffResults: MutableDiffResult DiffResults.jl:23` | `DiffResults.MutableDiffResult{1, Float64, Tuple{Vector{Float` | 1 | 12.8 |
| nuts_step | `AdvancedHMC: ∂H∂r hamiltonian.jl:58` | `Vector{Float64}` | 7756 | 31150.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:240` | `Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{` | 4408 | 15940.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:243` | `Memory{Float64}` | 4427 | 15910.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:237` | `Vector{Float64}` | 4446 | 15760.0 |
| nuts_step | `AdvancedHMC: ∂H∂θ hamiltonian.jl:47` | `DataType` | 3936 | 8085.0 |
| nuts_step | `AdvancedHMC: neg_energy hamiltonian.jl:170` | `Vector{Float64}` | 1673 | 8009.0 |
| nuts_step | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Memory{Float64}` | 1626 | 7810.0 |
| nuts_step | `AdvancedHMC: combine trajectory.jl:467` | `Memory{Float64}` | 1659 | 7699.0 |
| nuts_step | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Vector{Float64}` | 1668 | 7531.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:247` | `AdvancedHMC.Hamiltonian{AdvancedHMC.DiagEuclideanMetric{Floa` | 3862 | 1572.0 |

## Diagnosis

- **gc_share** (0.3237 vs 0.15): GC takes 32.4% of wall time with a zero-allocation tape: the garbage comes from the non-tape path — see the Profile.Allocs sites (DynamicPPL `LogDensityFunction` wrappers, the NUTS tree, chain storage). The profiler put 34.2% of awake sampler-thread samples in GC-stall frames.
- **efficiency_ratio** (4.219 vs 2.0): A leapfrog in situ costs 4.22× the bare compiled gradient. With high GC this is collection pauses; with low GC suspect per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead — see the Profile.Allocs sites and hot frames) or memory-bandwidth contention from a large tape across all threads — compare tape_instructions and AD guide §10.1.
