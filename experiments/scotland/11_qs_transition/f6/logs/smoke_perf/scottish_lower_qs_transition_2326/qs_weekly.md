# Sampling performance — qs_weekly

- **experiment**: scottish_lower_qs_transition_2326
- **candidate**: qs_weekly
- **recipe_hash**: aaed32f47a4c97e032bfb067b6debd2e4ccc14d934df58301955f4bccffae952
- **git_sha**: 91bcd7d8-dirty
- **fold**: 115 (the smoke's largest)
- **probe**: 4 chains × (50 warm-up + 50 draws) on 4 threads; 1060 rows, 975 parameters

## Verdict

**review** — efficiency_ratio.

## Metrics

| metric | value | review if |
|---|---|---|
| bytes per compiled gradient (tape audit) | 0 | > 1024 (hard check) |
| bare gradient latency (warmed min) | 0.3003 ms |  |
| tape instructions | 7627 |  |
| tape record + compile | 0.01266 s |  |
| wall time | 52.69 s |  |
| full-concurrency window | 47.15 s |  |
| leapfrogs (warm-up included) | 321472 |  |
| in-situ ms per leapfrog | 0.6267 |  |
| efficiency ratio | 2.087 | > 2.0 |
| bytes per leapfrog | 129800.0 |  |
| GC share (window, GC_Diff) | 9.7% | > 15.0% |
| GC share (whole run) | 9.5% |  |
| GC pauses / full sweeps (window) | 332 / 1 |  |
| GC allocated (window) | 36.03 GiB |  |
| time to safepoint (window, summed) | 2.975 s |  |
| max GC pause / max time to safepoint (process) | 506.6 ms / 2883.0 ms |  |
| sampler-thread utilisation, mean (profile) | 100.0% | < 75.0% |
| sampler-thread utilisation, min | 100.0% |  |
| sampler samples in GC-stall frames | 65.3% |  |
| sampler work share = utilisation × (1 − GC-stall share) | 34.7% | (reported; spinning at a safepoint counts as awake) |
| sampler samples in JIT/LLVM frames | 3.1% |  |
| profiled sub-window | 5.009 s |  |
| /proc sampler-thread CPU ÷ (window × threads) (fallback) | 82.1% |  |
| /proc process CPU ÷ (window × threads) | 84.6% |  |
| /proc non-sampler CPU (cores) | 0.09925 |  |

## Threads (statistical profiler)

| thread | pool | samples | utilisation |
|---|---|---|---|
| 1 | interactive | 290 | 1.4% |
| 2 | default | 291 | 100.0% |
| 3 | default | 293 | 100.0% |
| 4 | default | 292 | 100.0% |
| 5 | default | 292 | 100.0% |
| 6 | foreign: gc | 291 | 100.0% |
| 7 | foreign: gc | 292 | 100.0% |
| 8 | foreign: gc | 292 | 100.0% |

### Top frames by self samples (sampler threads, awake)

| frame | self | share |
|---|---|---|
| ` libc.so.6:-1` | 606 | 51.9% |
| ` :-1` | 63 | 5.4% |
| `getproperty Base_compiler.jl:54` | 48 | 4.1% |
| `gc_mark_obj8 gc-stock.c:1702` | 40 | 3.4% |
| `CallWrapper FunctionWrappers.jl:65` | 36 | 3.1% |
| `gc_sweep_sysimg staticdata.c:2347` | 31 | 2.7% |
| `pull_deriv! tracked.jl:188` | 26 | 2.2% |
| `gc_sweep_page gc-stock.c:911` | 25 | 2.1% |
| `getindex essentials.jl:920` | 16 | 1.4% |
| `gc_sweep_pool gc-stock.c:1421` | 14 | 1.2% |
| `_setindex! array.jl:991` | 14 | 1.2% |
| `gc_try_claim_and_push gc-stock.c:1682` | 13 | 1.1% |
| `gc_mark_objarray gc-stock.c:1804` | 13 | 1.1% |
| `* float.jl:497` | 12 | 1.0% |
| `setproperty! Base_compiler.jl:58` | 10 | 0.9% |

## Allocating tape instructions (tape audit)

None — the compiled tape replays without allocating.

## Allocation outside the tape (Profile.Allocs)

Estimated bytes per gradient (`gradient`: `logdensity_and_gradient` through Turing's `LogDensityFunction`) or per leapfrog (`nuts_step`: NUTS transitions after setup). Tape allocation has no source frame here — see the tape audit.

| source | first non-Base frame | type | samples | est. bytes per unit |
|---|---|---|---|---|
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Memory{Float64}` | 8 | 6310.0 |
| gradient | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Memory{Float64}` | 6 | 6282.0 |
| gradient | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:290` | `DynamicPPL.LogDensityAt{DynamicPPL.Model{typeof(BayesianFoot` | 8 | 595.2 |
| gradient | `DiffResults: MutableDiffResult DiffResults.jl:23` | `DiffResults.MutableDiffResult{1, Float64, Tuple{Vector{Float` | 1 | 12.8 |
| nuts_step | `AdvancedHMC: ∂H∂r hamiltonian.jl:58` | `Vector{Float64}` | 6529 | 29870.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:243` | `Vector{Float64}` | 4443 | 16740.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:237` | `Vector{Float64}` | 4364 | 15890.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:240` | `Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{` | 4390 | 15680.0 |
| nuts_step | `DifferentiationInterface: value_and_gradient onearg.jl:113` | `Profile.Allocs.BufferType` | 1738 | 8517.0 |
| nuts_step | `AdvancedHMC: ∂H∂θ hamiltonian.jl:47` | `Profile.Allocs.BufferType` | 3837 | 8203.0 |
| nuts_step | `AdvancedHMC: combine trajectory.jl:467` | `Memory{Float64}` | 1693 | 8178.0 |
| nuts_step | `AdvancedHMC: neg_energy hamiltonian.jl:170` | `Vector{Float64}` | 1722 | 8053.0 |
| nuts_step | `DynamicPPL: logdensity_and_gradient logdensityfunction.jl:286` | `Memory{Float64}` | 1664 | 7826.0 |
| nuts_step | `AdvancedHMC: #step#11 integrator.jl:247` | `AdvancedHMC.DualValue{Float64, Vector{Float64}}` | 3990 | 1647.0 |

## Diagnosis

- **efficiency_ratio** (2.087 vs 2.0): A leapfrog in situ costs 2.09× the bare compiled gradient. With high GC this is collection pauses; with low GC suspect per-leapfrog work outside the tape (DynamicPPL/AdvancedHMC overhead — see the Profile.Allocs sites and hot frames) or memory-bandwidth contention from a large tape across all threads — compare tape_instructions and AD guide §10.1.
