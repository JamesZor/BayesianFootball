# ==============================================================================
# r08 — Experiment 08 gradient-path allocation probe (TODO 003 root cause)
# ==============================================================================
#
# WHY. The TODO 003 GC benchmark measured 11,873 GiB allocated per 16-chain
# config-B batch (~742 GiB per 1,000-iteration chain), whatever the GC flags, while
# `l08_gradient_checks` reports a compiled model gradient of ~0.2 ms and 0 bytes. This
# probe measures, on one thread and on the same Fold 1 m00 FeatureSet, where between
# those two numbers the allocation arises:
#
#   1. the experiment's own compiled ReverseDiff tape over `logdensity(ldf, θ)`;
#   2. `logdensity_and_gradient` on a `LogDensityFunction` carrying exactly the
#      adtype `Samplers.run_sampler` passes to `sample` (`AutoReverseDiff(compile = true)`);
#   3. a whole `Turing.sample(NUTS)` run without adaptation, divided by the leapfrog
#      steps its chain records, which is the per-gradient cost the sampler actually pays;
#   4. the per-iteration work outside the trajectory: plain log density, VarInfo
#      deepcopy, and the `Transition` model re-evaluation Turing performs every step;
#   5. bare `AdvancedHMC.transition` on the Hamiltonian and kernel Turing builds, which
#      separates trajectory cost from Turing's per-iteration wrapping;
#   6. Turing's own `AbstractMCMC.step` by hand and `Turing.sample` with the adtype in the
#      `NUTS` constructor. The finding: Turing 0.41 ignores an `adtype` keyword on
#      `sample`, so `run_sampler` (which passes it there) has been running ForwardDiff.
#
# Single-threaded and read-only: no database, no persistence beyond one CSV.
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================
using BayesianFootball
using CSV
using DataFrames
using LinearAlgebra
using MCMCChains
using Printf
using Random
import DynamicPPL
import LogDensityProblems
import ReverseDiff
import Turing

LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l08_workflow.jl"))
include(joinpath(@__DIR__, "l08_incident_data.jl"))
include(joinpath(@__DIR__, "l08_decomposed_models.jl"))
include(joinpath(@__DIR__, "l08_model_checks.jl"))

const R08P_MODEL = get(ENV, "L08_PROBE_MODEL", "m00_recombined_control")
const R08P_OUT = joinpath(@__DIR__, "results", "sampling_budget_fold1", "gradient_path_probe_" * R08P_MODEL)
const R08P_ADTYPE = Turing.AutoReverseDiff(compile = true)  # the binding run_sampler passes to `sample`
mkpath(R08P_OUT)

# %%
# ==============================================================================
# 2. The Fold 1 FeatureSet, exactly as the benchmark builds it
# ==============================================================================
l08_load_runtime_env!()
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)
splitter = l08_splitter()
incident_registry, incident_snapshot_hash = GoalDecompositionIncidentData.load_registry(joinpath(@__DIR__, "results"))
entry = only(e for e in l08_models(incident_registry, incident_snapshot_hash;
                                    registry_hash = incident_snapshot_hash)
             if String(e.name) == R08P_MODEL)
model = entry.model
boundary = Data.create_id_boundaries(ds, splitter)[1:1]
original_features = Features.create_features(boundary, ds, model, splitter)
oos = [Data.get_next_matches(ds, feature, splitter) for feature in original_features]
fs = first(first(l08_declare_prediction_teams(original_features, oos)))

"Mean bytes and seconds per call of `f`, after a warm-up call and a full collection."
function r08p_bench(f, n)
    f()
    GC.gc()
    bytes = @allocated for _ in 1:n
        f()
    end
    seconds = @elapsed for _ in 1:n
        f()
    end
    return (bytes = bytes / n, seconds = seconds / n)
end

# %%
# ==============================================================================
# 3. The three measurements
# ==============================================================================
p = l08_logdensity_problem(model, fs)
tape = ReverseDiff.compile(ReverseDiff.GradientTape(p.f, p.θ))
g = similar(p.θ)
own = r08p_bench(() -> ReverseDiff.gradient!(g, tape, p.θ), 2_000)

ldf_ad = DynamicPPL.LogDensityFunction(p.turing_model, DynamicPPL.getlogjoint_internal, p.vi;
                                       adtype = R08P_ADTYPE)
ld_own, _ = LogDensityProblems.logdensity_and_gradient(ldf_ad, p.θ)
isapprox(ld_own, p.f(p.θ); rtol = 1e-10) || error("sampler-path log density disagrees with the l08 problem")
sampler_grad = r08p_bench(() -> LogDensityProblems.logdensity_and_gradient(ldf_ad, p.θ), 2_000)
@printf("  l08 compiled tape          : %10.0f bytes  %8.4f ms per gradient\n", own.bytes, 1e3 * own.seconds)
@printf("  sampler adtype gradient    : %10.0f bytes  %8.4f ms per gradient\n", sampler_grad.bytes, 1e3 * sampler_grad.seconds)
flush(stdout)

# Per-iteration work Turing 0.41's `AbstractMCMC.step(::Hamiltonian)` does outside the
# trajectory: `Transition(model, vi, t)` deep-copies the VarInfo and re-evaluates the
# whole model with `ValuesAsInModelAccumulator(true)` to record values.
plain_ld = r08p_bench(() -> LogDensityProblems.logdensity(ldf_ad, p.θ), 200)
copy_vi = r08p_bench(() -> deepcopy(p.vi), 200)
transition_eval() = DynamicPPL.evaluate!!(p.turing_model, DynamicPPL.setaccs!!(deepcopy(p.vi), (
    DynamicPPL.ValuesAsInModelAccumulator(true),
    DynamicPPL.LogPriorAccumulator(),
    DynamicPPL.LogLikelihoodAccumulator())))
transition = r08p_bench(transition_eval, 50)
values_recorded = length(DynamicPPL.getacc(last(transition_eval()), Val(:ValuesAsInModel)).values)
@printf("  plain logdensity (no AD)   : %10.0f bytes  %8.4f ms per call\n", plain_ld.bytes, 1e3 * plain_ld.seconds)
@printf("  deepcopy(varinfo)          : %10.0f bytes  %8.4f ms per call\n", copy_vi.bytes, 1e3 * copy_vi.seconds)
@printf("  Transition re-evaluation   : %10.0f bytes  %8.4f ms per call (%d recorded values)\n",
    transition.bytes, 1e3 * transition.seconds, values_recorded)
flush(stdout)

# Bare AdvancedHMC: the Hamiltonian and NUTS kernel exactly as Turing's `initialstep`
# builds them, stepped with `AHMC.transition` and no Turing wrapper around it.
const R08P_AHMC = Turing.Inference.AHMC
ahmc_h = R08P_AHMC.Hamiltonian(R08P_AHMC.DiagEuclideanMetric(length(p.θ)),
    Base.Fix1(LogDensityProblems.logdensity, ldf_ad),
    Base.Fix1(LogDensityProblems.logdensity_and_gradient, ldf_ad))
ahmc_rng = MersenneTwister(20_260_910)
ahmc_eps = R08P_AHMC.find_good_stepsize(ahmc_rng, ahmc_h, copy(p.θ))
ahmc_kernel = Turing.Inference.make_ahmc_kernel(Turing.NUTS(0, 0.95; adtype = R08P_ADTYPE), ahmc_eps)
function r08p_ahmc_run(n)
    z = R08P_AHMC.phasepoint(ahmc_rng, copy(p.θ), ahmc_h)
    steps = 0
    for _ in 1:n
        t = R08P_AHMC.transition(ahmc_rng, ahmc_h, ahmc_kernel, z)
        z = t.z
        steps += t.stat.n_steps
    end
    return steps
end
r08p_ahmc_run(5)
GC.gc()
ahmc_bytes = @allocated ahmc_steps = r08p_ahmc_run(200)
ahmc_seconds = @elapsed ahmc_steps_timed = r08p_ahmc_run(200)
@printf("  bare AHMC.transition       : %10.0f bytes  %8.4f ms per leapfrog (%d / %d leapfrogs, ε %.3g)\n",
    ahmc_bytes / ahmc_steps, 1e3 * ahmc_seconds / ahmc_steps_timed, ahmc_steps, ahmc_steps_timed, ahmc_eps)
flush(stdout)

# Turing's own state: step the sampler by hand, time the gradient of the Hamiltonian
# Turing built (from its own VarInfo), and time whole `AbstractMCMC.step` calls.
const R08P_AMCMC = parentmodule(Turing.Inference.AbstractSampler)
turing_spl = Turing.NUTS(0, 0.95; adtype = R08P_ADTYPE)
turing_rng = MersenneTwister(20_260_910)
_, turing_state = R08P_AMCMC.step(turing_rng, p.turing_model, turing_spl;
                                  initial_params = DynamicPPL.InitFromUniform())
turing_ldf = turing_state.hamiltonian.∂ℓπ∂θ.x
turing_grad = r08p_bench(() -> turing_state.hamiltonian.∂ℓπ∂θ(turing_state.z.θ), 500)
@printf("  Turing-state gradient      : %10.0f bytes  %8.4f ms per gradient\n", turing_grad.bytes, 1e3 * turing_grad.seconds)
println("  same varinfo type as probe : ", typeof(turing_ldf.varinfo) == typeof(p.vi))
println("  same prep type as probe    : ", typeof(turing_ldf.prep) == typeof(ldf_ad.prep))
println("  Turing varinfo type        : ", first(string(typeof(turing_ldf.varinfo)), 400))
println("  probe  varinfo type        : ", first(string(typeof(p.vi)), 400))
function r08p_turing_steps(n)
    state = turing_state
    steps = 0
    for _ in 1:n
        t, state = R08P_AMCMC.step(turing_rng, p.turing_model, turing_spl, state)
        steps += t.stat.n_steps
    end
    return steps
end
r08p_turing_steps(3)
GC.gc()
turing_step_bytes = @allocated turing_step_leapfrogs = r08p_turing_steps(50)
@printf("  AbstractMCMC.step (Turing) : %10.0f bytes per leapfrog, %.1f MiB per step (%d leapfrogs / 50 steps)\n",
    turing_step_bytes / turing_step_leapfrogs, turing_step_bytes / 50 / 2^20, turing_step_leapfrogs)
flush(stdout)

# Whole-sampler cost per leapfrog step. `NUTS(0, δ)` disables adaptation so every
# transition is retained and its `n_steps` recorded; the first call compiles.
# `nuts_run` passes `adtype` to `sample` exactly as `Samplers.run_sampler` does. Turing
# 0.41 reads the backend only from the sampler object (`spl.adtype`), so this silently
# runs the `NUTS` default, AutoForwardDiff. `nuts_fixed_run` puts it in the constructor.
nuts_run() = Turing.sample(p.turing_model, Turing.NUTS(0, 0.95), 200;
                           adtype = R08P_ADTYPE, progress = false)
nuts_fixed_run() = Turing.sample(p.turing_model, Turing.NUTS(0, 0.95; adtype = R08P_ADTYPE), 200;
                                 progress = false)
println("  NUTS default adtype        : ", Turing.NUTS(0, 0.95).adtype)
Random.seed!(20_260_910)
nuts_fixed_run()
GC.gc()
Random.seed!(20_260_910)
fixed_gc0 = Base.gc_num()
fixed_seconds = @elapsed fixed_chain = nuts_fixed_run()
fixed_gc = Base.GC_Diff(Base.gc_num(), fixed_gc0)
fixed_leapfrogs = Int(sum(skipmissing(vec(Array(fixed_chain[:n_steps])))))
@printf("  sample, adtype in NUTS()   : %10.0f bytes  %8.4f ms per leapfrog (%d leapfrogs, %.1f s, GC %.1f s)\n",
    fixed_gc.allocd / fixed_leapfrogs, 1e3 * fixed_seconds / fixed_leapfrogs, fixed_leapfrogs,
    fixed_seconds, fixed_gc.total_time / 1e9)
flush(stdout)
Random.seed!(20_260_910)
nuts_run()
GC.gc()
Random.seed!(20_260_910)
gc0 = Base.gc_num()
nuts_seconds = @elapsed chain = nuts_run()
gc1 = Base.gc_num()
nuts_gc = Base.GC_Diff(gc1, gc0)
steps = vec(Array(chain[:n_steps]))
n_missing_steps = count(ismissing, steps)
leapfrogs = Int(sum(skipmissing(steps)))
iterations = size(chain, 1)
n_missing_steps == 0 || @warn "$n_missing_steps of $iterations transitions record no n_steps; per-leapfrog figures use the rest"

rows = [
    (path = "l08 compiled tape (ReverseDiff.gradient!)", bytes_per_gradient = own.bytes,
     ms_per_gradient = 1e3 * own.seconds, gradients = 2_000),
    (path = "sampler adtype (logdensity_and_gradient)", bytes_per_gradient = sampler_grad.bytes,
     ms_per_gradient = 1e3 * sampler_grad.seconds, gradients = 2_000),
    (path = "Turing.sample, adtype as sample kwarg (run_sampler) per leapfrog", bytes_per_gradient = nuts_gc.allocd / leapfrogs,
     ms_per_gradient = 1e3 * nuts_seconds / leapfrogs, gradients = leapfrogs),
    (path = "Turing.sample, adtype in NUTS constructor per leapfrog", bytes_per_gradient = fixed_gc.allocd / fixed_leapfrogs,
     ms_per_gradient = 1e3 * fixed_seconds / fixed_leapfrogs, gradients = fixed_leapfrogs),
    (path = "AbstractMCMC.step by hand, adtype in NUTS constructor per leapfrog", bytes_per_gradient = turing_step_bytes / turing_step_leapfrogs,
     ms_per_gradient = NaN, gradients = turing_step_leapfrogs),
    (path = "bare AHMC.transition per leapfrog", bytes_per_gradient = ahmc_bytes / ahmc_steps,
     ms_per_gradient = 1e3 * ahmc_seconds / ahmc_steps_timed, gradients = ahmc_steps),
    (path = "per call: plain logdensity (no AD)", bytes_per_gradient = plain_ld.bytes,
     ms_per_gradient = 1e3 * plain_ld.seconds, gradients = 200),
    (path = "per call: deepcopy(varinfo)", bytes_per_gradient = copy_vi.bytes,
     ms_per_gradient = 1e3 * copy_vi.seconds, gradients = 200),
    (path = "per iteration: Transition re-evaluation", bytes_per_gradient = transition.bytes,
     ms_per_gradient = 1e3 * transition.seconds, gradients = 50),
    (path = "per iteration: Turing.sample, adtype as sample kwarg", bytes_per_gradient = nuts_gc.allocd / iterations,
     ms_per_gradient = 1e3 * nuts_seconds / iterations, gradients = iterations),
]
probe = DataFrame(rows)
CSV.write(joinpath(R08P_OUT, "gradient_path_probe.csv"), probe)

println("\n", "="^100)
println(" GRADIENT-PATH ALLOCATION PROBE · ", R08P_MODEL, " · fold 1 · 1 thread · Julia ", VERSION)
println("="^100)
for r in eachrow(probe)
    @printf("  %-68s %12.0f bytes  %9.4f ms  (n = %d)\n", r.path, r.bytes_per_gradient, r.ms_per_gradient, r.gradients)
end
@printf("  NUTS run: %d iterations, %d leapfrog steps, %.1f s, %.2f GiB allocated, GC %.1f s (%.0f%%), %d pauses\n",
    iterations, leapfrogs, nuts_seconds, nuts_gc.allocd / 2^30, nuts_gc.total_time / 1e9,
    100 * nuts_gc.total_time / 1e9 / nuts_seconds, nuts_gc.pause)
@printf("  allocation per NUTS iteration: %.1f MiB\n", nuts_gc.allocd / iterations / 2^20)
println("Outputs in ", R08P_OUT)
