# ==============================================================================
# r01 — Smoke gate: MultiScaleGRW × PlayerLineupPillar, folds 1–2
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# A mechanical gate, not a result. It answers one question per ladder model: can
# this composition be compiled, sampled, audited, extracted and persisted end to
# end? Nothing it prints is a proper score, and 100 retained draws per chain
# cannot say anything about convergence at production scale beyond "no geometry
# pathology is visible yet".
#
# The one genuinely new composition is `PlayerLineupPillar` beside the (team, time)
# indexed GRW state. The builder adds the pillar's `(h, a)` shift after the dynamics
# gather, so the pillar never sees the time axis — but that is exactly the kind of
# claim a smoke run exists to check rather than assume.
#
# GATES — every one must pass for every model before `r02` may be launched:
#
#   G1  ReverseDiff tape compiles; compiled == fresh RD ≤ 1e-8; RD == ForwardDiff
#       ≤ 1e-6; compiled tape correct at three perturbed points (folds 1 AND 2 —
#       fold 1 has no target steps, fold 2 does, so both GRW branches are taped)
#   G2  zero divergences across all chains
#   G3  max R̂ < 1.05 across all sites
#   G4  CountLatents extract with finite, strictly positive λ draws and finite,
#       non-zero per-fixture means and variances
#   G5  save_fit → load_fit round-trip through PostgresStorage("smoke_grw_player")
#       reproduces fold count, latents and every chain value exactly
#
#   Advisory: bulk/tail ESS. With 2 chains × 100 draws there are 200 draws in total,
#   so ESS ≥ 400 is arithmetically unreachable at this budget; it is reported, not
#   gated.
#
# PERSISTENCE CAVEAT. Each invocation writes four runs into `smoke_grw_player`,
# named with a per-invocation suffix so `save_fit`'s config-hash deduplication can
# never hand back an older smoke run's UUID for the round-trip comparison.
#
# USAGE (mcmc-beast, from /root/BF_grw_player_hybrid)
#
#   /root/.juliaup/bin/julia --project -t 16 current_development/grw_player_hybrid/r01_smoke.jl
# ==============================================================================

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball
using CSV
using DataFrames
using Dates
using Printf

include(joinpath(@__DIR__, "l01_loader.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R01_CONFIG = GPHConfig()
const R01_SUFFIX = "_smoke_" * Dates.format(now(), "yyyymmddHHMMSS")
const R01_OUT_DIR = joinpath(R01_CONFIG.save_root, "smoke")
const R01_GIT = try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end

println("\n" * "="^96)
println("  r01 SMOKE GATE — MultiScaleGRW × PlayerLineupPillar, folds 1–$(R01_CONFIG.smoke_folds)")
println("  sampler    : QueuedNUTS  $(R01_CONFIG.smoke_chains) chains × ",
        "$(R01_CONFIG.smoke_warmup) warmup + $(R01_CONFIG.smoke_samples) retained, ",
        "δ = $(R01_CONFIG.accept_rate), max depth $(R01_CONFIG.max_depth)")
println("  experiment : ", R01_CONFIG.smoke_experiment, "   (name suffix ", R01_SUFFIX, ")")
println("  git        : ", R01_GIT, "   threads: ", Threads.nthreads())
println("="^96)

# %%
# ===================================================================
# 3. Runtime and output directory
# ===================================================================
mkpath(R01_OUT_DIR)
r01_db = gph_database(R01_CONFIG.smoke_experiment)

# %%
# ===================================================================
# 4. Data snapshot and temporal splits
# ===================================================================
r01_ds = gph_load_data()
r01_splitter = gph_splitter(R01_CONFIG.target_seasons)
println("\n  matches in store : ", nrow(r01_ds.matches),
        "   latest kickoff: ", maximum(r01_ds.matches.match_date))

# %%
# ===================================================================
# 5. Engine / model construction
# ===================================================================
r01_models = gph_models()
r01_sampler = gph_smoke_sampler(R01_CONFIG)
r01_configs = gph_fit_configs(R01_CONFIG, r01_models, r01_splitter, r01_sampler;
                              name_suffix = R01_SUFFIX)
for (name, model) in r01_models
    println("  ", rpad(name, 32), " covariates: ",
            join(string.(nameof.(typeof.(model.covariates))), ", "),
            " | obs: ", nameof(typeof(model.observation)))
end

# %%
# ===================================================================
# 6. Feature construction and preflight gates (G1)
# ===================================================================
r01_rows = NamedTuple[]
r01_gradients = NamedTuple[]
r01_inputs = Dict{String,Any}()

for (name, model) in r01_models
    println("\n--- features + gradient audit: ", name)
    inputs = gph_fold_inputs(r01_ds, r01_splitter, model; limit = R01_CONFIG.smoke_folds)
    r01_inputs[name] = inputs
    filtration = gph_filtration_report(r01_ds, inputs)
    show(stdout, MIME"text/plain"(), filtration; allcols = true)
    println()
    all(filtration.ordered) || error("$name: a fold's last training kickoff is not before its first OOS kickoff")

    for (fold, fs) in enumerate(inputs.feature_sets)
        audit = gph_gradient_audit(model, fs; replays = R01_CONFIG.gradient_replays)
        push!(r01_gradients, (; model = name, fold,
                                n_target = Int(first(fs).data[:n_target_steps]),
                                audit...))
        @printf("  fold %d  θ=%4d  tape=%5d  grad=%.3f ms  alloc=%7d B  RD/FD=%.1e  perturbed=%.1e\n",
                fold, audit.n_parameters, audit.tape_instructions, audit.gradient_ms,
                audit.allocated_bytes, audit.compiled_forward_error, audit.worst_perturbed_error)
    end
end

# %%
# ===================================================================
# 7. Training — two folds per model
# ===================================================================
r01_fits = Dict{String,Any}()
for (name, _) in r01_models
    println("\n--- sampling: ", name)
    r01_fits[name] = gph_sample(r01_configs[name], r01_inputs[name], R01_CONFIG)
end

# %%
# ===================================================================
# 8. Convergence diagnostics (G2, G3), latents (G4), round-trip (G5)
# ===================================================================
for (name, _) in r01_models
    fit = r01_fits[name]
    d = fit.diagnostics
    failures = String[]

    d.n_divergent == 0 || push!(failures, "G2 divergences=$(d.n_divergent)")
    d.max_rhat < R01_CONFIG.max_rhat || push!(failures,
        @sprintf("G3 max R̂ %.4f (fold %d)", d.max_rhat, d.worst_rhat_fold))

    latent = try
        gph_latent_audit(fit)
    catch err
        push!(failures, "G4 " * sprint(showerror, err))
        nothing
    end

    run_id = try
        gph_save_and_verify(r01_db, fit)
    catch err
        push!(failures, "G5 " * sprint(showerror, err))
        nothing
    end

    row = gph_convergence_row(name, fit, R01_CONFIG; run_id)
    push!(r01_rows, (; row...,
                       latent_mean_lambda_h = latent === nothing ? NaN : latent.mean_lambda_h,
                       latent_mean_lambda_a = latent === nothing ? NaN : latent.mean_lambda_a,
                       latent_min_sd = latent === nothing ? NaN : latent.min_sd,
                       gate_pass = isempty(failures),
                       gate_failures = join(failures, "; ")))
    @printf("  %-32s R̂=%.4f  ESS bulk=%6.1f tail=%6.1f  div=%d  latents=%s  run=%s  %s\n",
            name, d.max_rhat, d.min_ess_bulk, d.min_ess_tail, d.n_divergent,
            latent === nothing ? "FAIL" : string(latent.n_matches, "×", latent.n_draws),
            run_id === nothing ? "FAIL" : string(run_id),
            isempty(failures) ? "PASS" : "FAIL: " * join(failures, "; "))
end

# %%
# ===================================================================
# 9. Final report
# ===================================================================
r01_summary = DataFrame(r01_rows)
r01_gradient_frame = DataFrame(r01_gradients)
CSV.write(joinpath(R01_OUT_DIR, "r01_smoke_gates.csv"), r01_summary)
CSV.write(joinpath(R01_OUT_DIR, "r01_gradient_audit.csv"), r01_gradient_frame)

open(joinpath(R01_OUT_DIR, "r01_smoke_report.md"), "w") do io
    println(io, "# r01 smoke gate — Task 013\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R01_GIT,
            "` on ", gethostname(), " with ", Threads.nthreads(), " threads.\n")
    println(io, "Sampler: QueuedNUTS, ", R01_CONFIG.smoke_chains, " chains × ",
            R01_CONFIG.smoke_warmup, " warmup + ", R01_CONFIG.smoke_samples,
            " retained, δ = ", R01_CONFIG.accept_rate, ". Folds 1–", R01_CONFIG.smoke_folds, ".\n")
    println(io, "## Gradient audit (G1)\n")
    print(io, gph_markdown_table(select(r01_gradient_frame,
        :model, :fold, :n_target, :n_parameters, :tape_instructions, :gradient_ms,
        :allocated_bytes, :compiled_forward_error, :worst_perturbed_error);
        formats = Dict(:gradient_ms => v -> gph_num(v; digits = 3),
                       :compiled_forward_error => v -> @sprintf("%.1e", v),
                       :worst_perturbed_error => v -> @sprintf("%.1e", v))))
    println(io, "\n## Sampling, latents and persistence (G2–G5)\n")
    print(io, gph_markdown_table(select(r01_summary,
        :model, :folds, :oos, :draws, :max_rhat, :min_ess_bulk, :min_ess_tail,
        :n_divergent, :latent_min_sd, :gate_pass, :run_id)))
    failed = filter(:gate_pass => !, r01_summary)
    if nrow(failed) > 0
        println(io, "\n### Failures\n")
        for r in eachrow(failed)
            println(io, "* `", r.model, "` — ", r.gate_failures)
        end
    end
end

r01_verdict = all(r01_summary.gate_pass)
println("\nR01_VERDICT ", r01_verdict ? "PASS" : "FAIL", "  (",
        count(r01_summary.gate_pass), "/", nrow(r01_summary), " models)")
println("R01_DONE report=", joinpath(R01_OUT_DIR, "r01_smoke_report.md"))
