# ==============================================================================
# r01 — Smoke gate: HierarchicalTeamHomeAdvantage × three production tiers, folds 1–2
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# A mechanical gate, not a result. It answers one question per candidate: can the
# hierarchical home advantage be compiled, sampled, audited, extracted and persisted
# end to end beside each production pillar? Nothing printed here is a proper score.
#
# The one new composition is a `(n_teams)` vector HA gathered by `home_ids` beside
# the (team, time) GRW state and the lineup pillar. The non-centred z_i sit beside
# a scale σ_γ whose prior puts most mass below 0.1, which is the classic funnel
# geometry — divergences, not a tape failure, are the likely way this breaks.
#
# GATES — every one must pass for every candidate before `r02` may be launched:
#
#   G1  ReverseDiff tape compiles; compiled == fresh RD ≤ 1e-8; RD == ForwardDiff
#       ≤ 1e-6; compiled tape correct at three perturbed points; folds 1 AND 2.
#       Allocation per warmed gradient is measured beside the flat-HA twin on the same
#       fold. The work package asks for zero; Task 013 measured 35–130 KB for every
#       model on this ReverseDiff stack, so the gate is Δalloc(hier − flat) = 0 —
#       the hierarchical slot must not ADD allocation — and the absolute is reported.
#   G2  zero divergences across all chains
#   G3  max R̂ ≤ 1.05 over every site AND over every `ha.*` site named explicitly;
#       bulk/tail ESS ≥ 400 (the 4 × 400 budget holds 1,600 draws per fold)
#   G4  CountLatents extract with finite, strictly positive λ draws and finite,
#       non-zero per-fixture means and variances
#   G5  save_fit → load_fit round-trip through PostgresStorage("smoke_hier_ha")
#       reproduces fold count, latents and every chain value exactly
#
# Also reported, not gated: held-out fixtures whose home club is absent from the
# fold's team map (priced at γ = 0 by the hierarchical extraction — ticket T003).
#
# PERSISTENCE CAVEAT. Each invocation writes three runs into `smoke_hier_ha`, named
# with a per-invocation suffix so `save_fit`'s config-hash deduplication can never
# hand back an older smoke run's UUID for the round-trip comparison.
#
# USAGE (mcmc-beast, from /root/BF_hier_ha)
#
#   /root/.juliaup/bin/julia --project -t 16 current_development/hierarchical_home_advantage/r01_smoke.jl
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
const R01_CONFIG = let env(k, d) = parse(Int, get(ENV, k, string(d)))
    base = HHAConfig()
    HHAConfig(smoke_samples = env("R01_SAMPLES", base.smoke_samples),
              smoke_warmup = env("R01_WARMUP", base.smoke_warmup),
              smoke_chains = env("R01_CHAINS", base.smoke_chains))
end
const R01_GATE_ESS = R01_CONFIG.smoke_chains * R01_CONFIG.smoke_samples >= 2 * R01_CONFIG.min_ess
const R01_SUFFIX = "_smoke_" * Dates.format(now(), "yyyymmddHHMMSS")
const R01_OUT_DIR = joinpath(R01_CONFIG.save_root, "smoke",
    "$(R01_CONFIG.smoke_chains)x$(R01_CONFIG.smoke_warmup)w$(R01_CONFIG.smoke_samples)s")
const R01_GIT = try readchomp(`git rev-parse --short HEAD`) catch; "unsynced-rsync" end

println("\n" * "="^96)
println("  r01 SMOKE GATE — HierarchicalTeamHomeAdvantage, folds 1–$(R01_CONFIG.smoke_folds)")
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
r01_models = hha_models()
r01_flat_twins = Dict(hha_models(home_advantage = GlobalHomeAdvantage()))
r01_sampler = hha_smoke_sampler(R01_CONFIG)
r01_configs = hha_fit_configs(R01_CONFIG, r01_models, r01_splitter, r01_sampler;
                              name_suffix = R01_SUFFIX)
for (name, model) in r01_models
    println("  ", rpad(name, 38), " HA: ", nameof(typeof(model.home_advantage)),
            " | dynamics: ", nameof(typeof(model.dynamics)),
            " | covariates: ", join(string.(nameof.(typeof.(model.covariates))), ", "))
end

# %%
# ===================================================================
# 6. Feature construction and preflight gates (G1)
# ===================================================================
r01_rows = NamedTuple[]
r01_gradients = NamedTuple[]
r01_inputs = Dict{String,Any}()
r01_unmapped = DataFrame[]

for (name, model) in r01_models
    println("\n--- features + gradient audit: ", name)
    inputs = gph_fold_inputs(r01_ds, r01_splitter, model; limit = R01_CONFIG.smoke_folds)
    r01_inputs[name] = inputs
    filtration = gph_filtration_report(r01_ds, inputs)
    show(stdout, MIME"text/plain"(), filtration; allcols = true)
    println()
    all(filtration.ordered) || error("$name: a fold's last training kickoff is not before its first OOS kickoff")

    unmapped = hha_unmapped_home_report(inputs)
    unmapped.model = fill(name, nrow(unmapped))
    push!(r01_unmapped, unmapped)
    println("  held-out fixtures with an unmapped home club: ", nrow(unmapped))

    flat = r01_flat_twins[name]
    for (fold, fs) in enumerate(inputs.feature_sets)
        audit = gph_gradient_audit(model, fs; replays = R01_CONFIG.gradient_replays)
        twin = gph_gradient_audit(flat, fs; replays = R01_CONFIG.gradient_replays)
        push!(r01_gradients, (; model = name, fold,
                                n_target = Int(first(fs).data[:n_target_steps]),
                                n_teams = Int(first(fs).data[:n_teams]),
                                audit...,
                                flat_n_parameters = twin.n_parameters,
                                flat_tape_instructions = twin.tape_instructions,
                                flat_gradient_ms = twin.gradient_ms,
                                flat_allocated_bytes = twin.allocated_bytes,
                                delta_allocated_bytes = audit.allocated_bytes - twin.allocated_bytes))
        @printf("  fold %d  θ=%4d (+%d)  tape=%5d (+%d)  grad=%.3f ms (flat %.3f)  alloc=%7d B (Δ %+d)  RD/FD=%.1e  perturbed=%.1e\n",
                fold, audit.n_parameters, audit.n_parameters - twin.n_parameters,
                audit.tape_instructions, audit.tape_instructions - twin.tape_instructions,
                audit.gradient_ms, twin.gradient_ms,
                audit.allocated_bytes, audit.allocated_bytes - twin.allocated_bytes,
                audit.compiled_forward_error, audit.worst_perturbed_error)
    end
end

# %%
# ===================================================================
# 7. Training — two folds per candidate
# ===================================================================
r01_fits = Dict{String,Any}()
for (name, _) in r01_models
    println("\n--- sampling: ", name)
    r01_fits[name] = hha_sample(r01_configs[name], r01_inputs[name], R01_CONFIG)
end

# %%
# ===================================================================
# 8. Convergence diagnostics (G2, G3), latents (G4), round-trip (G5)
# ===================================================================
r01_gradient_frame = DataFrame(r01_gradients)
r01_ha_sites = DataFrame[]
r01_ground = DataFrame[]
r01_hyper = NamedTuple[]

for (name, _) in r01_models
    fit = r01_fits[name]
    d = fit.diagnostics
    failures = String[]

    grads = filter(:model => ==(name), r01_gradient_frame)
    all(grads.delta_allocated_bytes .== 0) || push!(failures,
        "G1 hierarchical HA adds allocation: Δ = " * join(grads.delta_allocated_bytes, ", ") * " B")

    d.n_divergent == 0 || push!(failures, "G2 divergences=$(d.n_divergent)")
    d.max_rhat <= R01_CONFIG.max_rhat || push!(failures,
        @sprintf("G3 max R̂ %.4f at %s (fold %d)", d.max_rhat,
                 d.folds[d.worst_rhat_fold].worst_rhat_param, d.worst_rhat_fold))
    R01_GATE_ESS && min(d.min_ess_bulk, d.min_ess_tail) < R01_CONFIG.min_ess && push!(failures,
        @sprintf("G3 min ESS bulk %.0f / tail %.0f", d.min_ess_bulk, d.min_ess_tail))

    sites = hha_ha_site_report(fit)
    sites.model = fill(name, nrow(sites))
    push!(r01_ha_sites, sites)
    worst_ha = sites[argmax(sites.max_rhat), :]
    worst_ha.max_rhat <= R01_CONFIG.max_rhat || push!(failures,
        @sprintf("G3 ha site %s R̂ %.4f (fold %d)", worst_ha.site, worst_ha.max_rhat,
                 worst_ha.worst_rhat_fold))

    for fold in eachindex(fit.folds)
        clubs, hyper = hha_ground_effects(fit, fold, r01_inputs[name].feature_sets[fold])
        clubs.model = fill(name, nrow(clubs))
        clubs.fold = fill(fold, nrow(clubs))
        push!(r01_ground, clubs)
        push!(r01_hyper, (; model = name, hyper...))
    end

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

    push!(r01_rows, (; model = name,
                       folds = length(fit.folds),
                       oos = fit.latents === nothing ? 0 : n_matches(fit.latents),
                       draws = fit.latents === nothing ? 0 : n_draws(fit.latents),
                       max_rhat = d.max_rhat,
                       max_ha_rhat = worst_ha.max_rhat,
                       min_ha_ess_bulk = minimum(sites.min_ess_bulk),
                       min_ess_bulk = d.min_ess_bulk,
                       min_ess_tail = d.min_ess_tail,
                       n_divergent = d.n_divergent,
                       min_bfmi = d.min_bfmi,
                       treedepth_rate = d.treedepth_rate,
                       latent_min_sd = latent === nothing ? NaN : latent.min_sd,
                       wall_min = fit.metadata.elapsed_seconds / 60,
                       gate_pass = isempty(failures),
                       gate_failures = join(failures, "; "),
                       run_id = run_id === nothing ? "" : string(run_id)))
    @printf("  %-38s R̂=%.4f  ha R̂=%.4f  ESS bulk=%6.1f tail=%6.1f  div=%d  latents=%s  run=%s  %s\n",
            name, d.max_rhat, worst_ha.max_rhat, d.min_ess_bulk, d.min_ess_tail, d.n_divergent,
            latent === nothing ? "FAIL" : string(latent.n_matches, "×", latent.n_draws),
            run_id === nothing ? "FAIL" : string(run_id),
            isempty(failures) ? "PASS" : "FAIL: " * join(failures, "; "))
end

r01_hyper_frame = DataFrame(r01_hyper)
println("\n=== HOME-ADVANTAGE HYPERPARAMETERS (per fold) ===")
show(stdout, MIME"text/plain"(),
     select(r01_hyper_frame, :model, :fold, :gamma_base_mean, :gamma_base_sd,
            :sigma_q05, :sigma_q50, :sigma_q95, :p_sigma_below_0p02, :prior_p_sigma_below_0p02);
     allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 9. Final report
# ===================================================================
r01_summary = DataFrame(r01_rows)
r01_site_frame = vcat(r01_ha_sites...)
r01_ground_frame = vcat(r01_ground...)
r01_unmapped_frame = vcat(r01_unmapped...)
CSV.write(joinpath(R01_OUT_DIR, "r01_smoke_gates.csv"), r01_summary)
CSV.write(joinpath(R01_OUT_DIR, "r01_gradient_audit.csv"), r01_gradient_frame)
CSV.write(joinpath(R01_OUT_DIR, "r01_ha_sites.csv"), r01_site_frame)
CSV.write(joinpath(R01_OUT_DIR, "r01_ground_effects.csv"), r01_ground_frame)
CSV.write(joinpath(R01_OUT_DIR, "r01_ha_hyper.csv"), r01_hyper_frame)
CSV.write(joinpath(R01_OUT_DIR, "r01_unmapped_home.csv"), r01_unmapped_frame)

open(joinpath(R01_OUT_DIR, "r01_smoke_report.md"), "w") do io
    println(io, "# r01 smoke gate — Task 008 Phase 1\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R01_GIT,
            "` on ", gethostname(), " with ", Threads.nthreads(), " threads.\n")
    println(io, "Sampler: QueuedNUTS, ", R01_CONFIG.smoke_chains, " chains × ",
            R01_CONFIG.smoke_warmup, " warmup + ", R01_CONFIG.smoke_samples,
            " retained, δ = ", R01_CONFIG.accept_rate, ". Folds 1–", R01_CONFIG.smoke_folds, ".\n")
    println(io, "## Gradient audit (G1), hierarchical vs flat twin on the same fold\n")
    print(io, gph_markdown_table(select(r01_gradient_frame,
        :model, :fold, :n_teams, :n_parameters, :flat_n_parameters, :tape_instructions,
        :flat_tape_instructions, :gradient_ms, :flat_gradient_ms, :allocated_bytes,
        :flat_allocated_bytes, :delta_allocated_bytes, :compiled_forward_error, :worst_perturbed_error);
        formats = Dict(:gradient_ms => v -> gph_num(v; digits = 3),
                       :flat_gradient_ms => v -> gph_num(v; digits = 3),
                       :compiled_forward_error => v -> @sprintf("%.1e", v),
                       :worst_perturbed_error => v -> @sprintf("%.1e", v))))
    println(io, "\n## Sampling, latents and persistence (G2–G5)\n")
    print(io, gph_markdown_table(select(r01_summary,
        :model, :folds, :oos, :draws, :max_rhat, :max_ha_rhat, :min_ess_bulk, :min_ess_tail,
        :n_divergent, :min_bfmi, :wall_min, :gate_pass, :run_id)))
    println(io, "\n## HA hyperparameters\n")
    print(io, gph_markdown_table(select(r01_hyper_frame, :model, :fold, :gamma_base_mean,
        :gamma_base_sd, :sigma_q05, :sigma_q50, :sigma_q95, :p_sigma_below_0p02)))
    println(io, "\n## Held-out fixtures with an unmapped home club (T003)\n")
    println(io, nrow(r01_unmapped_frame) == 0 ? "None.\n" :
        gph_markdown_table(r01_unmapped_frame))
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
