# ==============================================================================
# r01 — Smoke gate: fast & slow Poisson MultiScaleGRW arms, folds 1 / 20 / 40
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# A mechanical gate plus one early directional read. Three folds (~50 held-out
# fixtures) cannot settle whether rate pooling beats the handrail; they can say
# whether each arm compiles, samples, audits, persists and prices coherently,
# and whether the loose arms move supremacy in the intended direction at all.
#
# GATES — every one must pass for every arm before `r02` may be launched:
#
#   G1  ReverseDiff tape compiles; compiled == fresh RD ≤ 1e-8; RD == ForwardDiff
#       ≤ 1e-6; stable under perturbation (every smoke fold — fold 1 has no
#       target steps, so both GRW branches are taped)
#   G2  zero divergences
#   G3  max R̂ ≤ 1.05
#   G4  min bulk ESS ≥ 200 (reported target 300)
#   G5  save_fit → load_fit round-trip through PostgresStorage("smoke_fast_slow_grw")
#       reproduces folds, latents and every chain value exactly
#   G6  the loose arms' supremacy slope vs the inverted Betfair close exceeds the
#       tight arm's on the same fixtures (directional; 3 folds)
#   G7  rate-pooled latents (tight ⊕ each loose arm, w = 0.4) price 1X2, O/U 2.5
#       and BTTS with every probability in [0, 1] and every book summing, draw by
#       draw, to the 0–11 goal grid's retained Poisson mass within 1e-10
#
# USAGE (mcmc-beast, from /root/BF_fast_slow_grw)
#
#   /root/.juliaup/bin/julia --project -t 16 current_development/fast_slow_grw/r01_fast_slow_smoke.jl
#   R01_SAMPLES=100 R01_WARMUP=100 R01_CHAINS=2 julia ... r01_fast_slow_smoke.jl   # quick mechanics
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

include(joinpath(@__DIR__, "l01_fast_slow_grw_loader.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R01_CONFIG = let env(k, d) = parse(Int, get(ENV, k, string(d)))
    base = FSGConfig()
    FSGConfig(smoke_samples = env("R01_SAMPLES", base.smoke_samples),
              smoke_warmup = env("R01_WARMUP", base.smoke_warmup),
              smoke_chains = env("R01_CHAINS", base.smoke_chains))
end
const R01_GPH = fsg_gph_config(R01_CONFIG)
const R01_W = 0.4
const R01_SUFFIX = "_smoke_" * Dates.format(now(), "yyyymmddHHMMSS")
const R01_OUT_DIR = joinpath(R01_CONFIG.save_root, "smoke",
    "$(R01_CONFIG.smoke_chains)x$(R01_CONFIG.smoke_warmup)w$(R01_CONFIG.smoke_samples)s")
const R01_GIT = try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end

println("\n" * "="^96)
println("  r01 SMOKE GATE — fast & slow Poisson MultiScaleGRW, folds ",
        join(R01_CONFIG.smoke_folds, "/"))
println("  sampler    : QueuedNUTS  $(R01_CONFIG.smoke_chains) chains × ",
        "$(R01_CONFIG.smoke_warmup) warmup + $(R01_CONFIG.smoke_samples) retained, ",
        "δ = $(R01_CONFIG.accept_rate)")
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
# 4. Data snapshot, temporal splits, market
# ===================================================================
r01_ds = gph_load_data()
r01_splitter = gph_splitter(R01_CONFIG.target_seasons)
r01_book = fsg_closing_book(r01_ds)
println("\n  matches in store : ", nrow(r01_ds.matches),
        "   latest kickoff: ", maximum(r01_ds.matches.match_date),
        "   closing-book rows: ", nrow(r01_book))

# %%
# ===================================================================
# 5. Model construction
# ===================================================================
r01_models = fsg_models()
r01_sampler = QueuedNUTSConfig(
    n_samples = R01_CONFIG.smoke_samples, n_warmup = R01_CONFIG.smoke_warmup,
    n_chains = R01_CONFIG.smoke_chains, accept_rate = R01_CONFIG.accept_rate,
    max_depth = R01_CONFIG.max_depth, show_progress = false)
r01_configs = fsg_fit_configs(R01_CONFIG, r01_models, r01_splitter, r01_sampler;
                              name_suffix = R01_SUFFIX)
for (name, _) in r01_models
    println("  ", rpad(name, 30), " ", FSG_DESCRIPTIONS[name])
end

# %%
# ===================================================================
# 6. Features and gradient audit (G1)
# ===================================================================
r01_gradients = NamedTuple[]
r01_inputs = Dict{String,Any}()
r01_g1 = Dict{String,String}()

for (name, model) in r01_models
    println("\n--- features + gradient audit: ", name)
    inputs = fsg_fold_inputs(r01_ds, r01_splitter, model; folds = R01_CONFIG.smoke_folds)
    r01_inputs[name] = inputs
    filtration = gph_filtration_report(r01_ds, inputs)
    filtration.fold = inputs.fold_ids
    show(stdout, MIME"text/plain"(), filtration; allcols = true)
    println()
    all(filtration.ordered) || error("$name: a fold's last training kickoff is not before its first OOS kickoff")

    r01_g1[name] = ""
    for (k, fs) in enumerate(inputs.feature_sets)
        fold = inputs.fold_ids[k]
        audit = try
            gph_gradient_audit(model, fs; replays = R01_CONFIG.gradient_replays)
        catch err
            r01_g1[name] *= "fold $fold: " * sprint(showerror, err) * "; "
            continue
        end
        push!(r01_gradients, (; model = name, fold,
                                n_target = Int(first(fs).data[:n_target_steps]),
                                audit...))
        @printf("  fold %2d  θ=%4d  tape=%5d  grad=%.3f ms  RD/FD=%.1e  perturbed=%.1e\n",
                fold, audit.n_parameters, audit.tape_instructions, audit.gradient_ms,
                audit.compiled_forward_error, audit.worst_perturbed_error)
    end
end

# %%
# ===================================================================
# 7. Training — three folds per arm
# ===================================================================
r01_fits = Dict{String,Any}()
for (name, _) in r01_models
    println("\n--- sampling: ", name)
    t0 = time()
    r01_fits[name] = gph_sample(r01_configs[name], r01_inputs[name], R01_GPH)
    @printf("  sampled %s in %.1f min\n", name, (time() - t0) / 60)
end

# %%
# ===================================================================
# 8. Convergence (G2–G4), persistence (G5), supremacy (G6)
# ===================================================================
r01_oos_ids = sort!(collect(intersect((Set(r01_fits[n].latents.match_ids) for n in FSG_MODEL_NAMES)...)))
r01_market = fsg_market_frame(r01_book, r01_oos_ids)
println("\n  OOS fixtures: ", length(r01_oos_ids), "   with an accepted market inversion: ",
        nrow(r01_market))

r01_rows = NamedTuple[]
for (name, model) in r01_models
    fit = r01_fits[name]
    d = fit.diagnostics
    failures = String[]
    isempty(r01_g1[name]) || push!(failures, "G1 " * r01_g1[name])
    d.n_divergent == 0 || push!(failures, "G2 divergences=$(d.n_divergent)")
    d.max_rhat <= R01_CONFIG.max_rhat || push!(failures,
        @sprintf("G3 max R̂ %.4f (fold %d)", d.max_rhat, d.worst_rhat_fold))
    d.min_ess_bulk >= R01_CONFIG.min_ess || push!(failures,
        @sprintf("G4 min bulk ESS %.0f", d.min_ess_bulk))

    try
        gph_latent_audit(fit)
    catch err
        push!(failures, "latents " * sprint(showerror, err))
    end

    run_id = try
        gph_save_and_verify(r01_db, fit)
    catch err
        push!(failures, "G5 " * sprint(showerror, err))
        nothing
    end

    sup = fsg_supremacy_report(fit.latents, model, r01_market)
    σ₀ = fsg_sigma0(fit)
    push!(r01_rows, (; model = name,
                       folds = length(fit.folds),
                       oos = n_matches(fit.latents),
                       draws = n_draws(fit.latents),
                       max_rhat = d.max_rhat,
                       min_ess_bulk = d.min_ess_bulk,
                       min_ess_tail = d.min_ess_tail,
                       n_divergent = d.n_divergent,
                       min_bfmi = d.min_bfmi,
                       wall_min = fit.metadata.elapsed_seconds / 60,
                       sigma0_att = σ₀["α_σ₀"],
                       sigma0_def = σ₀["β_σ₀"],
                       sup_slope = sup.slope,
                       sup_r2 = sup.r2,
                       sup_n = sup.n,
                       sup_sd = sup.sup_sd,
                       max_win_prob = sup.max_win_prob,
                       n_fav70 = sup.n_fav70,
                       fav70_model = sup.fav70_model,
                       fav70_market = sup.fav70_market,
                       run_id = run_id === nothing ? "" : string(run_id),
                       failures = join(failures, "; ")))
end
r01_summary = DataFrame(r01_rows)

# G6 — directional: each loose arm must beat the tight arm's slope.
tight_slope = only(filter(:model => ==(FSG_TIGHT), r01_summary).sup_slope)
r01_summary.g6_pass = [r.model == FSG_TIGHT ? true : r.sup_slope > tight_slope
                       for r in eachrow(r01_summary)]

# %%
# ===================================================================
# 9. Rate pooling (G7) and a small weight ladder
# ===================================================================
r01_blend_rows = NamedTuple[]
r01_g7 = Dict{String,String}()
tight_model = last(first(r01_models))
for loose in (FSG_LOOSE_VAR, FSG_LOOSE_T)
    r01_g7[loose] = ""
    for w in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        blended = blend_latents(r01_fits[FSG_TIGHT].latents, r01_fits[loose].latents, w)
        if w == R01_W
            try
                for (label, lat) in (("tight", r01_fits[FSG_TIGHT].latents),
                                     (loose, r01_fits[loose].latents), ("blend", blended))
                    g = fsg_grid_audit(lat, tight_model)
                    @printf("  G7 %-28s w=%.1f %-6s rows=%d  p∈[%.2e, %.4f]  book dev %.1e  truncation %.1e\n",
                            label, w, "", g.n_rows,
                            g.min_prob, g.max_prob, g.worst_book_dev, g.worst_truncation)
                end
            catch err
                r01_g7[loose] = sprint(showerror, err)
            end
        end
        s = fsg_supremacy_report(blended, tight_model, r01_market)
        push!(r01_blend_rows, (; loose, w, sup_slope = s.slope, sup_sd = s.sup_sd,
                                 max_win_prob = s.max_win_prob,
                                 fav70_model = s.fav70_model, fav70_market = s.fav70_market))
    end
end
r01_blends = DataFrame(r01_blend_rows)
r01_summary.g7_failures = [get(r01_g7, r.model, "") for r in eachrow(r01_summary)]
r01_summary.gate_pass = isempty.(r01_summary.failures) .& r01_summary.g6_pass .&
                        isempty.(r01_summary.g7_failures)

# %%
# ===================================================================
# 10. Report
# ===================================================================
println("\n", "="^96)
show(stdout, MIME"text/plain"(), select(r01_summary, :model, :max_rhat, :min_ess_bulk,
     :n_divergent, :wall_min, :sigma0_att, :sigma0_def, :sup_slope, :sup_sd, :max_win_prob,
     :n_fav70, :fav70_model, :fav70_market, :gate_pass); allcols = true)
println("\n")
show(stdout, MIME"text/plain"(), r01_blends; allcols = true, allrows = true)
println()

CSV.write(joinpath(R01_OUT_DIR, "r01_smoke_gates.csv"), r01_summary)
CSV.write(joinpath(R01_OUT_DIR, "r01_gradient_audit.csv"), DataFrame(r01_gradients))
CSV.write(joinpath(R01_OUT_DIR, "r01_blend_ladder.csv"), r01_blends)

open(joinpath(R01_OUT_DIR, "r01_smoke_report.md"), "w") do io
    println(io, "# r01 smoke gate — TODO 021 fast & slow GRW\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R01_GIT,
            "` on ", gethostname(), " with ", Threads.nthreads(), " threads. ",
            "QueuedNUTS ", R01_CONFIG.smoke_chains, " × (", R01_CONFIG.smoke_warmup, " + ",
            R01_CONFIG.smoke_samples, "), δ = ", R01_CONFIG.accept_rate, ". Folds ",
            join(R01_CONFIG.smoke_folds, ", "), ". ", length(r01_oos_ids), " OOS fixtures, ",
            nrow(r01_market), " with an accepted market inversion.\n")
    println(io, "## Gates and supremacy\n")
    print(io, gph_markdown_table(select(r01_summary, :model, :max_rhat, :min_ess_bulk,
        :n_divergent, :sigma0_att, :sigma0_def, :sup_slope, :sup_sd, :max_win_prob,
        :n_fav70, :fav70_model, :fav70_market, :gate_pass, :run_id)))
    println(io, "\n## Rate-pooling ladder (tight ⊕ loose)\n")
    print(io, gph_markdown_table(r01_blends))
    failed = filter(:gate_pass => !, r01_summary)
    if nrow(failed) > 0
        println(io, "\n### Failures\n")
        for r in eachrow(failed)
            println(io, "* `", r.model, "` — ", r.failures, " ",
                    r.g6_pass ? "" : "G6 slope not above tight; ", r.g7_failures)
        end
    end
end

r01_verdict = all(r01_summary.gate_pass)
println("\nR01_VERDICT ", r01_verdict ? "PASS" : "FAIL", "  (",
        count(r01_summary.gate_pass), "/", nrow(r01_summary), " arms)")
println("R01_DONE report=", joinpath(R01_OUT_DIR, "r01_smoke_report.md"))
