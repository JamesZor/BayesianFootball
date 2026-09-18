# ==============================================================================
# r01 — Smoke gate: market-anchored MultiScaleGRW ladder, folds 1–2
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# A mechanical gate, not a result. It answers, for each of the work package's three
# ladder rungs — baseline, supremacy, smile + supremacy (moderate weights) — whether the
# composition compiles, samples, converges, extracts, prices and persists end to end.
# Nothing it prints is a proper score.
#
# GATES — every one must pass for every rung before `r02` may be launched:
#
#   G0  likelihood parity. (a) With both pillar slots empty, the wrapper's log density is
#       BIT-IDENTICAL to Task 013's m05 builder model (Δ == 0.0) at a prior draw and three
#       displaced points. (b) With pillars on, log density − base log density equals an
#       independent `Distributions.logpdf` re-derivation of the pillar terms (≤ 1e-9 rel).
#       Folds 1 and 2 — fold 1 has no GRW target steps, fold 2 does.
#   GB  market coverage: each pillar reads a non-empty set of training matches per fold.
#   G1  ReverseDiff tape: compiled == fresh ≤ 1e-8, RD == ForwardDiff ≤ 1e-6, compiled tape
#       exact at three perturbed points. Allocation per warmed gradient and tape length are
#       REPORTED against the baseline, not gated at zero: Task 007/008/013 measured every
#       model in this repository at 129–182 KB per call, the baseline included.
#   G2  zero divergences across all chains.
#   G3  max R̂ ≤ 1.05 and bulk/tail ESS ≥ 400 over EVERY sampled site; the pillar sites
#       (σ_sup, σ_smile, log_φ[1:5]) are also listed individually.
#   G4  latents: finite positive λ; for the smile rung a `SmileLatents` whose O/U 2.5 price
#       through the typed kernel AND the legacy MatchDay row route both equal
#       mean cdf(Poisson(λ_tot·φ₂), 2) to ≤ 1e-12.
#   G5  save_fit → load_fit through PostgresStorage("smoke_grw_smile"): chains identical;
#       for the smile rung, latents rebuilt from the PERSISTED chains equal the fitted ones
#       (the T010 detour — see the loader header).
#   G6  config registry: every rung's model, splitter, sampler and FitConfig register.
#
# SAMPLER. The work package's smoke budget: 4 chains × (500 warmup + 500 retained),
# δ = 0.80. 2,000 retained draws per fold can reach ESS 400; the gate is informative.
#
# PERSISTENCE CAVEAT. Each invocation writes three runs into `smoke_grw_smile`, suffixed
# with a timestamp so config-hash deduplication never hands back an older smoke run.
#
# USAGE (mcmc-beast, from /root/BF_grw_market_smile)
#
#   julia --project -t 16 current_development/grw_market_smile/r01_smoke.jl
#   R01_SAMPLES=100 R01_WARMUP=100 R01_CHAINS=2 julia ... r01_smoke.jl    # plumbing only
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
    base = GMSConfig()
    GMSConfig(smoke_samples = env("R01_SAMPLES", base.smoke_samples),
              smoke_warmup = env("R01_WARMUP", base.smoke_warmup),
              smoke_chains = env("R01_CHAINS", base.smoke_chains))
end
const R01_GATE_ESS = R01_CONFIG.smoke_chains * R01_CONFIG.smoke_samples >= 2 * R01_CONFIG.min_ess
const R01_SUFFIX = "_smoke_" * Dates.format(now(), "yyyymmddHHMMSS")
const R01_OUT_DIR = joinpath(R01_CONFIG.save_root, "smoke",
    "$(R01_CONFIG.smoke_chains)x$(R01_CONFIG.smoke_warmup)w$(R01_CONFIG.smoke_samples)s")
const R01_PILLAR_SITES = ["σ_sup", "σ_smile", "log_φ[1]", "log_φ[2]", "log_φ[3]", "log_φ[4]", "log_φ[5]"]
const R01_GIT = try readchomp(`git rev-parse --short HEAD`) catch; "unknown" end

println("\n" * "="^96)
println("  r01 SMOKE GATE — market-anchored MultiScaleGRW ladder, folds 1–$(R01_CONFIG.smoke_folds)")
println("  rungs      : ", join(GMS_SMOKE_MODEL_NAMES, ", "))
println("  sampler    : QueuedNUTS  $(R01_CONFIG.smoke_chains) chains × ",
        "$(R01_CONFIG.smoke_warmup) warmup + $(R01_CONFIG.smoke_samples) retained, ",
        "δ = $(R01_CONFIG.accept_rate), max depth $(R01_CONFIG.max_depth)",
        R01_GATE_ESS ? "" : "   (ESS NOT gated: budget cannot reach $(R01_CONFIG.min_ess))")
println("  experiment : ", R01_CONFIG.smoke_experiment, "   (suffix ", R01_SUFFIX, ")")
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
        "   latest kickoff: ", maximum(r01_ds.matches.match_date),
        "   rows dated 2026-09-12: ", count(==(Date(2026, 9, 12)), Date.(r01_ds.matches.match_date)))

# %%
# ===================================================================
# 5. Engine / model construction
# ===================================================================
r01_models = gms_select(gms_models(), GMS_SMOKE_MODEL_NAMES)
r01_base = last(first(r01_models))
r01_sampler = gms_smoke_sampler(R01_CONFIG)
r01_configs = gms_fit_configs(R01_CONFIG, r01_models, r01_splitter, r01_sampler;
                              name_suffix = R01_SUFFIX)
for (name, model) in r01_models
    println("  ", rpad(name, 36), " ", nameof(typeof(model)),
            model isa MarketAnchoredCountModel ?
                " | supremacy " * string(nameof(typeof(model.supremacy))) *
                " | smile " * string(nameof(typeof(model.smile))) : "",
            " | family ", nameof(typeof(latent_family(model))))
end

# %%
# ===================================================================
# 6. Features, filtration and market coverage (GB)
# ===================================================================
r01_inputs = Dict{String,Any}()
r01_coverage = DataFrame[]
for (name, model) in r01_models
    println("\n--- features: ", name)
    inputs = gph_fold_inputs(r01_ds, r01_splitter, model; limit = R01_CONFIG.smoke_folds)
    r01_inputs[name] = inputs
    filtration = gph_filtration_report(r01_ds, inputs)
    show(stdout, MIME"text/plain"(), filtration; allcols = true)
    println()
    all(filtration.ordered) ||
        error("$name: a fold's last training kickoff is not before its first OOS kickoff")
    if model isa MarketAnchoredCountModel
        cov = gms_market_coverage(model, inputs)
        show(stdout, MIME"text/plain"(), cov; allcols = true)
        println()
        push!(r01_coverage, insertcols(cov, 1, :model => name))
        all(cov.supremacy_observed .> 0) || error("GB FAILED: $name supremacy pillar reads no match")
        model.smile isa MarketSmilePillar && (all(cov.smile_matches .> 0) ||
            error("GB FAILED: $name smile pillar reads no match"))
    end
end

# %%
# ===================================================================
# 7. Likelihood parity (G0)
# ===================================================================
# First, and on its own. If the density is wrong, every later number is a well-converged
# posterior for the wrong model.
r01_parity = NamedTuple[]
r01_parity_cases = vcat(
    [("null_anchor", gms_null_anchor(r01_base), GMS_SMOKE_MODEL_NAMES[2])],
    [(name, model, name) for (name, model) in r01_models if model isa MarketAnchoredCountModel],
)
for (label, model, inputs_key) in r01_parity_cases, fold in 1:R01_CONFIG.smoke_folds
    res = gms_parity_check(model, r01_base, r01_inputs[inputs_key].feature_sets[fold])
    is_null = label == "null_anchor"
    pass = is_null ? (res.n_pillar_sites == 0 && all(==(0.0), res.base_deltas)) :
                     res.worst_rel <= 1.0e-9
    push!(r01_parity, (; case = label, fold, points = res.n_points,
                         base_sites = res.n_base_sites, pillar_sites = res.pillar_sites,
                         max_abs_base_delta = maximum(abs.(res.base_deltas)),
                         worst_abs = res.worst_abs, worst_rel = res.worst_rel, pass))
    @printf("  G0 %-36s fold %d  θ_base=%d  +[%s]  |Δ_base|max=%.3e  worst_rel=%.2e  %s\n",
            label, fold, res.n_base_sites, res.pillar_sites,
            maximum(abs.(res.base_deltas)), res.worst_rel, pass ? "PASS" : "FAIL")
end
all(r -> r.pass, r01_parity) ||
    error("G0 FAILED: the wrapper's log density is not the base model plus the stated pillars")

# %%
# ===================================================================
# 8. Gradient audit (G1)
# ===================================================================
r01_gradients = NamedTuple[]
for (name, model) in r01_models, fold in 1:R01_CONFIG.smoke_folds
    fs = r01_inputs[name].feature_sets[fold]
    audit = gph_gradient_audit(model, fs; replays = R01_CONFIG.gradient_replays)
    push!(r01_gradients, (; model = name, fold,
                            n_target = Int(first(fs).data[:n_target_steps]), audit...))
    @printf("  G1 %-36s fold %d  θ=%4d  tape=%5d  grad=%.3f ms  alloc=%7d B  RD/FD=%.1e  perturbed=%.1e\n",
            name, fold, audit.n_parameters, audit.tape_instructions, audit.gradient_ms,
            audit.allocated_bytes, audit.compiled_forward_error, audit.worst_perturbed_error)
end
r01_gradient_frame = DataFrame(r01_gradients)
r01_base_grad = Dict(r.fold => r for r in eachrow(r01_gradient_frame) if r.model == GMS_SMOKE_MODEL_NAMES[1])
r01_gradient_frame.delta_tape = [r.tape_instructions - r01_base_grad[r.fold].tape_instructions
                                 for r in eachrow(r01_gradient_frame)]
r01_gradient_frame.delta_alloc = [r.allocated_bytes - r01_base_grad[r.fold].allocated_bytes
                                  for r in eachrow(r01_gradient_frame)]

# %%
# ===================================================================
# 9. Config registry (G6) — before sampling, so a serialisation failure costs nothing
# ===================================================================
r01_registry = try
    gms_register!(r01_db, r01_models, r01_splitter, r01_sampler, r01_configs)
catch err
    error("G6 FAILED: config registry refused a rung: " * sprint(showerror, err))
end
println("  G6 registered models ", r01_registry.model_ids)

# %%
# ===================================================================
# 10. Training — two folds per rung
# ===================================================================
# NUTS chains are single-threaded; QueuedExecution flattens 2 folds × 4 chains into one
# queue over the pinned threads.
r01_fits = Dict{String,Any}()
for (name, _) in r01_models
    println("\n--- sampling: ", name)
    r01_fits[name] = gph_sample(r01_configs[name], r01_inputs[name], gms_gph_config(R01_CONFIG))
end

# %%
# ===================================================================
# 11. Convergence (G2, G3), latents (G4), round-trip (G5)
# ===================================================================
r01_rows = NamedTuple[]
r01_site_rows = NamedTuple[]
r01_pricing = DataFrame[]

for (name, model) in r01_models
    fit = r01_fits[name]
    d = fit.diagnostics
    failures = String[]

    d.n_divergent == 0 || push!(failures, "G2 divergences=$(d.n_divergent)")
    d.max_rhat <= R01_CONFIG.max_rhat || push!(failures,
        @sprintf("G3 max R̂ %.4f (fold %d)", d.max_rhat, d.worst_rhat_fold))
    R01_GATE_ESS && min(d.min_ess_bulk, d.min_ess_tail) < R01_CONFIG.min_ess && push!(failures,
        @sprintf("G3 min ESS bulk %.0f / tail %.0f", d.min_ess_bulk, d.min_ess_tail))

    # The pillar sites, listed one by one — they are inside the audit above already.
    for f in fit.folds
        rh = DataFrame(MCMCChains.rhat(f.chain))
        eb = DataFrame(MCMCChains.ess(f.chain; kind = :bulk))
        et = DataFrame(MCMCChains.ess(f.chain; kind = :tail))
        for site in R01_PILLAR_SITES
            i = findfirst(==(Symbol(site)), rh.parameters)
            i === nothing && continue
            push!(r01_site_rows, (; model = name, fold = f.fold, site,
                                    mean = mean(vec(Array(f.chain[Symbol(site)]))),
                                    rhat = rh.rhat[i], ess_bulk = eb.ess[i], ess_tail = et.ess[i]))
        end
    end

    latent = try
        gms_latent_audit(fit)
    catch err
        push!(failures, "G4 " * sprint(showerror, err))
        nothing
    end

    if fit.latents isa SmileLatents
        try
            g = gms_smile_pricing_gate(fit)
            push!(r01_pricing, insertcols(g, 1, :model => name))
            maximum(abs.(g.p_under_typed .- g.p_under_ref)) <= 1e-12 ||
                error("typed smile O/U price ≠ cdf(Poisson(λ_tot·φ))")
            maximum(abs.(g.p_under_legacy .- g.p_under_ref)) <= 1e-12 ||
                error("legacy MatchDay row route smile O/U price ≠ cdf(Poisson(λ_tot·φ))")
            maximum(abs.(g.smile_shift)) > 1e-6 ||
                error("smile price is indistinguishable from the plain grid; φ never reached pricing")
        catch err
            push!(failures, "G4 smile " * sprint(showerror, err))
        end
    elseif model isa MarketAnchoredCountModel || model === r01_base
        fit.latents isa CountLatents || push!(failures, "G4 expected CountLatents")
    end

    run_id = try
        gms_save_and_verify(r01_db, fit, r01_inputs[name]; latent_dir = joinpath(R01_OUT_DIR, "latents"))
    catch err
        push!(failures, "G5 " * sprint(showerror, err))
        nothing
    end

    row = gms_convergence_row(name, fit, R01_CONFIG; run_id)
    push!(r01_rows, (; row...,
                       latent_family = latent === nothing ? "n/a" : latent.family,
                       latent_mean_lambda_h = latent === nothing ? NaN : latent.mean_lambda_h,
                       latent_mean_lambda_a = latent === nothing ? NaN : latent.mean_lambda_a,
                       latent_φ_mean = latent === nothing ? "" : latent.φ_mean,
                       gate_pass = isempty(failures),
                       gate_failures = join(failures, "; ")))
    @printf("  %-36s R̂=%.4f  ESS=%6.1f/%6.1f  div=%d  σ_sup=%.3f  σ_smile=%.3f  κ=%.3f  φ=%s  %s\n",
            name, d.max_rhat, d.min_ess_bulk, d.min_ess_tail, d.n_divergent,
            row.σ_sup_median, row.σ_smile_median, row.κ_median, row.φ_median,
            isempty(failures) ? "PASS" : "FAIL: " * join(failures, "; "))
end

# %%
# ===================================================================
# 12. Final report
# ===================================================================
r01_summary = DataFrame(r01_rows)
r01_parity_frame = DataFrame(r01_parity)
r01_site_frame = DataFrame(r01_site_rows)
r01_coverage_frame = isempty(r01_coverage) ? DataFrame() : vcat(r01_coverage...)
r01_pricing_frame = isempty(r01_pricing) ? DataFrame() : vcat(r01_pricing...)

CSV.write(joinpath(R01_OUT_DIR, "r01_smoke_gates.csv"), r01_summary)
CSV.write(joinpath(R01_OUT_DIR, "r01_parity.csv"), select(r01_parity_frame, Not(:pillar_sites)))
CSV.write(joinpath(R01_OUT_DIR, "r01_gradient_audit.csv"), r01_gradient_frame)
nrow(r01_site_frame) > 0 && CSV.write(joinpath(R01_OUT_DIR, "r01_pillar_sites.csv"), r01_site_frame)
nrow(r01_coverage_frame) > 0 && CSV.write(joinpath(R01_OUT_DIR, "r01_market_coverage.csv"), r01_coverage_frame)
nrow(r01_pricing_frame) > 0 && CSV.write(joinpath(R01_OUT_DIR, "r01_smile_pricing.csv"), r01_pricing_frame)

open(joinpath(R01_OUT_DIR, "r01_smoke_report.md"), "w") do io
    println(io, "# r01 smoke gate — Task 015 (market-anchored MultiScaleGRW)\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R01_GIT,
            "` on ", gethostname(), " with ", Threads.nthreads(), " threads. Store latest kickoff ",
            maximum(r01_ds.matches.match_date), ".\n")
    println(io, "Sampler: QueuedNUTS, ", R01_CONFIG.smoke_chains, " chains × ",
            R01_CONFIG.smoke_warmup, " warmup + ", R01_CONFIG.smoke_samples,
            " retained, δ = ", R01_CONFIG.accept_rate, ". Folds 1–", R01_CONFIG.smoke_folds, ".\n")

    println(io, "## G0 likelihood parity\n")
    print(io, gph_markdown_table(r01_parity_frame;
        formats = Dict(:max_abs_base_delta => v -> @sprintf("%.3e", v),
                       :worst_abs => v -> @sprintf("%.2e", v),
                       :worst_rel => v -> @sprintf("%.2e", v))))

    if nrow(r01_coverage_frame) > 0
        println(io, "\n## GB market coverage (training matches each pillar reads)\n")
        print(io, gph_markdown_table(r01_coverage_frame;
            formats = Dict(:supremacy_share => v -> gph_num(v; digits = 3),
                           :smile_share => v -> gph_num(v; digits = 3))))
    end

    println(io, "\n## G1 gradient audit\n")
    println(io, "Δ columns are against the baseline rung on the same fold.\n")
    print(io, gph_markdown_table(select(r01_gradient_frame,
        :model, :fold, :n_target, :n_parameters, :tape_instructions, :delta_tape, :gradient_ms,
        :allocated_bytes, :delta_alloc, :compiled_forward_error, :worst_perturbed_error);
        formats = Dict(:gradient_ms => v -> gph_num(v; digits = 3),
                       :compiled_forward_error => v -> @sprintf("%.1e", v),
                       :worst_perturbed_error => v -> @sprintf("%.1e", v))))

    println(io, "\n## G2–G5 sampling, latents, persistence\n")
    print(io, gph_markdown_table(select(r01_summary,
        :model, :folds, :oos, :draws, :max_rhat, :min_ess_bulk, :min_ess_tail, :n_divergent,
        :min_bfmi, :σ_sup_median, :σ_smile_median, :κ_median, :φ_median, :latent_family,
        :gate_pass, :run_id)))

    if nrow(r01_site_frame) > 0
        println(io, "\n## Pillar sites, per fold\n")
        print(io, gph_markdown_table(r01_site_frame;
            formats = Dict(:mean => v -> gph_num(v; digits = 4),
                           :ess_bulk => v -> gph_num(v; digits = 0),
                           :ess_tail => v -> gph_num(v; digits = 0))))
    end

    if nrow(r01_pricing_frame) > 0
        println(io, "\n## G4 smile O/U 2.5 pricing — three routes and the plain grid\n")
        print(io, gph_markdown_table(r01_pricing_frame;
            formats = Dict(:p_under_ref => v -> gph_num(v; digits = 6),
                           :p_under_typed => v -> gph_num(v; digits = 6),
                           :p_under_legacy => v -> gph_num(v; digits = 6),
                           :p_under_grid => v -> gph_num(v; digits = 6),
                           :smile_shift => v -> gph_signed(v; digits = 5))))
    end

    failed = filter(:gate_pass => !, r01_summary)
    if nrow(failed) > 0
        println(io, "\n### Failures\n")
        for r in eachrow(failed)
            println(io, "* `", r.model, "` — ", r.gate_failures)
        end
    end
end

r01_verdict = all(r01_summary.gate_pass) && all(r01_parity_frame.pass)
println("\nR01_VERDICT ", r01_verdict ? "PASS" : "FAIL", "  (",
        count(r01_summary.gate_pass), "/", nrow(r01_summary), " rungs)")
println("R01_DONE report=", joinpath(R01_OUT_DIR, "r01_smoke_report.md"))
