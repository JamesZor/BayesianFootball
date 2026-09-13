# ==============================================================================
# r01 — Smoke gate: 1-parameter smile spine on market-anchored MultiScaleGRW, folds 1–2
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# A mechanical gate, not a result. It answers, for the two spine rungs
# (`m05_joint_grw_smile_spine_w020`, `…_w040`), whether the composition is the model the
# work package writes down, differentiates exactly, samples without divergences, extracts a
# `SmileLatents`, stakes coherently off the anti-diagonal-reweighted grid (ticket T011), and
# persists. Nothing it prints is a proper score or a benchmark verdict: two folds cannot
# settle H1–H5. The β posterior and the wall time are PRINTED as early signal only.
#
# The four pinned rungs (baseline, supremacy, five-strike smile @0.20/@0.40) are NOT sampled.
# The baseline and five-strike @0.40 are built here only as references for G0 and G1.
#
# GATES — every one must pass for both rungs before `r02` may be launched:
#
#   GA  filtration: last training kickoff < first held-out kickoff, no overlap.
#   GB  market coverage: both pillars read a non-empty training set on each fold.
#   G4a reweighting, pure (BEFORE sampling), on a synthetic spine container:
#       reweighted totals marginal = smile CDF ≤ 1e-9 per draw and in the mean, each draw sums
#       to 1, no negative mass, anti-diagonals rescaled uniformly (≤ 1e-12); φ ≡ 1 shortcut
#       bit-identical and the un-shortcut path ≤ 1e-6; a non-monotone curve REFUSED.
#   G0a null anchor: this file's engine copy with both slots empty is BIT-IDENTICAL to the
#       m05 builder model (Δ == 0.0) at four prior draws.
#   G0b spine rung log density − base = independent `logpdf` re-derivation, ≤ 1e-14 relative.
#   G0c spine rung ≡ five-strike rung at log φ_K = β(K − 2): likelihoods equal up to their
#       shape priors, ≤ 1e-14 relative.
#   G1  ReverseDiff: compiled == fresh ≤ 1e-8, compiled == ForwardDiff ≤ 1e-6, compiled tape
#       exact at three perturbed points ≤ 1e-8. Tape length, gradient time and allocation are
#       REPORTED against the baseline and the five-strike smile.
#   G6  config registry.
#   G2  zero divergences.
#   G3  max R̂ ≤ 1.05 and bulk/tail ESS ≥ 400 over every site (when the budget can reach it).
#   G4b fitted latents: `SmileLatents` with φ(2.5) ≡ 1; O/U through the typed kernel and the
#       legacy row route = mean cdf(Poisson(λ_tot·φ_K), K) ≤ 1e-12; the G4a checks on the
#       fitted container.
#   G4c portfolio (T011): the φ ≡ 1 twin stakes the bit-identical ledger of its grid twin; every
#       staked book's p_grid implies the smile totals CDF ≤ 1e-9.
#   G5  save_fit → load_fit through PostgresStorage("smoke_grw_smile_spine"): chains identical,
#       `SmileLatents` rebuilt from the PERSISTED chains equal the fitted ones (T010 path).
#
# THRESHOLDS are the work package's (G0 ≤ 1e-14, G1 ≤ 1e-6) or ticket T011's (≤ 1e-9, tighter
# than the work package's 1e-6 G4). None was tuned to pass.
#
# SAMPLER. 4 chains × (500 warmup + 1000 retained), δ = 0.80 — the production budget. Task 015's
# smoke failed its pinned baseline on tail ESS at 500 retained draws; see `gss_config`.
#
# PERSISTENCE CAVEAT. Each invocation writes two runs into `smoke_grw_smile_spine`, suffixed with a
# timestamp so config-hash deduplication never returns an older smoke run. Reports go to
# `results/smoke/<budget>/`. Loading a spine artefact later requires `include("l01_loader.jl")`.
#
# USAGE (mcmc-beast, from the task worktree checkout)
#
#   julia --project -t 16 current_development/grw_smile_spine/r01_smoke.jl
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
using Statistics

include(joinpath(@__DIR__, "l01_loader.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R01_CONFIG = let env(k, d) = parse(Int, get(ENV, k, string(d)))
    base = gss_config()
    gss_config(smoke_samples = env("R01_SAMPLES", base.smoke_samples),
               smoke_warmup = env("R01_WARMUP", base.smoke_warmup),
               smoke_chains = env("R01_CHAINS", base.smoke_chains))
end

const R01_SAMPLED = GSS_GRID_MODEL_NAMES
const R01_BASELINE = "m05_joint_grw_baseline"
const R01_FIVE_STRIKE = "m05_joint_grw_smile_supremacy_w040"
const R01_AUDITED = [R01_BASELINE, R01_FIVE_STRIKE, R01_SAMPLED...]

const R01_G0_TOL = 1.0e-14
const R01_G4_TOL = 1.0e-9
const R01_SPREAD_TOL = 1.0e-12
const R01_FORCED_IDENTITY_TOL = 1.0e-6
const R01_PRICING_TOL = 1.0e-12

const R01_GATE_ESS = R01_CONFIG.smoke_chains * R01_CONFIG.smoke_samples >= 2 * R01_CONFIG.min_ess
const R01_SUFFIX = "_smoke_" * Dates.format(now(), "yyyymmddHHMMSS")
const R01_BUDGET = "$(R01_CONFIG.smoke_chains)x$(R01_CONFIG.smoke_warmup)w$(R01_CONFIG.smoke_samples)s"
const R01_OUT_DIR = joinpath(R01_CONFIG.save_root, "smoke", R01_BUDGET)
const R01_TASK015_SMOKE = joinpath(@__DIR__, "..", "grw_market_smile", "results", "smoke",
                                   "4x500w1000s", "r01_smoke_gates.csv")
const R01_GIT = try
    readchomp(`git rev-parse --short HEAD`)
catch
    "unknown"
end

println("\n" * "="^96)
println("  r01 SMOKE GATE — 1-parameter smile spine, folds 1–$(R01_CONFIG.smoke_folds)")
println("  sampled    : ", join(R01_SAMPLED, ", "))
println("  references : ", R01_BASELINE, ", ", R01_FIVE_STRIKE, "   (G0/G1 only, not sampled)")
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
r01_failures = Dict(name => String[] for name in R01_SAMPLED)

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
r01_ladder = gss_models()
r01_models = Dict{String,Any}(name => model for (name, model) in r01_ladder)
r01_base = r01_models[R01_BASELINE]
r01_sampled = gms_select(r01_ladder, R01_SAMPLED)
r01_sampler = gms_smoke_sampler(R01_CONFIG)
r01_configs = gss_fit_configs(R01_CONFIG, r01_sampled, r01_splitter, r01_sampler;
                              name_suffix = R01_SUFFIX)

for name in R01_AUDITED
    model = r01_models[name]
    shape = model isa SpineAnchoredCountModel ?
        @sprintf(" | spine w=%.2f β~%s pivot=%d", model.smile.weight, model.smile.β_prior, model.smile.pivot) :
        model isa MarketAnchoredCountModel ? " | five-strike smile" : ""
    println("  ", rpad(name, 36), " ", nameof(typeof(model)), shape,
            " | family ", nameof(typeof(latent_family(model))))
end

# %%
# ===================================================================
# 6. Features, filtration (GA) and market coverage (GB)
# ===================================================================
# One feature build serves every audited model: the spine and five-strike rungs declare the same
# features, and the baseline reads a subset of them.
r01_feature_names = Dict(name => sort(string.(GMS_FEATURES.required_features(r01_models[name])))
                         for name in R01_AUDITED if r01_models[name] !== r01_base)
length(unique(values(r01_feature_names))) == 1 ||
    error("GB FAILED: the market rungs declare different features: $r01_feature_names")

r01_inputs = gph_fold_inputs(r01_ds, r01_splitter, r01_models[last(R01_SAMPLED)];
                             limit = R01_CONFIG.smoke_folds)
r01_filtration = gph_filtration_report(r01_ds, r01_inputs)
show(stdout, MIME"text/plain"(), r01_filtration; allcols = true)
println()
all(r01_filtration.ordered) ||
    error("GA FAILED: a fold's last training kickoff is not before its first OOS kickoff")

r01_coverage = vcat([insertcols(gss_market_coverage(r01_models[name], r01_inputs), 1, :model => name)
                     for name in R01_SAMPLED]...)
show(stdout, MIME"text/plain"(), r01_coverage; allcols = true)
println()
all(r01_coverage.supremacy_observed .> 0) || error("GB FAILED: a supremacy pillar reads no match")
all(r01_coverage.smile_matches .> 0) || error("GB FAILED: a spine pillar reads no match")

# %%
# ===================================================================
# 7. Anti-diagonal reweighting on a synthetic container (G4a) — before any sampling
# ===================================================================
# Pure arithmetic, no chain. If the T011 pricer is wrong it is wrong for every run below, so it
# is checked first and cheapest.
r01_synthetic = gss_synthetic_spine_latents()
r01_g4a = gss_reweight_gate(r01_synthetic)
r01_g4a_identity = gss_identity_path_gate(r01_synthetic)
r01_g4a_refusal = gss_refusal_gate()
show(stdout, MIME"text/plain"(), r01_g4a; allcols = true)
println()
@printf("  G4a identity shortcut bit-identical=%s   forced path max |Δ|=%.2e   non-monotone refused=%s\n",
        r01_g4a_identity.shortcut_bit_identical, r01_g4a_identity.forced_max_abs_gap,
        r01_g4a_refusal.refused)

r01_g4a_failures = gss_reweight_failures(r01_g4a; tol = R01_G4_TOL, spread_tol = R01_SPREAD_TOL)
r01_g4a_identity.shortcut_bit_identical || push!(r01_g4a_failures, "φ ≡ 1 shortcut changed the grid")
r01_g4a_identity.forced_max_abs_gap <= R01_FORCED_IDENTITY_TOL || push!(r01_g4a_failures,
    @sprintf("un-shortcut φ ≡ 1 path moved the grid by %.2e", r01_g4a_identity.forced_max_abs_gap))
r01_g4a_refusal.refused || push!(r01_g4a_failures,
    "a non-monotone smile curve was not refused: " * r01_g4a_refusal.message)
isempty(r01_g4a_failures) || error("G4a FAILED: " * join(r01_g4a_failures, "; "))

# %%
# ===================================================================
# 8. Likelihood parity (G0)
# ===================================================================
# If the density is wrong, every later number is a well-converged posterior for the wrong model.
r01_parity = NamedTuple[]
for fold in 1:R01_CONFIG.smoke_folds
    fs = r01_inputs.feature_sets[fold]

    null = gss_parity_check(gss_null_anchor(r01_base), r01_base, fs)
    push!(r01_parity, (; check = "G0a", model = "null_anchor", fold, sites = null.pillar_sites,
                         max_abs_base_delta = maximum(abs.(null.base_deltas)),
                         worst_abs = null.worst_abs, worst_rel = null.worst_rel,
                         pass = null.n_pillar_sites == 0 && all(==(0.0), null.base_deltas)))

    for name in R01_SAMPLED
        res = gss_parity_check(r01_models[name], r01_base, fs)
        push!(r01_parity, (; check = "G0b", model = name, fold, sites = res.pillar_sites,
                             max_abs_base_delta = maximum(abs.(res.base_deltas)),
                             worst_abs = res.worst_abs, worst_rel = res.worst_rel,
                             pass = res.worst_rel <= R01_G0_TOL))
    end

    for (spine_name, five_name) in GSS_LINE_PAIRS
        res = gss_spine_line_identity(r01_models[spine_name], r01_models[five_name], fs)
        push!(r01_parity, (; check = "G0c", model = spine_name, fold, sites = "vs " * five_name,
                             max_abs_base_delta = NaN,
                             worst_abs = res.worst_abs, worst_rel = res.worst_rel,
                             pass = res.worst_rel <= R01_G0_TOL))
    end
end
r01_parity_frame = DataFrame(r01_parity)
for r in eachrow(r01_parity_frame)
    @printf("  %s %-34s fold %d  [%s]  |Δ_base|max=%.3e  worst_abs=%.2e  worst_rel=%.2e  %s\n",
            r.check, r.model, r.fold, r.sites, r.max_abs_base_delta, r.worst_abs, r.worst_rel,
            r.pass ? "PASS" : "FAIL")
end
all(r01_parity_frame.pass) ||
    error("G0 FAILED: the spine model is not the base model plus the stated pillars")

# %%
# ===================================================================
# 9. Gradient audit (G1)
# ===================================================================
# `gph_gradient_audit` throws on any threshold breach; the table is the cost comparison.
r01_gradients = NamedTuple[]
for name in R01_AUDITED, fold in 1:R01_CONFIG.smoke_folds
    fs = r01_inputs.feature_sets[fold]
    audit = gph_gradient_audit(r01_models[name], fs; replays = R01_CONFIG.gradient_replays)
    push!(r01_gradients, (; model = name, fold,
                            n_target = Int(first(fs).data[:n_target_steps]), audit...))
    @printf("  G1 %-36s fold %d  θ=%4d  tape=%6d  grad=%.3f ms  alloc=%7d B  RD/FD=%.1e  perturbed=%.1e\n",
            name, fold, audit.n_parameters, audit.tape_instructions, audit.gradient_ms,
            audit.allocated_bytes, audit.compiled_forward_error, audit.worst_perturbed_error)
end
r01_gradient_frame = DataFrame(r01_gradients)
r01_gradient_by = Dict((r.model, r.fold) => r for r in r01_gradients)
r01_reference_row(model, fold) = r01_gradient_by[(model, fold)]
r01_gradient_frame.tape_vs_baseline = [r.tape_instructions - r01_reference_row(R01_BASELINE, r.fold).tape_instructions
                                       for r in eachrow(r01_gradient_frame)]
r01_gradient_frame.tape_vs_five_strike = [r.tape_instructions - r01_reference_row(R01_FIVE_STRIKE, r.fold).tape_instructions
                                          for r in eachrow(r01_gradient_frame)]
r01_gradient_frame.grad_ratio_vs_five_strike = [r.gradient_ms / r01_reference_row(R01_FIVE_STRIKE, r.fold).gradient_ms
                                                for r in eachrow(r01_gradient_frame)]

# %%
# ===================================================================
# 10. Config registry (G6) — before sampling, so a serialisation failure costs nothing
# ===================================================================
r01_registry = try
    gss_register!(r01_db, r01_sampled, r01_splitter, r01_sampler, r01_configs)
catch err
    error("G6 FAILED: config registry refused a rung: " * sprint(showerror, err))
end
println("  G6 registered models ", r01_registry.model_ids)

# %%
# ===================================================================
# 11. Training — two folds per spine rung
# ===================================================================
# NUTS chains are single-threaded; QueuedExecution flattens 2 folds × 4 chains into one queue
# over the pinned threads. The rungs run one after the other, so each wall time is its own.
r01_fits = Dict{String,Any}()
for name in R01_SAMPLED
    println("\n--- sampling: ", name)
    r01_fits[name] = gph_sample(r01_configs[name], r01_inputs, gms_gph_config(R01_CONFIG))
end

# %%
# ===================================================================
# 12. Convergence (G2, G3) and β recovery (H2, early signal)
# ===================================================================
r01_beta = DataFrame[]
for name in R01_SAMPLED
    d = r01_fits[name].diagnostics
    failures = r01_failures[name]
    d.n_divergent == 0 || push!(failures, "G2 divergences=$(d.n_divergent)")
    d.max_rhat <= R01_CONFIG.max_rhat || push!(failures,
        @sprintf("G3 max R̂ %.4f (fold %d)", d.max_rhat, d.worst_rhat_fold))
    if R01_GATE_ESS && min(d.min_ess_bulk, d.min_ess_tail) < R01_CONFIG.min_ess
        push!(failures, @sprintf("G3 min ESS bulk %.0f / tail %.0f", d.min_ess_bulk, d.min_ess_tail))
    end
    push!(r01_beta, insertcols(gss_beta_by_fold(r01_fits[name]), 1, :model => name))
end
r01_beta_frame = vcat(r01_beta...)
r01_line = gss_task015_line_fit()
show(stdout, MIME"text/plain"(), r01_beta_frame; allcols = true)
println()
@printf("  Task 015 five-strike medians imply β_LS = %.4f; line residuals K=0…4: %s\n",
        r01_line.β_least_squares, join(round.(r01_line.residuals; digits = 3), " / "))

# %%
# ===================================================================
# 13. OOS latents and smile pricing (G4b)
# ===================================================================
r01_latent_summary = Dict{String,Any}()
r01_pricing = DataFrame[]
r01_reweight = DataFrame[]
r01_identity = NamedTuple[]
for name in R01_SAMPLED
    fit = r01_fits[name]
    failures = r01_failures[name]

    r01_latent_summary[name] = try
        gms_latent_audit(fit)
    catch err
        push!(failures, "G4b latent audit: " * sprint(showerror, err))
        nothing
    end
    if !(fit.latents isa SmileLatents)
        push!(failures, "G4b expected SmileLatents, got $(typeof(fit.latents))")
        continue
    end
    all(==(1.0), view(fit.latents.φ, :, 3, :)) ||
        push!(failures, "G4b φ at the 2.5 strike is not exactly 1")

    pricing = gms_smile_pricing_gate(fit)
    push!(r01_pricing, insertcols(pricing, 1, :model => name))
    maximum(abs.(pricing.p_under_typed .- pricing.p_under_ref)) <= R01_PRICING_TOL ||
        push!(failures, "G4b typed smile O/U price ≠ cdf(Poisson(λ_tot·φ))")
    maximum(abs.(pricing.p_under_legacy .- pricing.p_under_ref)) <= R01_PRICING_TOL ||
        push!(failures, "G4b legacy row route smile O/U price ≠ cdf(Poisson(λ_tot·φ))")

    reweight = gss_reweight_gate(fit.latents)
    push!(r01_reweight, insertcols(reweight, 1, :model => name))
    append!(failures, "G4b " .* gss_reweight_failures(reweight; tol = R01_G4_TOL,
                                                      spread_tol = R01_SPREAD_TOL))

    identity = gss_identity_path_gate(fit.latents)
    push!(r01_identity, (; model = name, identity...))
    identity.shortcut_bit_identical || push!(failures, "G4b φ ≡ 1 shortcut changed the grid")
    identity.forced_max_abs_gap <= R01_FORCED_IDENTITY_TOL || push!(failures,
        @sprintf("G4b un-shortcut φ ≡ 1 path moved the grid by %.2e", identity.forced_max_abs_gap))

    @printf("  G4b %-34s fixtures=%d  totals gap=%.2e  mass gap=%.2e  Δp_home=%+.5f  Δp_under25=%+.5f\n",
            name, nrow(reweight), maximum(reweight.max_draw_cdf_gap), maximum(reweight.max_mass_gap),
            mean(reweight.Δp_home), mean(reweight.Δp_under25))
end

# %%
# ===================================================================
# 14. Portfolio staking off the reweighted grid (G4c, ticket T011)
# ===================================================================
# Option B book spec, fold 1–2 OOS fixtures against their closing quotes. This is a coherence
# check of the stake vector, not a backtest: two folds settle nothing.
r01_book_spec = gss_option_b_book()
r01_staking = NamedTuple[]
for name in R01_SAMPLED
    lat = r01_fits[name].latents
    lat isa SmileLatents || continue
    failures = r01_failures[name]

    ledger = gss_identity_ledger_gate(r01_book_spec, lat, r01_ds.odds, r01_ds)
    staking = gss_staking_gate(r01_book_spec, lat, r01_ds.odds, r01_ds)
    push!(r01_staking, (; model = name, ledger..., staking...))

    ledger.n_books_flat > 0 || push!(failures, "G4c no books built for the φ ≡ 1 twin")
    ledger.n_books_flat == ledger.n_books_grid == ledger.n_identical || push!(failures,
        "G4c φ ≡ 1 ledger differs from the grid twin ($(ledger.n_identical) of " *
        "$(ledger.n_books_grid) identical, max stake gap $(ledger.max_flat_stake_gap))")
    staking.max_totals_gap <= R01_G4_TOL || push!(failures,
        @sprintf("G4c staked p_grid totals ≠ smile CDF (%.2e)", staking.max_totals_gap))

    @printf("  G4c %-34s books=%d  φ≡1 identical=%d/%d  totals gap=%.2e  stakes changed by φ=%d\n",
            name, staking.n_books, ledger.n_identical, ledger.n_books_grid,
            staking.max_totals_gap, staking.n_stake_changed)
end

# %%
# ===================================================================
# 15. Persistence round-trip (G5)
# ===================================================================
r01_run_ids = Dict{String,Any}()
for name in R01_SAMPLED
    r01_run_ids[name] = try
        gms_save_and_verify(r01_db, r01_fits[name], r01_inputs;
                            latent_dir = joinpath(R01_OUT_DIR, "latents"))
    catch err
        push!(r01_failures[name], "G5 " * sprint(showerror, err))
        nothing
    end
end

# %%
# ===================================================================
# 16. Final report
# ===================================================================
r01_summary = DataFrame([(; gss_convergence_row(name, r01_fits[name], R01_CONFIG;
                                                run_id = r01_run_ids[name])...,
                            gate_pass = isempty(r01_failures[name]),
                            gate_failures = join(r01_failures[name], "; "))
                         for name in R01_SAMPLED])
r01_pricing_frame = isempty(r01_pricing) ? DataFrame() : vcat(r01_pricing...)
r01_reweight_frame = isempty(r01_reweight) ? DataFrame() : vcat(r01_reweight...)
r01_staking_frame = DataFrame(r01_staking)
r01_identity_frame = DataFrame(r01_identity)

# Task 015's smoke at the same budget, for the wall-time comparison. Same host and thread count
# only if both reports say so — read the header lines before quoting a ratio.
r01_task015_smoke = isfile(R01_TASK015_SMOKE) ?
    select(CSV.read(R01_TASK015_SMOKE, DataFrame), :model, :max_rhat, :min_ess_bulk, :min_ess_tail,
           :n_divergent, :wall_min) : DataFrame()

CSV.write(joinpath(R01_OUT_DIR, "r01_smoke_gates.csv"), r01_summary)
CSV.write(joinpath(R01_OUT_DIR, "r01_parity.csv"), r01_parity_frame)
CSV.write(joinpath(R01_OUT_DIR, "r01_gradient_audit.csv"), r01_gradient_frame)
CSV.write(joinpath(R01_OUT_DIR, "r01_beta_by_fold.csv"), r01_beta_frame)
CSV.write(joinpath(R01_OUT_DIR, "r01_market_coverage.csv"), r01_coverage)
CSV.write(joinpath(R01_OUT_DIR, "r01_reweight_synthetic.csv"), r01_g4a)
nrow(r01_pricing_frame) > 0 && CSV.write(joinpath(R01_OUT_DIR, "r01_smile_pricing.csv"), r01_pricing_frame)
nrow(r01_reweight_frame) > 0 && CSV.write(joinpath(R01_OUT_DIR, "r01_reweight_fitted.csv"), r01_reweight_frame)
nrow(r01_staking_frame) > 0 && CSV.write(joinpath(R01_OUT_DIR, "r01_staking.csv"), r01_staking_frame)

open(joinpath(R01_OUT_DIR, "r01_smoke_report.md"), "w") do io
    println(io, "# r01 smoke gate — Task 016 (1-parameter smile spine)\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), " at `", R01_GIT,
            "` on ", gethostname(), " with ", Threads.nthreads(), " threads. Store latest kickoff ",
            maximum(r01_ds.matches.match_date), ".\n")
    println(io, "Sampler: QueuedNUTS, ", R01_CONFIG.smoke_chains, " chains × ",
            R01_CONFIG.smoke_warmup, " warmup + ", R01_CONFIG.smoke_samples,
            " retained, δ = ", R01_CONFIG.accept_rate, ". Folds 1–", R01_CONFIG.smoke_folds, ".\n")

    println(io, "## G4a anti-diagonal reweighting, synthetic container\n")
    print(io, gph_markdown_table(r01_g4a;
        formats = Dict(:max_draw_cdf_gap => v -> @sprintf("%.2e", v),
                       :max_mean_cdf_gap => v -> @sprintf("%.2e", v),
                       :max_mass_gap => v -> @sprintf("%.2e", v),
                       :min_cell => v -> @sprintf("%.2e", v),
                       :max_diag_spread => v -> @sprintf("%.2e", v),
                       :Δp_home => v -> gph_signed(v), :Δp_draw => v -> gph_signed(v),
                       :Δp_away => v -> gph_signed(v), :Δp_under25 => v -> gph_signed(v))))
    println(io, @sprintf("\nφ ≡ 1 shortcut bit-identical: %s. Un-shortcut path max |Δ|: %.2e. Non-monotone curve refused: %s.\n",
                         r01_g4a_identity.shortcut_bit_identical, r01_g4a_identity.forced_max_abs_gap,
                         r01_g4a_refusal.refused))

    println(io, "## G0 likelihood parity\n")
    print(io, gph_markdown_table(r01_parity_frame;
        formats = Dict(:max_abs_base_delta => v -> @sprintf("%.3e", v),
                       :worst_abs => v -> @sprintf("%.2e", v),
                       :worst_rel => v -> @sprintf("%.2e", v))))

    println(io, "\n## GB market coverage\n")
    print(io, gph_markdown_table(r01_coverage;
        formats = Dict(:supremacy_share => v -> gph_num(v; digits = 3),
                       :smile_share => v -> gph_num(v; digits = 3))))

    println(io, "\n## G1 gradient audit\n")
    println(io, "Tape and gradient-time columns compare each row with the baseline and the five-strike smile @0.40 on the same fold.\n")
    print(io, gph_markdown_table(select(r01_gradient_frame,
        :model, :fold, :n_parameters, :tape_instructions, :tape_vs_baseline, :tape_vs_five_strike,
        :gradient_ms, :grad_ratio_vs_five_strike, :allocated_bytes, :compiled_forward_error,
        :worst_perturbed_error);
        formats = Dict(:gradient_ms => v -> gph_num(v; digits = 3),
                       :grad_ratio_vs_five_strike => v -> gph_num(v; digits = 2),
                       :compiled_forward_error => v -> @sprintf("%.1e", v),
                       :worst_perturbed_error => v -> @sprintf("%.1e", v))))

    println(io, "\n## G2–G5 sampling, latents, persistence\n")
    print(io, gph_markdown_table(select(r01_summary,
        :model, :folds, :oos, :draws, :max_rhat, :min_ess_bulk, :min_ess_tail, :n_divergent,
        :min_bfmi, :wall_min, :σ_sup_median, :σ_smile_median, :κ_median, :β_median, :φ_median,
        :gate_pass, :run_id);
        formats = Dict(:wall_min => v -> gph_num(v; digits = 1),
                       :β_median => v -> gph_num(v; digits = 4))))

    println(io, "\n## β_spine per fold (H2, early signal)\n")
    println(io, @sprintf("Task 015's five-strike medians imply β_LS = %.4f, with line residuals %s at K = 0…4.\n",
                         r01_line.β_least_squares, join(round.(r01_line.residuals; digits = 3), " / ")))
    print(io, gph_markdown_table(r01_beta_frame;
        formats = Dict(:β_median => v -> gph_num(v; digits = 4), :β_q05 => v -> gph_num(v; digits = 4),
                       :β_q95 => v -> gph_num(v; digits = 4), :β_sd => v -> gph_num(v; digits = 4),
                       :rhat => v -> gph_num(v; digits = 4), :ess_bulk => v -> gph_num(v; digits = 0),
                       :ess_tail => v -> gph_num(v; digits = 0))))

    if nrow(r01_task015_smoke) > 0
        println(io, "\n## Task 015 smoke at the same budget (wall time and ESS, for comparison)\n")
        print(io, gph_markdown_table(r01_task015_smoke;
            formats = Dict(:wall_min => v -> gph_num(v; digits = 1))))
    end

    if nrow(r01_reweight_frame) > 0
        println(io, "\n## G4b reweighting on the fitted containers\n")
        summary = combine(groupby(r01_reweight_frame, :model),
                          nrow => :fixtures,
                          :max_draw_cdf_gap => maximum => :max_draw_cdf_gap,
                          :max_mass_gap => maximum => :max_mass_gap,
                          :max_diag_spread => maximum => :max_diag_spread,
                          :Δp_home => mean => :mean_Δp_home,
                          :Δp_draw => mean => :mean_Δp_draw,
                          :Δp_away => mean => :mean_Δp_away,
                          :Δp_under25 => mean => :mean_Δp_under25)
        print(io, gph_markdown_table(summary;
            formats = Dict(:max_draw_cdf_gap => v -> @sprintf("%.2e", v),
                           :max_mass_gap => v -> @sprintf("%.2e", v),
                           :max_diag_spread => v -> @sprintf("%.2e", v),
                           :mean_Δp_home => v -> gph_signed(v), :mean_Δp_draw => v -> gph_signed(v),
                           :mean_Δp_away => v -> gph_signed(v), :mean_Δp_under25 => v -> gph_signed(v))))
        println(io)
        print(io, gph_markdown_table(r01_identity_frame;
            formats = Dict(:forced_max_abs_gap => v -> @sprintf("%.2e", v))))
    end

    if nrow(r01_staking_frame) > 0
        println(io, "\n## G4c portfolio staking off the reweighted grid (T011)\n")
        print(io, gph_markdown_table(r01_staking_frame;
            formats = Dict(:max_flat_stake_gap => v -> @sprintf("%.2e", v),
                           :max_totals_gap => v -> @sprintf("%.2e", v))))
    end

    if nrow(r01_pricing_frame) > 0
        println(io, "\n## G4b smile O/U 2.5 pricing — three routes and the plain grid\n")
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
for r in eachrow(r01_summary)
    r.gate_pass || println("  ", r.model, ": ", r.gate_failures)
end
println("R01_DONE report=", joinpath(R01_OUT_DIR, "r01_smoke_report.md"))
