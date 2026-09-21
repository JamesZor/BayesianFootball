# ==============================================================================
# r01 — Feature compression & player ratings EDA (Scottish Lower, 56/57)
# ==============================================================================
#
# WHAT THIS IS
#   A read-only empirical audit of WHERE the supremacy signal of the production
#   Scottish Lower models is lost. The model's supremacy is four scalars,
#
#     η_h − η_a = γ + (α_h − α_a) + (β_a − β_h) + (w_att + w_def)·ΔL + 2·w_W·ΔW
#
#   and the market's is one, log(λ_mkt_h / λ_mkt_a). This runner measures the
#   spread of each term against the market's, at every stage of the pipeline that
#   could compress it:
#
#     H1  double shrinkage   Ridge λ = 1000 on the RAPM, then N(0, 0.3) on w_att/w_def
#     H2  cannibalisation    α/β, ΔL and ΔW all measure team quality
#     H3  wealth saturation  the Richards curve in ProductionWealthFeature
#
# WHAT THIS IS NOT
#   Not a fit, not a portfolio study, and not a claim that decompression is
#   profitable. Nothing here re-samples; chains are read from `mcmc_experiments`
#   by UUID and every number is a function of those chains, the cached DataStore
#   and the Betfair close.
#
# FILTRATION CONTRACT
#   Folds and feature sets are rebuilt with the Experiment 06 splitter, so fold k
#   here is fold k there. Every RAPM fit — production and sweep — is fit on that
#   fold's frozen history block (`fit_on = :history`). The Betfair close is used
#   only as a yardstick, never as a feature.
#
# USAGE
#   julia --project -t 6
#   include("current_development/feature_compression_eda/r01_feature_compression_eda.jl")
#
#   The expensive objects (DataStore, feature panels, fits, market inversion) are
#   built behind `@isdefined` guards so the file can be re-included section by
#   section in a warm REPL without rebuilding them.
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================
using CSV
using DataFrames
using Dates
using Distributions
using LinearAlgebra
using Printf
using Statistics
using ThreadPinning
using UUIDs

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

const FCE_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(FCE_ROOT, "experiments", "scottish_lower",
                 "06_joint_player_lineup_fusion", "l66_hierarchical_kappa_eval_loader.jl"))
include(joinpath(@__DIR__, "l01_feature_compression_loader.jl"))

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================
const R01_SEGMENT        = Data.ScottishLower()          # tournaments 56 + 57
const R01_N_FOLDS        = FCE_N_FOLDS                   # 40 walk-forward folds
const R01_EXPECTED_OOS   = FCE_N_OOS                     # 710 held-out fixtures
const R01_LAMBDAS        = FCE_LAMBDAS                   # ridge sweep
const R01_W_BENCH        = FCE_W_BENCH                   # production bench weight
const R01_MIN_MINUTES    = 450.0                         # "has a real rating" floor
const R01_FAVOURITE_N    = 15                            # fixtures in the attribution table
const R01_PRIOR_W_SD     = 0.30                          # N(0, 0.3) on w_att and w_def
const R01_PRIOR_WEALTH   = truncated(Normal(0.10, 0.05), lower = 0.0)

@info "output directory" FCE_RESULTS

# %%
# ==============================================================================
# 3. Data snapshot, models and fold panel
# ==============================================================================
if !@isdefined(r01_ds)
    const r01_ds = Data.load_datastore_cached(R01_SEGMENT)
end
@info "DataStore" matches=nrow(r01_ds.matches) lineups=nrow(r01_ds.lineups) tournaments=sort(unique(r01_ds.matches.tournament_id))

const R01_MODELS = Dict(name => model for (name, model) in l60_models())
const R01_M12 = R01_MODELS["m12_joint_hybrid_synergy"]
const R01_M05 = R01_MODELS["m05_joint_production_wealth"]
const R01_SPLITTER = l60_splitter()

if !@isdefined(r01_panel12)
    @info "building m12 feature panel (40 folds; each rebuilds the history-fit RAPM)"
    const r01_panel12 = fce_fold_panel(r01_ds, R01_M12, R01_SPLITTER; n_folds = R01_N_FOLDS)
end
if !@isdefined(r01_panel05)
    @info "building m05 feature panel (control: no lineup pillar)"
    const r01_panel05 = fce_fold_panel(r01_ds, R01_M05, R01_SPLITTER; n_folds = R01_N_FOLDS)
end
const r01_fs12, r01_oos12 = r01_panel12
const r01_fs05, r01_oos05 = r01_panel05

@info "fold panel" folds=length(r01_oos12) oos_fixtures=sum(nrow, r01_oos12)

# %%
# ==============================================================================
# 4. Posterior artefacts — the two stored runs, by UUID
# ==============================================================================
if !@isdefined(r01_fit12)
    const r01_db = PostgresStorage(FCE_RUNS.m12.experiment)
    const r01_fit12 = load_fit(UUID(FCE_RUNS.m12.run_id), r01_db)
    const r01_fit05 = load_fit(UUID(FCE_RUNS.m05.run_id), r01_db)
end
@info "runs" m12_folds=length(r01_fit12.folds) m05_folds=length(r01_fit05.folds) m12_latents=n_matches(r01_fit12.latents)

# The stored runs were later extended to 43 folds. Folds 1..40 are the canonical
# 24/25 + 25/26 cohort this study reports; 41..43 are the 26/27 extension.
const r01_chains12 = [r01_fit12.folds[i].chain for i in 1:R01_N_FOLDS]
const r01_chains05 = [r01_fit05.folds[i].chain for i in 1:R01_N_FOLDS]

# %%
# ==============================================================================
# 5. Per-fixture decomposition of the model's supremacy
# ==============================================================================
const r01_dec12 = reduce(vcat, [fce_decompose_fold(R01_M12, r01_chains12[i], r01_fs12[i][1],
                                                   r01_oos12[i]; fold = i,
                                                   w_bench = R01_W_BENCH)
                                for i in 1:R01_N_FOLDS])
const r01_dec05 = reduce(vcat, [fce_decompose_fold(R01_M05, r01_chains05[i], r01_fs05[i][1],
                                                   r01_oos05[i]; fold = i)
                                for i in 1:R01_N_FOLDS])

nrow(r01_dec12) == R01_EXPECTED_OOS || @warn "m12 OOS cohort is $(nrow(r01_dec12)), expected $R01_EXPECTED_OOS"

const r01_params12 = reduce(vcat, [fce_fold_parameters(R01_M12, r01_chains12[i]; fold = i)
                                   for i in 1:R01_N_FOLDS])
const r01_params05 = reduce(vcat, [fce_fold_parameters(R01_M05, r01_chains05[i]; fold = i)
                                   for i in 1:R01_N_FOLDS])

# VERIFICATION. The decomposition must reproduce the stored latents: the sum of
# the four terms is log(λ_h/λ_a) up to the posterior-mean-versus-mean-of-logs gap,
# and the shared κ cancels in the ratio.
let l = r01_fit12.latents
    idx = [match_index(l, m) for m in r01_dec12.match_id]
    stored = [mean(log.(l.λ_home[i, :] ./ l.λ_away[i, :])) for i in idx]
    Δ = abs.(stored .- r01_dec12.sup_model)
    @info "decomposition vs stored latents" max_abs_diff=maximum(Δ) mean_abs_diff=mean(Δ)
    global r01_latent_check = (; max = maximum(Δ), mean = mean(Δ))
end

# %%
# ==============================================================================
# 6. Market yardstick — the de-vigged close, inverted to (λ_mkt_h, λ_mkt_a)
# ==============================================================================
if !@isdefined(r01_book)
    const r01_book = fce_closing_book(r01_ds)
    const r01_market = fce_market_rates(r01_book, r01_dec12.match_id)
end
@info "market inversion" fixtures=nrow(r01_market) accepted=sum(r01_market.mkt_accepted)

const r01_extra = fce_extra_features(r01_ds, r01_dec12.match_id)

# One row per held-out fixture: model terms, market supremacy, extra covariates.
r01_panel = leftjoin(r01_dec12, r01_market; on = :match_id)
r01_panel = leftjoin(r01_panel, r01_extra; on = :match_id)
r01_panel = leftjoin(r01_panel,
                     select(r01_dec05, :match_id,
                            :sup_model => :sup_model_m05, :sup_team => :sup_team_m05,
                            :sup_wealth => :sup_wealth_m05, :mu_total => :mu_total_m05);
                     on = :match_id)
sort!(r01_panel, [:fold, :match_id])

# A left join makes every joined column `Union{Missing,T}`; an absent market is
# `NaN` from here on, which every helper in the loader already filters on.
for col in names(r01_panel)
    v = r01_panel[!, col]
    eltype(v) <: Union{Missing,Real} || continue
    Missing <: eltype(v) || continue
    r01_panel[!, col] = Float64[ismissing(x) ? NaN : Float64(x) for x in v]
end

# Model 1X2 probabilities, independent Poisson over the stored goal-rate draws.
# This is the goals arm of the joint observation, which is exactly what the score
# grid prices; it is stated as a diagnostic, not as the production pricing path.
# The pmf is built by the Poisson recurrence `p_k = p_{k-1}·λ/k` rather than by
# `pdf`/`cdf` per cell — 710 fixtures × 3,200 draws makes that difference minutes.
function r01_poisson_home_probs(latents, match_ids; max_goals::Int = 15)
    ph = Vector{Float64}(undef, length(match_ids))
    pd = similar(ph); pa = similar(ph)
    ph_buf = zeros(max_goals + 1); pa_buf = zeros(max_goals + 1)
    pmf!(buf, λ) = begin
        buf[1] = exp(-λ)
        for k in 2:length(buf)
            buf[k] = buf[k - 1] * λ / (k - 1)
        end
        buf
    end
    for (k, mid) in enumerate(match_ids)
        i = match_index(latents, mid)
        λh = @view latents.λ_home[i, :]
        λa = @view latents.λ_away[i, :]
        h = 0.0; d = 0.0; a = 0.0
        for s in eachindex(λh)
            pmf!(ph_buf, λh[s]); pmf!(pa_buf, λa[s])
            ph_s = 0.0; pd_s = 0.0; cum_a = 0.0
            for x in 0:max_goals
                px = ph_buf[x + 1]
                pd_s += px * pa_buf[x + 1]
                ph_s += px * cum_a              # away strictly fewer than x
                cum_a += pa_buf[x + 1]
            end
            h += ph_s; d += pd_s; a += 1 - ph_s - pd_s
        end
        n = length(λh)
        ph[k] = h / n; pd[k] = d / n; pa[k] = a / n
    end
    return ph, pd, pa
end

let (ph, pd, pa) = r01_poisson_home_probs(r01_fit12.latents, r01_panel.match_id)
    r01_panel.p_home_m12 = ph
    r01_panel.p_draw_m12 = pd
    r01_panel.p_away_m12 = pa
end
let (ph, pd, pa) = r01_poisson_home_probs(r01_fit05.latents, r01_panel.match_id)
    r01_panel.p_home_m05 = ph
    r01_panel.p_draw_m05 = pd
    r01_panel.p_away_m05 = pa
end

fce_write(r01_panel, "r01_fixture_panel.csv")
fce_write(r01_params12, "r01_fold_parameters_m12.csv")
fce_write(r01_params05, "r01_fold_parameters_m05.csv")

# %%
# ==============================================================================
# 7. Compression measured — supremacy responsiveness against the market
# ==============================================================================
const r01_mkt_ok = findall(isfinite, r01_panel.sup_market)
@info "fixtures with an inverted market supremacy" n=length(r01_mkt_ok)

function r01_slope(y, x)
    f = fce_ols(y, reshape(Float64.(x), :, 1))
    return (; slope = f.coef[2], se = f.se[2], intercept = f.coef[1],
            r2 = f.r2, n = f.n)
end

r01_slopes = DataFrame(
    quantity = String[], slope = Float64[], se = Float64[], intercept = Float64[],
    r2 = Float64[], sd_quantity = Float64[], sd_market = Float64[], n = Int[])

let sub = r01_panel[r01_mkt_ok, :]
    for (label, col) in (("m12 total supremacy", :sup_model),
                         ("m12 team term", :sup_team),
                         ("m12 lineup term", :sup_lineup),
                         ("m12 wealth term", :sup_wealth),
                         ("m05 total supremacy", :sup_model_m05),
                         ("m05 team term", :sup_team_m05))
        s = r01_slope(sub[!, col], sub.sup_market)
        push!(r01_slopes, (label, s.slope, s.se, s.intercept, s.r2,
                           fce_std(sub[!, col]), fce_std(sub.sup_market), s.n))
    end
end
fce_write(r01_slopes, "r01_supremacy_slopes.csv")
println("\n── 7.1 Supremacy responsiveness (regressed on market log-rate supremacy) ──")
println(fce_round(r01_slopes))

# Probability-scale ceiling: the market's favourites against the models'.
r01_tail = DataFrame(cohort = String[], n = Int[], market_mean = Float64[],
                     m12_mean = Float64[], m05_mean = Float64[])
let sub = dropmissing(select(r01_panel, :p_home_mkt, :p_home_m12, :p_home_m05), :p_home_mkt)
    for (label, mask) in (("home favourite ≥ 0.60", sub.p_home_mkt .>= 0.60),
                          ("home favourite ≥ 0.70", sub.p_home_mkt .>= 0.70),
                          ("home favourite ≥ 0.80", sub.p_home_mkt .>= 0.80),
                          ("home longshot < 0.15", sub.p_home_mkt .< 0.15))
        any(mask) || continue
        push!(r01_tail, (label, sum(mask), mean(sub.p_home_mkt[mask]),
                         mean(sub.p_home_m12[mask]), mean(sub.p_home_m05[mask])))
    end
    push!(r01_tail, ("maximum over cohort", nrow(sub), maximum(sub.p_home_mkt),
                     maximum(sub.p_home_m12), maximum(sub.p_home_m05)))
end
fce_write(r01_tail, "r01_probability_tails.csv")
println("\n── 7.2 Probability tails (independent-Poisson home win) ──")
println(fce_round(r01_tail))

# %%
# ==============================================================================
# 8. H1 — the two-stage variance loss in the player ratings
# ==============================================================================
#
# Stage 0  raw plus-minus            no adjustment, no penalty
# Stage 1  ridge RAPM (λ = 1000)     the production rating
# Stage 2  w_att·L                   the rating after the Bayesian weight
#
# measured per fold on that fold's own permitted history, and reported as the
# standard deviation that survives each stage.
const r01_fold_history = [Int.(r01_fs12[i][1].data[:history_match_ids]) for i in 1:R01_N_FOLDS]

if !@isdefined(r01_designs)
    @info "building one ridge design per fold (reused by every λ)"
    const r01_designs = [fce_ridge_design(r01_ds, r01_fold_history[i]) for i in 1:R01_N_FOLDS]
    # Stage 0 and the exposure floor are also fold properties, not λ properties.
    const r01_raw = [fce_raw_plus_minus(r01_ds, r01_fold_history[i]) for i in 1:R01_N_FOLDS]
    const r01_exposed = [[p for (p, m) in r01_raw[i][2] if m >= R01_MIN_MINUTES]
                         for i in 1:R01_N_FOLDS]
end
@info "ridge designs" built=count(!isnothing, r01_designs)

r01_variance_chain = DataFrame(
    fold = Int[], n_segments = Int[], n_players = Int[], n_players_exposed = Int[],
    sd_raw_pm = Float64[], sd_raw_pm_exposed = Float64[],
    sd_ridge = Float64[], sd_ridge_exposed = Float64[],
    ridge_retention = Float64[],
    sd_delta_lineup = Float64[], sd_sup_lineup = Float64[],
    w_lineup_sum = Float64[], prior_sd_sup_lineup = Float64[],
    sd_sup_team = Float64[], sd_sup_market = Float64[])

for i in 1:R01_N_FOLDS
    design = r01_designs[i]
    design === nothing && continue
    raw, _ = r01_raw[i]
    ridge = fce_ridge_ratings(design, FCE_PRODUCTION_LAMBDA)
    exposed = r01_exposed[i]

    sub = r01_panel[r01_panel.fold .== i, :]
    w_sum = r01_params12.w_lineup_sum[i]
    sd_ΔL = fce_std(sub.delta_lineup)

    push!(r01_variance_chain, (
        i, design.n_segments, design.n_players, length(exposed),
        fce_std(collect(values(raw))),
        fce_std([raw[p] for p in exposed if haskey(raw, p)]),
        fce_std(collect(values(ridge))),
        fce_std([get(ridge, p, NaN) for p in exposed]),
        fce_std(collect(values(ridge))) / fce_std(collect(values(raw))),
        sd_ΔL, fce_std(sub.sup_lineup), w_sum,
        # what the PRIOR alone would have allowed this term to be:
        # sd((w_att + w_def)·ΔL) with w ~ N(0, 0.3) independent ⇒ √2 · 0.3 · sd(ΔL)
        sqrt(2) * R01_PRIOR_W_SD * sd_ΔL,
        fce_std(sub.sup_team), fce_std(sub.sup_market)))
end
fce_write(r01_variance_chain, "r01_variance_chain.csv")
println("\n── 8.1 Two-stage variance loss, first and last fold and the mean ──")
println(fce_round(r01_variance_chain[[1, R01_N_FOLDS], :]))
println(fce_round(DataFrame(describe(r01_variance_chain, :mean)[:, [:variable, :mean]])))

# %%
# ==============================================================================
# 8.2 How often is the rating actually refreshed?
# ==============================================================================
#
# `fit_on = :history` fits the ridge on the fold's frozen HISTORY block, which is
# the last `history_seasons` complete seasons — not "everything before kickoff".
# If that block is identical across a run of folds, so is the rating vector, and
# ΔL moves within a season only because the teamsheets move. This is the
# staleness the compression argument has to be read against.
r01_history_blocks = DataFrame(
    fold = collect(1:R01_N_FOLDS),
    n_history = [length(ids) for ids in r01_fold_history],
    block_id = [hash(sort(ids)) for ids in r01_fold_history],
    last_history_date = [maximum(r01_ds.matches.match_date[
        in.(Int.(r01_ds.matches.match_id), Ref(Set(ids)))]) for ids in r01_fold_history],
    first_oos_date = [minimum(r01_oos12[i].match_date) for i in 1:R01_N_FOLDS])
r01_history_blocks.rating_age_days =
    Dates.value.(r01_history_blocks.first_oos_date .- r01_history_blocks.last_history_date)
r01_history_blocks.block_index =
    [findfirst(==(b), unique(r01_history_blocks.block_id)) for b in r01_history_blocks.block_id]
fce_write(r01_history_blocks, "r01_history_blocks.csv")
println("\n── 8.2 Distinct history blocks across the 40 folds: ",
        length(unique(r01_history_blocks.block_id)), " ──")
println(fce_round(combine(groupby(r01_history_blocks, :block_index),
                          nrow => :folds,
                          :n_history => first => :n_history,
                          :last_history_date => first => :last_history_match,
                          :rating_age_days => minimum => :min_age_days,
                          :rating_age_days => maximum => :max_age_days)))

# %%
# ==============================================================================
# 9. H1 — ridge λ sensitivity sweep
# ==============================================================================
#
# Same folds, same aggregation, same held-out fixtures; only λ moves. The two
# questions are how much spread the penalty removes (σ of the ratings and of ΔL)
# and whether the removed spread was signal — measured against the market's own
# supremacy, both raw and after the team and wealth terms are partialled out.
r01_lambda_sweep = DataFrame(
    lambda = Float64[], fold = Int[], sd_rating = Float64[],
    sd_rating_exposed = Float64[], sd_delta_lineup = Float64[])

r01_lambda_pooled = DataFrame(
    lambda = Float64[], n = Int[], sd_rating_mean = Float64[],
    sd_delta_lineup = Float64[], cor_market = Float64[],
    partial_cor_market = Float64[], ols_weight = Float64[], ols_se = Float64[],
    posterior_weight = Float64[])

const r01_sweep_deltas = Dict{Float64,Vector{Float64}}()

for λ in R01_LAMBDAS
    ΔL_all = fill(NaN, nrow(r01_panel))
    for i in 1:R01_N_FOLDS
        design = r01_designs[i]
        design === nothing && continue
        ratings = fce_ridge_ratings(design, λ)
        exposed = r01_exposed[i]

        rows = findall(==(i), r01_panel.fold)
        ΔL = fce_lineup_delta_from_ratings(r01_ds, ratings, r01_panel.match_id[rows];
                                           w_bench = R01_W_BENCH)
        ΔL_all[rows] = ΔL
        push!(r01_lambda_sweep, (λ, i, fce_std(collect(values(ratings))),
                                 fce_std([get(ratings, p, NaN) for p in exposed]),
                                 fce_std(ΔL)))
    end
    r01_sweep_deltas[λ] = ΔL_all

    ok = findall(i -> isfinite(ΔL_all[i]) && isfinite(r01_panel.sup_market[i]), 1:nrow(r01_panel))
    Z = hcat(r01_panel.sup_team[ok], r01_panel.delta_prod_wealth[ok])
    fit = fce_ols(r01_panel.sup_market[ok], reshape(ΔL_all[ok], :, 1))
    push!(r01_lambda_pooled, (
        λ, length(ok),
        mean(r01_lambda_sweep.sd_rating[r01_lambda_sweep.lambda .== λ]),
        fce_std(ΔL_all[ok]),
        fce_cor(ΔL_all[ok], r01_panel.sup_market[ok]),
        fce_partial_corr(ΔL_all[ok], r01_panel.sup_market[ok], Z),
        fit.coef[2], fit.se[2],
        λ == FCE_PRODUCTION_LAMBDA ? mean(r01_params12.w_lineup_sum) : NaN))
end
fce_write(r01_lambda_sweep, "r01_lambda_sweep_by_fold.csv")
fce_write(r01_lambda_pooled, "r01_lambda_sweep_pooled.csv")
println("\n── 9.1 Ridge λ sweep, pooled over the 40 folds ──")
println(fce_round(r01_lambda_pooled))

# %%
# ==============================================================================
# 10. H2 — collinearity between the three quality measures
# ==============================================================================
const R01_DESIGN_COLS = [:sup_team, :delta_lineup, :delta_prod_wealth, :log_dist_z]
const R01_DESIGN_LABELS = ["team α/β contrast", "lineup ΔL", "production wealth ΔW",
                           "travel log-dist z"]

let sub = r01_panel[r01_mkt_ok, :]
    X = hcat((Float64.(coalesce.(sub[!, c], 0.0)) for c in R01_DESIGN_COLS)...)
    C = cor(X)
    global r01_corr = DataFrame(C, Symbol.(R01_DESIGN_LABELS))
    insertcols!(r01_corr, 1, :feature => R01_DESIGN_LABELS)
    global r01_vif = DataFrame(feature = R01_DESIGN_LABELS, vif = fce_vif(X))
    vd = fce_variance_decomposition(X)
    global r01_vdp = DataFrame(vd.proportions, Symbol.(R01_DESIGN_LABELS))
    insertcols!(r01_vdp, 1, :condition_index => vd.condition_indices)
    global r01_condition = fce_condition_number(X)

    # The market-implied weights on the same design, which is what the posterior
    # weights are being compared against.
    fit = fce_ols(sub.sup_market, X; names = R01_DESIGN_LABELS)
    global r01_market_weights = DataFrame(
        term = fit.labels, coef = fit.coef, se = fit.se, t = fit.t)
    global r01_market_fit_r2 = fit.r2
end
fce_write(r01_corr, "r01_design_correlations.csv")
fce_write(r01_vif, "r01_design_vif.csv")
fce_write(r01_vdp, "r01_design_variance_decomposition.csv")
fce_write(r01_market_weights, "r01_market_implied_weights.csv")
println("\n── 10.1 Design correlations ──");  println(fce_round(r01_corr))
println("\n── 10.2 VIF (condition number $(round(r01_condition, digits = 2))) ──")
println(fce_round(r01_vif))
println("\n── 10.3 Belsley variance decomposition ──"); println(fce_round(r01_vdp))
println("\n── 10.4 Market-implied weights (R² = $(round(r01_market_fit_r2, digits = 4))) ──")
println(fce_round(r01_market_weights))

# How much of each fixture-level feature is pure team identity — the quantity
# that decides whether a feature can be told apart from α/β at all. Measured on
# each fold's TRAINING matches, where there are enough fixtures per team.
r01_team_absorption = DataFrame(fold = Int[], n_train = Int[], n_teams = Int[],
                                r2_delta_lineup = Float64[], r2_delta_wealth = Float64[])
for i in 1:R01_N_FOLDS
    local d = r01_fs12[i][1].data
    ids = Int.(d[:ordered_match_ids])
    h = Int.(d[:flat_home_ids]); a = Int.(d[:flat_away_ids])
    n_teams = Int(d[:n_teams])
    ΔL = fce_lineup_delta(d[:player_lineup_ratings_map], ids; w_bench = R01_W_BENCH)
    ΔW = Float64.(d[:flat_delta_production_wealth])
    push!(r01_team_absorption, (i, length(ids), n_teams,
                                fce_team_dummy_r2(ΔL, h, a, n_teams),
                                fce_team_dummy_r2(ΔW, h, a, n_teams)))
end
fce_write(r01_team_absorption, "r01_team_absorption.csv")
println("\n── 10.5 Share of each feature explained by team identity alone (R²) ──")
println(fce_round(combine(r01_team_absorption,
                          :r2_delta_lineup => mean, :r2_delta_wealth => mean,
                          :r2_delta_lineup => minimum, :r2_delta_lineup => maximum)))

# %%
# ==============================================================================
# 11. Attribution on the market's heaviest favourites
# ==============================================================================
r01_fav = sort(r01_panel[r01_mkt_ok, :], :sup_market; rev = true)[1:R01_FAVOURITE_N, :]
r01_fav_table = select(r01_fav,
    :fold, :match_date, :home_team, :away_team,
    :sup_market, :sup_model, :sup_ha, :sup_team, :sup_lineup, :sup_wealth,
    :p_home_mkt, :p_home_m12, :p_home_m05,
    [:sup_market, :sup_model] => ByRow((m, s) -> m - s) => :supremacy_gap)
fce_write(r01_fav_table, "r01_favourite_attribution.csv")
println("\n── 11.1 The $(R01_FAVOURITE_N) heaviest market favourites ──")
println(fce_round(r01_fav_table))

# Average attribution by market-supremacy decile: which term stops responding.
r01_by_decile = let sub = r01_panel[r01_mkt_ok, :]
    q = [quantile(sub.sup_market, p) for p in range(0, 1; length = 11)]
    bin = [searchsortedlast(q, v) for v in sub.sup_market]
    bin = clamp.(bin, 1, 10)
    sub = copy(sub); sub.decile = bin
    combine(groupby(sub, :decile),
            nrow => :n,
            :sup_market => mean => :sup_market,
            :sup_model => mean => :sup_model,
            :sup_team => mean => :sup_team,
            :sup_lineup => mean => :sup_lineup,
            :sup_wealth => mean => :sup_wealth,
            :p_home_mkt => fce_mean => :p_home_mkt,
            :p_home_m12 => fce_mean => :p_home_m12)
end
sort!(r01_by_decile, :decile)
fce_write(r01_by_decile, "r01_supremacy_by_decile.csv")
println("\n── 11.2 Supremacy attribution by market decile ──")
println(fce_round(r01_by_decile))

# %%
# ==============================================================================
# 12. H3 — the wealth transform
# ==============================================================================
#
# `RichardsSigmoid` is an AGE-productivity curve inside the squad valuation, not a
# saturation applied to the wealth differential: the differential itself is a log
# ratio. §12.1 shows what the curve does to a player's value by age; §12.2
# contrasts the age-weighted differential with the raw log-sum one on the same
# fixtures.
const R01_CURVE = RichardsSigmoid(23.0, 0.80, 2.0)
r01_age_curve = DataFrame(age = [16.0, 18.0, 20.0, 23.0, 26.0, 29.0, 33.0, 37.0])
r01_age_curve.phi = [age_weight(R01_CURVE, a) for a in r01_age_curve.age]
r01_age_curve.phi_vs_prime = r01_age_curve.phi ./ age_weight(R01_CURVE, 27.0)
fce_write(r01_age_curve, "r01_age_curve.csv")
println("\n── 12.1 Richards age weight ──"); println(fce_round(r01_age_curve))

r01_wealth_contrast = let sub = r01_panel[r01_mkt_ok, :]
    prod = Float64.(coalesce.(sub.delta_prod_wealth, 0.0))
    logs = Float64.(coalesce.(sub.delta_wealth_logsum, 0.0))
    DataFrame(
        transform_name = ["production wealth (age-weighted)", "raw log-sum wealth"],
        sd = [fce_std(prod), fce_std(logs)],
        p05 = [quantile(prod, 0.05), quantile(logs, 0.05)],
        p95 = [quantile(prod, 0.95), quantile(logs, 0.95)],
        max_abs = [maximum(abs, prod), maximum(abs, logs)],
        cor_market = [fce_cor(prod, sub.sup_market), fce_cor(logs, sub.sup_market)],
        partial_cor_market = [
            fce_partial_corr(prod, sub.sup_market, reshape(sub.sup_team, :, 1)),
            fce_partial_corr(logs, sub.sup_market, reshape(sub.sup_team, :, 1))],
        ols_weight = [fce_ols(sub.sup_market, reshape(prod, :, 1)).coef[2],
                      fce_ols(sub.sup_market, reshape(logs, :, 1)).coef[2]],
        cor_between = fill(fce_cor(prod, logs), 2))
end
fce_write(r01_wealth_contrast, "r01_wealth_transform_contrast.csv")
println("\n── 12.2 Wealth transform contrast ──"); println(fce_round(r01_wealth_contrast))

# %%
# ==============================================================================
# 13. Prior-versus-posterior for the weights this study argues about
# ==============================================================================
r01_weight_trajectory = select(r01_params12, :fold, :w_att, :w_att_sd, :w_def, :w_def_sd,
                               :w_lineup_sum, :w_wealth, :w_wealth_sd, :sigma_a, :sigma_d)
fce_write(r01_weight_trajectory, "r01_weight_trajectory.csv")

r01_prior_posterior = DataFrame(
    parameter = ["lineup.w_att", "lineup.w_def", "production_wealth.w (m12)",
                 "production_wealth.w (m05)"],
    prior = ["Normal(0, 0.3)", "Normal(0, 0.3)",
             "truncated(Normal(0.10, 0.05), 0, ∞)", "truncated(Normal(0.10, 0.05), 0, ∞)"],
    prior_sd = [R01_PRIOR_W_SD, R01_PRIOR_W_SD, std(R01_PRIOR_WEALTH), std(R01_PRIOR_WEALTH)],
    posterior_mean = [mean(r01_params12.w_att), mean(r01_params12.w_def),
                      mean(r01_params12.w_wealth), mean(r01_params05.w_wealth)],
    posterior_sd_within_fold = [mean(r01_params12.w_att_sd), mean(r01_params12.w_def_sd),
                                mean(r01_params12.w_wealth_sd), mean(r01_params05.w_wealth_sd)],
    fold1 = [r01_params12.w_att[1], r01_params12.w_def[1],
             r01_params12.w_wealth[1], r01_params05.w_wealth[1]],
    fold40 = [r01_params12.w_att[end], r01_params12.w_def[end],
              r01_params12.w_wealth[end], r01_params05.w_wealth[end]])
r01_prior_posterior.posterior_over_prior_sd =
    r01_prior_posterior.posterior_mean ./ r01_prior_posterior.prior_sd
fce_write(r01_prior_posterior, "r01_prior_vs_posterior.csv")
println("\n── 13.1 Prior versus posterior ──"); println(fce_round(r01_prior_posterior))

# %%
# ==============================================================================
# 13.2 What a decompressed supremacy would price
# ==============================================================================
#
# A counterfactual, not a proposal: hold each fixture's total rate
# `μ = √(λ_h λ_a)` fixed and rescale ONLY the supremacy by `1/slope`, so the model
# responds one-for-one to the market. It answers the question the ceiling claim is
# really asking — is the 57.8% cap a property of the supremacy spread, or of the
# total rate? — without refitting anything.
function r01_rescaled_probs(latents, match_ids, scale::Float64; max_goals::Int = 15)
    ph = Vector{Float64}(undef, length(match_ids))
    for (k, mid) in enumerate(match_ids)
        i = match_index(latents, mid)
        λh = @view latents.λ_home[i, :]
        λa = @view latents.λ_away[i, :]
        acc = 0.0
        for s in eachindex(λh)
            μ = sqrt(λh[s] * λa[s])
            sup = log(λh[s] / λa[s]) * scale
            h = μ * exp(sup / 2); a = μ * exp(-sup / 2)
            ph_buf = [exp(-h)]; pa_buf = [exp(-a)]
            for x in 1:max_goals
                push!(ph_buf, ph_buf[end] * h / x)
                push!(pa_buf, pa_buf[end] * a / x)
            end
            p = 0.0; cum = 0.0
            for x in 0:max_goals
                p += ph_buf[x + 1] * cum
                cum += pa_buf[x + 1]
            end
            acc += p
        end
        ph[k] = acc / length(λh)
    end
    return ph
end

let scale = 1.0 / r01_slopes.slope[1]
    p = r01_rescaled_probs(r01_fit12.latents, r01_panel.match_id, scale)
    r01_panel.p_home_m12_rescaled = p
    sub = r01_panel[r01_mkt_ok, :]
    global r01_counterfactual = DataFrame(
        cohort = ["all inverted fixtures", "home favourite ≥ 0.70", "maximum"],
        n = [nrow(sub), sum(sub.p_home_mkt .>= 0.70), nrow(sub)],
        market = [mean(sub.p_home_mkt), mean(sub.p_home_mkt[sub.p_home_mkt .>= 0.70]),
                  maximum(sub.p_home_mkt)],
        m12 = [mean(sub.p_home_m12), mean(sub.p_home_m12[sub.p_home_mkt .>= 0.70]),
               maximum(sub.p_home_m12)],
        m12_rescaled = [mean(sub.p_home_m12_rescaled),
                        mean(sub.p_home_m12_rescaled[sub.p_home_mkt .>= 0.70]),
                        maximum(sub.p_home_m12_rescaled)])
    @info "supremacy rescaling factor" scale
end
fce_write(r01_counterfactual, "r01_counterfactual_rescale.csv")
println("\n── 13.2 Supremacy rescaled to market responsiveness ──")
println(fce_round(r01_counterfactual))

# %%
# ==============================================================================
# 14. Headline summary
# ==============================================================================
println("\n" * "="^78)
println("FEATURE COMPRESSION EDA — headline numbers")
println("="^78)
@printf("cohort                          %d folds, %d held-out fixtures, %d with an inverted book\n",
        R01_N_FOLDS, nrow(r01_panel), length(r01_mkt_ok))
@printf("decomposition vs stored latents max |Δ| = %.2e\n", r01_latent_check.max)
@printf("m12 supremacy slope vs market   %.4f   (m05 control %.4f)\n",
        r01_slopes.slope[1], r01_slopes.slope[5])
@printf("sd(model supremacy) / sd(market) %.4f  (m05 %.4f)\n",
        r01_slopes.sd_quantity[1] / r01_slopes.sd_market[1],
        r01_slopes.sd_quantity[5] / r01_slopes.sd_market[5])
@printf("ridge retention σ(RAPM)/σ(raw)  %.4f at λ = 1000\n",
        mean(r01_variance_chain.ridge_retention))
@printf("lineup term sd                  %.4f   (prior would allow %.4f)\n",
        mean(r01_variance_chain.sd_sup_lineup), mean(r01_variance_chain.prior_sd_sup_lineup))
@printf("team-identity R² of ΔL          %.4f\n", mean(r01_team_absorption.r2_delta_lineup))
@printf("design condition number         %.2f\n", r01_condition)
println("="^78)
