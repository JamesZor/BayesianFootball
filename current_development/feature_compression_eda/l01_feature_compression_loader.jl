# ==============================================================================
# Feature compression & player-ratings EDA — shared loader
# ==============================================================================
#
# Loader. Definitions only; `r01_feature_compression_eda.jl` executes.
#
# WHAT THIS FILE IS FOR. The production Scottish Lower models price a fixture with
#
#     η_h − η_a  =  γ  +  (α_h − α_a)  +  (β_a − β_h)  +  (w_att + w_def)·ΔL  +  2·w_W·ΔW
#                   ^HA   ^team attack   ^team defence   ^lineup RAPM pillar    ^production wealth
#
# so the whole supremacy of the model is FOUR scalars per fixture. Everything here
# measures where the spread of that sum is lost relative to the market's own
# supremacy `log(λ_mkt_h / λ_mkt_a)`: in the ridge that builds the player ratings,
# in the zero-mean priors that weight them, or in the collinearity between the
# three quality measures that makes them share one variance.
#
# NOTHING HERE SAMPLES. Chains are read from `mcmc_experiments` (run UUIDs in §1);
# feature sets are rebuilt from the cached DataStore with the SAME splitter the
# production grid used, so fold k's features here are fold k's features there.
#
# LEAK DISCIPLINE. Every RAPM fit — the production one and every λ on the sweep —
# is fit on `F_data[:history_match_ids]`, i.e. the frozen history block of that
# fold, exactly as `ShotsPlusMinusFeature(fit_on = :history)` does. The λ sweep
# reuses the production fold boundaries rather than inventing its own.
# ==============================================================================

using BayesianFootball
using CSV
using DataFrames
using Dates
using Distributions
using LinearAlgebra
using Printf
using SparseArrays
using Statistics
using UUIDs

import Serialization

const FCE_FEATURES = BayesianFootball.Features
const FCE_PG       = BayesianFootball.Models.PreGame
const FCE_INF      = BayesianFootball.Training.Inference
const FCE_CAL      = BayesianFootball.Calibration

const FCE_DIR     = @__DIR__
const FCE_RESULTS = joinpath(FCE_DIR, "results")
isdir(FCE_RESULTS) || mkpath(FCE_RESULTS)

# %%
# ==============================================================================
# 1. THE RUN MANIFEST  (posterior artefacts this study reads)
# ==============================================================================
#
# Addressed by UUID, never by name: `load_fit(db, name)` takes the NEWEST row with
# that name, and both of these names have newer rows. `m12` here is the 43-fold
# extension of the Experiment 06 grid; folds 1..40 are the canonical 710-fixture
# 24/25 + 25/26 cohort and §2 keeps only those.
const FCE_RUNS = (
    m12 = (name       = "m12_joint_hybrid_synergy",
           run_id     = "132df5c2-c742-4e95-8693-3aeb2b2cbaef",
           experiment = "scottish_lower_joint_player_2426",
           note       = "team time decay + shots-RAPM lineup (bench 0.10) + production wealth"),
    m05 = (name       = "m05_joint_production_wealth",
           run_id     = "ed541a7c-01e2-447e-a771-783517728d47",
           experiment = "scottish_lower_joint_player_2426",
           note       = "same recipe with the lineup pillar removed — the attribution control"),
)

const FCE_N_FOLDS = 40          # the canonical 24/25 + 25/26 walk-forward grid
const FCE_N_OOS   = 710         # held-out fixtures across those folds

"The Experiment 06 λ and the sweep around it. 1.0 is the near-unpenalised reference."
const FCE_LAMBDAS = [1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0]

const FCE_PRODUCTION_LAMBDA = 1000.0
const FCE_W_BENCH = 0.10

# %%
# ==============================================================================
# 2. FOLD PANEL — features and held-out fixtures, rebuilt from the cached store
# ==============================================================================
"""
    fce_fold_panel(ds, model, splitter; n_folds = FCE_N_FOLDS)
        -> (feature_sets, oos_frames)

`feature_sets[i]` is the `(FeatureSet, meta)` tuple fold `i` was fitted with and
`oos_frames[i]` the fixtures it was scored on — the same two objects `fit_model`
builds, from the same DataStore and splitter, so they align with the stored chains
fold for fold.
"""
function fce_fold_panel(ds, model, splitter; n_folds::Int = FCE_N_FOLDS)
    boundaries = Data.create_id_boundaries(ds, splitter)
    length(boundaries) >= n_folds || error(
        "splitter produced $(length(boundaries)) folds; need at least $n_folds")
    wanted = boundaries[1:n_folds]
    feature_sets = FCE_FEATURES.create_features(wanted, ds, model, splitter)
    oos = [Data.get_next_matches(ds, feature_sets[i], splitter) for i in 1:n_folds]
    return feature_sets, oos
end

"""
    fce_lineup_delta(lineup_map, match_ids; w_bench = FCE_W_BENCH) -> Vector{Float64}

`ΔL`, the lineup design the model actually multiplies by `w_att`/`w_def`:
starters plus `w_bench` × bench, home minus away. Absent lineups are a hard zero,
which is what `predictor_oos` does with a missing bridge entry.
"""
function fce_lineup_delta(lineup_map, match_ids; w_bench::Float64 = FCE_W_BENCH)
    neutral = FCE_FEATURES._pm_empty_lineup_aggregate()
    out = Vector{Float64}(undef, length(match_ids))
    for (i, mid) in enumerate(match_ids)
        v = get(lineup_map, Int(mid), neutral)
        home = v.home_outfield + w_bench * v.home_bench
        away = v.away_outfield + w_bench * v.away_bench
        out[i] = home - away
    end
    return out
end

"Per-side lineup strength (starters + w_bench × bench), for tail diagnostics."
function fce_lineup_sides(lineup_map, match_ids; w_bench::Float64 = FCE_W_BENCH)
    neutral = FCE_FEATURES._pm_empty_lineup_aggregate()
    h = Vector{Float64}(undef, length(match_ids))
    a = similar(h)
    for (i, mid) in enumerate(match_ids)
        v = get(lineup_map, Int(mid), neutral)
        h[i] = v.home_outfield + w_bench * v.home_bench
        a[i] = v.away_outfield + w_bench * v.away_bench
    end
    return h, a
end

"""
    fce_bridge_column(fs, key, match_ids) -> Vector{Float64}

A point-in-time covariate bridge (`:production_wealth_oos_bridge_by_match_id`,
`:wealth_oos_bridge_by_match_id`, …) read in fixture order, missing ⇒ 0.0, which
is the neutral value the covariate contract imputes.
"""
function fce_bridge_column(fs, key::Symbol, match_ids)
    bridge = get(fs.data, key, Dict{Int,Float64}())
    return Float64[Float64(get(bridge, Int(mid), 0.0)) for mid in match_ids]
end

"""
    fce_extra_features(ds, match_ids) -> DataFrame

Covariates the m12 recipe does NOT carry but the collinearity section needs:
raw (un-age-weighted) log-sum squad wealth and standardised log travel distance.
Both extractors are self-contained functions of the store and the id list.
"""
function fce_extra_features(ds, match_ids)
    ids = Int.(match_ids)
    team_map = Dict{String,Int}()

    # `LogSumWealthFeature` is declared beside its covariate in the builder module,
    # not in `Features`; its `add_feature!` is the Features one.
    logsum = Dict{Symbol,Any}()
    FCE_FEATURES.add_feature!(logsum, FCE_PG.LogSumWealthFeature(), ids, team_map, ds)

    dist = Dict{Symbol,Any}()
    FCE_FEATURES.add_feature!(dist, FCE_FEATURES.DistanceFeature(metric = :log_dist_z),
                              ids, team_map, ds)

    return DataFrame(
        match_id            = ids,
        delta_wealth_logsum = Float64.(logsum[:flat_delta_wealth_logsum]),
        wealth_fallback     = Int.(logsum[:flat_wealth_fallback]),
        log_dist_z          = Float64.(dist[:flat_log_distance_z]),
        dist_miles          = Float64.(dist[:flat_distance_miles]),
    )
end

# %%
# ==============================================================================
# 3. POSTERIOR DECOMPOSITION — the four scalars, per fixture, per draw
# ==============================================================================
#
# Read the same way `extract_parameters` does (same extractors, same site names),
# but keeping the TERMS apart instead of summing them into η. `sup_*` columns are
# posterior means of each term's contribution to `η_h − η_a`; `*_sd` is that
# term's posterior standard deviation, which is what a "the prior is doing the
# work" claim has to be read against.

"`(w_att, w_def, w_bench)` draws for the lineup pillar, or `nothing` when the model has none."
function fce_lineup_weights(model, chain)
    for term in model.covariates
        term isa PlayerLineupPillar || continue
        return predictor_extract(chain, term, "lineup")
    end
    return nothing
end

"Posterior draws of a scalar covariate weight by site prefix, or `nothing`."
function fce_covariate_weight(model, chain, name::Symbol)
    for term in model.covariates
        term isa AbstractCovariateConfig || continue
        covariate_name(term) === name || continue
        return vec(Array(chain[Symbol("$(name).w")]))
    end
    return nothing
end

"""
    fce_decompose_fold(model, chain, fs, oos; fold, w_bench) -> DataFrame

One row per held-out fixture: the market-free half of the study. Columns

| group | columns |
|---|---|
| identity | `fold`, `match_id`, `match_date`, `home_team`, `away_team` |
| design   | `delta_lineup`, `delta_prod_wealth`, `lineup_home`, `lineup_away` |
| terms    | `sup_ha`, `sup_team`, `sup_lineup`, `sup_wealth` (+ `_sd`) |
| total    | `sup_model`, `sup_model_sd`, `mu_total` |

`sup_team = (α_h − α_a) + (β_a − β_h)` — the sign convention of the engine, where
`β` enters the OPPONENT's rate, so a large `β` is a leaky defence.
"""
function fce_decompose_fold(model, chain, fs, oos; fold::Int,
                            w_bench::Float64 = FCE_W_BENCH)
    d = fs.data
    n_teams   = Int(d[:n_teams])
    n_seasons = Int(d[:n_seasons])
    team_map  = d[:team_map]
    lineup_map = get(d, :player_lineup_ratings_map, Dict{Int,Any}())
    n_draws = size(chain, 1) * size(chain, 3)

    inter = FCE_PG.extract_interception(chain, model.interception, n_seasons)
    ha    = FCE_PG.extract_home_advantage(chain, model.home_advantage, n_teams)
    dyn   = FCE_PG.extract_dynamics(chain, model.dynamics, "dyn", n_teams)
    lw    = fce_lineup_weights(model, chain)
    ww    = fce_covariate_weight(model, chain, :production_wealth)

    ids = Int.(oos.match_id)
    wealth = fce_bridge_column(fs, :production_wealth_oos_bridge_by_match_id, ids)
    ΔL = fce_lineup_delta(lineup_map, ids; w_bench = w_bench)
    Lh, La = fce_lineup_sides(lineup_map, ids; w_bench = w_bench)

    rows = Vector{NamedTuple}(undef, nrow(oos))
    zero_draws = zeros(n_draws)
    for (i, row) in enumerate(eachrow(oos))
        h_idx = get(team_map, row.home_team, 0)
        a_idx = get(team_map, row.away_team, 0)
        α_h = h_idx > 0 ? dyn.α[:, h_idx] : zero_draws
        α_a = a_idx > 0 ? dyn.α[:, a_idx] : zero_draws
        β_h = h_idx > 0 ? dyn.β[:, h_idx] : zero_draws
        β_a = a_idx > 0 ? dyn.β[:, a_idx] : zero_draws

        γ = model.home_advantage isa FCE_PG.GlobalHomeAdvantage ? ha[:, 1] :
            (h_idx > 0 ? ha[:, h_idx] : zero_draws)
        team = (α_h .- α_a) .+ (β_a .- β_h)
        lineup = lw === nothing ? zero_draws : (lw.w_att .+ lw.w_def) .* ΔL[i]
        wealth_term = ww === nothing ? zero_draws : 2.0 .* ww .* wealth[i]
        total = γ .+ team .+ lineup .+ wealth_term

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        base = inter.μ_base[:, s_idx] .+ inter.δ_month[:, Dates.month(row.match_date)]
        μ_total = mean(exp.(base .+ γ .+ α_h .+ β_a) .+ exp.(base .+ α_a .+ β_h))

        rows[i] = (; fold, match_id = ids[i], match_date = row.match_date,
                   home_team = String(row.home_team), away_team = String(row.away_team),
                   home_idx = h_idx, away_idx = a_idx,
                   delta_lineup = ΔL[i], lineup_home = Lh[i], lineup_away = La[i],
                   delta_prod_wealth = wealth[i],
                   sup_ha = mean(γ), sup_team = mean(team),
                   sup_lineup = mean(lineup), sup_wealth = mean(wealth_term),
                   sup_ha_sd = std(γ), sup_team_sd = std(team),
                   sup_lineup_sd = std(lineup), sup_wealth_sd = std(wealth_term),
                   sup_model = mean(total), sup_model_sd = std(total),
                   mu_total = μ_total)
    end
    return DataFrame(rows)
end

"""
    fce_fold_parameters(model, chain; fold) -> DataFrame

One row per fold of the posterior weights this study argues about, plus what the
prior alone would have said. `*_prior_sd` is the prior standard deviation of the
same site, so `w_att` at 0.10 against a prior SD of 0.30 is a *fifth* of a prior
standard deviation, not "a small number".
"""
function fce_fold_parameters(model, chain; fold::Int)
    lw = fce_lineup_weights(model, chain)
    ww = fce_covariate_weight(model, chain, :production_wealth)
    σ_a = vec(Array(chain[Symbol("dyn.σ_a")]))
    σ_d = vec(Array(chain[Symbol("dyn.σ_d")]))
    γ   = vec(Array(chain[Symbol("ha.γ_global")]))

    w_att = lw === nothing ? Float64[] : lw.w_att
    w_def = lw === nothing ? Float64[] : lw.w_def

    return DataFrame(
        fold = fold,
        sigma_a = mean(σ_a), sigma_d = mean(σ_d), gamma_ha = mean(γ),
        w_att = lw === nothing ? NaN : mean(w_att),
        w_att_sd = lw === nothing ? NaN : std(w_att),
        w_def = lw === nothing ? NaN : mean(w_def),
        w_def_sd = lw === nothing ? NaN : std(w_def),
        w_lineup_sum = lw === nothing ? NaN : mean(w_att .+ w_def),
        w_wealth = ww === nothing ? NaN : mean(ww),
        w_wealth_sd = ww === nothing ? NaN : std(ww),
        p_w_att_pos = lw === nothing ? NaN : mean(w_att .> 0),
    )
end

# %%
# ==============================================================================
# 4. MARKET SUPREMACY — the yardstick
# ==============================================================================
"""
    fce_closing_book(ds) -> DataFrame

Betfair time-weighted-average closes over (−20m, 0m], de-vigged proportionally
within each (match, market, line). Same construction as
`l66_betfair_closing_odds`, which is what every Experiment 06 score was read
against — the point is that this study's benchmark is the same price.
"""
function fce_closing_book(ds)
    raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator(); window = (-20.0, 0.0))
    odds = DataFrame(
        match_id = Int.(raw.match_id),
        market_name = String.(raw.market_name),
        market_line = Float64.(raw.market_line),
        selection = Symbol.(raw.selection),
        odds_close = Float64.(raw.odds),
    )
    filter!(row -> isfinite(row.odds_close) && row.odds_close > 1.0, odds)
    odds.prob_implied_close = 1.0 ./ odds.odds_close
    transform!(groupby(odds, [:match_id, :market_name, :market_line]),
               :prob_implied_close => (p -> p ./ sum(p)) => :prob_fair_close)
    sort!(odds, [:match_id, :market_name, :market_line, :selection])
    return odds
end

"""
    fce_market_rates(book, match_ids) -> DataFrame

Nelder–Mead inversion of the de-vigged book back to `(λ_mkt_h, λ_mkt_a)` —
`Calibration.invert_market_rates`, the production inverter — plus the de-vigged
1X2 probabilities. `sup_market = log(λ_mkt_h / λ_mkt_a)` is the market's own
supremacy on the model's scale, which is the only way to compare the two without
the scale mismatch that makes a logit-on-logit slope uninterpretable.
"""
function fce_market_rates(book::AbstractDataFrame, match_ids)
    ids = Int.(match_ids)
    fits = FCE_CAL.invert_market_rates(book; match_ids = ids)
    probs = Dict{Int,Dict{Symbol,Float64}}()
    for r in eachrow(book)
        r.market_name == "1X2" || continue
        p = r.prob_fair_close
        (p === missing || !isfinite(p)) && continue
        get!(() -> Dict{Symbol,Float64}(), probs, Int(r.match_id))[Symbol(r.selection)] = Float64(p)
    end

    rows = NamedTuple[]
    for mid in ids
        f = get(fits, mid, nothing)
        pm = get(probs, mid, Dict{Symbol,Float64}())
        ph = get(pm, :home, NaN); pd = get(pm, :draw, NaN); pa = get(pm, :away, NaN)
        if f === nothing
            push!(rows, (; match_id = mid, lambda_mkt_h = NaN, lambda_mkt_a = NaN,
                         sup_market = NaN, mkt_accepted = false, mkt_sse = NaN,
                         p_home_mkt = ph, p_draw_mkt = pd, p_away_mkt = pa))
            continue
        end
        ok = f.accepted && isfinite(f.lambda_home) && isfinite(f.lambda_away)
        push!(rows, (; match_id = mid,
                     lambda_mkt_h = f.lambda_home, lambda_mkt_a = f.lambda_away,
                     sup_market = ok ? log(f.lambda_home / f.lambda_away) : NaN,
                     mkt_accepted = ok, mkt_sse = f.sse,
                     p_home_mkt = ph, p_draw_mkt = pd, p_away_mkt = pa))
    end
    return DataFrame(rows)
end

# %%
# ==============================================================================
# 5. THE RIDGE — one design per fold, every λ off the same normal equations
# ==============================================================================
#
# `Features.fit_ratings` rebuilds the design for every call. A λ sweep over 40
# folds does not need 320 designs, it needs 40: `XᵀWX` and `XᵀWy` do not depend on
# λ. `fce_ridge_design` builds them once and `fce_ridge_ratings` solves. The
# arithmetic is `fit_ratings`'s own — same `build_design`, same `penalty_matrix`,
# same `ridge_solve` — so λ = 1000 here reproduces the production rating vector.

struct FCERidgeDesign
    A::Matrix{Float64}            # XᵀWX
    b::Vector{Float64}            # XᵀWy
    RtR::SparseMatrixCSC{Float64,Int}
    cols::Any                     # Features.DesignCols
    n_segments::Int
    n_players::Int
    T_rating::Date
end

"""
    fce_ridge_design(ds, fit_ids; target, half_life) -> FCERidgeDesign | nothing

The weighted normal equations for one fold's permitted history, anchored at the
last match the fold may see. `nothing` when the subset is below `fit_ratings`'s
own 500-segment floor, which is the same refusal the extractor honours.
"""
function fce_ridge_design(ds, fit_ids; target::Symbol = :y_shots,
                          half_life::Float64 = 730.0)
    prep = FCE_FEATURES.pm_prepared(ds)
    nrow(prep.segments) == 0 && return nothing
    ids = Set(Int.(fit_ids))
    segs = prep.segments[in.(Int.(prep.segments.match_id), Ref(ids)), :]
    if target in FCE_FEATURES.PM_SHOT_TARGETS && hasproperty(segs, :covered)
        segs = segs[segs.covered, :]
    end
    nrow(segs) < 500 && return nothing

    date_of = Dict{Int,Date}(Int(r.match_id) => r.match_date for r in eachrow(ds.matches))
    fit_dates = [d for (i, d) in date_of if i in ids]
    T_rating = isempty(fit_dates) ? maximum(segs.match_date) : maximum(fit_dates)

    X, y, w, cols = FCE_FEATURES.build_design(
        segs; target = target,
        weights = FCE_FEATURES.SegmentWeights(; half_life_days = half_life),
        T_rating = T_rating,
        comp_sets = FCE_FEATURES.competition_sets(ds; match_ids = ids))

    A = Matrix(Symmetric(Matrix(X' * Diagonal(w) * X)))
    b = Vector(X' * (w .* y))
    RtR = FCE_FEATURES.penalty_matrix(cols, nothing, 0.0)   # w_sim = 0: plain ridge
    return FCERidgeDesign(A, b, RtR, cols, nrow(segs), length(cols.player_ids), T_rating)
end

"`player_id => rating` at this λ, off a design already built."
function fce_ridge_ratings(design::FCERidgeDesign, λ::Float64)
    β = FCE_FEATURES.ridge_solve(design.A, design.b, design.RtR, λ)
    np = design.n_players
    return Dict{Int,Float64}(design.cols.player_ids[i] => β[i] for i in 1:np)
end

"""
    fce_raw_plus_minus(ds, fit_ids; target, half_life) -> (ratings, exposure)

Stage 0 of the variance chain: the UNADJUSTED plus-minus. Each player's
duration-and-recency-weighted on-pitch shot differential per 90, with no
teammate/opponent adjustment and no penalty —

    r_p = Σ_s w_s · sign_{p,s} · y_s  /  Σ_s w_s · (d_s / 90)

over the segments `p` was on the pitch for. `exposure[p]` is `Σ_s d_s`, the
minutes behind that number, which is what separates a real rating from a
one-substitute-appearance artefact.
"""
function fce_raw_plus_minus(ds, fit_ids; target::Symbol = :y_shots,
                            half_life::Float64 = 730.0)
    prep = FCE_FEATURES.pm_prepared(ds)
    ids = Set(Int.(fit_ids))
    segs = prep.segments[in.(Int.(prep.segments.match_id), Ref(ids)), :]
    if target in FCE_FEATURES.PM_SHOT_TARGETS && hasproperty(segs, :covered)
        segs = segs[segs.covered, :]
    end
    nrow(segs) < 500 && return (Dict{Int,Float64}(), Dict{Int,Float64}())

    date_of = Dict{Int,Date}(Int(r.match_id) => r.match_date for r in eachrow(ds.matches))
    fit_dates = [d for (i, d) in date_of if i in ids]
    T_rating = isempty(fit_dates) ? maximum(segs.match_date) : maximum(fit_dates)
    wcfg = FCE_FEATURES.SegmentWeights(; half_life_days = half_life)

    num = Dict{Int,Float64}(); den = Dict{Int,Float64}(); minutes = Dict{Int,Float64}()
    for seg in eachrow(segs)
        w = FCE_FEATURES.segment_weight(seg, wcfg, T_rating)
        y = Float64(seg[target])
        scale = seg.duration / 90.0
        for (players, sgn) in ((seg.home_players, 1.0), (seg.away_players, -1.0))
            for p in players
                num[p] = get(num, p, 0.0) + w * sgn * y
                den[p] = get(den, p, 0.0) + w * scale
                minutes[p] = get(minutes, p, 0.0) + seg.duration
            end
        end
    end
    ratings = Dict{Int,Float64}(p => (den[p] > 0 ? num[p] / den[p] : 0.0) for p in keys(num))
    return ratings, minutes
end

"""
    fce_lineup_delta_from_ratings(ds, ratings, match_ids; w_bench) -> Vector{Float64}

`ΔL` for an ARBITRARY rating vector, through the same aggregation the extractor
uses (`pm_lineup_aggregates`: starters at 1.0, bench at `w_bench`, keepers
dropped). This is what makes the λ sweep comparable to production — the ratings
change, the aggregation does not.
"""
function fce_lineup_delta_from_ratings(ds, ratings::Dict{Int,Float64}, match_ids;
                                       w_bench::Float64 = FCE_W_BENCH)
    aggregates = FCE_FEATURES.pm_lineup_aggregates(ds.lineups, ds.matches, ratings)
    return fce_lineup_delta(aggregates, match_ids; w_bench = w_bench)
end

# %%
# ==============================================================================
# 6. STATISTICS — OLS, VIF, conditioning, variance decomposition
# ==============================================================================
"""
    fce_ols(y, X; names) -> NamedTuple

Least squares with an intercept, returning `coef`, `se`, `t`, `r2`, `r2_adj`,
`sigma`, `n`. Written out rather than pulled from GLM.jl because every regression
in this study is a handful of columns and the diagnostic value is in `se` and
`r2` sitting beside the coefficient, not in a formula interface.
"""
function fce_ols(y::AbstractVector{<:Real}, X::AbstractMatrix{<:Real};
                 names::Vector{String} = String[])
    keep = findall(i -> isfinite(y[i]) && all(isfinite, @view X[i, :]), 1:length(y))
    yk = Float64.(y[keep]); Xk = Float64.(X[keep, :])
    n, k = size(Xk)
    D = hcat(ones(n), Xk)
    coef = D \ yk
    resid = yk .- D * coef
    dof = n - k - 1
    s2 = dof > 0 ? sum(abs2, resid) / dof : NaN
    covm = dof > 0 ? s2 .* inv(Symmetric(D' * D)) : fill(NaN, k + 1, k + 1)
    se = sqrt.(max.(diag(covm), 0.0))
    ss_tot = sum(abs2, yk .- mean(yk))
    r2 = 1.0 - sum(abs2, resid) / ss_tot
    labels = isempty(names) ? ["x$i" for i in 1:k] : names
    return (; coef, se, t = coef ./ se, r2,
            r2_adj = 1 - (1 - r2) * (n - 1) / max(dof, 1),
            sigma = sqrt(s2), n, labels = vcat("(intercept)", labels), resid, keep)
end

"Variance inflation factors, one per column of `X` (no intercept column)."
function fce_vif(X::AbstractMatrix{<:Real})
    n, k = size(X)
    out = Vector{Float64}(undef, k)
    for j in 1:k
        others = X[:, setdiff(1:k, j)]
        fit = fce_ols(X[:, j], others)
        out[j] = 1.0 / max(1.0 - fit.r2, eps())
    end
    return out
end

"Scaled condition number of the design (columns to unit length, Belsley's form)."
function fce_condition_number(X::AbstractMatrix{<:Real})
    S = fce_unit_columns(X)
    sv = svdvals(S)
    return maximum(sv) / minimum(sv)
end

function fce_unit_columns(X::AbstractMatrix{<:Real})
    S = Float64.(X)
    for j in 1:size(S, 2)
        nrm = norm(@view S[:, j])
        nrm > 0 && (S[:, j] ./= nrm)
    end
    return S
end

"""
    fce_variance_decomposition(X) -> (; condition_indices, proportions)

Belsley–Kuh–Welsch. `proportions[k, j]` is the share of column `j`'s coefficient
variance associated with singular value `k`; a near-dependency is a row with a
large condition index carrying > 0.5 of the variance of two or more columns.
"""
function fce_variance_decomposition(X::AbstractMatrix{<:Real})
    S = fce_unit_columns(X)
    F = svd(S)
    μ = F.S
    V = F.V
    k = length(μ)
    Φ = zeros(k, size(S, 2))
    for j in 1:size(S, 2)
        φ = [V[j, i]^2 / μ[i]^2 for i in 1:k]
        Φ[:, j] = φ ./ sum(φ)
    end
    return (; condition_indices = maximum(μ) ./ μ, proportions = Φ)
end

"Correlation of `x` and `y` after both are residualised on `Z` (with intercept)."
function fce_partial_corr(x::AbstractVector, y::AbstractVector, Z::AbstractMatrix)
    keep = findall(i -> isfinite(x[i]) && isfinite(y[i]) && all(isfinite, @view Z[i, :]),
                   1:length(x))
    rx = fce_ols(x[keep], Z[keep, :]).resid
    ry = fce_ols(y[keep], Z[keep, :]).resid
    return cor(rx, ry)
end

"Pearson correlation over the rows where both series are finite."
function fce_cor(x::AbstractVector, y::AbstractVector)
    keep = findall(i -> isfinite(x[i]) && isfinite(y[i]), 1:length(x))
    length(keep) < 3 && return NaN
    return cor(Float64.(x[keep]), Float64.(y[keep]))
end

"Mean over the finite entries; a cohort with no quoted market is `NaN`, not an error."
fce_mean(x) = (v = Float64[t for t in x if isfinite(t)]; isempty(v) ? NaN : mean(v))

"Standard deviation over the finite entries."
fce_std(x) = (v = Float64[t for t in x if isfinite(t)]; length(v) < 2 ? NaN : std(v))

"""
    fce_team_dummy_r2(values, home_idx, away_idx, n_teams) -> Float64

How much of a fixture-level DIFFERENCE column is explained by team identity alone:
regress it on the signed team-indicator design (+1 home, −1 away, one column per
team, first team dropped for identification). A column with `R² ≈ 1` carries no
information a team effect does not already carry — it cannot be identified beside
`α`/`β` except through the priors.
"""
function fce_team_dummy_r2(values::AbstractVector{<:Real}, home_idx::AbstractVector{<:Integer},
                           away_idx::AbstractVector{<:Integer}, n_teams::Int)
    n = length(values)
    n_teams < 2 && return NaN
    D = zeros(n, n_teams - 1)
    for i in 1:n
        h = home_idx[i]; a = away_idx[i]
        1 < h <= n_teams && (D[i, h - 1] += 1.0)
        1 < a <= n_teams && (D[i, a - 1] -= 1.0)
    end
    return fce_ols(values, D).r2
end

"`x` rescaled to zero mean and unit variance; constant columns pass through."
function fce_standardise(x::AbstractVector{<:Real})
    v = Float64.(x)
    s = std(v)
    return s > 0 ? (v .- mean(v)) ./ s : v .- mean(v)
end

"Round every Float64 column of a frame for printing."
function fce_round(df::AbstractDataFrame; digits::Int = 4)
    out = copy(df)
    for c in names(out)
        eltype(out[!, c]) <: Union{Missing,AbstractFloat} || continue
        out[!, c] = round.(out[!, c]; digits = digits)
    end
    return out
end

"Write a frame to `results/` and say so; every table in the report has a CSV."
function fce_write(df::AbstractDataFrame, name::AbstractString)
    path = joinpath(FCE_RESULTS, name)
    CSV.write(path, df)
    @info "wrote $(basename(path))  ($(nrow(df)) rows)"
    return path
end
