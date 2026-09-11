# ==============================================================================
# Task 013 loader — proper scores, paired bootstrap, portfolio attribution
# ==============================================================================
#
# Definitions only. `r04_evaluate.jl` and `r05_portfolio_attribution.jl` execute.
# Nothing here samples: every posterior is loaded from `mcmc_experiments` by UUID.
#
# ONE PANEL, ONE BOOK. Every arm — the four GRW ladder models and the TimeDecay
# controls — is scored on the identical 710 fixtures of the 24/25 + 25/26 walk-
# forward grid, against the identical de-vigged Betfair TWA[-20, 0] close. A run
# extended into 2026/27 carries extra fixtures; they are removed by restriction
# BEFORE scoring, never by filtering a score afterwards.
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Printf
using Random
using Statistics
using UUIDs

const GPH_EVAL = BayesianFootball.Evaluation
const GPH_PORTFOLIO = BayesianFootball.Portfolio
const GPH_BACKTESTING = BayesianFootball.BackTesting
const GPH_MD = BayesianFootball.MatchDay

# ==============================================================================
# 1. Arms
# ==============================================================================

"One scored posterior: where it lives, what it is, and which ladder rung it fills."
struct GPHArm
    label::String
    experiment::String
    run_id::UUID
    dynamics::String
    role::String
end

"The four ladder models by name from this task's namespace, then the pinned controls."
function gph_arms(c::GPHConfig)
    db = PostgresStorage(c.experiment)
    arms = GPHArm[]
    for name in GPH_MODEL_NAMES
        run_id = gph_run_by_name(db, name)
        run_id === nothing && error("no completed run named $name in $(c.experiment)")
        push!(arms, GPHArm(name, c.experiment, run_id, "MultiScaleGRW", "ladder"))
    end
    for ctl in GPH_CONTROLS
        dyn = occursin("grw", ctl.label) ? "MultiScaleGRW" : "TimeDecay(180)"
        push!(arms, GPHArm(ctl.label, ctl.experiment, ctl.run_id, dyn, "control"))
    end
    return arms
end

"""
    gph_load_arm(arm) -> Fit

Load by UUID and refuse anything that could make a comparison describe nothing: a
non-`Fit`, a non-`CountLatents` container, duplicate fixtures, or a synthetic
placeholder.
"""
function gph_load_arm(arm::GPHArm)
    fit = load_fit(PostgresStorage(arm.experiment), arm.run_id)
    fit isa Fit || error("$(arm.label) did not deserialize to a Fit")
    fit.latents isa CountLatents || error("$(arm.label) carries $(typeof(fit.latents))")
    fit.metadata.git_commit == "synthetic-no-mcmc" && error("$(arm.label) is synthetic")
    allunique(fit.latents.match_ids) || error("$(arm.label) has duplicate latent fixtures")
    return fit
end

# ==============================================================================
# 2. The book and the panel
# ==============================================================================

"""
    gph_betfair_closing_odds(ds) -> DataFrame

De-vigged Betfair time-weighted-average close over the (−20, 0] minute window —
byte-for-byte the frame Experiment 06 `r62` scored against, so its published
LogLoss/ECE are reproducible here.
"""
function gph_betfair_closing_odds(ds)
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
    winners = unique(select(ds.odds, :match_id, :market_name, :market_line, :selection, :is_winner))
    odds = leftjoin(odds, winners; on = [:match_id, :market_name, :market_line, :selection])
    sort!(odds, [:match_id, :market_name, :market_line, :selection])
    return odds
end

"The canonical walk-forward panel: every fixture of `seasons` the fit holds a latent for."
function gph_season_panel(ds, fit, seasons::Vector{String})
    season_of = Dict(Int(r.match_id) => String(r.season) for r in eachrow(ds.matches))
    return sort!([Int(m) for m in fit.latents.match_ids if get(season_of, Int(m), "") in seasons])
end

"The same run carrying only `panel`'s fixtures — the restriction precedes scoring."
function gph_restrict(fit, panel::Vector{Int})
    lat = restrict_latents(fit.latents, panel)
    n_matches(lat) == length(panel) || error(
        "restricted container holds $(n_matches(lat)) of $(length(panel)) panel fixtures")
    return Fit(fit.config, fit.folds, lat, fit.diagnostics, fit.metadata, fit.save_path)
end

"The market family of a scored row, for per-market cuts."
function gph_family(market_name::AbstractString, line::Real)
    m = lowercase(market_name)
    m == "1x2" && return "1X2"
    occursin("btts", m) && return "BTTS"
    (occursin("over", m) || occursin("under", m) || occursin("total", m)) &&
        return "OU" * string(line)
    return String(market_name)
end

"`family => selections` for the three families the work package scores."
function gph_family_selections(odds::AbstractDataFrame)
    out = Dict{String,Vector{Symbol}}()
    for r in eachrow(unique(select(odds, :market_name, :market_line, :selection)))
        fam = gph_family(r.market_name, r.market_line)
        fam in ("1X2", "OU2.5", "BTTS") || continue
        push!(get!(out, fam, Symbol[]), r.selection)
    end
    for (k, v) in out
        out[k] = sort!(unique(v))
    end
    return out
end

# ==============================================================================
# 3. Proper scores
# ==============================================================================

const GPH_METRICS = GPH_EVAL.AbstractScoringRule[
    GPH_EVAL.LogLoss(), GPH_EVAL.CRPS(), GPH_EVAL.PredictionScore()]

gph_context(fit, odds, ds) = build_evaluation_context(
    fit_latents(fit), odds, ds.matches, GPH_METRICS; threaded = true)

"""
    gph_scores(label, ctx, families) -> Vector{NamedTuple}

One row per scope (`all`, `1X2`, `OU2.5`, `BTTS`), each carrying the model's LogLoss,
Brier, ECE, MCE and RPS beside the Betfair close's on the SAME rows. RPS is ordered
and defined on 1X2 only; it is the same 1X2 number in every scope row.
"""
function gph_scores(label::AbstractString, ctx, families::Dict{String,Vector{Symbol}})
    rows = NamedTuple[]
    for (scope, sels) in vcat([("all", nothing)],
                              [(f, families[f]) for f in ("1X2", "OU2.5", "BTTS")
                               if haskey(families, f)])
        s = evaluate_predictions(ctx; selections = sels, n_bins = 10)
        push!(rows, (; model = String(label), scope,
                       n_obs = s.model.n_obs,
                       logloss = s.model.logloss, market_logloss = s.market.logloss,
                       brier = s.model.brier, market_brier = s.market.brier,
                       ece = s.model.ece, market_ece = s.market.ece,
                       mce = s.model.mce, market_mce = s.market.mce,
                       rps = s.model.rps, market_rps = s.market.rps))
    end
    return rows
end

"Per-observation frame: one row per scored (fixture, selection), for pairing."
function gph_observation_frame(label::AbstractString, ctx, odds::AbstractDataFrame)
    rows = evaluation_rows(ctx)
    family_of = Dict((Int(r.match_id), r.selection) => gph_family(r.market_name, r.market_line)
                     for r in eachrow(odds))
    df = DataFrame(
        match_id = [r.match_id for r in rows],
        selection = [r.selection for r in rows],
        p_model = [r.model_prob for r in rows],
        p_market = [r.market_prob for r in rows],
        y = [Float64(r.outcome) for r in rows],
    )
    df.family = [get(family_of, (m, s), "other") for (m, s) in zip(df.match_id, df.selection)]
    df.ll_model = GPH_EVAL.calc_logloss.(df.p_model, df.y)
    df.ll_market = GPH_EVAL.calc_logloss.(df.p_market, df.y)
    df.model .= String(label)
    return df
end

# ==============================================================================
# 4. Paired, fixture-clustered bootstrap
# ==============================================================================

"""
    gph_paired_bootstrap(a, b; B, seed, family) -> NamedTuple

ΔLogLoss = mean(ll_a − ll_b) over the rows both frames score, with a 95% interval
from `B` resamples of FIXTURES (not rows).

2,899 scored rows come from 710 fixtures, and a home, draw and away price on one
fixture are three readings of one scoreline. Resampling rows would treat them as
independent and shrink the interval by roughly √4; resampling fixtures keeps every
fixture's rows together, which is the honest unit of evidence. Each resample is the
row-weighted mean `Σ_g S_g / Σ_g n_g` over the drawn fixtures, so the point estimate
and the resampled statistic are the same functional.

`b === :market` pairs the model against the Betfair close on its own rows.
"""
function gph_paired_bootstrap(a::AbstractDataFrame, b; B::Int = 10_000, seed::Int = 20260911,
                              family::Union{Nothing,String} = nothing)
    sel = family === nothing ? a : a[a.family .== family, :]
    joined = if b === :market
        DataFrame(match_id = sel.match_id, d = sel.ll_model .- sel.ll_market)
    else
        bb = family === nothing ? b : b[b.family .== family, :]
        j = innerjoin(select(sel, :match_id, :selection, :ll_model => :ll_a),
                      select(bb, :match_id, :selection, :ll_model => :ll_b);
                      on = [:match_id, :selection])
        DataFrame(match_id = j.match_id, d = j.ll_a .- j.ll_b)
    end
    nrow(joined) == 0 && return (; n_obs = 0, n_fixtures = 0, delta = NaN, lo = NaN,
                                   hi = NaN, p_negative = NaN)
    g = combine(groupby(joined, :match_id), :d => sum => :s, nrow => :n)
    S = g.s
    N = Float64.(g.n)
    G = length(S)
    rng = MersenneTwister(seed)
    stats = Vector{Float64}(undef, B)
    idx = Vector{Int}(undef, G)
    @inbounds for k in 1:B
        rand!(rng, idx, 1:G)
        s = 0.0
        n = 0.0
        for i in idx
            s += S[i]
            n += N[i]
        end
        stats[k] = s / n
    end
    return (; n_obs = nrow(joined), n_fixtures = G,
              delta = sum(S) / sum(N),
              lo = quantile(stats, 0.025), hi = quantile(stats, 0.975),
              p_negative = mean(stats .< 0.0))
end

# ==============================================================================
# 5. Portfolio under the production Option B contract
# ==============================================================================

"""
    gph_option_b() -> (book, policy)

`MatchDay.option_b_system()` exactly: canonical markets, de-arbed prices, 30%
fractional Kelly, 2% commission; TieredTrust (Home, Under 2.5 at 1.0; Draw, Away,
Over 1.5 at 1/1.4; all else 0), SlateDrawdown(8.0), FixedCap(0.25), DailySlate.

The work package sketches `TieredTrust(base = 0.25)`, `FixedCap(0.20)` and
Baker-McHale on 1X2 + O/U 2.5. No `TieredTrust(base = …)` constructor exists, and
the audited production system is the one a live slate would stake under, so that
is the contract used here — for every arm, identically.
"""
function gph_option_b()
    system = GPH_MD.option_b_system()
    return getproperty(system, :book), getproperty(system, :policy)
end

"""
    gph_buildable_panel(book, fits, odds, ds, panel) -> (Vector{Int}, DataFrame)

Narrow `panel` to fixtures every arm can build a staking book for, and name every
fixture dropped and why. Book refusals (unplayed, no quotes, no usable selection)
hit every arm identically; removing their union once is what makes that true
rather than assumed.
"""
function gph_buildable_panel(book, fits::AbstractDict, odds, ds, panel::Vector{Int})
    dropped = Dict{Int,String}()
    for (_, fit) in fits
        _, report = GPH_PORTFOLIO.build_books_reported(
            book, gph_restrict(fit, panel), odds, ds; require_converged = false, quiet = true)
        for (ids, why) in ((report.skipped_no_fixture, "no fixture row"),
                           (report.skipped_unplayed, "unplayed"),
                           (report.skipped_no_quotes, "no quotes"),
                           (report.skipped_no_selections, "no usable selections"))
            for m in ids
                dropped[Int(m)] = why
            end
        end
        for (m, msg) in report.errored
            dropped[Int(m)] = "error: " * msg
        end
    end
    keep = sort!(collect(setdiff(Set(panel), keys(dropped))))
    frame = DataFrame(match_id = sort!(collect(keys(dropped))))
    frame.reason = [dropped[m] for m in frame.match_id]
    return keep, frame
end

"""
    gph_simulate(book, policy, fit, odds, ds, panel; label) -> PortfolioResult

Reprice one restricted posterior and simulate the policy. Refuses a book that
skipped a panel fixture: an arm that staked fewer fixtures than its comparator is
not comparable with it.
"""
function gph_simulate(book, policy, fit, odds, ds, panel::Vector{Int};
                      label::AbstractString, B::Int = 4000, seed::Int = 1)
    books, report = GPH_PORTFOLIO.build_books_reported(
        book, gph_restrict(fit, panel), odds, ds; require_converged = false, quiet = true)
    GPH_PORTFOLIO.n_skipped(report) == 0 || error(
        "$label skipped $(GPH_PORTFOLIO.n_skipped(report)) panel fixtures")
    length(books) == length(panel) || error(
        "$label built $(length(books)) books for $(length(panel)) fixtures")
    return GPH_PORTFOLIO.simulate_portfolio(
        policy, books, report;
        bootstrap = true, B = B, seed = seed,
        metrics = GPH_BACKTESTING.AbstractWealthMetric[
            GPH_BACKTESTING.CalmarRatio(), GPH_BACKTESTING.SharpeRatio()])
end

"One headline row. `stake`/`pnl` are fractions of the bankroll at each slate."
function gph_portfolio_row(label::AbstractString, arm_dynamics::AbstractString, result;
                           n_panel::Int)
    s = result.summary
    ci = result.bootstrap_ci
    bets = result.trajectory.bets
    return (; model = String(label), dynamics = String(arm_dynamics),
              n_panel, n_slates = s.n_slates, n_bets = s.n_bets,
              total_return_pct = s.total_return_pct,
              cagr_pct = 100 * s.cagr,
              growth_per_slate = s.growth_per_slate,
              growth_lo = ci === nothing ? NaN : ci.growth_lo,
              growth_hi = ci === nothing ? NaN : ci.growth_hi,
              roi_pct = s.roi,
              p_roi_positive = ci === nothing ? NaN : ci.p_roi_positive,
              sharpe_ann = s.sharpe_ann, calmar = s.calmar,
              max_drawdown_pct = s.mdd,
              win_rate_pct = 100 * s.win_rate,
              mean_edge_pp = nrow(bets) == 0 ? NaN : 100 * mean(bets.p_model .- bets.p_market),
              mean_exposure = s.mean_exposure)
end

# ==============================================================================
# 6. Bet-level attribution (Task 012 standards)
# ==============================================================================
#
# The ledger's units: `stake` is a FRACTION OF THE BANKROLL AT THAT SLATE and
# `pnl = stake × settle` in the same units. Summing `pnl` across a compounding path
# pools fractions of different bankrolls, so ROI (`Σpnl / Σstake`, path-normalised)
# and win rate are the comparators; `pnl` sums are reported as secondary.
#
# These definitions are Task 007 `l04_attribution_markets.jl` verbatim in substance,
# so the capture ratios here are on the same scale as the r04 numbers Task 012 cites.

gph_bet_key(r) = (Int(r.match_id), String(r.family), Symbol(r.selection))

"""
    gph_partition_bets(a, b) -> (both_a, both_b, only_a, only_b)

Shared bets (same fixture, family, selection — and therefore the same price and
settlement) seen from each model's ledger, plus the two exclusive sets. The shared
frames are row-aligned on the key, which is asserted.
"""
function gph_partition_bets(a::AbstractDataFrame, b::AbstractDataFrame)
    ak = [gph_bet_key(r) for r in eachrow(a)]
    bk = [gph_bet_key(r) for r in eachrow(b)]
    allunique(ak) || error("ledger A has duplicate bet keys")
    allunique(bk) || error("ledger B has duplicate bet keys")
    aset, bset = Set(ak), Set(bk)
    shared = intersect(aset, bset)
    A = copy(DataFrame(a)); A.key = ak
    Bf = copy(DataFrame(b)); Bf.key = bk
    both_a = sort!(filter(r -> r.key in shared, A), :key)
    both_b = sort!(filter(r -> r.key in shared, Bf), :key)
    only_a = sort!(filter(r -> !(r.key in bset), A), :key)
    only_b = sort!(filter(r -> !(r.key in aset), Bf), :key)
    both_a.key == both_b.key || error("shared sets are not row-aligned")
    return both_a, both_b, only_a, only_b
end

"""
    gph_edge_summary(bets) -> NamedTuple

Edge, confidence and sizing over one set of staked bets. `edge = p_model − p_market`
is positive on every staked bet (the allocator only stakes positive edges), so
`capture_ratio = E[edge | won] / E[edge | lost]` compares two positive means: > 1
says the model sits further from the market when it is right than when it is wrong.
`cap_weighted_win_rate` weights outcomes by stake — above the plain win rate means
the money landed on the winners.
"""
function gph_edge_summary(bets::AbstractDataFrame)
    n = nrow(bets)
    z = NaN
    n == 0 && return (; n_bets = 0, n_wins = 0, win_rate = z, cap_weighted_win_rate = z,
                        stake_sum = 0.0, pnl_sum = 0.0, roi = z, edge_mean = z,
                        edge_win = z, edge_loss = z, capture_ratio = z,
                        stake_mean = z, odds_mean = z, p_model_mean = z, p_market_mean = z)
    won = bets.payoff .> 0
    edge = bets.p_model .- bets.p_market
    stake_sum = sum(bets.stake)
    e_win = any(won) ? mean(edge[won]) : NaN
    e_loss = any(.!won) ? mean(edge[.!won]) : NaN
    return (; n_bets = n, n_wins = count(won), win_rate = mean(won),
              cap_weighted_win_rate = stake_sum > 0 ? sum(bets.stake .* won) / stake_sum : NaN,
              stake_sum, pnl_sum = sum(bets.pnl),
              roi = stake_sum > 0 ? 100 * sum(bets.pnl) / stake_sum : NaN,
              edge_mean = 100 * mean(edge), edge_win = 100 * e_win, edge_loss = 100 * e_loss,
              capture_ratio = (isfinite(e_loss) && e_loss > 0) ? e_win / e_loss : NaN,
              stake_mean = mean(bets.stake), odds_mean = mean(bets.odds),
              p_model_mean = mean(bets.p_model), p_market_mean = mean(bets.p_market))
end

"""
    gph_pair_attribution(name_a, name_b, bets_a, bets_b) -> (rows, sizing)

The three-way partition as reportable rows, plus the controlled sizing contrast.

On the shared set fixtures, selections, prices and outcomes are identical, so
`sizing_delta_pnl = Σ (s_a − s_b) · settle` is attributable to stake size and to
nothing else. `settle` is recovered as `pnl / stake` from each ledger and asserted
equal across the two, which is the check that "shared" really means the same bet.
"""
function gph_pair_attribution(name_a::AbstractString, name_b::AbstractString,
                              bets_a::AbstractDataFrame, bets_b::AbstractDataFrame)
    both_a, both_b, only_a, only_b = gph_partition_bets(bets_a, bets_b)
    rows = NamedTuple[]
    for (set, owner, frame) in (("shared", name_a, both_a), ("shared", name_b, both_b),
                                ("exclusive", name_a, only_a), ("exclusive", name_b, only_b))
        s = gph_edge_summary(frame)
        push!(rows, (; pair = "$(name_a) vs $(name_b)", bet_set = set, owner = String(owner),
                       n_bets = s.n_bets, n_wins = s.n_wins,
                       win_rate_pct = 100 * s.win_rate,
                       cap_weighted_win_rate_pct = 100 * s.cap_weighted_win_rate,
                       roi_pct = s.roi, stake_sum = s.stake_sum, pnl_sum = s.pnl_sum,
                       stake_mean = s.stake_mean, odds_mean = s.odds_mean,
                       edge_mean_pp = s.edge_mean, edge_win_pp = s.edge_win,
                       edge_loss_pp = s.edge_loss, capture_ratio = s.capture_ratio))
    end
    settle_a = both_a.pnl ./ both_a.stake
    settle_b = both_b.pnl ./ both_b.stake
    nrow(both_a) == 0 || maximum(abs.(settle_a .- settle_b)) < 1e-9 || error(
        "shared bets settle differently in the two ledgers — not the same bet")
    Δs = both_a.stake .- both_b.stake
    larger_a = Δs .> 0
    sizing = (; pair = "$(name_a) vs $(name_b)",
                n_shared = nrow(both_a),
                n_only_a = nrow(only_a), n_only_b = nrow(only_b),
                overlap_pct = 100 * nrow(both_a) / max(1, nrow(both_a) + nrow(only_a) + nrow(only_b)),
                stake_mean_a = nrow(both_a) == 0 ? NaN : mean(both_a.stake),
                stake_mean_b = nrow(both_b) == 0 ? NaN : mean(both_b.stake),
                n_a_larger = count(larger_a),
                roi_when_a_larger_pct = any(larger_a) ?
                    100 * sum(settle_a[larger_a] .* Δs[larger_a]) / sum(Δs[larger_a]) : NaN,
                roi_when_b_larger_pct = any(.!larger_a) ?
                    100 * sum(settle_a[.!larger_a] .* -Δs[.!larger_a]) / sum(-Δs[.!larger_a]) : NaN,
                sizing_delta_pnl = sum(Δs .* settle_a),
                shared_roi_a_pct = gph_edge_summary(both_a).roi,
                shared_roi_b_pct = gph_edge_summary(both_b).roi,
                capture_ratio_a = gph_edge_summary(bets_a).capture_ratio,
                capture_ratio_b = gph_edge_summary(bets_b).capture_ratio)
    return rows, sizing
end
