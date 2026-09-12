# ==============================================================================
# Task 014 loader — proper scores, paired bootstrap, portfolio attribution
# ==============================================================================
#
# Definitions only. `r04_evaluate.jl` and `r05_portfolio.jl` execute. Nothing here
# samples: every posterior is loaded from `mcmc_experiments` by UUID.
#
# ONE PANEL, ONE BOOK. Every arm — the four NegBin ladder models and the four Task 013
# Poisson controls — is scored on the IDENTICAL fixtures of the 24/25 + 25/26 walk-
# forward grid, against the IDENTICAL de-vigged Betfair TWA(−20, 0] close. The controls
# were extended in place to 43 folds / 769 fixtures, so they carry fixtures this study's
# runs do not; those are removed by restriction BEFORE scoring, never by filtering a
# score afterwards.
#
# WHAT IS DIFFERENT FROM TASK 013. The market scope. Task 013 scored `1X2`, `OU2.5` and
# `BTTS`, because its mechanism (team state, lineup ratings) moves the PREDICTOR and a
# predictor shift shows up in the result market. This task's mechanism moves the SHAPE
# of the count distribution at a fixed mean, which is invisible to 1X2 by construction
# and visible in the tail. So every totals line the book quotes is scored — 1.5, 2.5,
# 3.5, 4.5 — and 1X2 is carried as the control that says the mean did not move.
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Printf
using Random
using Statistics
using UUIDs

const GJN_EVAL = BayesianFootball.Evaluation
const GJN_PORTFOLIO = BayesianFootball.Portfolio
const GJN_BACKTESTING = BayesianFootball.BackTesting
const GJN_MD = BayesianFootball.MatchDay

# The scoring scopes, in report order. `all` is every scored row pooled; the rest are
# the market families the ladder is being judged on.
const GJN_SCOPES = ["1X2", "OU1.5", "OU2.5", "OU3.5", "OU4.5", "BTTS"]

"""
    GJN_MARKETS

The markets the evaluation context PRICES.

This has to be stated explicitly, and it is the single most important line in this file.
`Evaluation.DEFAULT_SCORED_MARKETS` is 1X2, O/U 2.5 and BTTS — that is what a `LogLoss()`
with no selection filter asks for, and it is what Task 013 and Experiment 06 scored. A
context built on the default would silently price no O/U 1.5, 3.5 or 4.5 at all, and the
totals hypothesis would be tested on the one line least able to show it: 2.5 sits nearest
the mean, which is exactly where a mean-preserving change of shape does the least.

The Betfair archive quotes O/U 0.5 through 5.5 on this league, so the lines below are
available rather than invented. 0.5 and 5.5 are left out deliberately: both are heavily
one-sided, and adding markets after seeing which ones moved is how a null becomes a
finding by accident. This set is the work package's, fixed before any score was computed.
"""
const GJN_MARKETS = Data.AbstractMarket[
    Data.Market1X2(), Data.MarketBTTS(),
    Data.MarketOverUnder(1.5), Data.MarketOverUnder(2.5),
    Data.MarketOverUnder(3.5), Data.MarketOverUnder(4.5),
]

"""
    GJN_LEGACY_MARKETS

`Evaluation.DEFAULT_SCORED_MARKETS` — 1X2, O/U 2.5, BTTS — used for ONE purpose: the
reproduction gate.

Task 013's published `m12` LogLoss of 0.64437 is a pooled score over these three markets
and their 2,899 rows. Reproducing it requires scoring on the same basis; pooling six
markets and comparing to a three-market figure would fail the gate for a reason that has
nothing to do with whether the control loaded correctly.
"""
const GJN_LEGACY_MARKETS = Data.AbstractMarket[
    Data.Market1X2(), Data.MarketOverUnder(2.5), Data.MarketBTTS(),
]

# ==============================================================================
# 1. Arms
# ==============================================================================

"One scored posterior: where it lives, what it is, and which ladder rung it fills."
struct GJNArm
    label::String
    experiment::String
    run_id::UUID
    likelihood::String
    role::String
end

"The four ladder models by name from this task's namespace, then the pinned controls."
function gjn_arms(c::GJNConfig)
    db = PostgresStorage(c.experiment)
    arms = GJNArm[]
    for name in GJN_MODEL_NAMES
        run_id = gjn_run_by_name(db, name)
        run_id === nothing && error("no completed run named $name in $(c.experiment)")
        joint = occursin("m05", name) || occursin("m12", name)
        push!(arms, GJNArm(name, c.experiment, run_id,
                           joint ? "Joint Gamma-NegBin" : "NegBin", "ladder"))
    end
    for ctl in GJN_CONTROLS
        joint = occursin("m05", ctl.label) || occursin("m12", ctl.label)
        push!(arms, GJNArm(ctl.label, ctl.experiment, ctl.run_id,
                           joint ? "Joint Gamma-Poisson" : "Poisson", "control"))
    end
    return arms
end

"""
    gjn_load_arm(arm) -> Fit

Load by UUID and refuse anything that could make a comparison describe nothing: a
non-`Fit`, a non-`CountLatents` container, duplicate fixtures, or a synthetic placeholder.

The observation-family check is the Task 014 addition. A ladder arm MUST carry
`observation_params` (it prices on the NegBin grid) and a control arm must NOT (it
prices on the double-Poisson grid). Getting this backwards would silently compare a
model against itself, which is the one failure mode a proper score cannot reveal.
"""
function gjn_load_arm(arm::GJNArm)
    fit = load_fit(PostgresStorage(arm.experiment), arm.run_id)
    fit isa Fit || error("$(arm.label) did not deserialize to a Fit")
    fit.latents isa CountLatents || error("$(arm.label) carries $(typeof(fit.latents))")
    fit.metadata.git_commit == "synthetic-no-mcmc" && error("$(arm.label) is synthetic")
    allunique(fit.latents.match_ids) || error("$(arm.label) has duplicate latent fixtures")

    has_dispersion = fit.latents.observation_params !== nothing
    if arm.role == "ladder"
        has_dispersion || error(
            "$(arm.label) is a NegBin ladder arm but carries no observation_params; " *
            "it would be priced on the double-Poisson grid")
    else
        has_dispersion && error(
            "$(arm.label) is a Poisson control but carries observation_params")
    end
    return fit
end

# ==============================================================================
# 2. The book and the panel
# ==============================================================================

"""
    gjn_betfair_closing_odds(ds) -> DataFrame

De-vigged Betfair time-weighted-average close over the (−20, 0] minute window —
byte-for-byte the frame Task 013 `r04` and Experiment 06 `r62` scored against, so their
published LogLoss/ECE are reproducible here as a gate.
"""
function gjn_betfair_closing_odds(ds)
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
function gjn_season_panel(ds, fit, seasons::Vector{String})
    season_of = Dict(Int(r.match_id) => String(r.season) for r in eachrow(ds.matches))
    return sort!([Int(m) for m in fit.latents.match_ids if get(season_of, Int(m), "") in seasons])
end

"""
    gjn_common_panel(ds, fits, seasons) -> Vector{Int}

The fixtures EVERY arm holds a latent for, inside `seasons`.

This is where the 43-fold controls are cut back to this study's 40-fold grid. Doing it
as a set intersection rather than by trusting a fold count means a control that was
extended again tomorrow still scores on the same rows as the ladder.
"""
function gjn_common_panel(ds, fits::AbstractDict, seasons::Vector{String})
    panels = [Set(gjn_season_panel(ds, fit, seasons)) for (_, fit) in fits]
    isempty(panels) && error("no arms to intersect")
    return sort!(collect(reduce(intersect, panels)))
end

"The same run carrying only `panel`'s fixtures — the restriction precedes scoring."
function gjn_restrict(fit, panel::Vector{Int})
    lat = restrict_latents(fit.latents, panel)
    n_matches(lat) == length(panel) || error(
        "restricted container holds $(n_matches(lat)) of $(length(panel)) panel fixtures")
    return Fit(fit.config, fit.folds, lat, fit.diagnostics, fit.metadata, fit.save_path)
end

"""
    gjn_family(market_name, line) -> String

The market family of a scored row, for per-market cuts. Totals keep their LINE in the
family name (`OU2.5`), because "does the negative binomial help on totals?" has a
different answer at 1.5 than at 3.5 — that difference IS the hypothesis.
"""
function gjn_family(market_name::AbstractString, line::Real)
    m = lowercase(market_name)
    m == "1x2" && return "1X2"
    occursin("btts", m) && return "BTTS"
    (occursin("over", m) || occursin("under", m) || occursin("total", m)) &&
        return "OU" * string(line)
    return String(market_name)
end

"`family => selections` for every family in `GJN_SCOPES` the book actually quotes."
function gjn_family_selections(odds::AbstractDataFrame)
    out = Dict{String,Vector{Symbol}}()
    for r in eachrow(unique(select(odds, :market_name, :market_line, :selection)))
        fam = gjn_family(r.market_name, r.market_line)
        fam in GJN_SCOPES || continue
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

const GJN_METRICS = GJN_EVAL.AbstractScoringRule[
    GJN_EVAL.LogLoss(), GJN_EVAL.CRPS(), GJN_EVAL.PredictionScore()]

"""
    gjn_context(fit, odds, ds; markets) -> EvaluationContext

`markets` is passed explicitly rather than derived from the metrics — see `GJN_MARKETS`
for why the derived default would quietly drop the lines this study exists to score.
"""
gjn_context(fit, odds, ds; markets = GJN_MARKETS) = build_evaluation_context(
    fit_latents(fit), odds, ds.matches, GJN_METRICS; markets, threaded = true)

"""
    gjn_scores(label, ctx, families) -> Vector{NamedTuple}

One row per scope (`all`, then each quoted family in `GJN_SCOPES`), each carrying the
model's LogLoss, Brier, ECE, MCE and RPS beside the Betfair close's on the SAME rows.

RPS is ordered and defined on 1X2 only; it is the same 1X2 number in every scope row and
should be read only from the `1X2` row.
"""
function gjn_scores(label::AbstractString, ctx, families::Dict{String,Vector{Symbol}})
    rows = NamedTuple[]
    scopes = vcat([("all", nothing)],
                  [(f, families[f]) for f in GJN_SCOPES if haskey(families, f)])
    for (scope, sels) in scopes
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
function gjn_observation_frame(label::AbstractString, ctx, odds::AbstractDataFrame)
    rows = evaluation_rows(ctx)
    family_of = Dict((Int(r.match_id), r.selection) => gjn_family(r.market_name, r.market_line)
                     for r in eachrow(odds))
    df = DataFrame(
        match_id = [r.match_id for r in rows],
        selection = [r.selection for r in rows],
        p_model = [r.model_prob for r in rows],
        p_market = [r.market_prob for r in rows],
        y = [Float64(r.outcome) for r in rows],
    )
    df.family = [get(family_of, (m, s), "other") for (m, s) in zip(df.match_id, df.selection)]
    df.ll_model = GJN_EVAL.calc_logloss.(df.p_model, df.y)
    df.ll_market = GJN_EVAL.calc_logloss.(df.p_market, df.y)
    df.brier_model = (df.p_model .- df.y) .^ 2
    df.brier_market = (df.p_market .- df.y) .^ 2
    df.model .= String(label)
    return df
end

# ==============================================================================
# 4. Paired, fixture-clustered bootstrap
# ==============================================================================

"""
    gjn_paired_bootstrap(a, b; B, seed, family, column) -> NamedTuple

Δ(score) = mean(score_a − score_b) over the rows both frames score, with a 95% interval
from `B` resamples of FIXTURES (not rows).

A home, draw and away price on one fixture are three readings of one scoreline, and the
Over and the Under of one line are two readings of the same number. Resampling rows
would treat them as independent and shrink the interval; resampling fixtures keeps every
fixture's rows together, which is the honest unit of evidence. Each resample is the
row-weighted mean `Σ_g S_g / Σ_g n_g` over the drawn fixtures, so the point estimate and
the resampled statistic are the same functional.

`b === :market` pairs the model against the Betfair close on its own rows.
`column` selects the scoring rule: `:ll` for LogLoss, `:brier` for Brier.
"""
function gjn_paired_bootstrap(a::AbstractDataFrame, b; B::Int = 10_000, seed::Int = 20260914,
                              family::Union{Nothing,String} = nothing,
                              column::Symbol = :ll)
    mcol = Symbol(column, "_model")
    kcol = Symbol(column, "_market")
    sel = family === nothing ? a : a[a.family .== family, :]
    joined = if b === :market
        DataFrame(match_id = sel.match_id, d = sel[!, mcol] .- sel[!, kcol])
    else
        bb = family === nothing ? b : b[b.family .== family, :]
        j = innerjoin(select(sel, :match_id, :selection, mcol => :s_a),
                      select(bb, :match_id, :selection, mcol => :s_b);
                      on = [:match_id, :selection])
        DataFrame(match_id = j.match_id, d = j.s_a .- j.s_b)
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

"""
    gjn_pair_table(frames; B, seed, column) -> DataFrame

Every `GJN_PAIRS` contrast (NegBin rung − its Poisson control), on every quoted scope.

`p_negative` is the posterior-style read: the share of resamples in which the NegBin arm
scored BETTER (lower) than its control. A value near 0.5 is a wash, which — given
Experiment 02's Δ = +0.0001 on 1X2 — is the null this study expects to have to argue
against on the totals markets.
"""
function gjn_pair_table(frames::AbstractDict; B::Int = 10_000, seed::Int = 20260914,
                        column::Symbol = :ll)
    rows = NamedTuple[]
    for (negbin, poisson) in GJN_PAIRS
        haskey(frames, negbin) && haskey(frames, poisson) || continue
        a = frames[negbin]
        b = frames[poisson]
        scopes = vcat([nothing], [f for f in GJN_SCOPES if any(a.family .== f)])
        for fam in scopes
            r = gjn_paired_bootstrap(a, b; B, seed, family = fam, column)
            push!(rows, (; contrast = "$negbin − $poisson",
                           scope = fam === nothing ? "all" : fam,
                           rule = String(column),
                           n_obs = r.n_obs, n_fixtures = r.n_fixtures,
                           delta = r.delta, lo = r.lo, hi = r.hi,
                           p_better = r.p_negative))
        end
    end
    return DataFrame(rows)
end

# ==============================================================================
# 5. Portfolio under the production Option B contract
# ==============================================================================

"""
    gjn_option_b() -> (book, policy)

`MatchDay.option_b_system()` exactly: canonical markets, de-arbed prices, 30% fractional
Kelly, 2% commission; TieredTrust, SlateDrawdown(8.0), FixedCap(0.25), DailySlate.

The audited production system is the one a live slate would stake under, so that is the
contract used here — for every arm, identically. Task 013 used the same call, which is
what makes its bankroll figures comparable with these.
"""
function gjn_option_b()
    system = GJN_MD.option_b_system()
    return getproperty(system, :book), getproperty(system, :policy)
end

"""
    gjn_buildable_panel(book, fits, odds, ds, panel) -> (Vector{Int}, DataFrame)

Narrow `panel` to fixtures every arm can build a staking book for, and name every fixture
dropped and why. Book refusals (unplayed, no quotes, no usable selection) hit every arm
identically; removing their union once is what makes that true rather than assumed.
"""
function gjn_buildable_panel(book, fits::AbstractDict, odds, ds, panel::Vector{Int})
    dropped = Dict{Int,String}()
    for (_, fit) in fits
        _, report = GJN_PORTFOLIO.build_books_reported(
            book, gjn_restrict(fit, panel), odds, ds; require_converged = false, quiet = true)
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
    gjn_simulate(book, policy, fit, odds, ds, panel; label) -> PortfolioResult

Reprice one restricted posterior and simulate the policy. Refuses a book that skipped a
panel fixture: an arm that staked fewer fixtures than its comparator is not comparable
with it.
"""
function gjn_simulate(book, policy, fit, odds, ds, panel::Vector{Int};
                      label::AbstractString, B::Int = 4000, seed::Int = 1)
    books, report = GJN_PORTFOLIO.build_books_reported(
        book, gjn_restrict(fit, panel), odds, ds; require_converged = false, quiet = true)
    GJN_PORTFOLIO.n_skipped(report) == 0 || error(
        "$label skipped $(GJN_PORTFOLIO.n_skipped(report)) panel fixtures")
    length(books) == length(panel) || error(
        "$label built $(length(books)) books for $(length(panel)) fixtures")
    return GJN_PORTFOLIO.simulate_portfolio(
        policy, books, report;
        bootstrap = true, B = B, seed = seed,
        metrics = GJN_BACKTESTING.AbstractWealthMetric[
            GJN_BACKTESTING.CalmarRatio(), GJN_BACKTESTING.SharpeRatio()])
end

"One headline row. `stake`/`pnl` are fractions of the bankroll at each slate."
function gjn_portfolio_row(label::AbstractString, likelihood::AbstractString, result;
                           n_panel::Int)
    s = result.summary
    ci = result.bootstrap_ci
    bets = result.trajectory.bets
    return (; model = String(label), likelihood = String(likelihood),
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
# `pnl = stake × settle` in the same units. Summing `pnl` across a compounding path pools
# fractions of different bankrolls, so ROI (`Σpnl / Σstake`, path-normalised) and win rate
# are the comparators; `pnl` sums are reported as secondary.
#
# These definitions are Task 013 `l02_evaluation.jl` verbatim in substance, so the capture
# ratios here are on the same scale as the numbers that task's README reports.

gjn_bet_key(r) = (Int(r.match_id), String(r.family), Symbol(r.selection))

"""
    gjn_partition_bets(a, b) -> (both_a, both_b, only_a, only_b)

Shared bets (same fixture, family, selection — and therefore the same price and
settlement) seen from each model's ledger, plus the two exclusive sets. The shared frames
are row-aligned on the key, which is asserted.
"""
function gjn_partition_bets(a::AbstractDataFrame, b::AbstractDataFrame)
    ak = [gjn_bet_key(r) for r in eachrow(a)]
    bk = [gjn_bet_key(r) for r in eachrow(b)]
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
    gjn_edge_summary(bets) -> NamedTuple

Edge, confidence and sizing over one set of staked bets. `edge = p_model − p_market` is
positive on every staked bet (the allocator only stakes positive edges), so
`capture_ratio = E[edge | won] / E[edge | lost]` compares two positive means: > 1 says the
model sits further from the market when it is right than when it is wrong.
`cap_weighted_win_rate` weights outcomes by stake — above the plain win rate means the
money landed on the winners.
"""
function gjn_edge_summary(bets::AbstractDataFrame)
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
    gjn_pair_attribution(name_a, name_b, bets_a, bets_b) -> (rows, sizing)

The three-way partition as reportable rows, plus the controlled sizing contrast.

On the shared set fixtures, selections, prices and outcomes are identical, so
`sizing_delta_pnl = Σ (s_a − s_b) · settle` is attributable to stake size and to nothing
else. `settle` is recovered as `pnl / stake` from each ledger and asserted equal across
the two, which is the check that "shared" really means the same bet.
"""
function gjn_pair_attribution(name_a::AbstractString, name_b::AbstractString,
                              bets_a::AbstractDataFrame, bets_b::AbstractDataFrame)
    both_a, both_b, only_a, only_b = gjn_partition_bets(bets_a, bets_b)
    rows = NamedTuple[]
    for (set, owner, frame) in (("shared", name_a, both_a), ("shared", name_b, both_b),
                                ("exclusive", name_a, only_a), ("exclusive", name_b, only_b))
        s = gjn_edge_summary(frame)
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
                shared_roi_a_pct = gjn_edge_summary(both_a).roi,
                shared_roi_b_pct = gjn_edge_summary(both_b).roi,
                capture_ratio_a = gjn_edge_summary(bets_a).capture_ratio,
                capture_ratio_b = gjn_edge_summary(bets_b).capture_ratio)
    return rows, sizing
end
