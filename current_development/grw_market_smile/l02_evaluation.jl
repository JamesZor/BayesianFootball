# ==============================================================================
# Task 015 loader — proper scores, paired bootstrap, home-favourite compression
# ==============================================================================
#
# Definitions only. `r04_evaluate.jl` executes. Nothing here samples: every posterior is
# loaded from `mcmc_experiments` by UUID (smile rungs rebuilt from their chains, T010).
#
# Adapted from Task 014's `grw_joint_negbin/l02_evaluation.jl` (book, panel, families,
# fixture-clustered bootstrap), with two changes:
#
# 1. `restrict_latents` accepts only `CountLatents`, so `gms_restrict` carries a
#    `SmileLatents` method that keeps `λ_tot` and `φ` row-aligned.
# 2. The market scope and the contrasts are this task's, fixed here before any score was
#    computed (see `GMS_MARKETS`, `GMS_PAIRS`, `gms_compression_table`).
# ==============================================================================

if !isdefined(@__MODULE__, :GMSConfig)
    include(joinpath(@__DIR__, "l01_loader.jl"))
end

const GMS_EVAL = BayesianFootball.Evaluation

"Report scopes. `all` pools every scored row of the PRIMARY markets only."
const GMS_SCOPES = ["1X2", "OU2.5", "BTTS", "OU1.5", "OU3.5"]

"""
    GMS_PRIMARY_MARKETS

The work package's scored set — 1X2, O/U 2.5, BTTS — which is also Task 013's published
basis (2,899 rows on the 710-fixture panel), so the baseline control's published LogLoss is
reproducible on it as a gate.
"""
const GMS_PRIMARY_MARKETS = Data.AbstractMarket[
    Data.Market1X2(), Data.MarketOverUnder(2.5), Data.MarketBTTS(),
]

"""
    GMS_SECONDARY_MARKETS

O/U 1.5 and 3.5, declared before scoring. The smile pillar learns a per-strike curve for
K = 0…4, so its effect need not sit at 2.5; these two lines are where a smile that moved the
shoulders but not the centre would show. Reported separately, never pooled into `all`.
"""
const GMS_SECONDARY_MARKETS = Data.AbstractMarket[
    Data.MarketOverUnder(1.5), Data.MarketOverUnder(3.5),
]

const GMS_METRICS = GMS_EVAL.AbstractScoringRule[
    GMS_EVAL.LogLoss(), GMS_EVAL.CRPS(), GMS_EVAL.PredictionScore()]

"Task 013's published figures for the baseline control on the primary basis (README §4)."
const GMS_BASELINE_PUBLISHED = (logloss = 0.64315, ece = 0.0123, n_obs = 2899)

"""
The contrasts, `(candidate, reference)`. Fixed before scoring.

* every candidate − baseline      does the market anchor help at all?
* smile@0.40 − supremacy@0.40     what the smile adds on top of supremacy
* smile@0.20 / @0.70 − smile@0.40 the weight grid, against the Ireland default
"""
const GMS_PAIRS = [
    ("m05_joint_grw_supremacy_w040",       "m05_joint_grw_baseline"),
    ("m05_joint_grw_smile_supremacy_w020", "m05_joint_grw_baseline"),
    ("m05_joint_grw_smile_supremacy_w040", "m05_joint_grw_baseline"),
    ("m05_joint_grw_smile_supremacy_w070", "m05_joint_grw_baseline"),
    ("m05_joint_grw_smile_supremacy_w040", "m05_joint_grw_supremacy_w040"),
    ("m05_joint_grw_smile_supremacy_w020", "m05_joint_grw_smile_supremacy_w040"),
    ("m05_joint_grw_smile_supremacy_w070", "m05_joint_grw_smile_supremacy_w040"),
]

# ==============================================================================
# 1. Arms
# ==============================================================================

struct GMSArm
    label::String
    experiment::String
    run_id::UUID
    role::String
end

function gms_arms(c::GMSConfig)
    arms = GMSArm[GMSArm(GMS_BASELINE_CONTROL.label, GMS_BASELINE_CONTROL.experiment,
                         GMS_BASELINE_CONTROL.run_id, "baseline")]
    db = PostgresStorage(c.experiment)
    for name in GMS_GRID_MODEL_NAMES
        run_id = gms_run_by_name(db, name)
        run_id === nothing && error("no completed run named $name in $(c.experiment) — run r02")
        push!(arms, GMSArm(name, c.experiment, run_id, "candidate"))
    end
    return arms
end

"""
    gms_load_arm(arm, ds; splitter, latent_dir) -> Fit

Load by UUID. A smile rung comes back with no panel and is rebuilt from its chains; if the
r02 file copy exists it must equal the rebuilt panel. Refuses a non-converged or synthetic
run and a latent family that does not match the model.
"""
function gms_load_arm(arm::GMSArm, ds; splitter, latent_dir::AbstractString)
    fit = gms_load_fit(PostgresStorage(arm.experiment), arm.run_id, ds; splitter)
    fit.diagnostics.passed || error("$(arm.label) failed its convergence audit")
    fit.metadata.git_commit == "synthetic-no-mcmc" && error("$(arm.label) is synthetic")
    expected = latent_family(fit.config.model) isa GMS_LATENTS.SmilePoissonFamily ? SmileLatents : CountLatents
    fit.latents isa expected || error("$(arm.label) carries $(typeof(fit.latents)); expected $expected")
    copy_path = joinpath(latent_dir, string(arm.run_id))
    if fit.latents isa SmileLatents && isdir(copy_path)
        gms_latents_equal(load_latents(copy_path), fit.latents) || error(
            "$(arm.label): latents rebuilt from PostgreSQL chains differ from the r02 file copy")
    end
    return fit
end

# ==============================================================================
# 2. Book, panel, restriction
# ==============================================================================

"De-vigged Betfair TWA(−20, 0] close — Task 013 `r04`'s frame exactly."
function gms_betfair_closing_odds(ds)
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

"The fixtures of `seasons` every arm holds a latent for."
function gms_common_panel(ds, fits::AbstractDict, seasons::Vector{String})
    season_of = Dict(Int(r.match_id) => String(r.season) for r in eachrow(ds.matches))
    panels = [Set(m for m in Int.(f.latents.match_ids) if get(season_of, m, "") in seasons)
              for f in values(fits)]
    return sort!(collect(reduce(intersect, panels)))
end

gms_restrict_latents(l::CountLatents, keep) = restrict_latents(l, keep)

function gms_restrict_latents(l::SmileLatents, keep)
    want = Set{Int}(Int.(keep))
    rows = findall(i -> l.match_ids[i] in want, eachindex(l.match_ids))
    isempty(rows) && error("gms_restrict_latents: no fixture kept")
    return SmileLatents(l.match_ids[rows], l.λ_home[rows, :], l.λ_away[rows, :], nothing,
                        l.λ_tot[rows, :], l.φ[rows, :, :], l.strikes)
end

function gms_restrict(fit, panel::Vector{Int})
    lat = gms_restrict_latents(fit.latents, panel)
    n_matches(lat) == length(panel) || error(
        "restricted container holds $(n_matches(lat)) of $(length(panel)) panel fixtures")
    return Fit(fit.config, fit.folds, lat, fit.diagnostics, fit.metadata, fit.save_path)
end

function gms_family(market_name::AbstractString, line::Real)
    m = lowercase(market_name)
    m == "1x2" && return "1X2"
    occursin("btts", m) && return "BTTS"
    (occursin("over", m) || occursin("under", m) || occursin("total", m)) && return "OU" * string(line)
    return String(market_name)
end

function gms_family_selections(odds::AbstractDataFrame)
    out = Dict{String,Vector{Symbol}}()
    for r in eachrow(unique(select(odds, :market_name, :market_line, :selection)))
        fam = gms_family(r.market_name, r.market_line)
        fam in GMS_SCOPES || continue
        push!(get!(out, fam, Symbol[]), r.selection)
    end
    for (k, v) in out
        out[k] = sort!(unique(v))
    end
    return out
end

# ==============================================================================
# 3. Scores
# ==============================================================================

gms_context(fit, odds, ds; markets) = build_evaluation_context(
    fit_latents(fit), odds, ds.matches, GMS_METRICS; markets, threaded = true)

"One row per scope. `all` is computed on the primary context only."
function gms_scores(label, primary_ctx, secondary_ctx, families)
    rows = NamedTuple[]
    # Explicit element type: a literal `[("all", ctx, nothing)]` infers a `Nothing` third slot and
    # refuses the per-family `Vector{Symbol}` selections pushed after it (first r04 attempt).
    scopes = Tuple{String,Any,Union{Nothing,Vector{Symbol}}}[("all", primary_ctx, nothing)]
    for f in GMS_SCOPES
        haskey(families, f) || continue
        push!(scopes, (f, f in ("OU1.5", "OU3.5") ? secondary_ctx : primary_ctx, families[f]))
    end
    for (scope, ctx, sels) in scopes
        s = evaluate_predictions(ctx; selections = sels, n_bins = 10)
        push!(rows, (; model = String(label), scope, n_obs = s.model.n_obs,
                       logloss = s.model.logloss, market_logloss = s.market.logloss,
                       brier = s.model.brier, market_brier = s.market.brier,
                       ece = s.model.ece, market_ece = s.market.ece,
                       rps = s.model.rps, market_rps = s.market.rps))
    end
    return rows
end

"Per-observation frame (one row per scored fixture × selection) for pairing."
function gms_observation_frame(label, ctx, odds::AbstractDataFrame)
    rows = evaluation_rows(ctx)
    family_of = Dict((Int(r.match_id), r.selection) => gms_family(r.market_name, r.market_line)
                     for r in eachrow(odds))
    df = DataFrame(match_id = [r.match_id for r in rows],
                   selection = [r.selection for r in rows],
                   p_model = [r.model_prob for r in rows],
                   p_market = [r.market_prob for r in rows],
                   y = [Float64(r.outcome) for r in rows])
    df.family = [get(family_of, (m, s), "other") for (m, s) in zip(df.match_id, df.selection)]
    df.ll_model = GMS_EVAL.calc_logloss.(df.p_model, df.y)
    df.ll_market = GMS_EVAL.calc_logloss.(df.p_market, df.y)
    df.brier_model = (df.p_model .- df.y) .^ 2
    df.brier_market = (df.p_market .- df.y) .^ 2
    df.model .= String(label)
    return df
end

# ==============================================================================
# 4. Paired, fixture-clustered bootstrap
# ==============================================================================

"""
    gms_paired_bootstrap(a, b; B, seed, family, column) -> NamedTuple

Δ = mean(score_a − score_b) over shared rows, 95% interval from `B` resamples of FIXTURES
(a fixture's rows move together). `b === :market` pairs against the Betfair close.
"""
function gms_paired_bootstrap(a::AbstractDataFrame, b; B::Int = 10_000, seed::Int = 20260915,
                              family::Union{Nothing,String} = nothing, column::Symbol = :ll)
    mcol = Symbol(column, "_model")
    kcol = Symbol(column, "_market")
    sel = family === nothing ? a : a[a.family .== family, :]
    joined = if b === :market
        DataFrame(match_id = sel.match_id, d = sel[!, mcol] .- sel[!, kcol])
    else
        bb = family === nothing ? b : b[b.family .== family, :]
        j = innerjoin(select(sel, :match_id, :selection, mcol => :s_a),
                      select(bb, :match_id, :selection, mcol => :s_b); on = [:match_id, :selection])
        DataFrame(match_id = j.match_id, d = j.s_a .- j.s_b)
    end
    nrow(joined) == 0 && return (; n_obs = 0, n_fixtures = 0, delta = NaN, lo = NaN, hi = NaN,
                                   p_negative = NaN)
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
    return (; n_obs = nrow(joined), n_fixtures = G, delta = sum(S) / sum(N),
              lo = quantile(stats, 0.025), hi = quantile(stats, 0.975),
              p_negative = mean(stats .< 0.0))
end

function gms_pair_table(frames::AbstractDict; B::Int = 10_000, seed::Int = 20260915,
                        column::Symbol = :ll)
    rows = NamedTuple[]
    for (cand, ref) in GMS_PAIRS
        haskey(frames, cand) && haskey(frames, ref) || continue
        a = frames[cand]
        scopes = vcat([nothing], [f for f in GMS_SCOPES if any(a.family .== f)])
        for fam in scopes
            r = gms_paired_bootstrap(a, frames[ref]; B, seed, family = fam, column)
            push!(rows, (; contrast = "$cand − $ref", scope = fam === nothing ? "all" : fam,
                           rule = String(column), n_obs = r.n_obs, n_fixtures = r.n_fixtures,
                           delta = r.delta, lo = r.lo, hi = r.hi, p_better = r.p_negative))
        end
    end
    return DataFrame(rows)
end

# ==============================================================================
# 5. The mechanism: home-favourite compression
# ==============================================================================

"""
    gms_compression_table(frames) -> DataFrame

The work package's motivating failure, measured on the whole panel rather than one card:
bin the 1X2 HOME selection by the Betfair close's fair p_home and report each arm's mean
p_model − p_market per bin. A compressed model is negative in the top bins and positive in
the bottom ones; the supremacy pillar's claim is that it moves both toward zero. Bins fixed
here: [0, 0.30), [0.30, 0.40), [0.40, 0.50), [0.50, 0.60), [0.60, 1].
"""
function gms_compression_table(frames::AbstractDict)
    edges = [0.0, 0.30, 0.40, 0.50, 0.60, 1.0001]
    rows = NamedTuple[]
    for (label, df) in frames
        home = df[(df.family .== "1X2") .& (df.selection .== :home), :]
        for b in 1:(length(edges) - 1)
            sub = home[(home.p_market .>= edges[b]) .& (home.p_market .< edges[b + 1]), :]
            nrow(sub) == 0 && continue
            push!(rows, (; model = label,
                           market_bin = @sprintf("[%.2f, %.2f)", edges[b], min(edges[b + 1], 1.0)),
                           n = nrow(sub), mean_p_market = mean(sub.p_market),
                           mean_p_model = mean(sub.p_model),
                           mean_gap = mean(sub.p_model .- sub.p_market),
                           home_win_rate = mean(sub.y)))
        end
    end
    return sort!(DataFrame(rows), [:market_bin, :model])
end

# ==============================================================================
# 6. Closing-line portfolio under the Option B contract
# ==============================================================================
#
# Task 014's `grw_joint_negbin/l02_evaluation.jl` §5–§6, carried with a `gms_` prefix. The
# one behavioural change is restriction: `gms_restrict` keeps `λ_tot` and `φ` row-aligned, so
# a smile arm reaches `Portfolio.build_books_reported` as a `SmileLatents` and its O/U books
# are priced through `λ_tot·φ(K)` (`src/Portfolio/pricing.jl:115`), not the plain grid.
# `gms_smile_book_gate` proves that on the staked ledger itself.

const GMS_PORTFOLIO = BayesianFootball.Portfolio
const GMS_BACKTESTING = BayesianFootball.BackTesting
const GMS_MD = BayesianFootball.MatchDay

"`MatchDay.option_b_system()` exactly, for every arm."
function gms_option_b()
    system = GMS_MD.option_b_system()
    return getproperty(system, :book), getproperty(system, :policy)
end

"Narrow `panel` to fixtures every arm can build a book for; name every drop and why."
function gms_buildable_panel(book, fits::AbstractDict, odds, ds, panel::Vector{Int})
    dropped = Dict{Int,String}()
    for (_, fit) in fits
        _, report = GMS_PORTFOLIO.build_books_reported(
            book, gms_restrict(fit, panel), odds, ds; require_converged = false, quiet = true)
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

"Reprice one restricted posterior and simulate. Refuses a book that skipped a panel fixture."
function gms_simulate(book, policy, fit, odds, ds, panel::Vector{Int};
                      label::AbstractString, B::Int = 4000, seed::Int = 1)
    restricted = gms_restrict(fit, panel)
    books, report = GMS_PORTFOLIO.build_books_reported(
        book, restricted, odds, ds; require_converged = false, quiet = true)
    GMS_PORTFOLIO.n_skipped(report) == 0 || error(
        "$label skipped $(GMS_PORTFOLIO.n_skipped(report)) panel fixtures")
    length(books) == length(panel) || error(
        "$label built $(length(books)) books for $(length(panel)) fixtures")
    result = GMS_PORTFOLIO.simulate_portfolio(
        policy, books, report;
        bootstrap = true, B = B, seed = seed,
        metrics = GMS_BACKTESTING.AbstractWealthMetric[
            GMS_BACKTESTING.CalmarRatio(), GMS_BACKTESTING.SharpeRatio()])
    return result, restricted
end

"One headline row. `stake`/`pnl` are fractions of the bankroll at each slate."
function gms_portfolio_row(label::AbstractString, result; n_panel::Int)
    s = result.summary
    ci = result.bootstrap_ci
    bets = result.trajectory.bets
    return (; model = String(label), n_panel, n_slates = s.n_slates, n_bets = s.n_bets,
              total_return_pct = s.total_return_pct, cagr_pct = 100 * s.cagr,
              growth_per_slate = s.growth_per_slate,
              growth_lo = ci === nothing ? NaN : ci.growth_lo,
              growth_hi = ci === nothing ? NaN : ci.growth_hi,
              roi_pct = s.roi,
              p_roi_positive = ci === nothing ? NaN : ci.p_roi_positive,
              sharpe_ann = s.sharpe_ann, calmar = s.calmar, max_drawdown_pct = s.mdd,
              win_rate_pct = 100 * s.win_rate,
              mean_edge_pp = nrow(bets) == 0 ? NaN : 100 * mean(bets.p_model .- bets.p_market),
              mean_exposure = s.mean_exposure)
end

"""
    gms_edge_summary(bets) -> NamedTuple

Win rate, flat ROI (Σpnl / Σstake, both in bankroll fractions) and edge over one set of
staked bets. `capture_ratio = E[edge | won] / E[edge | lost]`.
"""
function gms_edge_summary(bets::AbstractDataFrame)
    n = nrow(bets)
    z = NaN
    n == 0 && return (; n_bets = 0, n_wins = 0, win_rate = z, stake_sum = 0.0, pnl_sum = 0.0,
                        roi = z, edge_mean = z, capture_ratio = z, odds_mean = z)
    won = bets.payoff .> 0
    edge = bets.p_model .- bets.p_market
    stake_sum = sum(bets.stake)
    e_win = any(won) ? mean(edge[won]) : NaN
    e_loss = any(.!won) ? mean(edge[.!won]) : NaN
    return (; n_bets = n, n_wins = count(won), win_rate = mean(won),
              stake_sum, pnl_sum = sum(bets.pnl),
              roi = stake_sum > 0 ? 100 * sum(bets.pnl) / stake_sum : NaN,
              edge_mean = 100 * mean(edge),
              capture_ratio = (isfinite(e_loss) && e_loss > 0) ? e_win / e_loss : NaN,
              odds_mean = mean(bets.odds))
end

"The market a ledger family belongs to: `1X2` or `totals` (every O/U line)."
gms_market_group(family::AbstractString) = startswith(family, "1X2") ? "1X2" :
    (occursin("O/U", family) || occursin("over", lowercase(family)) ||
     occursin("under", lowercase(family))) ? "totals" : String(family)

"""
    gms_breakdown(label, bets) -> Vector{NamedTuple}

One row per market group (1X2, totals) and one per ledger family, each with bets, win rate,
flat ROI, edge and share of stake.
"""
function gms_breakdown(label::AbstractString, bets::AbstractDataFrame)
    rows = NamedTuple[]
    nrow(bets) == 0 && return rows
    frame = DataFrame(bets)
    frame.market = gms_market_group.(String.(frame.family))
    total_stake = sum(frame.stake)
    for (level, col) in (("market", :market), ("family", :family))
        for sub in groupby(sort(frame, col), col)
            s = gms_edge_summary(sub)
            push!(rows, (; model = String(label), level, group = String(first(sub[!, col])),
                           n_bets = s.n_bets, win_rate_pct = 100 * s.win_rate, roi_pct = s.roi,
                           stake_share_pct = 100 * s.stake_sum / total_stake,
                           edge_mean_pp = s.edge_mean, odds_mean = s.odds_mean,
                           capture_ratio = s.capture_ratio))
        end
    end
    return rows
end

"""
    gms_smile_book_gate(bets, latents) -> NamedTuple

For every staked totals bet of a smile arm, recompute the model probability two ways from the
restricted container the books were built from:

* smile — `mean_s cdf(Poisson(λ_tot·φ_K), K)` (or its complement for an Over);
* grid  — the plain double-Poisson grid at the same λ draws.

The ledger's `p_model` must equal the smile price (≤ 1e-9) on every bet. A book priced off the
grid would pass every other check in r06 and still not be the model r04 scored.
"""
function gms_smile_book_gate(bets::AbstractDataFrame, latents::SmileLatents)
    twin = CountLatents(latents.match_ids, latents.λ_home, latents.λ_away, nothing)
    row_of = Dict(m => i for (i, m) in enumerate(latents.match_ids))
    worst_smile = 0.0
    min_gap_grid = Inf
    n = 0
    for r in eachrow(bets)
        gms_market_group(String(r.family)) == "totals" || continue
        sel = String(Symbol(r.selection))
        m = match(r"^(over|under)_(\d)(\d)$", sel)
        m === nothing && error("unrecognised totals selection $sel")
        K = parse(Int, m.captures[2])
        is_under = m.captures[1] == "under"
        i = row_of[Int(r.match_id)]
        under = mean(cdf(Poisson(latents.λ_tot[i, s] * latents.φ[i, K + 1, s]), K)
                     for s in 1:size(latents.λ_home, 2))
        smile_p = is_under ? under : 1.0 - under
        market = Data.MarketOverUnder(K + 0.5)
        grid_book = GMS_PRED.price_market(GMS_PRED.compute_score_grid(twin, i), market)
        keys_ = Data.outcomes(market)
        grid_p = mean(grid_book[is_under ? keys_.under : keys_.over])
        worst_smile = max(worst_smile, abs(r.p_model - smile_p))
        min_gap_grid = min(min_gap_grid, abs(r.p_model - grid_p))
        n += 1
    end
    return (; n_totals_bets = n, max_abs_vs_smile = worst_smile, min_abs_vs_grid = min_gap_grid)
end
