module MultiScaleGRWPortfolio

# ==============================================================================
# Task 007 follow-up loader — portfolio + Layer 2 calibration definitions
# ==============================================================================
#
# Definitions only. `r02_portfolio_calibration.jl` owns execution.
#
# This file exists beside `l01_loader.jl` rather than inside it because the two
# answer different questions. `l01` is the inference recipe: it builds models,
# samples them and audits chains. Nothing here samples anything. This is a
# no-MCMC repricing layer that loads seven already-persisted posteriors by
# immutable UUID and pushes them through one fixed staking contract.
#
# THE CONTRACT IS FIXED AND SHARED. Every arm — raw or calibrated, GRW or
# TimeDecay control or `m12` — receives the identical de-vigged Betfair
# TWA[-20,0] book, the identical 1X2 + O/U 2.5 market set, the identical
# Baker-McHale shrinkage, the identical 2% commission and the identical
# daily-slate policy, restricted to the identical fixture panel. A portfolio
# comparison in which two arms saw different prices measures the prices.
# ==============================================================================

using BayesianFootball
using CSV
using DataFrames
using Dates
using DotEnv
using Printf
using Statistics
using UUIDs

const L02_DATA = BayesianFootball.Data
const L02_PORTFOLIO = BayesianFootball.Portfolio
const L02_BACKTESTING = BayesianFootball.BackTesting
const L02_CALIBRATION = BayesianFootball.Calibration
const L02_EVALUATION = BayesianFootball.Evaluation

export L02_ARMS, L02_RESULTS_DIR
export l02_load_runtime_env!, l02_book_spec, l02_policy_spec
export l02_betfair_closing_odds, l02_load_fit_checked, l02_load_all_fits
export l02_common_match_ids, l02_restrict_fit, l02_buildable_panel
export l02_calibrators, l02_calibrate_checked
export l02_simulate_arm, l02_arm_row, l02_tearsheet_rows
export l02_coherence_check, l02_write_outputs!, l02_report, l02_weight_table

# ==============================================================================
# 1. The seven runs, pinned by immutable UUID
# ==============================================================================
#
# UUIDs, NOT names. Three of these model names resolve to more than one completed
# run in `mcmc_experiments` — `m00_baseline` and `m05_production_wealth` each have
# a `synthetic-no-mcmc` twin from 2026-09-01, and `m12_joint_hybrid_synergy` has a
# 43-fold run scoring mean LogLoss 1.018 beside the 40-fold run scoring 0.643.
# Resolving by name would silently pick one of them. The UUIDs below are the runs
# Task 007 actually compared against, verified by direct query.

const L02_RESULTS_DIR = joinpath(@__DIR__, "results")

"""
One benchmarked arm: a persisted posterior plus the metadata the report groups by.

`family` separates the three GRW candidates from their matched TimeDecay controls
so a table can be read without knowing the naming convention, and `pair` names the
control each GRW candidate is a like-for-like substitute for.
"""
struct L02Arm
    label::String
    experiment::String
    run_id::UUID
    family::String
    pair::String
end

const L02_ARMS = (
    L02Arm("m00_baseline_grw", "scottish_lower_multiscale_grw_2426",
           UUID("f64a00a2-34a0-4f31-8c58-c093c92d54b7"), "GRW", "m00_baseline"),
    L02Arm("m05_production_wealth_grw", "scottish_lower_multiscale_grw_2426",
           UUID("b2d8036d-8fbd-45f9-92b5-cc7675926232"), "GRW", "m05_production_wealth"),
    L02Arm("m05_joint_production_wealth_grw", "scottish_lower_multiscale_grw_2426",
           UUID("f870dbb7-9df0-4dae-a84a-cf570cf8113e"), "GRW", "m05_joint_production_wealth"),
    L02Arm("m00_baseline", "scottish_lower_poisson_2426",
           UUID("2722f7e2-0ee6-4040-95cc-55420800b1c3"), "TimeDecay", "m00_baseline"),
    L02Arm("m05_production_wealth", "scottish_lower_poisson_2426",
           UUID("9239e392-897f-490e-aad4-9e3cc7c6cf5b"), "TimeDecay", "m05_production_wealth"),
    L02Arm("m05_joint_production_wealth", "scottish_lower_joint_2426",
           UUID("92a55d0b-86ba-4f08-a706-0684c95bec75"), "TimeDecay", "m05_joint_production_wealth"),
    L02Arm("m12_joint_hybrid_synergy", "scottish_lower_joint_player_2426",
           UUID("928dad3b-ccaf-4909-b6b7-4f1a815e1cab"), "Production", "m12_joint_hybrid_synergy"),
)

"Load the git-ignored operational DB environment; precompilation cannot retain it."
function l02_load_runtime_env!()
    env_file = joinpath(pkgdir(BayesianFootball), ".env")
    isfile(env_file) && DotEnv.load!(ENV, env_file)
    return nothing
end

# ==============================================================================
# 2. The fixed staking contract
# ==============================================================================

"""
    l02_book_spec() -> BookSpec

The canonical Betfair contract: 1X2 + O/U 2.5, de-arbed pricing, Kelly log-utility
allocation, Baker-McHale shrinkage and 2% per-bet commission.

Identical to `experiments/scottish_lower/08_goal_decomposition`'s `l08_book_spec`,
deliberately — a portfolio number produced under a different book is not
comparable with the published Generation 3/4 results, and the whole purpose of
this stage is to place the GRW candidates on that same scale.
"""
function l02_book_spec()
    return BayesianFootball.BookSpec(
        markets = L02_DATA.MarketConfig(L02_DATA.AbstractMarket[
            L02_DATA.Market1X2(),
            L02_DATA.MarketOverUnder(2.5),
        ]),
        price = BayesianFootball.DeArb(),
        allocator = BayesianFootball.KellyLogUtility(),
        shrink = L02_PORTFOLIO.BakerMcHale(),
        exec = BayesianFootball.ExecutionConfig(
            commission = BayesianFootball.PerBetCommission(0.02),
            budget = 0.99,
            min_selection_stake = 0.001,
        ),
    )
end

"""
    l02_policy_spec() -> PolicySpec

30% FlatTrust, SlateDrawdown(23), FixedCap(20%), DailySlate.

The task prompt offered this or a "0.25/20/0.25" variant. This one is chosen
because it is the contract `experiments/scottish_lower/08_goal_decomposition`
already published under, so the GRW rows land on a scale that existing numbers
can be read against instead of starting a third incompatible convention.
"""
function l02_policy_spec()
    return BayesianFootball.PolicySpec(
        trust = BayesianFootball.FlatTrust(0.30),
        risk = BayesianFootball.SlateDrawdown(23.0),
        cap = BayesianFootball.FixedCap(0.20),
        grouping = BayesianFootball.DailySlate(),
    )
end

"""
    l02_betfair_closing_odds(ds) -> DataFrame

The de-vigged Betfair time-weighted-average close over the [-20, 0] minute window.

`is_winner` is joined from `ds.odds`, which is the settlement record and carries no
price, so the join cannot leak a price the close could not see.
"""
function l02_betfair_closing_odds(ds)
    raw = L02_DATA.summarize_odds(ds.betfair_odds, L02_DATA.TWAEstimator(); window = (-20.0, 0.0))
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

# ==============================================================================
# 3. Loading the persisted posteriors
# ==============================================================================

"""
    l02_load_fit_checked(arm) -> (PostgresStorage, Fit)

Load one arm by immutable UUID and assert the facts the benchmark depends on:
that it deserialized to a `Fit`, that it carries a typed `CountLatents` container,
that its fixtures are unique, and that it is not a synthetic placeholder.

The synthetic check is not paranoia. `scottish_lower_poisson_2426` holds
`synthetic-no-mcmc` twins of `m00_baseline` and `m05_production_wealth` that were
never sampled; benchmarking against one would produce a plausible table of numbers
describing nothing.
"""
function l02_load_fit_checked(arm::L02Arm)
    db = BayesianFootball.PostgresStorage(arm.experiment)
    fit = BayesianFootball.load_fit(db, arm.run_id)
    fit isa BayesianFootball.Fit || error(
        "$(arm.label)/$(arm.run_id) did not deserialize to a Fit")
    lat = fit.latents
    lat isa BayesianFootball.CountLatents || error(
        "$(arm.label)/$(arm.run_id) carries $(typeof(lat)), not CountLatents")
    fit.metadata.git_commit == "synthetic-no-mcmc" && error(
        "$(arm.label)/$(arm.run_id) is a synthetic placeholder; refusing to benchmark it")
    ids = Int.(lat.match_ids)
    length(unique(ids)) == length(ids) || error(
        "$(arm.label)/$(arm.run_id) has duplicate latent match IDs")
    return db, fit
end

function l02_load_all_fits()
    dbs = Dict{String,Any}()
    fits = Dict{String,Any}()
    for arm in L02_ARMS
        db, fit = l02_load_fit_checked(arm)
        dbs[arm.label] = db
        fits[arm.label] = fit
        @printf("  %-34s %5d fixtures | %s\n", arm.label,
                BayesianFootball.n_matches(fit.latents), arm.run_id)
    end
    return dbs, fits
end

"""
    l02_common_match_ids(fits, odds) -> Vector{Int}

The frozen intersection: fixtures every arm holds a posterior for AND the Betfair
book actually quotes.

An intersection is used rather than one run's fixture list because the seven arms
do not all cover the same 710. Taking the intersection is what makes "every model
stakes the same panel" true rather than merely intended.
"""
function l02_common_match_ids(fits::AbstractDict, odds::AbstractDataFrame)
    isempty(fits) && error("cannot freeze a fixture panel from no fits")
    common = reduce(intersect, (Set(Int.(fit.latents.match_ids)) for fit in values(fits)))
    quoted = Set(Int.(odds.match_id))
    panel = sort!(collect(intersect(common, quoted)))
    isempty(panel) && error("the seven arms and the Betfair book share no fixtures")
    return panel
end

"""
    l02_buildable_panel(book, fits, odds, ds, panel) -> (Vector{Int}, DataFrame)

Narrow the panel to fixtures every arm can actually build a staking book for.

Quoted is not the same as buildable. A fixture can carry Betfair prices and still
be refused because it is unplayed, because a market's selection set is incomplete,
or because de-vigging left no usable selection. Those refusals are properties of
the BOOK, not of the model, so they hit every arm identically -- but taking the
union across all seven and removing it once is what guarantees that, rather than
assuming it.

Returns the surviving panel and a frame naming every dropped fixture with its
reason, so an exclusion is auditable instead of silent.
"""
function l02_buildable_panel(book, fits::AbstractDict, odds::AbstractDataFrame, ds,
                             panel::Vector{Int})
    dropped = Dict{Int,String}()
    for (_, fit) in fits
        _, report = L02_PORTFOLIO.build_books_reported(
            book, l02_restrict_fit(fit, panel), odds, ds;
            require_converged = false, quiet = true)
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
    isempty(keep) && error("no panel fixture is buildable by every arm")
    frame = DataFrame(match_id = sort!(collect(keys(dropped))))
    frame.reason = [dropped[m] for m in frame.match_id]
    return keep, frame
end

"""
    l02_restrict_fit(fit, panel) -> Fit

The same run, carrying only the panel's fixtures.

The seven arms each hold 710 posterior fixtures but the Betfair book quotes only
635 of them, so building straight from a `Fit` prices 710 and skips 75 for want of
a price. Restricting FIRST makes "every arm staked the identical panel" a property
of the containers rather than a hope about the builder, and turns `n_skipped == 0`
into a real assertion instead of one that can never pass.
"""
function l02_restrict_fit(fit, panel::Vector{Int})
    lat = BayesianFootball.restrict_latents(fit.latents, panel)
    BayesianFootball.n_matches(lat) == length(panel) || error(
        "restricted container holds $(BayesianFootball.n_matches(lat)) of " *
        "$(length(panel)) panel fixtures")
    return BayesianFootball.Fit(fit.config, fit.folds, lat, fit.diagnostics,
                                fit.metadata, fit.save_path)
end

# ==============================================================================
# 4. Layer 2 generative rate calibration
# ==============================================================================

"""
    l02_calibrators() -> Vector{NamedTuple}

The two instants under test, exactly as the work package specifies them.

`GenerativeRateCalibrator`'s keyword is `dispersion`, not `map`; the prompt's
`map = PoolDispersion()` would be an unsupported-keyword error. `PoolDispersion`
is the default and the validated production transform either way.
"""
function l02_calibrators()
    return [
        (; key = "t25_inv",
           as_of = -25.0,
           cal = BayesianFootball.GenerativeRateCalibrator(
               name = "scot_lower_t25_inv",
               law = BayesianFootball.InverseGaussianLaw(w_base = 0.25, sigma = 0.35),
               dispersion = BayesianFootball.PoolDispersion(),
               book_as_of_minutes = -25.0)),
        (; key = "close_std",
           as_of = 0.0,
           cal = BayesianFootball.GenerativeRateCalibrator(
               name = "scot_lower_close_std",
               law = BayesianFootball.StandardGaussianLaw(w_base = 0.30, sigma = 0.40),
               dispersion = BayesianFootball.PoolDispersion(),
               book_as_of_minutes = 0.0)),
    ]
end

"""
    l02_calibrate_checked(cal, fit, pit_book; rates) -> CalibratedFit

Calibrate one posterior against the point-in-time book of the calibrator's own
instant, reusing a precomputed market inversion.

The inversion depends on the BOOK ONLY — not the model, not the law — so the
caller inverts once per instant and passes it in rather than paying Nelder-Mead
seven times for an identical answer.
"""
function l02_calibrate_checked(cal, fit, pit_book::AbstractDataFrame; rates = nothing)
    return BayesianFootball.calibrate_fit(cal, fit, pit_book; rates = rates, quiet = true)
end

"""
    l02_coherence_check(source, markets) -> NamedTuple

Worst disagreement between market families read off one fixture's score tensor.

Generative rate calibration prices 1X2, O/U and BTTS as three partitions of a
single 12x12 tensor, so `max_family_spread` is zero to rounding by construction.
Asserting it is how the construction is verified rather than assumed.
"""
function l02_coherence_check(source, markets)
    lat = BayesianFootball.Evaluation.fit_latents(source)
    return BayesianFootball.coherence_report(lat, markets; threaded = true)
end

# ==============================================================================
# 5. Simulation and metric extraction
# ==============================================================================

"""
    l02_simulate_arm(book, policy, source, odds, ds, panel; label) -> (result, report)

Reprice one posterior on the frozen panel and simulate the staking policy over it.

Refuses rather than warns when the build skipped a panel fixture: an arm that
staked 600 of 622 fixtures is not comparable with one that staked all 622, and a
silently shorter book is exactly the failure that looks like alpha.
"""
function l02_simulate_arm(book, policy, source, odds::AbstractDataFrame, ds,
                          panel::Vector{Int}; label::AbstractString = "arm",
                          bootstrap::Bool = true, B::Int = 4000, seed::Int = 1)
    books, report = L02_PORTFOLIO.build_books_reported(
        book, source, odds, ds; require_converged = false, quiet = true)
    L02_PORTFOLIO.n_skipped(report) == 0 || error(
        "$label skipped $(L02_PORTFOLIO.n_skipped(report)) panel fixtures; " *
        "refusing an unequal book")
    length(books) == length(panel) || error(
        "$label built $(length(books)) books for $(length(panel)) panel fixtures")
    result = L02_PORTFOLIO.simulate_portfolio(
        policy, books, report;
        bootstrap = bootstrap, B = B, seed = seed,
        metrics = L02_BACKTESTING.AbstractWealthMetric[
            L02_BACKTESTING.CalmarRatio(),
            L02_BACKTESTING.SharpeRatio(),
        ])
    return result, report
end

"""
    l02_arm_row(arm, variant, result; n_panel) -> NamedTuple

One row of the headline summary: the wealth path, its uncertainty and its cost.

`growth_per_slate` and its bootstrap interval are the compounding quantities;
`cagr` annualises the same path. `expected_edge_pct` is the mean `p_model -
p_market` over STAKED selections only, in probability points — the edge the
allocator acted on, not the edge available.
"""
function l02_arm_row(arm::L02Arm, variant::AbstractString, result; n_panel::Int)
    s = result.summary
    ci = result.bootstrap_ci
    bets = result.trajectory.bets
    edge = nrow(bets) == 0 ? NaN : 100 * mean(bets.p_model .- bets.p_market)
    return (;
        model = arm.label,
        family = arm.family,
        control = arm.pair,
        variant = String(variant),
        n_panel_fixtures = n_panel,
        n_slates = s.n_slates,
        n_bets = s.n_bets,
        total_return_pct = s.total_return_pct,
        cagr_pct = 100 * s.cagr,
        growth_per_slate = s.growth_per_slate,
        growth_lo = ci === nothing ? NaN : ci.growth_lo,
        growth_hi = ci === nothing ? NaN : ci.growth_hi,
        roi_flat_pct = s.roi,
        roi_lo_pct = ci === nothing ? NaN : ci.roi_lo,
        roi_hi_pct = ci === nothing ? NaN : ci.roi_hi,
        p_roi_positive = ci === nothing ? NaN : ci.p_roi_positive,
        sharpe_ann = s.sharpe_ann,
        calmar = s.calmar,
        sortino = s.sortino,
        max_drawdown_pct = s.mdd,
        ulcer = s.ulcer,
        win_rate_pct = 100 * s.win_rate,
        expected_edge_pct = edge,
        mean_exposure = s.mean_exposure,
        worst_slate = s.worst_slate,
        span_days = s.span_days,
        run_id = string(arm.run_id),
    )
end

"""
    l02_tearsheet_rows(arm, variant, result) -> Vector{NamedTuple}

Per-selection `BackTesting` tearsheet rows, including the Bernoulli-Gamma hurdle fit.

The hurdle model decomposes per-bet ROI into a win probability and a Gamma body,
then integrates the parametric growth rate `G`. It is reported in BASIS POINTS
because at realistic stake fractions `G` lives in the fourth decimal place, where
a table rounded to two would show every arm as 0.00.
"""
function l02_tearsheet_rows(arm::L02Arm, variant::AbstractString, result;
                            hurdle = L02_BACKTESTING.BernoulliGammaHurdle())
    bets = result.trajectory.bets
    rows = NamedTuple[]
    nrow(bets) == 0 && return rows
    for sub in groupby(bets, :family)
        h = L02_BACKTESTING.compute_distributional_metric(hurdle, sub)
        stake = sum(sub.stake)
        pnl = sum(sub.pnl)
        push!(rows, (;
            model = arm.label,
            family = arm.family,
            variant = String(variant),
            selection_family = first(sub.family),
            bets = nrow(sub),
            turnover = stake,
            profit = pnl,
            roi_pct = stake > 0 ? 100 * pnl / stake : NaN,
            win_rate_pct = 100 * count(>(0.0), sub.pnl) / nrow(sub),
            avg_odds = mean(sub.odds),
            mean_edge_pct = 100 * mean(sub.p_model .- sub.p_market),
            hurdle_p = h.hurdle_p,
            hurdle_shape = h.hurdle_shape,
            hurdle_scale = h.hurdle_scale,
            hurdle_E_R = h.hurdle_E_R,
            hurdle_sharpe = h.hurdle_sharpe,
            hurdle_G_bps = 10_000 * h.hurdle_G,
            hurdle_G_emp_bps = 10_000 * h.hurdle_G_emp,
            hurdle_avg_stake = h.hurdle_avg_stake,
            hurdle_n_bets = h.hurdle_n_bets,
        ))
    end
    return rows
end

# ==============================================================================
# 6. Output
# ==============================================================================

l02_num(v; digits = 3) = isfinite(v) ? @sprintf("%.*f", digits, v) : "n/a"
l02_signed(v; digits = 3) = isfinite(v) ? @sprintf("%+.*f", digits, v) : "n/a"

function l02_write_outputs!(summary::DataFrame, tearsheet::DataFrame)
    mkpath(L02_RESULTS_DIR)
    summary_path = joinpath(L02_RESULTS_DIR, "r02_portfolio_summary.csv")
    tearsheet_path = joinpath(L02_RESULTS_DIR, "r02_portfolio_backtesting_tearsheet.csv")
    CSV.write(summary_path, summary)
    CSV.write(tearsheet_path, tearsheet)
    return summary_path, tearsheet_path
end

function l02_headline_table(summary::DataFrame, variant::AbstractString)
    part = filter(:variant => ==(variant), summary)
    nrow(part) == 0 && return "_No `$variant` rows._\n"
    sort!(part, :total_return_pct, rev = true)
    lines = [
        "| Model | Family | Bets | Return % | CAGR % | g / slate | g 95% CI | Sharpe ann | Calmar | MDD % | Win % | Edge pp |",
        "|---|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for r in eachrow(part)
        push!(lines, "| `$(r.model)` | $(r.family) | $(r.n_bets) | " *
            "$(l02_signed(r.total_return_pct; digits = 2)) | " *
            "$(l02_signed(r.cagr_pct; digits = 1)) | " *
            "$(l02_signed(r.growth_per_slate; digits = 5)) | " *
            "[$(l02_signed(r.growth_lo; digits = 4)), $(l02_signed(r.growth_hi; digits = 4))] | " *
            "$(l02_num(r.sharpe_ann)) | $(l02_num(r.calmar; digits = 2)) | " *
            "$(l02_num(r.max_drawdown_pct; digits = 1)) | " *
            "$(l02_num(r.win_rate_pct; digits = 1)) | " *
            "$(l02_signed(r.expected_edge_pct; digits = 2)) |")
    end
    return join(lines, "\n") * "\n"
end

function l02_paired_table(summary::DataFrame, variant::AbstractString)
    part = filter(:variant => ==(variant), summary)
    nrow(part) == 0 && return "_No `$variant` rows._\n"
    by_label = Dict(r.model => r for r in eachrow(part))
    lines = [
        "| GRW candidate | TimeDecay control | Δ Return pp | Δ CAGR pp | Δ g / slate | Δ Sharpe | Δ Calmar | Δ MDD pp |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in eachrow(filter(:family => ==("GRW"), part))
        haskey(by_label, r.control) || continue
        c = by_label[r.control]
        push!(lines, "| `$(r.model)` | `$(r.control)` | " *
            "$(l02_signed(r.total_return_pct - c.total_return_pct; digits = 2)) | " *
            "$(l02_signed(r.cagr_pct - c.cagr_pct; digits = 1)) | " *
            "$(l02_signed(r.growth_per_slate - c.growth_per_slate; digits = 5)) | " *
            "$(l02_signed(r.sharpe_ann - c.sharpe_ann)) | " *
            "$(l02_signed(r.calmar - c.calmar; digits = 2)) | " *
            "$(l02_signed(r.max_drawdown_pct - c.max_drawdown_pct; digits = 1)) |")
    end
    length(lines) == 2 && return "_No GRW/control pairs resolved._\n"
    return join(lines, "\n") * "\n"
end

function l02_variant_table(summary::DataFrame)
    variants = unique(summary.variant)
    lines = [
        "| Model | Variant | Bets | Return % | g / slate | Sharpe ann | MDD % | Edge pp |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in unique(summary.model), v in variants
        part = filter(r -> r.model == label && r.variant == v, summary)
        nrow(part) == 1 || continue
        r = only(eachrow(part))
        push!(lines, "| `$(r.model)` | $(r.variant) | $(r.n_bets) | " *
            "$(l02_signed(r.total_return_pct; digits = 2)) | " *
            "$(l02_signed(r.growth_per_slate; digits = 5)) | " *
            "$(l02_num(r.sharpe_ann)) | $(l02_num(r.max_drawdown_pct; digits = 1)) | " *
            "$(l02_signed(r.expected_edge_pct; digits = 2)) |")
    end
    return join(lines, "\n") * "\n"
end

function l02_hurdle_table(tearsheet::DataFrame, variant::AbstractString)
    part = filter(:variant => ==(variant), tearsheet)
    nrow(part) == 0 && return "_No `$variant` tearsheet rows._\n"
    agg = combine(groupby(part, [:model, :family]),
                  :bets => sum => :bets,
                  :turnover => sum => :turnover,
                  :profit => sum => :profit,
                  :hurdle_G_bps => (g -> mean(filter(isfinite, g))) => :mean_G_bps,
                  :hurdle_G_emp_bps => (g -> mean(filter(isfinite, g))) => :mean_G_emp_bps,
                  :hurdle_sharpe => (s -> mean(filter(isfinite, s))) => :mean_hurdle_sharpe)
    agg.roi_pct = 100 .* agg.profit ./ max.(agg.turnover, 1e-9)
    sort!(agg, :mean_G_bps, rev = true)
    lines = [
        "| Model | Family | Bets | Flat ROI % | Hurdle G (bps) | Empirical G (bps) | Hurdle Sharpe |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in eachrow(agg)
        push!(lines, "| `$(r.model)` | $(r.family) | $(r.bets) | " *
            "$(l02_signed(r.roi_pct; digits = 2)) | " *
            "$(l02_signed(r.mean_G_bps; digits = 2)) | " *
            "$(l02_signed(r.mean_G_emp_bps; digits = 2)) | " *
            "$(l02_num(r.mean_hurdle_sharpe)) |")
    end
    return join(lines, "\n") * "\n"
end

"""
    l02_calibration_note(summary, context) -> String

What the two calibrators actually did to the staked edge, read off the run.

The two instants behave nothing alike and the difference is the point of the
exercise, so this is computed from the measured weights and edges instead of
asserted from a prior about which law shrinks harder.
"""
function l02_calibration_note(summary::DataFrame, context)
    w = get(context, :weights, nothing)
    (w === nothing || nrow(w) == 0) && return ""
    io = IOBuffer()
    for v in ("t25_inv", "close_std")
        wv = filter(:variant => ==(v), w)
        sv = filter(:variant => ==(v), summary)
        (nrow(wv) == 0 || nrow(sv) == 0) && continue
        raw = filter(:variant => ==("raw"), summary)
        wmed = median(skipmissing(wv.w_median))
        share = median(skipmissing(wv.market_share_median))
        retained = median(skipmissing(wv.var_retention_median))
        d_edge = median(sv.expected_edge_pct) - median(raw.expected_edge_pct)
        d_mdd = median(sv.max_drawdown_pct) - median(raw.max_drawdown_pct)
        d_shp = median(sv.sharpe_ann) - median(raw.sharpe_ann)
        d_ret = median(sv.total_return_pct) - median(raw.total_return_pct)
        println(io, "* **`", v, "` pooled to a median weight of ", l02_num(wmed),
                "**, leaving the market ", l02_num(share), " of the location and ",
                l02_num(retained), " of the posterior log-variance. Across the seven ",
                "arms it moved median staked edge by ", l02_signed(d_edge),
                " pp, median return by ", l02_signed(d_ret), " pp, median drawdown by ",
                l02_signed(d_mdd), " pp and median annual Sharpe by ", l02_signed(d_shp),
                ". Shrinking toward the book removes edge the allocator would have ",
                "sized on, so a lower return here is the construction working rather ",
                "than the model failing — the question a calibrator answers is whether ",
                "the edge it removed was real.")
    end
    return String(take!(io))
end

"""
    l02_weight_table(weights) -> String

How much of each calibrated posterior's location the market actually supplied.

The Ireland post-mortem turned on exactly these numbers: a median `w` of 0.41
means the market supplied most of the location and most of the posterior
log-variance was destroyed. No headline return says that, so it is printed beside
every calibrated row rather than left in a diagnostics frame nobody opens.
"""
function l02_weight_table(weights::DataFrame)
    nrow(weights) == 0 && return "_No calibrated containers were produced._\n"
    lines = [
        "| Model | Instant | Fixtures shifted | median w | w p10 | w p90 | median var retained | median market share |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in eachrow(sort(weights, [:model, :variant]))
        push!(lines, "| `$(r.model)` | $(r.variant) | $(r.n_shifted) | " *
            "$(l02_num(r.w_median)) | $(l02_num(r.w_p10)) | $(l02_num(r.w_p90)) | " *
            "$(l02_num(r.var_retention_median)) | $(l02_num(r.market_share_median)) |")
    end
    return join(lines, "\n") * "\n"
end

"""
    l02_report(summary, tearsheet, context) -> String

The comparative markdown report.

Deliberately leads with what the numbers do NOT support. A 622-fixture backtest
over two seasons has wide enough intervals that a return ranking can invert on
noise, and the Task 007 proper-score result already showed the GRW edge shrinking
to near zero on the joint likelihood. A report that printed the ranking without
the interval would invite exactly the overread this stage exists to prevent.
"""
function l02_report(summary::DataFrame, tearsheet::DataFrame, context::NamedTuple)
    io = IOBuffer()
    println(io, "# Task 007 follow-up — portfolio backtest and Layer 2 calibration\n")
    println(io, "Generated ", Dates.now(), "\n")

    println(io, "## 1. What was run\n")
    println(io, "Seven persisted posteriors, repriced with no new MCMC, on one frozen panel of ",
                context.n_panel, " Betfair-quoted fixtures spanning ",
                context.span_days, " days.\n")
    println(io, "| Contract element | Value |")
    println(io, "|---|---|")
    println(io, "| Book | de-vigged Betfair TWA[-20,0] |")
    println(io, "| Markets | 1X2 + O/U 2.5 |")
    println(io, "| Shrinkage | Baker-McHale |")
    println(io, "| Commission | 2% per bet |")
    println(io, "| Policy | FlatTrust(0.30), SlateDrawdown(23.0), FixedCap(0.20), DailySlate |")
    println(io, "| Bootstrap | ", context.B, " resamples, match-clustered for ROI, slate-blocked for g |")
    println(io, "| Panel | ", context.n_panel, " fixtures, identical for every arm |\n")

    println(io, "**Every arm stakes the same book.** The calibrated variants differ from ",
                "the raw ones only in the posterior they carry; the prices, markets, ",
                "commission, shrinkage, policy and fixture panel are byte-identical. A ",
                "P&L difference is therefore attributable to the posterior and not to ",
                "the book.\n")

    println(io, "## 2. Raw posteriors — headline\n")
    print(io, l02_headline_table(summary, "raw"))
    println(io, "\n### 2.1 GRW versus its matched control\n")
    print(io, l02_paired_table(summary, "raw"))

    println(io, "\n## 3. Compounding growth and the hurdle model\n")
    println(io, "`g` is mean per-slate log growth — the quantity Kelly maximises — with a ",
                "95% block bootstrap over whole slates. Hurdle `G` is the parametric ",
                "growth rate from the Bernoulli-Gamma fit to per-bet ROI, in basis points.\n")
    print(io, l02_hurdle_table(tearsheet, "raw"))

    println(io, "\n## 4. Raw versus calibrated\n")
    print(io, l02_variant_table(summary))

    println(io, "\n## 5. Calibrated posteriors — headline\n")
    for v in filter(!=("raw"), unique(summary.variant))
        println(io, "\n### ", v, "\n")
        print(io, l02_headline_table(summary, v))
    end

    println(io, "\n### 5.1 How much the market supplied\n")
    print(io, get(context, :weight_table, "_Not recorded._\n"))

    println(io, "\n### 5.2 Derivative coherence\n")
    println(io, get(context, :coherence_note, "_Not recorded._"), "\n")

    println(io, "\n## 6. Reading this honestly\n")
    println(io, "* **The interval is wider than the ranking.** With ", context.n_panel,
                " fixtures the 95% band on per-slate growth overlaps zero for most arms. ",
                "Order the table by return and the top row is not reliably the best model; ",
                "it is the best sample path.")
    println(io, "* **Proper scores already warned about this.** Task 007 measured Δ LogLoss ",
                "of −0.00170 for the plain Poisson GRW but only −0.00031 for the joint ",
                "Gamma-Poisson arm. The GRW state and the proxy-xG arm are substitutive: ",
                "they compete to explain the same temporal signal, so a portfolio gain on ",
                "`m00` does not license one on the production-shaped joint model.")
    # This bullet is written FROM the measured weights. An earlier draft asserted
    # that the closing-instant calibrator collapses the edge; the run measured the
    # opposite (median w ~0.97 at the close, ~0.29 at T-25), so the prose is
    # derived rather than assumed.
    print(io, l02_calibration_note(summary, context))
    println(io, "* **Cost is not in these tables.** `m05_production_wealth_grw` sampled in ",
                "60.7 minutes against roughly 2.0 for its control — about 30x — and the ",
                "joint GRW took 90.9. Portfolio return per unit of compute is materially ",
                "worse than these rows alone suggest.")
    println(io, "* **This is a historical simulation**, carrying the portfolio's stated fill ",
                "assumptions on an exchange archive. It is not a prospective return claim.\n")

    return String(take!(io))
end

end # module
