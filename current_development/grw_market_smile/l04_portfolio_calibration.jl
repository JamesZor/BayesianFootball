# ==============================================================================
# Task 015 loader — T−25 calibrated portfolio and the trust-pruning sweep
# ==============================================================================
#
# Definitions only. `r07_t25_calibrated_portfolio.jl` and `r08_trust_sweep.jl` execute.
#
# THE SMILE × CALIBRATOR PROBLEM. `Calibration.calibrate_latents` refuses `SmileLatents`
# (`src/Calibration/rate_pool.jl`, §2.3): pooling only (λ_h, λ_a) would calibrate 1X2 and leave
# the totals ladder `λ_tot·φ(K)` uncalibrated, and the repository records that a smile
# calibration "has not been done". So a calibrated smile arm is not defined in `src`, and the
# definition changes the answer. Two defensible ones are run side by side, never one silently
# (Task 014 resolved its closing-calibrator ambiguity the same way):
#
#   t25_inv_pooltot   calibrate the grid with Option B, then rebuild SmileLatents with
#                     λ_tot := λ_h,cal + λ_a,cal draw by draw and φ UNCHANGED.
#                     1X2/BTTS and totals share ONE calibrated location; totals keep the
#                     fitted per-strike shape. Coherent by construction.
#   t25_inv_grid      calibrate the grid and price EVERYTHING off it — φ dropped. The
#                     "does the smile survive calibration at all" control.
#
# The third option — calibrated grid, RAW λ_tot — is the incoherent container the `src`
# error message warns against, and is not run.
#
# Count arms (baseline, supremacy) calibrate natively; their `t25_inv` row is the comparator
# for both smile variants.
# ==============================================================================

if !isdefined(@__MODULE__, :gms_option_b)
    include(joinpath(@__DIR__, "l02_evaluation.jl"))
end

const GMS_CAL = BayesianFootball.Calibration

# ==============================================================================
# 1. Books and simulation
# ==============================================================================

"The tradeable T−25 point-in-time book, built from the Betfair odds history at as_of = −25."
gms_t25_book(ds) = GMS_CAL.point_in_time_book(
    ds; config = GMS_CAL.PointInTimeBookConfig(as_of_minutes = -25.0))

"Build one already-restricted source's books over `panel`, refusing any skipped fixture."
function gms_build_books(book, source, odds, ds, panel::Vector{Int}; label::AbstractString)
    books, report = GMS_PORTFOLIO.build_books_reported(
        book, source, odds, ds; require_converged = false, quiet = true)
    GMS_PORTFOLIO.n_skipped(report) == 0 || error(
        "$label skipped $(GMS_PORTFOLIO.n_skipped(report)) panel fixtures")
    length(books) == length(panel) || error(
        "$label built $(length(books)) books for $(length(panel)) fixtures")
    return books, report
end

"Stake prebuilt books under `policy`. Books do not depend on the policy, so one build serves many."
gms_run_policy(policy, books, report; B::Int = 4000, seed::Int = 1) =
    GMS_PORTFOLIO.simulate_portfolio(
        policy, books, report; bootstrap = true, B = B, seed = seed,
        metrics = GMS_BACKTESTING.AbstractWealthMetric[
            GMS_BACKTESTING.CalmarRatio(), GMS_BACKTESTING.SharpeRatio()])

"Quoted fixtures per (market, line) inside `panel` — which strikes a book can actually offer."
function gms_line_coverage(env::AbstractString, odds::AbstractDataFrame, panel::Vector{Int})
    sub = filter(:match_id => in(Set(panel)), DataFrame(odds))
    g = combine(groupby(sub, [:market_name, :market_line]), :match_id => (x -> length(unique(x))) => :n_fixtures)
    g.environment .= String(env)
    return sort!(g, [:market_name, :market_line])
end

# ==============================================================================
# 2. Calibrated sources
# ==============================================================================

"""
    gms_calibrated_sources(cal, fit, inversion_book, rates) -> Vector{NamedTuple}

Every calibrated posterior this fit admits: one `t25_inv` for a count container; the two smile
definitions in the header for a `SmileLatents`. `fit` must already be restricted to the panel.
"""
function gms_calibrated_sources(cal, fit, inversion_book, rates)
    lat = fit.latents
    if lat isa SmileLatents
        twin = CountLatents(lat.match_ids, lat.λ_home, lat.λ_away, nothing)
        grid_fit = Fit(fit.config, fit.folds, twin, fit.diagnostics, fit.metadata, fit.save_path)
        cf = calibrate_fit(cal, grid_fit, inversion_book; rates = rates, quiet = true)
        cl = cf.fit.latents
        cl isa CountLatents || error("calibrated grid twin came back as $(typeof(cl))")
        cl.match_ids == lat.match_ids || error("calibration reordered the fixtures of a smile arm")
        pooled_latents = SmileLatents(cl.match_ids, cl.λ_home, cl.λ_away, nothing,
                                      cl.λ_home .+ cl.λ_away, lat.φ, lat.strikes)
        pooled = Fit(cf.fit.config, cf.fit.folds, pooled_latents, cf.fit.diagnostics,
                     cf.fit.metadata, cf.fit.save_path)
        return [(; variant = "t25_inv_pooltot", route = "smile, calibrated λ_tot × fitted φ",
                   source = pooled, latents = pooled_latents, diagnostics = cf.rate_diagnostics),
                (; variant = "t25_inv_grid", route = "calibrated grid, φ dropped",
                   source = cf.fit, latents = cl, diagnostics = cf.rate_diagnostics)]
    end
    cf = calibrate_fit(cal, fit, inversion_book; rates = rates, quiet = true)
    return [(; variant = "t25_inv", route = "native",
               source = cf.fit, latents = cf.fit.latents, diagnostics = cf.rate_diagnostics)]
end

"The calibration family a variant belongs to, for matching comparators across arms."
gms_calibration_family(variant::AbstractString) = startswith(variant, "t25_inv") ? "t25_inv" : String(variant)

# ==============================================================================
# 3. Paired comparisons
# ==============================================================================

"Per-slate log growth keyed by slate date."
function gms_slate_log_growth(result)
    t = result.trajectory
    b, d = t.bankroll, t.dates
    prior, post = if length(b) == length(d) + 1
        b[1:end-1], b[2:end]
    elseif length(b) == length(d)
        vcat(result.summary.initial_bankroll, b[1:end-1]), b
    else
        error("trajectory has $(length(b)) bankroll points for $(length(d)) slates")
    end
    return Dict(d[i] => log(post[i] / prior[i]) for i in eachindex(d))
end

"""
    gms_paired_growth(a, b; B, seed) -> NamedTuple

The paired test the per-arm growth intervals cannot give: bootstrap the slate-by-slate
difference in log growth, a − b, over the union of slates (a slate one arm did not stake
contributes 0 for it). `p_better` is the share of resamples with a positive mean difference.
"""
function gms_paired_growth(a, b; B::Int = 10_000, seed::Int = 20260915)
    ga, gb = gms_slate_log_growth(a), gms_slate_log_growth(b)
    dates = sort!(collect(union(keys(ga), keys(gb))))
    d = [get(ga, x, 0.0) - get(gb, x, 0.0) for x in dates]
    n = length(d)
    rng = MersenneTwister(seed)
    stats = Vector{Float64}(undef, B)
    idx = Vector{Int}(undef, n)
    for k in 1:B
        rand!(rng, idx, 1:n)
        s = 0.0
        for i in idx
            s += d[i]
        end
        stats[k] = s / n
    end
    return (; n_slates = n, delta_log_growth_per_slate = mean(d),
              lo = quantile(stats, 0.025), hi = quantile(stats, 0.975),
              p_better = mean(stats .> 0.0), delta_total_log_growth = sum(d))
end

gms_bet_key(r) = (Int(r.match_id), String(r.family), String(Symbol(r.selection)))

"Shared and exclusive bet sets of two ledgers, as reportable rows."
function gms_pair_sets(pair::AbstractString, name_a, bets_a, name_b, bets_b)
    a, b = DataFrame(bets_a), DataFrame(bets_b)
    ka = Set(gms_bet_key(r) for r in eachrow(a))
    kb = Set(gms_bet_key(r) for r in eachrow(b))
    shared = intersect(ka, kb)
    rows = NamedTuple[]
    for (set, owner, frame) in (("shared", name_a, filter(r -> gms_bet_key(r) in shared, a)),
                                ("shared", name_b, filter(r -> gms_bet_key(r) in shared, b)),
                                ("exclusive", name_a, filter(r -> !(gms_bet_key(r) in kb), a)),
                                ("exclusive", name_b, filter(r -> !(gms_bet_key(r) in ka), b)))
        s = gms_edge_summary(frame)
        push!(rows, (; pair = String(pair), bet_set = set, owner = String(owner),
                       n_bets = s.n_bets, win_rate_pct = 100 * s.win_rate, roi_pct = s.roi,
                       edge_mean_pp = s.edge_mean, capture_ratio = s.capture_ratio))
    end
    return rows
end

# ==============================================================================
# 4. Trust sweep
# ==============================================================================

"""
    gms_extended_book(base) -> BookSpec

Option B's book with O/U 4.5 added to the priced markets — `canonical_markets()` stops at
3.5, and 4.5 is the smile's K = 4 strike. Every other field is the base book's. With the
new market at trust 0 the staked ledger must be unchanged; r08 gates exactly that.
"""
function gms_extended_book(base)
    markets = Data.MarketConfig(reduce(vcat, (
        Data.AbstractMarket[Data.Market1X2(), Data.MarketBTTS()],
        [Data.MarketOverUnder(i + 0.5) for i in 0:4])))
    return GMS_PORTFOLIO.BookSpec(
        markets = markets, price = base.price, allocator = base.allocator,
        shrink = base.shrink, exec = base.exec, trust = GMS_PORTFOLIO.book_trust(base))
end

"""
Option B's policy with `additions` enabled at `weight`; risk, cap, selection filter and grouping
are carried from `base` field for field. (`PolicySpec` has a `filter` slot whose default is
`KeepAll()` — omitting it would change the policy in a second way the sweep does not intend.)
"""
function gms_policy_with(base, additions; weight::Real)
    table = Dict{Any,Float64}(k => v for (k, v) in base.trust.table)
    for key in additions
        table[key] = Float64(weight)
    end
    return GMS_PORTFOLIO.PolicySpec(
        trust = GMS_PORTFOLIO.TieredTrust(table; default = base.trust.default),
        risk = base.risk, cap = base.cap, filter = base.filter, grouping = base.grouping)
end

"The single-line additions and their unions, fixed before any sweep result was seen."
const GMS_SWEEP_ADDITIONS = let
    single = [
        ("+U0.5", [("over_under", 0.5, :under)]),
        ("+U1.5", [("over_under", 1.5, :under)]),
        ("+U3.5", [("over_under", 3.5, :under)]),
        ("+U4.5", [("over_under", 4.5, :under)]),
        ("+O2.5", [("over_under", 2.5, :over)]),
        ("+O3.5", [("over_under", 3.5, :over)]),
        ("+O4.5", [("over_under", 4.5, :over)]),
        ("+BTTS_yes", [("btts", 0.0, :btts_yes)]),
        ("+BTTS_no", [("btts", 0.0, :btts_no)]),
    ]
    unders = reduce(vcat, [k for (name, k) in single if startswith(name, "+U")])
    everything = reduce(vcat, [k for (_, k) in single])
    vcat(single, [("+all_unders", unders), ("+all_fringe", everything)])
end

"""
    gms_sweep_row(env, model, policy_key, result, p0; core_families) -> NamedTuple

Headline metrics, the added families' own ROI, and the capacity effect on the core basket:
`core_stake_vs_p0` is the core families' total stake relative to Option B's, and
`delta_core_roi_pp` their ROI change — the cannibalisation the EDA measured.
"""
function gms_sweep_row(env, model, policy_key, result, p0; core_families)
    s = result.summary
    bets = DataFrame(result.trajectory.bets)
    is_core = [f in core_families for f in String.(bets.family)]
    core = gms_edge_summary(bets[is_core, :])
    added = gms_edge_summary(bets[.!is_core, :])
    p0core = gms_edge_summary(DataFrame(p0.trajectory.bets))
    total_stake = sum(bets.stake; init = 0.0)
    return (; environment = String(env), model = String(model), policy = String(policy_key),
              n_bets = s.n_bets, total_return_pct = s.total_return_pct, roi_pct = s.roi,
              sharpe_ann = s.sharpe_ann, max_drawdown_pct = s.mdd, win_rate_pct = 100 * s.win_rate,
              n_capped = result.trajectory.n_capped, mean_exposure = s.mean_exposure,
              delta_return_pp = s.total_return_pct - p0.summary.total_return_pct,
              delta_sharpe = s.sharpe_ann - p0.summary.sharpe_ann,
              core_n_bets = core.n_bets, core_roi_pct = core.roi,
              core_stake_vs_p0 = p0core.stake_sum > 0 ? core.stake_sum / p0core.stake_sum : NaN,
              delta_core_roi_pp = core.roi - p0core.roi,
              added_n_bets = added.n_bets, added_win_rate_pct = 100 * added.win_rate,
              added_roi_pct = added.roi,
              added_stake_share_pct = total_stake > 0 ? 100 * added.stake_sum / total_stake : NaN,
              added_edge_pp = added.edge_mean)
end

"""
    gms_family_calibration(env, model, policy, bets) -> Vector{NamedTuple}

Per staked family: mean p_model, mean p_market and the realised win rate. The EDA's Jensen
finding is a statement about exactly this — a deep Under whose mean p_model sits well above
its realised rate. Staked bets are a selected sample (positive edge only), so this measures
the bets the policy took, not the model's unconditional calibration.
"""
function gms_family_calibration(env, model, policy, bets::AbstractDataFrame)
    rows = NamedTuple[]
    nrow(bets) == 0 && return rows
    for sub in groupby(sort(DataFrame(bets), :family), :family)
        won = sub.payoff .> 0
        push!(rows, (; environment = String(env), model = String(model), policy = String(policy),
                       family = String(first(sub.family)), n_bets = nrow(sub),
                       mean_p_model = mean(sub.p_model), mean_p_market = mean(sub.p_market),
                       realised_win_rate = mean(won),
                       model_minus_realised = mean(sub.p_model) - mean(won),
                       roi_pct = 100 * sum(sub.pnl) / sum(sub.stake)))
    end
    return rows
end
