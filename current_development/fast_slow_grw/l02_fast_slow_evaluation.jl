# ==============================================================================
# TODO 021 evaluation loader — draw mixtures, headline metrics, tails, derivatives
# ==============================================================================
#
# Definitions only; `r03_fast_slow_evaluation_and_blend.jl` executes.
#
# Reuses Task 013's evaluation helpers (`gph_betfair_closing_odds`, `gph_restrict`,
# `gph_buildable_panel`, `gph_simulate`, `gph_portfolio_row`) so the closing book,
# the panel narrowing and the portfolio path are the ones the 40-fold GRW ladder
# was benchmarked with.
# ==============================================================================

include(joinpath(@__DIR__, "l01_fast_slow_grw_loader.jl"))
include(joinpath(@__DIR__, "..", "grw_player_hybrid", "l02_evaluation.jl"))

# ==============================================================================
# 1. Arms and mixtures
# ==============================================================================

"Every arm by name from this package's namespace, loaded and restricted to the 710 panel."
function fsg_load_arms(ds, c::FSGConfig)
    db = PostgresStorage(c.experiment)
    fits = Dict{String,Any}()
    run_ids = Dict{String,UUID}()
    for name in FSG_MODEL_NAMES
        run_id = gph_run_by_name(db, name)
        run_id === nothing && error("no completed run named $name in $(c.experiment)")
        fit = gph_load_arm(GPHArm(name, c.experiment, run_id, "MultiScaleGRW", "arm"))
        panel = gph_season_panel(ds, fit, c.target_seasons)
        length(panel) == c.expected_oos || error("$name: $(length(panel)) panel fixtures")
        fits[name] = gph_restrict(fit, panel)
        run_ids[name] = run_id
    end
    return fits, run_ids
end

"`fit` carrying `latents` in place of its own — the portfolio path reads latents only."
fsg_with_latents(fit, latents) =
    Fit(fit.config, fit.folds, latents, fit.diagnostics, fit.metadata, fit.save_path)

"""
    fsg_mixture_grid(fits, c) -> Vector{NamedTuple}

One entry per (loose arm, ρ). ρ = 0 is the pure tight arm and appears once, labelled
`m01_poisson_grw_tight`; every other entry is `<loose>@ρ`.
"""
function fsg_mixture_grid(fits, c::FSGConfig)
    tight = fits[FSG_TIGHT]
    out = NamedTuple[(; label = FSG_TIGHT, loose = "—", rho = 0.0,
                        fit = tight)]
    for loose in FSG_LOOSE_NAMES, ρ in c.rhos
        ρ == 0.0 && continue
        lat = mixture_latents(tight.latents, fits[loose].latents, ρ)
        push!(out, (; label = ρ == 1.0 ? loose : @sprintf("%s@%.2f", loose, ρ),
                      loose, rho = ρ, fit = fsg_with_latents(tight, lat)))
    end
    return out
end

# ==============================================================================
# 2. The work-package staking system
# ==============================================================================

"""
`BookSpec(1X2, O/U 2.5, BakerMcHale)` and `PolicySpec(FlatTrust(0.25),
SlateDrawdown(20.0), FixedCap(0.25))`, as the work package specifies.
"""
function fsg_system()
    book = BookSpec(markets = Data.MarketConfig([Data.Market1X2(), Data.MarketOverUnder(2.5)]),
                    shrink = BakerMcHale())
    policy = PolicySpec(trust = FlatTrust(0.25), risk = SlateDrawdown(20.0), cap = FixedCap(0.25))
    return book, policy
end

"""
    fsg_ledger_metrics(bets) -> NamedTuple

Capital shares by price band (stake-weighted, `stake` being a bankroll fraction) and
flat-staking ROI: one unit on every bet the policy placed, `mean(pnl / stake)`.
"""
function fsg_ledger_metrics(bets::AbstractDataFrame)
    nrow(bets) == 0 && return (; capital_ge4_pct = NaN, capital_le18_pct = NaN,
                                 flat_roi_pct = NaN, median_odds = NaN)
    total = sum(bets.stake)
    return (; capital_ge4_pct = 100 * sum(bets.stake[bets.odds .>= 4.0]) / total,
              capital_le18_pct = 100 * sum(bets.stake[bets.odds .<= 1.8]) / total,
              flat_roi_pct = 100 * mean(bets.pnl ./ bets.stake),
              median_odds = median(bets.odds))
end

# ==============================================================================
# 3. Favourite tail
# ==============================================================================

const FSG_FAV_BANDS = [(0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.00)]

"""
    fsg_favourite_tail(latents, model, market) -> DataFrame

For each band of the market's favourite probability (the larger of home and away in
the de-vigged close), the mean market and model probability on that same side.
"""
function fsg_favourite_tail(latents, model, market::AbstractDataFrame)
    probs, _ = fsg_fixture_probs(latents, model)
    wide = unstack(select(filter(r -> r.market_name == "1X2", probs), :match_id, :selection, :prob),
                   :selection, :prob)
    j = innerjoin(wide, select(market, :match_id, :p_mkt_home, :p_mkt_away); on = :match_id)
    home_fav = j.p_mkt_home .>= j.p_mkt_away
    p_mkt = ifelse.(home_fav, j.p_mkt_home, j.p_mkt_away)
    p_mod = ifelse.(home_fav, j.home, j.away)
    rows = NamedTuple[]
    for (lo, hi) in FSG_FAV_BANDS
        m = (p_mkt .>= lo) .& (p_mkt .< hi)
        push!(rows, (; band = @sprintf("[%.2f, %.2f)", lo, hi), n = count(m),
                       p_market = count(m) == 0 ? NaN : mean(p_mkt[m]),
                       p_model = count(m) == 0 ? NaN : mean(p_mod[m])))
    end
    return DataFrame(rows)
end

# ==============================================================================
# 4. Derivative-market consistency
# ==============================================================================

"""
    fsg_family_scores(latents, model, odds) -> DataFrame

Per market family (1X2, O/U 2.5, BTTS) on fixtures the close quotes and settled:
model log loss, the de-vigged close's log loss on the same rows, and the mean model
vs market probability of the headline selection (home / over 2.5 / BTTS yes) — a
distortion of totals or BTTS by the mixture shows up as a drift in the last two.
"""
function fsg_family_scores(latents, model, odds::AbstractDataFrame)
    probs, _ = fsg_fixture_probs(latents, model)
    fam(m, l) = m == "1X2" ? "1X2" : m == "BTTS" ? "BTTS" : "OU" * string(l)
    probs.family = fam.(probs.market_name, probs.market_line)
    mk = filter(r -> !ismissing(r.is_winner), odds)
    mk = select(mk, :match_id, :market_name, :market_line, :selection, :prob_fair_close, :is_winner)
    mk.family = [lowercase(r.market_name) == "1x2" ? "1X2" :
                 occursin("btts", lowercase(r.market_name)) ? "BTTS" :
                 "OU" * string(r.market_line) for r in eachrow(mk)]
    j = innerjoin(select(probs, :match_id, :family, :selection, :prob),
                  select(mk, :match_id, :family, :selection, :prob_fair_close, :is_winner);
                  on = [:match_id, :family, :selection])
    headline = Dict("1X2" => :home, "OU2.5" => :over_25, "BTTS" => :btts_yes)
    rows = NamedTuple[]
    for f in ("1X2", "OU2.5", "BTTS")
        g = filter(r -> r.family == f, j)
        w = filter(r -> r.is_winner == true, g)
        h = filter(r -> r.selection == headline[f], g)
        push!(rows, (; family = f, n_fixtures = length(unique(w.match_id)),
                       logloss_model = -mean(log.(clamp.(w.prob, 1e-12, 1.0))),
                       logloss_close = -mean(log.(clamp.(w.prob_fair_close, 1e-12, 1.0))),
                       headline = String(headline[f]),
                       mean_p_model = mean(h.prob),
                       mean_p_close = mean(h.prob_fair_close)))
    end
    return DataFrame(rows)
end
