# l01_clv_napkin.jl — loader for the CLV napkin study (see BRIEF.md, r01_clv_napkin.jl).
#
# Everything here is plumbing: books at an instant, model probabilities from held-out
# latents, one joined selection frame per run, flat-stake bets, commission, and a
# slate-clustered bootstrap. The runner reads as the experiment; this file does the work.
#
# Definitions are the brief's, verbatim:
#   entry    last traded price at or before T−n (LTP; the archive holds no ladder)
#   close    de-vigged Betfair TWA over (−20, 0] minutes, multiplicative, within market
#   p_entry  = 1/odds_entry            (the taken price, NOT de-vigged)
#   clv_pp   = 100·(p_close − p_entry)
#   clv_pct  = 100·(odds_entry·p_close − 1)
#   edge_pp  = 100·(p_model − p_entry_fair)      (p_entry_fair = de-vigged entry book)
#
# Only `SELECT`s touch a database. No sampling, no fits are written.

using BayesianFootball
using DataFrames
using Dates
using Printf
using Random
using Serialization
using Statistics
using UUIDs
using CSV

const CN_PG = BayesianFootball.Models.PreGame

# Read-side compatibility for pre-2026-09-03 three-parameter joint-observation artefacts
# (same shim as scripts/score_runs.jl; see memory "old joint-obs fits won't deserialize").
function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   T::Type{<:CN_PG.JointGammaPoissonObservation})
    T isa DataType && return invoke(Serialization.deserialize,
        Tuple{Serialization.AbstractSerializer,DataType}, s, T)
    fields = Any[]
    for _ in 1:3
        tag = Int32(read(s.io, UInt8)::UInt8)
        push!(fields, Serialization.handle_deserialize(s, tag))
    end
    return CN_PG.JointGammaPoissonObservation(fields[1], fields[2], fields[3], CN_PG.SharedKappa())
end

"Include the prototype loaders whose model types some persisted fits reference (as score_runs.jl does)."
function cn_include_prototype_loaders!(root::AbstractString)
    Base.include(Main, joinpath(root, "current_development", "grw_pyramid_cups", "l01_loader.jl"))
    for (folder, loader) in (("10_momentum_multiscale_grw", "l10_momentum_grw_loader.jl"),
                             ("11_decompression_pxg_covariate", "l11_decompression_loader.jl"),
                             ("12_decoupled_generative_xg", "l12_loader.jl"))
        Base.include(Main, joinpath(root, "experiments", "scottish_lower", folder, loader))
    end
    return nothing
end


# ==============================================================================
# 1. MARKETS
# ==============================================================================

const CN_OU_LINES = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)

"Every market Betfair archives for tournaments 54–57 AND the score grid can price."
cn_market_config() = Data.MarketConfig(vcat(
    Data.AbstractMarket[Data.Market1X2()],
    Data.AbstractMarket[Data.MarketOverUnder(l) for l in CN_OU_LINES],
    Data.AbstractMarket[Data.MarketBTTS(), Data.MarketCorrectScore()]))

"Runner count of a complete Betfair market; 0 = not a market this study prices."
function cn_expected_runners(market_name::AbstractString)
    market_name == "1X2" && return 3
    market_name == "OverUnder" && return 2
    market_name == "BTTS" && return 2
    market_name == "CorrectScore" && return 19     # 0-0..3-3 + three "any other"
    return 0
end

"Short family label used in every table."
function cn_family(market_name::AbstractString, line::Real)
    market_name == "1X2" && return "1X2"
    market_name == "OverUnder" && return @sprintf("OU%.1f", line)
    market_name == "BTTS" && return "BTTS"
    market_name == "CorrectScore" && return "CS"
    return String(market_name)
end

"Overround sanity band. LTP-built correct-score books run wider than two-way books."
cn_overround_limits(market_name::AbstractString) =
    market_name == "CorrectScore" ? (0.90, 1.30) : (0.90, 1.10)

const CN_KEY = [:match_id, :market_name, :market_line, :selection]
const CN_MKEY = [:match_id, :market_name, :market_line]


# ==============================================================================
# 2. BOOKS
# ==============================================================================

"""
    cn_devig(prices; price_col) -> (book, refusals)

Keep only complete markets inside their overround band, then de-vig multiplicatively
within (match, market, line). Completeness is checked BEFORE normalisation.
"""
function cn_devig(prices::AbstractDataFrame; price_col::Symbol)
    keep = falses(nrow(prices))
    fair = fill(NaN, nrow(prices))
    refused = Dict{String,Int}()
    for g in groupby(prices, CN_MKEY)
        name = first(g.market_name)
        want = cn_expected_runners(name)
        if want == 0
            refused["unpriced family"] = get(refused, "unpriced family", 0) + 1
            continue
        end
        if nrow(g) != want
            refused["incomplete"] = get(refused, "incomplete", 0) + 1
            continue
        end
        implied = 1.0 ./ g[!, price_col]
        over = sum(implied)
        lo, hi = cn_overround_limits(name)
        if !(lo <= over <= hi)
            refused["overround"] = get(refused, "overround", 0) + 1
            continue
        end
        rows = parentindices(g)[1]
        keep[rows] .= true
        fair[rows] .= implied ./ over
    end
    book = prices[keep, :]
    book.p_fair = fair[keep]
    return book, refused
end

"""
    cn_close_book(ds) -> (book, refusals)

The harness scorecard's close: TWA over (−20, 0], de-vigged within market. The only
change from `Harness._betfair_closing_odds` is that incomplete markets are refused
rather than normalised (a one-sided close fabricates p = 1), and correct score gets a
wider overround band instead of the 1.10 filter that `summarize_odds` would apply.
"""
function cn_close_book(ds)
    raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator();
                              window = (-20.0, 0.0), overround_limits = (0.0, Inf))
    prices = DataFrame(match_id = Int.(raw.match_id),
                       market_name = String.(raw.market_name),
                       market_line = Float64.(raw.market_line),
                       selection = Symbol.(raw.selection),
                       odds_close = Float64.(coalesce.(raw.odds, NaN)))
    filter!(r -> isfinite(r.odds_close) && r.odds_close > 1.0, prices)
    book, refused = cn_devig(prices; price_col = :odds_close)
    rename!(book, :p_fair => :p_close)
    return book, refused
end

"""
    cn_entry_book(ds, as_of_minutes; max_staleness = 90.0) -> (book, refusals)

Last traded price at or before T`as_of_minutes` per selection
(`Calibration.point_in_time_prices`: the frame is filtered to `<= as_of` before the
argmax, so no later tick is reachable), staleness-gated, then completeness/overround
gated and de-vigged.
"""
function cn_entry_book(ds, as_of_minutes::Real; max_staleness::Real = 90.0)
    cfg = Calibration.PointInTimeBookConfig(as_of_minutes = Float64(as_of_minutes))
    prices = Calibration.point_in_time_prices(ds.betfair_odds; config = cfg)
    stale_groups = Set{Tuple{Int,String,Float64}}()
    for g in groupby(prices, CN_MKEY)
        maximum(g.staleness_minutes) > max_staleness &&
            push!(stale_groups, (first(g.match_id), first(g.market_name), first(g.market_line)))
    end
    fresh = filter(r -> (r.match_id, r.market_name, r.market_line) ∉ stale_groups, prices)
    book, refused = cn_devig(fresh; price_col = :odds_close)
    refused["stale"] = length(stale_groups)
    out = select(book, CN_KEY..., :odds_close => :odds_entry, :p_fair => :p_entry_fair,
                 :staleness_minutes)
    out.entry_minutes = fill(Float64(as_of_minutes), nrow(out))
    return out, refused
end

"""
    cn_earliest_book(ds; latest = -25.0) -> (book, refusals)

The earliest instant at which every runner of a market has traded at least once, and no
later than T`latest`; prices are the last tick at or before that instant. No staleness
gate (that is the point of the earliest book). `entry_minutes` is per market.
"""
function cn_earliest_book(ds; latest::Real = -25.0)
    visible = filter(r -> r.minutes_to_kickoff <= latest &&
                          isfinite(r.traded_price) && r.traded_price > 1.0, ds.betfair_odds)
    rows = NamedTuple[]
    for g in groupby(visible, CN_MKEY)
        want = cn_expected_runners(first(g.market_name))
        want == 0 && continue
        sels = groupby(g, :selection)
        length(sels) == want || continue
        t0 = maximum(minimum(s.minutes_to_kickoff) for s in sels)
        for s in sels
            ok = s.minutes_to_kickoff .<= t0
            i = argmax(ifelse.(ok, s.minutes_to_kickoff, -Inf))
            push!(rows, (; match_id = Int(first(g.match_id)), market_name = String(first(g.market_name)),
                         market_line = Float64(first(g.market_line)), selection = Symbol(first(s.selection)),
                         odds_close = Float64(s.traded_price[i]), entry_minutes = t0))
        end
    end
    prices = DataFrame(rows)
    book, refused = cn_devig(prices; price_col = :odds_close)
    out = select(book, CN_KEY..., :odds_close => :odds_entry, :p_fair => :p_entry_fair,
                 :entry_minutes)
    out.staleness_minutes = fill(NaN, nrow(out))
    return out, refused
end

"Match attributes and graded outcome for every priced key (settlement from final scores)."
function cn_outcomes(ds, keys::AbstractDataFrame)
    meta = Dict(Int(r.match_id) => r for r in eachrow(ds.matches))
    out = unique(select(keys, CN_KEY))
    won = Vector{Union{Missing,Bool}}(undef, nrow(out))
    date = Vector{Date}(undef, nrow(out))
    tid = Vector{Int}(undef, nrow(out))
    season = Vector{String}(undef, nrow(out))
    for (i, r) in enumerate(eachrow(out))
        m = meta[r.match_id]
        won[i] = Data.grade_selection(r.market_name, r.market_line, r.selection,
                                      m.home_score, m.away_score)
        date[i] = Date(m.match_date)
        tid[i] = Int(m.tournament_id)
        season[i] = String(m.season)
    end
    out.won = won
    out.date = date
    out.tournament_id = tid
    out.season = season
    return out
end


# ==============================================================================
# 3. RUNS AND MODEL PROBABILITIES
# ==============================================================================

"""
    cn_run_table(root) -> DataFrame

Every run named in the W2, W1 and W0 CSVs, one row per (panel, run_id). W1 and W0 share
the 710 panel, so a run listed in both is one row (labels joined with " = ").
"""
function cn_run_table(root::AbstractString)
    specs = (
        ("W2", joinpath(root, "experiments/scotland/04_pyramid_tier_transition_arms/W2_runs.csv"),
         ["23/24", "24/25", "25/26"]),
        ("W1", joinpath(root, "experiments/scotland/03_dynamics_scope_matrix/W1_runs.csv"),
         ["24/25", "25/26"]),
        ("W0", joinpath(root, "experiments/scotland/W0_runs.csv"),
         ["24/25", "25/26"]),
    )
    rows = NamedTuple[]
    for (wave, path, seasons) in specs
        frame = CSV.read(path, DataFrame)
        for r in eachrow(frame)
            push!(rows, (; wave, label = String(r.label), experiment = String(r.experiment),
                         run_id = UUID(string(r.run_id)),
                         panel = join(seasons, ","), seasons))
        end
    end
    all_rows = DataFrame(rows)
    out = NamedTuple[]
    for g in groupby(all_rows, [:panel, :run_id]; sort = false)
        push!(out, (; panel = first(g.panel), run_id = first(g.run_id),
                    wave = join(unique(g.wave), "+"), label = first(g.label),
                    aliases = join(unique(g.label), " = "), experiment = first(g.experiment),
                    seasons = first(g.seasons)))
    end
    return DataFrame(out)
end

"""
    cn_compression_slopes(run_ids) -> Dict{UUID,Float64}

1X2 target-subset compression slope from `harness_scores`, newest scorecard first
(v1.2, then v1.1, then v1). Read-only.
"""
function cn_compression_slopes(run_ids)
    storage = Training.PostgresStorage("harness")
    conn = Training.Inference._db_connect(storage)
    q = try
        DataFrame(Training.Inference.LibPQ.execute(conn, """
            SELECT run_id::text AS run_id, scorecard_version, value
            FROM harness_scores
            WHERE subset = 'target' AND market = '1X2' AND metric = 'compression_slope'"""))
    finally
        close(conn)
    end
    rank = Dict("v1.2" => 1, "v1.1" => 2, "v1" => 3)
    out = Dict{UUID,Float64}()
    best = Dict{UUID,Int}()
    for r in eachrow(q)
        id = UUID(r.run_id)
        id in run_ids || continue
        k = get(rank, coalesce(r.scorecard_version, ""), 9)
        if k < get(best, id, 99)
            best[id] = k
            out[id] = Float64(r.value)
        end
    end
    return out
end

"Load one persisted fit (prototype types must already be included)."
cn_load_fit(experiment::AbstractString, run_id::UUID) =
    Base.invokelatest(Training.load_fit, Training.PostgresStorage(experiment), run_id)

"""
    cn_model_probs(fit, match_ids; chunk = 200) -> DataFrame

Posterior-mean held-out probability for every selection of `cn_market_config()`, priced
from the fit's walk-forward latents via `Predictions.model_inference` in fixture chunks
(the PPD cache is emptied after each chunk so draw vectors do not accumulate).
"""
function cn_model_probs(fit, match_ids::AbstractVector{<:Integer}; chunk::Int = 200)
    mc = cn_market_config()
    model = fit.config.model
    frames = DataFrame[]
    ids = sort!(collect(Int.(match_ids)))
    for lo in 1:chunk:length(ids)
        part = ids[lo:min(lo + chunk - 1, length(ids))]
        lat = Calibration.restrict_latents(fit.latents, part)
        ppd = Base.invokelatest(Predictions.model_inference, lat, model; market_config = mc)
        df = ppd.df
        push!(frames, DataFrame(match_id = Int.(df.match_id), market_name = String.(df.market_name),
                                market_line = Float64.(df.market_line), selection = Symbol.(df.selection),
                                p_model = mean.(df.distribution)))
        empty!(Predictions._PPD_CACHE)
    end
    return vcat(frames...)
end

"Held-out fixtures of `fit` in `seasons` × `tournaments` (the harness panel rule)."
cn_panel(ds, fit, seasons; tournaments = (56, 57)) =
    Harness._season_panel(ds, fit, seasons; tournaments = tournaments)


# ==============================================================================
# 4. THE SELECTION FRAME AND BETS
# ==============================================================================

"""
    cn_selection_frame(probs, entry, close, outcomes) -> DataFrame

One row per selection with a model price, a valid entry book and a valid close.
The close never enters `edge_pp`; it is used only for `clv_pp`, `clv_pct`, `beat`.
"""
function cn_selection_frame(probs::AbstractDataFrame, entry::AbstractDataFrame,
                            close::AbstractDataFrame, outcomes::AbstractDataFrame)
    f = innerjoin(probs, select(entry, CN_KEY..., :odds_entry, :p_entry_fair, :entry_minutes);
                  on = CN_KEY)
    f = innerjoin(f, select(close, CN_KEY..., :p_close); on = CN_KEY)
    f = innerjoin(f, outcomes; on = CN_KEY)
    filter!(r -> !ismissing(r.won), f)
    f.won = Bool.(f.won)
    f.family = cn_family.(f.market_name, f.market_line)
    f.p_entry = 1.0 ./ f.odds_entry
    f.edge_pp = 100 .* (f.p_model .- f.p_entry_fair)
    f.clv_pp = 100 .* (f.p_close .- f.p_entry)
    f.clv_pct = 100 .* (f.odds_entry .* f.p_close .- 1.0)
    f.beat = f.clv_pp .> 0
    f.pnl_gross = ifelse.(f.won, f.odds_entry .- 1.0, -1.0)
    return f
end

"Flat 1-unit back bets: every selection with model edge ≥ `e` pp (e = -Inf is the market-only null)."
cn_bets(frame::AbstractDataFrame, e::Real) = frame[frame.edge_pp .>= e, :]

"""
    cn_market_pnl(bets, commission) -> DataFrame

Betfair charges commission on NET winnings per market, so P&L is settled per
(match, market, line): net = Σ gross; if net > 0, net·(1 − c). One row per market, with
its slate (`date`), stake and net.
"""
function cn_market_pnl(bets::AbstractDataFrame, commission::Real)
    g = combine(groupby(bets, [CN_MKEY..., :date]),
                :pnl_gross => sum => :gross, nrow => :stake)
    g.net = ifelse.(g.gross .> 0, g.gross .* (1 - commission), g.gross)
    return g
end


# ==============================================================================
# 5. SUMMARY WITH A SLATE-CLUSTERED BOOTSTRAP
# ==============================================================================

"""
    cn_summary(bets; B, seed, commissions = (0.02, 0.05)) -> NamedTuple

n_bets, mean clv_pp / clv_pct, beat share, realised ROI and close-implied expected ROI
(`ev*`: E[P&L] per unit if the de-vigged close is the true probability, after commission on
winnings — the low-variance twin of realised ROI) at each commission, each with a 95%
slate-clustered bootstrap CI (a slate = one match day). Slates are resampled with
replacement; ratio statistics are recomputed from resampled slate sums. `p_*` is the
one-sided bootstrap share of replicates at or below zero; `z_*` = point / bootstrap SD
(normal-tail z for a rough Bonferroni check beyond the bootstrap's 1/B resolution).
"""
function cn_summary(bets::AbstractDataFrame; B::Int = 2_000, seed::Int = 20260928,
                    commissions = (0.02, 0.05))
    n = nrow(bets)
    empty = (; n_bets = n, n_slates = 0, n_matches = 0, mean_edge = NaN,
             clv_pp = NaN, clv_pp_lo = NaN, clv_pp_hi = NaN, p_clv = NaN,
             clv_pct = NaN, beat = NaN, beat_lo = NaN, beat_hi = NaN,
             roi2 = NaN, roi2_lo = NaN, roi2_hi = NaN, p_roi2 = NaN,
             roi5 = NaN, roi5_lo = NaN, roi5_hi = NaN, p_roi5 = NaN, hit = NaN,
             mean_odds = NaN, z_clv = NaN, z_roi2 = NaN,
             ev2 = NaN, ev2_lo = NaN, ev2_hi = NaN, ev5 = NaN, ev5_lo = NaN, ev5_hi = NaN,
             z_ev2 = NaN)
    n < 5 && return empty

    slates = sort!(unique(bets.date))
    sidx = Dict(d => i for (i, d) in enumerate(slates))
    S = length(slates)
    s_n = zeros(S); s_clv = zeros(S); s_beat = zeros(S)
    for r in eachrow(bets)
        i = sidx[r.date]
        s_n[i] += 1
        s_clv[i] += r.clv_pp
        s_beat[i] += r.beat
    end
    # close-implied expected P&L per unit, commission on winnings: p·(o−1)·(1−c) − (1−p)
    s_ev = [zeros(S) for _ in commissions]
    for r in eachrow(bets), (k, c) in enumerate(commissions)
        s_ev[k][sidx[r.date]] += r.p_close * (r.odds_entry - 1) * (1 - c) - (1 - r.p_close)
    end
    s_net = [zeros(S) for _ in commissions]
    for (k, c) in enumerate(commissions)
        mp = cn_market_pnl(bets, c)
        for r in eachrow(mp)
            s_net[k][sidx[r.date]] += r.net
        end
    end
    # every bet is one unit, so stake per slate == s_n
    stat(w) = (sum(w .* s_clv) / sum(w .* s_n), sum(w .* s_beat) / sum(w .* s_n),
               [100 * sum(w .* x) / sum(w .* s_n) for x in s_net]...,
               [100 * sum(w .* x) / sum(w .* s_n) for x in s_ev]...)
    point = stat(ones(S))
    rng = MersenneTwister(seed)
    reps = Matrix{Float64}(undef, B, length(point))
    w = zeros(S)
    for b in 1:B
        fill!(w, 0.0)
        for _ in 1:S
            w[rand(rng, 1:S)] += 1.0
        end
        reps[b, :] .= stat(w)
    end
    q(j) = (quantile(view(reps, :, j), 0.025), quantile(view(reps, :, j), 0.975))
    pneg(j) = mean(view(reps, :, j) .<= 0)
    zval(j) = point[j] / std(view(reps, :, j))
    (c_lo, c_hi) = q(1); (b_lo, b_hi) = q(2); (r2_lo, r2_hi) = q(3); (r5_lo, r5_hi) = q(4)
    (v2_lo, v2_hi) = q(5); (v5_lo, v5_hi) = q(6)
    return (; n_bets = n, n_slates = S, n_matches = length(unique(bets.match_id)),
            mean_edge = mean(bets.edge_pp),
            clv_pp = point[1], clv_pp_lo = c_lo, clv_pp_hi = c_hi, p_clv = pneg(1),
            clv_pct = mean(bets.clv_pct),
            beat = 100 * point[2], beat_lo = 100 * b_lo, beat_hi = 100 * b_hi,
            roi2 = point[3], roi2_lo = r2_lo, roi2_hi = r2_hi, p_roi2 = pneg(3),
            roi5 = point[4], roi5_lo = r5_lo, roi5_hi = r5_hi, p_roi5 = pneg(4),
            hit = 100 * mean(bets.won), mean_odds = mean(bets.odds_entry),
            z_clv = zval(1), z_roi2 = zval(3),
            ev2 = point[5], ev2_lo = v2_lo, ev2_hi = v2_hi,
            ev5 = point[6], ev5_lo = v5_lo, ev5_hi = v5_hi, z_ev2 = zval(5))
end

"""
    cn_move_capture(frame) -> Float64

OLS slope of the entry→close fair move on the model edge, over all selections:
β = cov(p_close − p_entry_fair, p_model − p_entry_fair) / var(p_model − p_entry_fair).
β > 0 means the market moves toward the model between T−n and the close; β ≈ 0 means the
model's disagreements carry no information the market later prices.
"""
function cn_move_capture(frame::AbstractDataFrame)
    nrow(frame) < 10 && return NaN
    x = frame.p_model .- frame.p_entry_fair
    y = frame.p_close .- frame.p_entry_fair
    return cov(x, y) / var(x)
end


# ==============================================================================
# 5b. MARKET-ONLY PLACEBO
# ==============================================================================
#
# The entry is a last TRADED price, so it bounces between back and lay and can be stale.
# Any model centred near the fair price will then "find edge" where the LTP is noisily
# long and collect CLV when the (smoother) close reverts. The placebo measures that floor:
# a "model" that is nothing but the market's own recent history, all at or before T−25.

"""
    cn_placebo_book(ds; window = (-180.0, -25.0)) -> (book, refusals)

De-vigged TWA of the market's own prices over `window` (both ends inclusive, so no tick
after T−25), complete markets only. Used as `p_model` for the placebo.
"""
function cn_placebo_book(ds; window::Tuple{Float64,Float64} = (-180.0, -25.0))
    raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator();
                              window = window, overround_limits = (0.0, Inf))
    prices = DataFrame(match_id = Int.(raw.match_id),
                       market_name = String.(raw.market_name),
                       market_line = Float64.(raw.market_line),
                       selection = Symbol.(raw.selection),
                       odds_twa = Float64.(coalesce.(raw.odds, NaN)))
    filter!(r -> isfinite(r.odds_twa) && r.odds_twa > 1.0, prices)
    book, refused = cn_devig(prices; price_col = :odds_twa)
    return select(book, CN_KEY..., :p_fair => :p_placebo), refused
end

"The placebo as a probability frame for `match_ids` (drop-in for `cn_model_probs`)."
function cn_placebo_probs(placebo::AbstractDataFrame, match_ids)
    keep = Set(Int.(match_ids))
    out = placebo[in.(placebo.match_id, Ref(keep)), :]
    return rename(select(out, CN_KEY..., :p_placebo), :p_placebo => :p_model)
end

"""
    cn_joint_capture(frame, placebo; B = 500, seed) -> NamedTuple

Does the model's edge predict the entry→close move BEYOND the placebo's edge?
OLS of y = p_close − p_entry_fair on x_model = p_model − p_entry_fair and
x_placebo = p_placebo − p_entry_fair over the selections both price, with a slate
bootstrap for the model coefficient. `beta_model_joint > 0` with a CI above 0 is the
evidence that the model knows something the market's own recent prices do not.
"""
function cn_joint_capture(frame::AbstractDataFrame, placebo::AbstractDataFrame;
                          B::Int = 500, seed::Int = 20260928)
    j = innerjoin(select(frame, CN_KEY..., :date, :p_model, :p_entry_fair, :p_close),
                  select(placebo, CN_KEY..., :p_placebo); on = CN_KEY)
    nrow(j) < 30 && return (; n = nrow(j), beta_model_alone = NaN, beta_placebo_alone = NaN,
                            beta_model_joint = NaN, beta_model_joint_lo = NaN,
                            beta_model_joint_hi = NaN, beta_placebo_joint = NaN, r_edges = NaN)
    x1 = j.p_model .- j.p_entry_fair
    x2 = j.p_placebo .- j.p_entry_fair
    y = j.p_close .- j.p_entry_fair
    fitb(idx) = [ones(length(idx)) x1[idx] x2[idx]] \ y[idx]
    b = fitb(1:nrow(j))
    slates = unique(j.date)
    rows_of = Dict(d => findall(==(d), j.date) for d in slates)
    rng = MersenneTwister(seed)
    reps = Float64[]
    for _ in 1:B
        idx = reduce(vcat, (rows_of[slates[rand(rng, 1:length(slates))]] for _ in slates))
        push!(reps, fitb(idx)[2])
    end
    return (; n = nrow(j),
            beta_model_alone = cov(x1, y) / var(x1),
            beta_placebo_alone = cov(x2, y) / var(x2),
            beta_model_joint = b[2],
            beta_model_joint_lo = quantile(reps, 0.025),
            beta_model_joint_hi = quantile(reps, 0.975),
            beta_placebo_joint = b[3],
            r_edges = cor(x1, x2))
end


# ==============================================================================
# 6. SMALL PRINTING HELPERS
# ==============================================================================

cn_fmt(x; d = 2) = isfinite(x) ? string(round(x; digits = d)) : "—"
cn_ci(x, lo, hi; d = 2) = isfinite(x) ? @sprintf("%.*f [%.*f, %.*f]", d, x, d, lo, d, hi) : "—"

"Markdown table from a DataFrame (strings/numbers as-is)."
function cn_markdown(df::AbstractDataFrame)
    io = IOBuffer()
    cols = names(df)
    println(io, "| ", join(cols, " | "), " |")
    println(io, "|", join(fill("---", length(cols)), "|"), "|")
    for r in eachrow(df)
        println(io, "| ", join((string(r[c]) for c in cols), " | "), " |")
    end
    return String(take!(io))
end

"Crude ASCII scatter (x, y) with a label char per point."
function cn_ascii_scatter(x, y, labels; width = 60, height = 16, xlab = "x", ylab = "y")
    xs = collect(Float64, x); ys = collect(Float64, y)
    x0, x1 = extrema(xs); y0, y1 = extrema(ys)
    x1 == x0 && (x1 = x0 + 1); y1 == y0 && (y1 = y0 + 1)
    grid = fill(' ', height, width)
    if y0 < 0 < y1
        r0 = height - round(Int, (0 - y0) / (y1 - y0) * (height - 1))
        grid[r0, :] .= '·'
    end
    for (xi, yi, c) in zip(xs, ys, labels)
        col = 1 + round(Int, (xi - x0) / (x1 - x0) * (width - 1))
        row = height - round(Int, (yi - y0) / (y1 - y0) * (height - 1))
        grid[row, col] = c
    end
    io = IOBuffer()
    println(io, @sprintf("%s  %+.2f", ylab, y1))
    for r in 1:height
        println(io, "  |", String(grid[r, :]))
    end
    println(io, @sprintf("    %+.2f", y0), " "^max(1, width - 16), @sprintf("%s %.2f → %.2f", xlab, x0, x1))
    return String(take!(io))
end
