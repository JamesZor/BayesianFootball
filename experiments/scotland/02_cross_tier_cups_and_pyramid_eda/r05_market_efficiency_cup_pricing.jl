# r05 — Market efficiency in cross-tier cup ties.
#
#   include("experiments/scotland/02_cross_tier_cups_and_pyramid_eda/r05_market_efficiency_cup_pricing.jl")
#
# 1. De-vig the closing 1X2 (multiplicative and Shin) and invert it to independent-Poisson
#    rates (λ_h, λ_a) ⇒ market-implied supremacy Δλ_mkt = λ_h − λ_a.
# 2. Pricing error = realised GD − Δλ_mkt, by oriented tier pairing, vs league baseline.
# 3. Favourite–longshot bias: calibration slope, bins, flat-stake ROI by side.
# 4. Point-in-time challengers: (a) tier-only GLM refitted each season on earlier seasons,
#    (b) Dixon–Coles network refitted monthly on strictly earlier fixtures.  Does either
#    carry information the close has not priced (encompassing regression, log-loss)?
#
# Odds source: sofascore.match_odds 'Full time' (the only source covering the Challenge
# Cup).  Betfair exchange close is used as a robustness check where it exists.

# ── 1. Setup ────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "_common.jl"))
using GLM, Optim, Distributions, SpecialFunctions, Plots
include(joinpath(@__DIR__, "l04_dixon_coles_core.jl"))
gr(); default(fontfamily = "sans-serif", dpi = 150, framestyle = :axes, grid = :y, gridalpha = 0.15)

fx_all = load_fixtures(:long)
fx = fx_all[fx_all.in_primary_window .& .!ismissing.(fx_all.odds_home), :]

# ── 2. De-vig and Poisson inversion ─────────────────────────────────────────
"Shin (1993) de-vig: solve for insider share z so the adjusted probabilities sum to 1."
function shin(o::NTuple{3, Float64})
    π_ = 1 ./ collect(o); S = sum(π_)
    pz(z) = (sqrt.(z^2 .+ 4 * (1 - z) .* π_ .^ 2 ./ S) .- z) ./ (2 * (1 - z))
    lo, hi = 0.0, 0.4
    for _ in 1:80
        mid = (lo + hi) / 2
        sum(pz(mid)) > 1 ? (lo = mid) : (hi = mid)
    end
    p = pz((lo + hi) / 2)
    return (p ./ sum(p)..., (lo + hi) / 2)
end

"Independent-Poisson rates reproducing (p_home, p_away) exactly (2 equations, 2 unknowns)."
function invert_poisson(ph, pa)
    obj(v) = (q = probs_1x2(exp(v[1]), exp(v[2])); (q[1] - ph)^2 + (q[3] - pa)^2)
    r = optimize(obj, [log(1.4), log(1.1)], NelderMead(), Optim.Options(g_tol = 1e-14, iterations = 2000))
    v = Optim.minimizer(r)
    return exp(v[1]), exp(v[2]), sqrt(Optim.minimum(r))
end

o3 = [(Float64(a), Float64(b), Float64(c)) for (a, b, c) in zip(fx.odds_home, fx.odds_draw, fx.odds_away)]
fx.overround = [sum(1 ./ collect(o)) - 1 for o in o3]
mult = [(1 ./ collect(o)) ./ sum(1 ./ collect(o)) for o in o3]
sh = shin.(o3)
fx.p_home_mult = getindex.(mult, 1); fx.p_draw_mult = getindex.(mult, 2); fx.p_away_mult = getindex.(mult, 3)
fx.p_home_mkt = getindex.(sh, 1); fx.p_draw_mkt = getindex.(sh, 2); fx.p_away_mkt = getindex.(sh, 3)
fx.shin_z = getindex.(sh, 4)
inv_ = invert_poisson.(fx.p_home_mkt, fx.p_away_mkt)
fx.lam_home_mkt = getindex.(inv_, 1); fx.lam_away_mkt = getindex.(inv_, 2); fx.inv_resid = getindex.(inv_, 3)
fx.sup_mkt = fx.lam_home_mkt .- fx.lam_away_mkt
fx.gd = fx.home_goals .- fx.away_goals
fx.err = fx.gd .- fx.sup_mkt
@printf("odds rows %d | median overround %.3f | median Shin z %.4f | max inversion residual %.2e\n",
        nrow(fx), median(fx.overround), median(fx.shin_z), maximum(fx.inv_resid))

# ── 3. Pricing error by tier pairing ────────────────────────────────────────
# Oriented from the higher-tier club: err_hi = gd_hi − Δλ_mkt,hi.  Positive ⇒ the higher
# tier beat its market-implied margin (market under-priced the pyramid gap).
ox = oriented_cross_tier(fx)
ox.sup_mkt_hi = ox.lam_mkt_hi .- ox.lam_mkt_lo
ox.err_hi = ox.gd_hi .- ox.sup_mkt_hi
tstat(x) = (m = mean(x); s = std(x) / sqrt(length(x)); (mean = m, se = s, t = m / s, p = 2ccdf(Normal(), abs(m / s))))
function err_row(g)
    e = tstat(g.err_hi)
    (N = nrow(g), mkt_sup_hi = mean(g.sup_mkt_hi), real_gd_hi = mean(g.gd_hi), err_mean = e.mean, err_se = e.se, t = e.t, p = e.p,
     mkt_p_hi = mean(g.p_mkt_hi), real_hi_win = mean(g.res_hi .== 1), mkt_p_draw = mean(g.p_draw_mkt), real_draw = mean(g.res_hi .== 0),
     mkt_p_lo = mean(g.p_mkt_lo), real_upset = mean(g.res_hi .== -1))
end
e_pair = sort(combine(groupby(ox, [:hi_tier, :lo_tier]), err_row), [:hi_tier, :lo_tier])
e_gap = sort(combine(groupby(ox, :gap), err_row), :gap)
e_comp = sort(combine(groupby(ox, :competition), err_row), :competition)
e_all = combine(ox, err_row); insertcols!(e_all, 1, :group => "all cross-tier")
# baselines, home perspective: same-tier league, same-tier cup
base_rows = NamedTuple[]
for (lab, m) in (("same-tier league (home persp.)", .!fx.is_cup),
                 ("same-tier cup (home persp.)", fx.is_cup .& coalesce.(fx.tier_delta .== 0, false) .& .!fx.neutral))
    d = fx[m, :]; e = tstat(d.err)
    push!(base_rows, (group = lab, N = nrow(d), mkt_sup = mean(d.sup_mkt), real_gd = mean(d.gd), err_mean = e.mean, err_se = e.se, t = e.t, p = e.p))
end
base = DataFrame(base_rows)
for (n, t) in (("r05_error_by_tier_pair", e_pair), ("r05_error_by_gap", e_gap), ("r05_error_by_competition", e_comp),
               ("r05_error_all_cross_tier", e_all), ("r05_error_baselines", base))
    save_csv(n * ".csv", t); save_md(n * ".md", t; digits = 3)
end

# ── 4. Calibration / Mincer–Zarnowitz / favourite–longshot bias ─────────────
# (a) MZ: gd_hi = a + b·Δλ_mkt,hi.  b > 1 ⇒ the market compresses cross-tier supremacy.
mz_rows = NamedTuple[]
function mz!(lab, y, x)
    m = lm(hcat(ones(length(x)), x), y); c = coef(m); V = vcov(m)
    # HC1 for the slope
    X = hcat(ones(length(x)), x); e = y .- X * c; B = inv(X'X)
    Vh = B * (X' * Diagonal(e .^ 2) * X) * B .* (length(y) / (length(y) - 2))
    push!(mz_rows, (sample = lab, N = length(y), intercept = c[1], slope = c[2], slope_se_hc1 = sqrt(Vh[2, 2]),
                    t_slope_eq_1 = (c[2] - 1) / sqrt(Vh[2, 2]), p_slope_eq_1 = 2ccdf(Normal(), abs((c[2] - 1) / sqrt(Vh[2, 2])))))
end
mz!("cross-tier cup (hi persp.)", Float64.(ox.gd_hi), ox.sup_mkt_hi)
for c in ("Scottish Cup", "League Cup", "Challenge Cup")
    m = ox.competition .== c; mz!("cross-tier $(c)", Float64.(ox.gd_hi[m]), ox.sup_mkt_hi[m])
end
lgm = .!fx.is_cup; mz!("league (home persp.)", Float64.(fx.gd[lgm]), fx.sup_mkt[lgm])
for t in 1:4
    m = lgm .& coalesce.(fx.home_tier .== t, false); mz!("league T$t", Float64.(fx.gd[m]), fx.sup_mkt[m])
end
mz = DataFrame(mz_rows)
save_csv("r05_mincer_zarnowitz.csv", mz); save_md("r05_mincer_zarnowitz.md", mz; digits = 3)

# (b) Logistic calibration of the higher-tier win: logit P(win) = a + b·logit(p_mkt).
logit(p) = log(p / (1 - p))
cal_rows = NamedTuple[]
function cal!(lab, y, p)
    m = glm(hcat(ones(length(p)), logit.(p)), Float64.(y), Binomial(), LogitLink())
    c = coef(m); s = stderror(m)
    push!(cal_rows, (sample = lab, N = length(y), intercept = c[1], intercept_se = s[1], slope = c[2], slope_se = s[2],
                     mean_p = mean(p), realised = mean(y), brier = mean((y .- p) .^ 2)))
end
cal!("cross-tier: higher-tier win", ox.res_hi .== 1, ox.p_mkt_hi)
cal!("cross-tier: lower-tier win (upset)", ox.res_hi .== -1, ox.p_mkt_lo)
cal!("cross-tier: draw", ox.res_hi .== 0, ox.p_draw_mkt)
cal!("league: home win", fx.gd[lgm] .> 0, fx.p_home_mkt[lgm])
cal!("league: away win", fx.gd[lgm] .< 0, fx.p_away_mkt[lgm])
cal!("league: draw", fx.gd[lgm] .== 0, fx.p_draw_mkt[lgm])
cal = DataFrame(cal_rows)
save_csv("r05_logistic_calibration.csv", cal); save_md("r05_logistic_calibration.md", cal; digits = 3)

# (c) Implied-probability bins for the higher-tier side
edges = [0.0, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ox.pbin = [findlast(e -> p >= e, edges[1:end-1]) for p in ox.p_mkt_hi]
bins = sort(combine(groupby(ox, :pbin), nrow => :N, :p_mkt_hi => mean => :implied_hi_win,
                    :res_hi => (r -> mean(r .== 1)) => :realised_hi_win,
                    :res_hi => (r -> sqrt(mean(r .== 1) * (1 - mean(r .== 1)) / length(r))) => :se), :pbin)
bins.range = [@sprintf("%.2f–%.2f", edges[b], edges[b+1]) for b in bins.pbin]
bins.gap = bins.realised_hi_win .- bins.implied_hi_win
save_csv("r05_hi_win_calibration_bins.csv", bins); save_md("r05_hi_win_calibration_bins.md", bins[:, [:range, :N, :implied_hi_win, :realised_hi_win, :se, :gap]]; digits = 3)

# (d) Flat 1-unit stakes at the quoted closing price (includes the margin)
roi(win, odds) = (r = win .* odds .- 1; (N = length(r), roi = mean(r), se = std(r) / sqrt(length(r))))
roi_rows = NamedTuple[]
for (lab, d) in (("all cross-tier", ox), ("gap = 1", ox[ox.gap .== 1, :]), ("gap ≥ 2", ox[ox.gap .>= 2, :]),
                 ("Scottish Cup", ox[ox.competition .== "Scottish Cup", :]), ("League Cup", ox[ox.competition .== "League Cup", :]),
                 ("Challenge Cup", ox[ox.competition .== "Challenge Cup", :]))
    for (side, w, o) in (("higher tier", d.res_hi .== 1, d.odds_hi), ("draw", d.res_hi .== 0, d.odds_draw), ("lower tier", d.res_hi .== -1, d.odds_lo))
        r = roi(w, o); push!(roi_rows, (sample = lab, side = side, r.N, r.roi, r.se))
    end
end
for (side, w, o) in (("home", fx.gd[lgm] .> 0, fx.odds_home[lgm]), ("draw", fx.gd[lgm] .== 0, fx.odds_draw[lgm]), ("away", fx.gd[lgm] .< 0, fx.odds_away[lgm]))
    r = roi(w, o); push!(roi_rows, (sample = "league baseline", side = side, r.N, r.roi, r.se))
end
roidf = DataFrame(roi_rows)
save_csv("r05_flat_stake_roi.csv", roidf); save_md("r05_flat_stake_roi.md", roidf; digits = 3)

# (e) The favourite–longshot test proper: every selection (home/draw/away) pooled and
#     banded by its de-vigged probability, cross-tier ties vs same-tier league games.
#     The margin is common to both samples, so the *difference* in ROI per band is the bias.
function selections(d, win_h, win_d, win_a)
    DataFrame(p = vcat(d.p_home_mkt, d.p_draw_mkt, d.p_away_mkt),
              odds = Float64.(vcat(d.odds_home, d.odds_draw, d.odds_away)),
              win = vcat(win_h, win_d, win_a),
              side = vcat(fill("home", nrow(d)), fill("draw", nrow(d)), fill("away", nrow(d))))
end
xt = fx[in.(fx.match_id, Ref(Set(ox.match_id))), :]
sel = vcat(insertcols!(selections(xt, xt.gd .> 0, xt.gd .== 0, xt.gd .< 0), :sample => "cross-tier cup"),
           insertcols!(selections(fx[lgm, :], fx.gd[lgm] .> 0, fx.gd[lgm] .== 0, fx.gd[lgm] .< 0), :sample => "same-tier league"))
pb = [0.0, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0]
sel.band = [@sprintf("%.2f–%.2f", pb[b], pb[b+1]) for b in (findlast(e -> p >= e, pb[1:end-1]) for p in sel.p)]
flb = sort(combine(groupby(sel, [:band, :sample]), nrow => :N, :p => mean => :implied, :win => mean => :realised,
                   [:win, :odds] => ((w, o) -> mean(w .* o .- 1)) => :roi,
                   [:win, :odds] => ((w, o) -> std(w .* o .- 1) / sqrt(length(w))) => :roi_se), [:band, :sample])
save_csv("r05_flb_by_probability_band.csv", flb); save_md("r05_flb_by_probability_band.md", flb; digits = 3)
@printf("median overround: cross-tier %.3f | league %.3f\n", median(xt.overround), median(fx.overround[lgm]))

# ── 5. Point-in-time challengers ────────────────────────────────────────────
# (a) Tier-only Poisson GLM (r03 design, Old Firm split), refitted before each football
#     season on every earlier season back to 2008/09.
include(joinpath(@__DIR__, "l03_tier_design.jl"))
ox.sup_tier_hi = fill(NaN, nrow(ox)); ox.ptier_hi = fill(NaN, nrow(ox)); ox.ptier_d = fill(NaN, nrow(ox)); ox.ptier_lo = fill(NaN, nrow(ox))
for s in sort(unique(ox.fs))
    Dtr = long_design(fx_all[fx_all.fs .< s, :]; split_old_firm = true)
    m = glm(Dtr.X, Dtr.y, Poisson(), LogLink()); β = coef(m)
    idx = findall(ox.fs .== s)
    for i in idx
        r = ox[i, :]
        lev(t, team) = team in OLD_FIRM ? "T0" : "T$t"
        xh = design_row(Dtr.names, lev(r.hi_tier, r.hi_team), lev(r.lo_tier, r.lo_team), r.venue == "hi_home", r.competition)
        xl = design_row(Dtr.names, lev(r.lo_tier, r.lo_team), lev(r.hi_tier, r.hi_team), r.venue == "lo_home", r.competition)
        λh, λl = exp(dot(xh, β)), exp(dot(xl, β))
        q = probs_1x2(λh, λl)
        ox.sup_tier_hi[i] = λh - λl; ox.ptier_hi[i], ox.ptier_d[i], ox.ptier_lo[i] = q
    end
end

# (b) Dixon–Coles network, refitted on the first day of each month with a cross-tier tie,
#     on all fixtures from 2014/15 strictly before that day.
ox.sup_dc_hi = fill(NaN, nrow(ox)); ox.pdc_hi = fill(NaN, nrow(ox)); ox.pdc_d = fill(NaN, nrow(ox)); ox.pdc_lo = fill(NaN, nrow(ox))
months = sort(unique(firstdayofmonth.(ox.match_date)))
t0 = time()
for mth in months
    tr = fx_all[(fx_all.match_date .< mth) .& (fx_all.fs .>= 2014), :]
    N = build_network(tr)
    p, _ = fit_dc(N, 1.0, 0.20)
    for i in findall(firstdayofmonth.(ox.match_date) .== mth)
        r = ox[i, :]
        h_home = r.venue == "hi_home"
        home, away = h_home ? (r.hi_team, r.lo_team) : (r.lo_team, r.hi_team)
        λh, λa = predict_dc(N, p, home, away, r.fs, r.competition, r.venue == "neutral", 1.0)
        q = probs_1x2(λh, λa, p[7])
        if h_home
            ox.sup_dc_hi[i] = λh - λa; ox.pdc_hi[i], ox.pdc_d[i], ox.pdc_lo[i] = q
        else
            ox.sup_dc_hi[i] = λa - λh; ox.pdc_lo[i], ox.pdc_d[i], ox.pdc_hi[i] = q
        end
    end
end
@printf("walk-forward DC: %d monthly refits in %.1f s\n", length(months), time() - t0)

# Proper scores on the cross-tier ties (1X2 log-loss, RPS) and encompassing regressions
function scores(ph, pd, pl, res)
    p_obs = [r == 1 ? a : r == 0 ? b : c for (a, b, c, r) in zip(ph, pd, pl, res)]
    ll = -mean(log.(p_obs))
    rps = mean(0.5 .* ((ph .- (res .== 1)) .^ 2 .+ ((ph .+ pd) .- (res .>= 0)) .^ 2))
    (logloss = ll, rps = rps)
end
sc = DataFrame([(source = "market (Shin)", scores(ox.p_mkt_hi, ox.p_draw_mkt, ox.p_mkt_lo, ox.res_hi)...),
                (source = "tier-only GLM (walk-forward)", scores(ox.ptier_hi, ox.ptier_d, ox.ptier_lo, ox.res_hi)...),
                (source = "Dixon–Coles network (walk-forward)", scores(ox.pdc_hi, ox.pdc_d, ox.pdc_lo, ox.res_hi)...),
                (source = "50/50 market+DC", scores((ox.p_mkt_hi .+ ox.pdc_hi) ./ 2, (ox.p_draw_mkt .+ ox.pdc_d) ./ 2,
                                                   (ox.p_mkt_lo .+ ox.pdc_lo) ./ 2, ox.res_hi)...)])
insertcols!(sc, 2, :N => nrow(ox))
save_csv("r05_model_vs_market_scores.csv", sc); save_md("r05_model_vs_market_scores.md", sc; digits = 4)

enc_rows = NamedTuple[]
function enc!(lab, cols...)
    X = hcat(ones(nrow(ox)), cols...); y = Float64.(ox.gd_hi)
    b = X \ y; e = y .- X * b; B = inv(X'X); V = B * (X' * Diagonal(e .^ 2) * X) * B .* (length(y) / (length(y) - size(X, 2)))
    for (k, nm) in enumerate(("intercept", "market Δλ", lab))
        k > size(X, 2) && break
        push!(enc_rows, (model = lab, term = nm, coef = b[k], se_hc1 = sqrt(V[k, k]), p = 2ccdf(Normal(), abs(b[k] / sqrt(V[k, k])))))
    end
end
enc!("market only", ox.sup_mkt_hi)
enc!("tier GLM Δλ − market Δλ", ox.sup_mkt_hi, ox.sup_tier_hi .- ox.sup_mkt_hi)
enc!("DC Δλ − market Δλ", ox.sup_mkt_hi, ox.sup_dc_hi .- ox.sup_mkt_hi)
enc = DataFrame(enc_rows)
save_csv("r05_encompassing.csv", enc); save_md("r05_encompassing.md", enc; digits = 3)
sup_cmp = sort(combine(groupby(ox, [:hi_tier, :lo_tier]), nrow => :N, :gd_hi => mean => :realised_gd_hi,
                       :sup_mkt_hi => mean => :market_sup_hi, :sup_tier_hi => mean => :tier_glm_sup_hi,
                       :sup_dc_hi => mean => :dc_sup_hi), [:hi_tier, :lo_tier])
save_csv("r05_supremacy_market_vs_models.csv", sup_cmp); save_md("r05_supremacy_market_vs_models.md", sup_cmp; digits = 3)

# ── 6. Betfair robustness (Scottish Cup / League Cup cross-tier ties with an exchange close) ─
bfm = .!ismissing.(ox.bf_p_hi)
bf_rows = NamedTuple[]
if count(bfm) >= 10
    d = ox[bfm, :]
    bfinv = invert_poisson.(Float64.(d.bf_p_hi), Float64.(d.bf_p_lo))
    bsup = getindex.(bfinv, 1) .- getindex.(bfinv, 2)
    for (lab, s) in (("sofascore close (Shin)", d.sup_mkt_hi), ("betfair last pre-KO", bsup))
        e = tstat(d.gd_hi .- s)
        push!(bf_rows, (source = lab, N = nrow(d), mean_sup_hi = mean(s), err_mean = e.mean, err_se = e.se, p = e.p))
    end
end
bft = DataFrame(bf_rows)
save_csv("r05_betfair_robustness.csv", bft); save_md("r05_betfair_robustness.md", bft; digits = 3)

# ── 7. Relegated-club cold start: what the close believes in the first 5 league games ─
# Market-implied supremacy of a club in its first five league fixtures after relegation,
# minus the average market supremacy its opponents faced from the same venue (so the
# number is the market's view of the club vs an average club of its new tier).
lg = fx_all[.!fx_all.is_cup .& .!ismissing.(fx_all.odds_home), :]
tiers_fs = Dict{Tuple{String, Int}, Int}()
for r in eachrow(fx_all[.!fx_all.is_cup, :]); tiers_fs[(r.home_team, r.fs)] = r.home_tier; tiers_fs[(r.away_team, r.fs)] = r.away_tier; end
s3 = shin.([(Float64(a), Float64(b), Float64(c)) for (a, b, c) in zip(lg.odds_home, lg.odds_draw, lg.odds_away)])
li = invert_poisson.(getindex.(s3, 1), getindex.(s3, 3))
lg.sup = getindex.(li, 1) .- getindex.(li, 2)
lg.lh = log.(getindex.(li, 1)); lg.la = log.(getindex.(li, 2))
# league-season average home supremacy (the venue baseline)
lg.lsup = lg.lh .- lg.la
hbt = combine(groupby(lg, [:tournament_id, :fs]), :sup => mean => :m, :lsup => mean => :ml)
hb  = Dict((r.tournament_id, r.fs) => r.m for r in eachrow(hbt))
hbl = Dict((r.tournament_id, r.fs) => r.ml for r in eachrow(hbt))
cs_rows = NamedTuple[]
for (team, fs) in unique(vcat(collect(zip(lg.home_team, lg.fs)), collect(zip(lg.away_team, lg.fs))))
    prev = get(tiers_fs, (team, fs - 1), 0); cur = tiers_fs[(team, fs)]
    prev == 0 && continue
    m = findall(((lg.home_team .== team) .| (lg.away_team .== team)) .& (lg.fs .== fs))
    length(m) < 5 && continue
    m = m[sortperm(lg.match_date[m])][1:5]
    ex = [lg.home_team[i] == team ? lg.sup[i] - hb[(lg.tournament_id[i], fs)] : -(lg.sup[i] - hb[(lg.tournament_id[i], fs)]) for i in m]
    exl = [lg.home_team[i] == team ? lg.lsup[i] - hbl[(lg.tournament_id[i], fs)] : -(lg.lsup[i] - hbl[(lg.tournament_id[i], fs)]) for i in m]
    real = [lg.home_team[i] == team ? lg.home_goals[i] - lg.away_goals[i] : lg.away_goals[i] - lg.home_goals[i] for i in m]
    push!(cs_rows, (team = team, season = fs_label(fs), move = cur > prev ? "relegated" : cur < prev ? "promoted" : "stayed",
                    from = "T$prev", to = "T$cur", mkt_sup_vs_avg_first5 = mean(ex), mkt_logsup_vs_avg_first5 = mean(exl), real_gd_first5 = mean(real)))
end
cs = DataFrame(cs_rows)
save_csv("r05_first5_market_view_transitions.csv", cs[cs.move .!= "stayed", :])
css = sort(combine(groupby(cs, [:move, :to]), nrow => :n, :mkt_sup_vs_avg_first5 => mean => :mkt_sup_mean,
                   :mkt_sup_vs_avg_first5 => std => :mkt_sup_sd,
                   :mkt_logsup_vs_avg_first5 => mean => :mkt_theta_mean, :mkt_logsup_vs_avg_first5 => std => :mkt_theta_sd, :real_gd_first5 => mean => :real_gd_mean), [:move, :to])
save_csv("r05_first5_market_view_summary.csv", css); save_md("r05_first5_market_view_summary.md", css; digits = 3)

# ── 8. Figure: realised vs market-implied higher-tier win probability ───────
p = plot([0, 1], [0, 1]; color = :gray, ls = :dash, label = "perfect calibration", size = (560, 520),
         xlabel = "Market-implied P(higher tier wins), Shin", ylabel = "Realised frequency",
         title = "Cross-tier cup ties, 2021/22–2026/27: calibration of the close", titlefontsize = 10, legend = :topleft)
scatter!(p, bins.implied_hi_win, bins.realised_hi_win; yerror = 1.96 .* bins.se, ms = 6, color = colorant"#2a78d6",
         msc = colorant"#2a78d6", label = "binned ties (95% CI)")
for r in eachrow(bins); annotate!(p, r.implied_hi_win + 0.02, r.realised_hi_win - 0.04, text("n=$(r.N)", 7, :left)); end
savefig(p, joinpath(FIGS, "r05_hi_tier_calibration.png"))

CSV.write(joinpath(DATA, "r05_cross_tier_priced.csv"), ox)

# ── 9. Console summary ──────────────────────────────────────────────────────
println("Error by tier pair:"); show(e_pair[:, [:hi_tier, :lo_tier, :N, :mkt_sup_hi, :real_gd_hi, :err_mean, :err_se, :p]]; allrows = true); println()
println("All / baselines:"); show(e_all[:, [:group, :N, :err_mean, :err_se, :p]]); println(); show(base); println()
println("MZ:"); show(mz; allrows = true); println()
println("Calibration:"); show(cal; allrows = true); println()
println("ROI:"); show(roidf; allrows = true); println()
println("Scores:"); show(sc); println()
println("Encompassing:"); show(enc; allrows = true); println()
println("Supremacy compare:"); show(sup_cmp; allrows = true); println()
println("Betfair:"); show(bft); println()
println("First-5 market view:"); show(css; allrows = true); println()
