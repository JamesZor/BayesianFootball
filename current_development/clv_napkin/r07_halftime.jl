# r07_halftime.jl — half-time re-pricing: does the first half tell us more than the in-play market?
#
# QUESTION (BRIEF_07)
#   At half-time, does pre-match anchor + HT score/red cards + first-half BBC performance predict
#   the full-time result better than the Betfair in-play price during the break? Scored on the
#   outcome (FT 1X2 log-loss given HT; O/U 2.5 where the in-play book allows). Then: where, does it
#   survive the in-play spread (26/27 order book), and does model-guided closing beat holding?
#
# DESIGN
#   anchor   pre-match close (TWA (−20, 0], de-vigged) 1X2 + O/U 1.5/2.5/3.5 inverted with §14's
#            best grid (bivariate + COM, λ₃ 0.08, ν 1.08) → λ_pre = the grid's goal means.
#   HT book  median LTP of the ticks inside the break (BBC half_end → 2nd-half kick_off, else an
#            empirical fallback window), de-vigged; one missing runner imputed (flagged).
#   state    HT score (sofascore), first-half reds (sofascore incidents).
#   perf     BBC live text, minute ≤ 45: shots, on target, corners, proxy xG (src cell table fitted
#            on training matches only); gaps vs the anchor's expectation (GLM on training rows).
#   model    team-row Poisson GLM for second-half goals, offset log λ_pre; FT-draw inflation δ;
#            (1) state only, fit 21/22–23/24; (2) state + pxG/on-target gaps, fit 23/24 (BBC starts
#            23/24); (2b) + shots/corners; (3) log-pool of (2) with the HT market, weight fit on 23/24.
#   test     24/25–25/26; paired match-day-clustered bootstrap of log-loss differences.
#
# Read-only SELECTs; no sampling. USAGE (mcmc-beast):
#   julia --project -t 16 current_development/clv_napkin/r07_halftime.jl

# %%
# ===================================================================
# 1. Packages and configuration
# ===================================================================

include(joinpath(@__DIR__, "l07_halftime.jl"))
LinearAlgebra.BLAS.set_num_threads(1)

const R7_OUT = joinpath(@__DIR__, "out")
const R7_TRAIN = ["21/22", "22/23", "23/24"]
const R7_TRAIN_PERF = ["23/24"]
const R7_TEST = ["24/25", "25/26"]
const R7_LEAGUE = Dict(54 => "Prem", 55 => "Champ", 56 => "L1", 57 => "L2")
const R7_B = 2_000
const R7_SEED = 20260928
const R7_EDGE = 0.02
const R7_COMM = 0.02
println("threads: ", Threads.nthreads(), "  started ", now())
r7_write(name, df) = CSV.write(joinpath(R7_OUT, "r07_" * name * ".csv"), df)
r7_show(df) = (show(stdout, MIME"text/plain"(), df; allrows = true, allcols = true, truncate = 0); println())

# %%
# ===================================================================
# 2. Matches, HT windows, HT book, state, first-half performance, anchor
# ===================================================================

ds = deserialize(joinpath(C4_OUT, "r04_datastore_ScottishPyramid.jls"))
meta = Dict(Int(r.match_id) => r for r in eachrow(ds.matches))
kickoffs = c7_kickoffs()
windows, fb_window = c7_ht_windows(kickoffs)
@printf("HT window: %d BBC-timed; fallback window [%.1f, %.1f] min after the sofascore kick-off\n",
        count(v -> v[3] === :bbc, values(windows)), fb_window...)
bbc_w = [v for v in values(windows) if v[3] === :bbc]
@printf("BBC break: start median %.1f (p10 %.1f, p90 %.1f), end median %.1f (p10 %.1f, p90 %.1f)\n",
        median(first.(bbc_w)), quantile(first.(bbc_w), 0.1), quantile(first.(bbc_w), 0.9),
        median(getindex.(bbc_w, 2)), quantile(getindex.(bbc_w, 2), 0.1), quantile(getindex.(bbc_w, 2), 0.9))
ht_book = c7_ht_book(ds.betfair_odds, windows)
reds = c7_first_half_reds()
inc_cov = c7_incident_matches()

played = ds.matches[in.(ds.matches.tournament_id, Ref(C7_TIDS)) .& .!ismissing.(ds.matches.home_score) .&
                    .!ismissing.(ds.matches.home_score_ht), :]
fit_ids = Set(Int.(played.match_id[in.(played.season, Ref(R7_TRAIN))]))
perf, xg_model = c7_first_half_perf(ds, fit_ids)
perf_d = Dict(r.match_id => r for r in eachrow(perf))
@printf("proxy-xG cell table: %d cells, base rate %.3f, penalty %.2f (fitted on training shots)\n",
        length(xg_model.cells), xg_model.base_rate, xg_model.penalty_xg)

close_book, _ = cn_close_book(ds)
tg = c5_targets(close_book[in.(close_book.match_id, Ref(Set(Int.(played.match_id)))), :]; pcol = :p_close)
live_book, _ = c5_live_close(ds)
merge!(tg, c5_targets(live_book; pcol = :p_close))
anchor = Dict{Int,Tuple{Float64,Float64}}()
tgv = collect(values(tg))
lam = Vector{Tuple{Float64,Float64}}(undef, length(tgv))
Threads.@threads for i in eachindex(tgv)
    f = c5_fit(tgv[i], C7_ANCHOR_GRID..., C7_ANCHOR_LINES)
    lam[i] = (f.mh, f.ma)
end
for (t, l) in zip(tgv, lam)
    anchor[t.match_id] = l
end
pre_odds = Dict((Int(r.match_id), Symbol(r.selection)) => Float64(r.odds_close)
                for r in eachrow(close_book) if r.market_name == "1X2")

hx = Dict((r.match_id, r.selection) => r for r in eachrow(ht_book[ht_book.market_name .== "1X2", :]))
hou = Dict(r.match_id => r for r in eachrow(ht_book[(ht_book.market_name .== "OverUnder") .& (ht_book.selection .== :under_25), :]))
rows = NamedTuple[]
for r in eachrow(played)
    mid = Int(r.match_id)
    has_ht = haskey(hx, (mid, :home))
    pr = get(perf_d, mid, nothing)
    λ = get(anchor, mid, (NaN, NaN))
    fh, fa, hh, ha = Int(r.home_score), Int(r.away_score), Int(r.home_score_ht), Int(r.away_score_ht)
    push!(rows, (; match_id = mid, tid = Int(r.tournament_id), league = R7_LEAGUE[Int(r.tournament_id)],
                 season = String(r.season), date = Date(r.match_date), hh, ha, fh, fa,
                 res = fh > fa ? 1 : fh == fa ? 2 : 3, lam_h = λ[1], lam_a = λ[2],
                 red_h = get(reds, (mid, true), 0), red_a = get(reds, (mid, false), 0), has_inc = mid in inc_cov,
                 has_perf = pr !== nothing,
                 sh_h = pr === nothing ? NaN : pr.sh_h, sh_a = pr === nothing ? NaN : pr.sh_a,
                 sot_h = pr === nothing ? NaN : pr.sot_h, sot_a = pr === nothing ? NaN : pr.sot_a,
                 pxg_h = pr === nothing ? NaN : pr.pxg_h, pxg_a = pr === nothing ? NaN : pr.pxg_a,
                 cor_h = pr === nothing ? NaN : pr.cor_h, cor_a = pr === nothing ? NaN : pr.cor_a,
                 has_ht = has_ht,
                 mk_h = has_ht ? hx[(mid, :home)].p_ht : NaN, mk_d = has_ht ? hx[(mid, :draw)].p_ht : NaN,
                 mk_a = has_ht ? hx[(mid, :away)].p_ht : NaN,
                 o_h = has_ht ? hx[(mid, :home)].odds_ht : NaN, o_d = has_ht ? hx[(mid, :draw)].odds_ht : NaN,
                 o_a = has_ht ? hx[(mid, :away)].odds_ht : NaN,
                 ht_imputed = has_ht && any(hx[(mid, s)].imputed for s in (:home, :draw, :away)),
                 window = has_ht ? hx[(mid, :home)].window : windows[mid][3],
                 mk_under = haskey(hou, mid) ? hou[mid].p_ht : NaN,
                 pre_o_h = get(pre_odds, (mid, :home), NaN), pre_o_d = get(pre_odds, (mid, :draw), NaN),
                 pre_o_a = get(pre_odds, (mid, :away), NaN)))
end
mf = DataFrame(rows)
mf.state = [abs(r.hh - r.ha) == 0 ? "level" : abs(r.hh - r.ha) == 1 ? "one-goal lead" : "two+ goal lead" for r in eachrow(mf)]

# sanity: when a side leads by 2+ at HT, is it the HT market favourite?
lead2 = mf[mf.has_ht .& (abs.(mf.hh .- mf.ha) .>= 2), :]
agree = mean([(r.hh > r.ha) == (r.mk_h > r.mk_a) for r in eachrow(lead2)])
@printf("HT-book sanity: the 2+-goal leader is the HT market favourite in %.1f%% of %d matches\n", 100agree, nrow(lead2))

# coverage
stat_cov = BayesianFootball.MatchDay._query("""
    SELECT m.match_id, bool_or(s.period = '1ST') AS per1
    FROM sofascore.match_statistics s JOIN sofascore.matches m USING (match_id)
    WHERE m.tournament_id = ANY(\$1) GROUP BY 1""", (collect(C7_TIDS),))
per1 = Set(Int.(stat_cov.match_id[coalesce.(stat_cov.per1, false)]))
shotxg = Set(Int.(BayesianFootball.MatchDay._query("""
    SELECT DISTINCT s.match_id FROM sofascore.match_shotmap s JOIN sofascore.matches m USING (match_id)
    WHERE m.tournament_id = ANY(\$1) AND s.xg IS NOT NULL""", (collect(C7_TIDS),)).match_id))
mf.sofa_1st = in.(mf.match_id, Ref(per1)); mf.sofa_xg = in.(mf.match_id, Ref(shotxg))
coverage = combine(groupby(mf, [:league, :season]), nrow => :played,
                   :lam_h => (x -> count(isfinite, x)) => :anchor, :has_ht => sum => :ht_1x2,
                   :ht_imputed => sum => :ht_1x2_imputed, :mk_under => (x -> count(isfinite, x)) => :ht_ou25,
                   :window => (x -> count(==(:bbc), x)) => :bbc_timed_window, :has_perf => sum => :bbc_first_half,
                   :has_inc => sum => :sofa_incidents, :sofa_1st => sum => :sofa_period_stats, :sofa_xg => sum => :sofa_shot_xg)
sort!(coverage, [:league, :season])
r7_show(coverage)
r7_write("coverage", coverage)

# %%
# ===================================================================
# 3. First-half expectations from the anchor, and the team-row design
# ===================================================================
#
# E[stat_1H for a team] = exp(a + b·log λ_for + c·log λ_against + d·home), a quasi-Poisson GLM on
# training rows with BBC coverage (23/24). Gaps: pxG raw (obs − E); counts Pearson (obs − E)/√E.

ok_anchor(r) = isfinite(r.lam_h) && isfinite(r.lam_a)
"Two team rows per match (home first) with the columns every model needs."
function r7_team_rows(mf)
    rows = NamedTuple[]
    for r in eachrow(mf)
        ok_anchor(r) || continue
        for home in (true, false)
            gf, ga = home ? (r.hh, r.ha) : (r.ha, r.hh)
            push!(rows, (; match_id = r.match_id, season = r.season, date = r.date, home = Float64(home),
                         lam_for = home ? r.lam_h : r.lam_a, lam_against = home ? r.lam_a : r.lam_h,
                         g2 = Float64(home ? r.fh - r.hh : r.fa - r.ha), d = gf - ga, tot = r.hh + r.ha,
                         red_for = Float64(home ? r.red_h : r.red_a), red_against = Float64(home ? r.red_a : r.red_h),
                         has_perf = r.has_perf,
                         pxg_for = home ? r.pxg_h : r.pxg_a, pxg_against = home ? r.pxg_a : r.pxg_h,
                         sot_for = home ? r.sot_h : r.sot_a, sot_against = home ? r.sot_a : r.sot_h,
                         sh_for = home ? r.sh_h : r.sh_a, sh_against = home ? r.sh_a : r.sh_h,
                         cor_for = home ? r.cor_h : r.cor_a, cor_against = home ? r.cor_a : r.cor_h))
        end
    end
    return DataFrame(rows)
end
tr = r7_team_rows(mf)
expX(d) = hcat(ones(nrow(d)), log.(d.lam_for), log.(d.lam_against), d.home)
exp_fit = Dict{Symbol,Vector{Float64}}()
etrain = tr[tr.has_perf .& in.(tr.season, Ref(R7_TRAIN_PERF)), :]
for s in (:pxg, :sot, :sh, :cor)
    β, _ = c7_poisson(expX(etrain), Float64.(etrain[!, Symbol(s, "_for")]), zeros(nrow(etrain)))
    exp_fit[s] = β
end
for s in (:pxg, :sot, :sh, :cor), side in ("for", "against")
    # the expectation of the OTHER team's stat swaps λ_for/λ_against and home
    X = side == "for" ? expX(tr) : hcat(ones(nrow(tr)), log.(tr.lam_against), log.(tr.lam_for), 1 .- tr.home)
    E = exp.(X * exp_fit[s])
    obs = Float64.(tr[!, Symbol(s, "_", side)])
    tr[!, Symbol(s, "_e_", side)] = E
    tr[!, Symbol(s, "_gap_", side)] = s === :pxg ? obs .- E : (obs .- E) ./ sqrt.(E)
end
exp_tab = DataFrame([(; stat = String(s), intercept = b[1], log_lam_for = b[2], log_lam_against = b[3], home = b[4],
                      mean_obs = mean(etrain[!, Symbol(s, "_for")]), n_rows = nrow(etrain)) for (s, b) in exp_fit])
r7_show(exp_tab)
r7_write("first_half_expectation", exp_tab)

const R7_STATE = [("home", d -> d.home), ("lead 1", d -> Float64.(d.d .== 1)), ("lead 2+", d -> Float64.(d.d .>= 2)),
                  ("trail 1", d -> Float64.(d.d .== -1)), ("trail 2+", d -> Float64.(d.d .<= -2)),
                  ("red for", d -> min.(d.red_for, 1.0)), ("red against", d -> min.(d.red_against, 1.0)),
                  ("HT goals (cap 4)", d -> Float64.(min.(d.tot, 4)))]
const R7_PERF = [("pxG gap for", d -> d.pxg_gap_for), ("pxG gap against", d -> d.pxg_gap_against),
                 ("on-target z for", d -> d.sot_gap_for), ("on-target z against", d -> d.sot_gap_against)]
const R7_PERF_B = [("shots z for", d -> d.sh_gap_for), ("shots z against", d -> d.sh_gap_against),
                   ("corners z for", d -> d.cor_gap_for), ("corners z against", d -> d.cor_gap_against)]
design(d, feats) = hcat(ones(nrow(d)), [f[2](d) for f in feats]...)

# %%
# ===================================================================
# 4. Fit (1), (2), (2b); FT-draw inflation δ; blend weight
# ===================================================================

"Home/away second-half means for every match of `mf` under a fitted model (NaN if not applicable)."
function r7_means(mf, tr, β, feats)
    μ = exp.(log.(tr.lam_for) .+ design(tr, feats) * β)
    d = Dict{Tuple{Int,Bool},Float64}()
    for (i, r) in enumerate(eachrow(tr))
        d[(r.match_id, r.home == 1.0)] = μ[i]
    end
    return [(get(d, (m, true), NaN), get(d, (m, false), NaN)) for m in mf.match_id]
end

"FT probability matrix (n × 3) and under-2.5 vector for `mf` rows given means and δ."
function r7_probs(mf, means, δ)
    P = fill(NaN, nrow(mf), 3); U = fill(NaN, nrow(mf))
    for (i, r) in enumerate(eachrow(mf))
        μh, μa = means[i]
        isfinite(μh) && isfinite(μa) || continue
        p = c7_ft_probs(r.hh, r.ha, μh, μa; δ)
        P[i, :] .= p[1:3]; U[i] = p[4]
    end
    return P, U
end

models = Dict{String,Any}()
coef_rows = NamedTuple[]
for (name, feats, seasons, need_perf) in (("(1) state only", R7_STATE, R7_TRAIN, false),
                                         ("(2) state + pxG/on-target", vcat(R7_STATE, R7_PERF), R7_TRAIN_PERF, true),
                                         ("(2b) + shots/corners", vcat(R7_STATE, R7_PERF, R7_PERF_B), R7_TRAIN_PERF, true),
                                         ("(1′) state only, 23/24 fit", R7_STATE, R7_TRAIN_PERF, true))
    fit_rows = tr[in.(tr.season, Ref(seasons)) .& (need_perf ? tr.has_perf : trues(nrow(tr))), :]
    β, se = c7_poisson(design(fit_rows, feats), fit_rows.g2, log.(fit_rows.lam_for))
    # δ by maximum likelihood of the FT result on the same training matches
    mtrain = mf[in.(mf.match_id, Ref(Set(fit_rows.match_id))), :]
    trm = tr[in.(tr.match_id, Ref(Set(mtrain.match_id))), :]
    means = r7_means(mtrain, trm, β, feats)
    nll(δ) = (P = first(r7_probs(mtrain, means, δ)); mean(c7_ll(P, mtrain.res)))
    δ = Optim.minimizer(optimize(nll, -0.3, 1.0))
    models[name] = (; feats, β, se, δ, need_perf, n_train = nrow(mtrain))
    for (j, lab) in enumerate(vcat(["log s₂ (intercept)"], [f[1] for f in feats]))
        push!(coef_rows, (; model = name, term = lab, estimate = β[j], se = se[j], z = β[j] / se[j], n_matches = nrow(mtrain)))
    end
    push!(coef_rows, (; model = name, term = "FT-draw inflation δ", estimate = δ, se = NaN, z = NaN, n_matches = nrow(mtrain)))
end
coefs = DataFrame(coef_rows)
r7_show(coefs)
r7_write("coefficients", coefs)

for (name, m) in models
    tm = tr[.!m.need_perf .| tr.has_perf, :]
    P, U = r7_probs(mf, r7_means(mf, tm, m.β, m.feats), m.δ)
    mf[!, Symbol("P_", name)] = [P[i, :] for i in 1:nrow(mf)]
    mf[!, Symbol("U_", name)] = U
end
Pm(col) = reduce(vcat, [permutedims(v) for v in mf[!, col]])
MK = hcat(mf.mk_h, mf.mk_d, mf.mk_a)
wtrain = mf.has_ht .& mf.has_perf .& in.(mf.season, Ref(R7_TRAIN_PERF)) .& isfinite.(Pm(Symbol("P_(2) state + pxG/on-target"))[:, 1])
P2 = Pm(Symbol("P_(2) state + pxG/on-target"))
pool_nll(w) = mean(c7_ll(c7_pool(P2[wtrain, :], MK[wtrain, :], w), mf.res[wtrain]))
w_blend = Optim.minimizer(optimize(pool_nll, 0.0, 1.0))
@printf("blend weight on model (2): %.3f (fit on %d 23/24 matches)\n", w_blend, count(wtrain))
PB = fill(NaN, nrow(mf), 3)
okb = isfinite.(P2[:, 1]) .& mf.has_ht
PB[okb, :] .= c7_pool(P2[okb, :], MK[okb, :], w_blend)
mf[!, Symbol("P_(3) blend (2) × market")] = [PB[i, :] for i in 1:nrow(mf)]
push!(coef_rows, (; model = "(3) blend (2) × market", term = "log-pool weight on (2)", estimate = w_blend, se = NaN, z = NaN,
                  n_matches = count(wtrain)))
r7_write("coefficients", DataFrame(coef_rows))

# %%
# ===================================================================
# 5. Test: FT 1X2 log-loss given HT vs the HT market (24/25–25/26)
# ===================================================================

const R7_MODELS = ["(1) state only", "(1′) state only, 23/24 fit", "(2) state + pxG/on-target",
                   "(2b) + shots/corners", "(3) blend (2) × market"]
test = mf.has_ht .& in.(mf.season, Ref(R7_TEST)) .& isfinite.(mf.lam_h)
common = test .& mf.has_perf .& isfinite.(P2[:, 1])
ll_mk = c7_ll(MK, mf.res)
ll_m = Dict(name => c7_ll(Pm(Symbol("P_", name)), mf.res) for name in R7_MODELS)

"Log-loss rows (market and each model, Δ = model − market, paired clustered CI) for a mask."
function r7_ll_rows(mask, label, sample)
    out = NamedTuple[]
    any(mask) || return out
    push!(out, (; sample, group = label, model = "(0) HT market", n = count(mask), logloss = mean(ll_mk[mask]),
                d_vs_market = 0.0, lo = 0.0, hi = 0.0, se = 0.0))
    for name in R7_MODELS
        ok = mask .& isfinite.(ll_m[name])
        count(ok) < 20 && continue
        p = c7_paired(ll_m[name][ok], ll_mk[ok], mf.date[ok]; B = R7_B, seed = R7_SEED)
        push!(out, (; sample, group = label, model = name, n = count(ok), logloss = mean(ll_m[name][ok]),
                    d_vs_market = p.mean, lo = p.lo, hi = p.hi, se = p.se))
    end
    return out
end
llrows = NamedTuple[]
for (sample, base) in (("common (BBC + HT book)", common), ("all with HT book", test),
                       ("common, no imputed runner", common .& .!mf.ht_imputed))
    append!(llrows, r7_ll_rows(base, "pooled", sample))
    for tid in C7_TIDS
        append!(llrows, r7_ll_rows(base .& (mf.tid .== tid), R7_LEAGUE[tid], sample))
    end
end
lltab = DataFrame(llrows)
r7_show(lltab)
r7_write("logloss", lltab)

# game state and first-half dominance without a lead (common sample)
trd = Dict((r.match_id, r.home == 1.0) => r for r in eachrow(tr))
mf.dom = [haskey(trd, (m, true)) && trd[(m, true)].has_perf ?
          (trd[(m, true)].pxg_gap_for - trd[(m, true)].pxg_gap_against) : NaN for m in mf.match_id]
dom_no_lead = [isfinite(r.dom) && abs(r.dom) >= 0.5 && sign(r.dom) * (r.hh - r.ha) <= 0 for r in eachrow(mf)]
strows = NamedTuple[]
for st in ("level", "one-goal lead", "two+ goal lead")
    append!(strows, r7_ll_rows(common .& (mf.state .== st), st, "common"))
end
append!(strows, r7_ll_rows(common .& dom_no_lead, "pxG-dominant side not leading (|gap| ≥ 0.5)", "common"))
append!(strows, r7_ll_rows(common .& .!dom_no_lead, "all other", "common"))
sttab = DataFrame(strows)
r7_show(sttab)
r7_write("logloss_state", sttab)

# O/U 2.5 FT given HT (binary log-loss) where the HT O/U book exists
ou = common .& isfinite.(mf.mk_under)
under_won = (mf.fh .+ mf.fa) .<= 2
bll(p, y) = [-log(max(y[i] ? p[i] : 1 - p[i], 1e-12)) for i in eachindex(y)]
ll_ou_mk = bll(mf.mk_under, under_won)
ourows = NamedTuple[]
push!(ourows, (; model = "(0) HT market", n = count(ou), logloss = mean(ll_ou_mk[ou]), d_vs_market = 0.0, lo = 0.0, hi = 0.0))
for name in R7_MODELS[1:4]
    l = bll(mf[!, Symbol("U_", name)], under_won)
    ok = ou .& isfinite.(l)
    p = c7_paired(l[ok], ll_ou_mk[ok], mf.date[ok]; B = R7_B, seed = R7_SEED)
    push!(ourows, (; model = name, n = count(ok), logloss = mean(l[ok]), d_vs_market = p.mean, lo = p.lo, hi = p.hi))
end
outab = DataFrame(ourows)
r7_show(outab)
r7_write("logloss_ou25", outab)

# power: minimum detectable Δ (80% power, two-sided 5%) = 2.8 × SE of the paired difference
pw = DataFrame([(; sample = r.sample, group = r.group, model = r.model, n = r.n, se = r.se, mde = 2.8 * r.se)
                for r in eachrow(lltab) if r.model != "(0) HT market" && r.group == "pooled"])
r7_show(pw)
r7_write("power", pw)

# %%
# ===================================================================
# 6. (4) Market-anchored: does first-half performance add to the HT MARKET's own rates?
# ===================================================================
#
# Invert the HT market's 1X2 (exactly identified; adding under 2.5 over-constrains the
# two-parameter Poisson and the β = 0 model would no longer reproduce the market) into second-half
# means given the HT score, then fit g2 ~ Poisson(μ_market · exp(β'perf)) on 23/24 and score on the test seasons.
# β = 0 reproduces the market; a gain here is information the market did not price.

mkt_mu = Dict{Int,Tuple{Float64,Float64}}()
hti = findall(mf.has_ht)
ht_inv = Vector{Tuple{Float64,Float64}}(undef, length(hti))
Threads.@threads for k in eachindex(hti)
    r = mf[hti[k], :]
    ht_inv[k] = c7_invert_ht(r.hh, r.ha, (r.mk_h, r.mk_d, r.mk_a), NaN)      # 1X2 only: exactly identified
end
for (k, i) in enumerate(hti)
    mkt_mu[mf.match_id[i]] = ht_inv[k]
end
tr.mu_mkt = [haskey(mkt_mu, r.match_id) ? mkt_mu[r.match_id][r.home == 1.0 ? 1 : 2] : NaN for r in eachrow(tr)]
R7_PERF4 = vcat([("home", d -> d.home)], R7_PERF)
f4 = tr[in.(tr.season, Ref(R7_TRAIN_PERF)) .& tr.has_perf .& isfinite.(tr.mu_mkt), :]
β4, se4 = c7_poisson(design(f4, R7_PERF4), f4.g2, log.(f4.mu_mkt))
m4 = mf[in.(mf.match_id, Ref(Set(f4.match_id))), :]
t4m = tr[in.(tr.match_id, Ref(Set(m4.match_id))), :]
means4(mfx, trx) = (μ = exp.(log.(trx.mu_mkt) .+ design(trx, R7_PERF4) * β4);
                    d = Dict((r.match_id, r.home == 1.0) => μ[i] for (i, r) in enumerate(eachrow(trx)));
                    [(get(d, (m, true), NaN), get(d, (m, false), NaN)) for m in mfx.match_id])
nll4(δ) = mean(c7_ll(first(r7_probs(m4, means4(m4, t4m), δ)), m4.res))
δ4 = Optim.minimizer(optimize(nll4, -0.3, 1.0))
for (j, lab) in enumerate(vcat(["log scale (intercept)"], [f[1] for f in R7_PERF4]))
    push!(coef_rows, (; model = "(4) HT market + pxG/on-target", term = lab, estimate = β4[j], se = se4[j],
                      z = β4[j] / se4[j], n_matches = nrow(m4)))
end
push!(coef_rows, (; model = "(4) HT market + pxG/on-target", term = "FT-draw inflation δ", estimate = δ4, se = NaN, z = NaN, n_matches = nrow(m4)))
r7_write("coefficients", DataFrame(coef_rows))
r7_show(DataFrame(coef_rows)[startswith.(DataFrame(coef_rows).model, "(4)"), :])
t4 = tr[tr.has_perf .& isfinite.(tr.mu_mkt), :]
P4, U4 = r7_probs(mf, means4(mf, t4), δ4)
# the pure inversion (β = 0, δ = 0) must reproduce the market: report its log-loss gap as a check
P4_0, _ = r7_probs(mf, [get(mkt_mu, m, (NaN, NaN)) for m in mf.match_id], 0.0)
mf[!, Symbol("P_(4) HT market + pxG/on-target")] = [P4[i, :] for i in 1:nrow(mf)]
mf[!, Symbol("U_(4) HT market + pxG/on-target")] = U4
ll_m["(4) HT market + pxG/on-target"] = c7_ll(P4, mf.res)
ll_inv0 = c7_ll(P4_0, mf.res)
@printf("inversion check (β = 0): mean |LL(inverted) − LL(market)| = %.2e on the test sample\n",
        mean(abs.(ll_inv0[common] .- ll_mk[common])))
push!(R7_MODELS, "(4) HT market + pxG/on-target")
llrows4 = NamedTuple[]
for (sample, base) in (("common (BBC + HT book)", common), ("common, no imputed runner", common .& .!mf.ht_imputed))
    append!(llrows4, r7_ll_rows(base, "pooled", sample)[end:end])
    for tid in C7_TIDS
        rr = r7_ll_rows(base .& (mf.tid .== tid), R7_LEAGUE[tid], sample)
        isempty(rr) || push!(llrows4, rr[end])
    end
end
for st in ("level", "one-goal lead", "two+ goal lead")
    rr = r7_ll_rows(common .& (mf.state .== st), st, "common"); isempty(rr) || push!(llrows4, rr[end])
end
rr = r7_ll_rows(common .& dom_no_lead, "pxG-dominant side not leading (|gap| ≥ 0.5)", "common"); isempty(rr) || push!(llrows4, rr[end])
ll4tab = DataFrame(llrows4)
r7_show(ll4tab)
lltab = vcat(lltab, ll4tab[.!in.(ll4tab.group, Ref(("level", "one-goal lead", "two+ goal lead", "pxG-dominant side not leading (|gap| ≥ 0.5)"))), :])
sttab = vcat(sttab, ll4tab[in.(ll4tab.group, Ref(("level", "one-goal lead", "two+ goal lead", "pxG-dominant side not leading (|gap| ≥ 0.5)"))), :])
r7_write("logloss", lltab); r7_write("logloss_state", sttab)
l4 = bll(U4, under_won); ok4 = ou .& isfinite.(l4)
p = c7_paired(l4[ok4], ll_ou_mk[ok4], mf.date[ok4]; B = R7_B, seed = R7_SEED)
push!(ourows, (; model = "(4) HT market + pxG/on-target", n = count(ok4), logloss = mean(l4[ok4]), d_vs_market = p.mean, lo = p.lo, hi = p.hi))
r7_write("logloss_ou25", DataFrame(ourows))
pw = DataFrame([(; sample = r.sample, group = r.group, model = r.model, n = r.n, se = r.se, mde = 2.8 * r.se)
                for r in eachrow(lltab) if r.model != "(0) HT market" && r.group == "pooled"])
r7_write("power", pw)

# %%
# ===================================================================
# 7. Tradability: the 2 pp rule against the HT LTP (test seasons), and the 26/27 order book
# ===================================================================

const R7_SELS = (:home, :draw, :away)
"Flat-stake bets where |p_model − p_HT| > R7_EDGE: back at the HT price above, lay at it below."
function r7_rule(mask, P, label)
    rows = NamedTuple[]
    for i in findall(mask .& isfinite.(P[:, 1]))
        r = mf[i, :]
        for (k, s) in enumerate(R7_SELS)
            pm, pk, o = P[i, k], (r.mk_h, r.mk_d, r.mk_a)[k], (r.o_h, r.o_d, r.o_a)[k]
            won = r.res == k
            if pm - pk > R7_EDGE
                push!(rows, (; model = label, side = "back", match_id = r.match_id, date = r.date, state = r.state,
                             odds = o, pnl = won ? (o - 1) * (1 - R7_COMM) : -1.0, risk = 1.0, edge = pm - pk))
            elseif pk - pm > R7_EDGE
                push!(rows, (; model = label, side = "lay", match_id = r.match_id, date = r.date, state = r.state,
                             odds = o, pnl = won ? -(o - 1) : 1.0 * (1 - R7_COMM), risk = o - 1, edge = pk - pm))
            end
        end
    end
    return DataFrame(rows)
end
bets = vcat([r7_rule(common, Pm(Symbol("P_", name)), name) for name in
             ("(1) state only", "(2) state + pxG/on-target", "(3) blend (2) × market", "(4) HT market + pxG/on-target")]...)
trade = NamedTuple[]
for g in groupby(bets, [:model, :side])
    m = c5_mean_ci(g.pnl, g.date; B = R7_B, seed = R7_SEED)
    push!(trade, (; model = first(g.model), side = first(g.side), n_bets = nrow(g), mean_edge_pp = 100mean(g.edge),
                  pnl_per_bet = m.mean, lo = m.lo, hi = m.hi, roi_on_risk = sum(g.pnl) / sum(g.risk), median_odds = median(g.odds)))
end
for g in groupby(bets, :model)
    m = c5_mean_ci(g.pnl, g.date; B = R7_B, seed = R7_SEED)
    push!(trade, (; model = first(g.model), side = "all", n_bets = nrow(g), mean_edge_pp = 100mean(g.edge),
                  pnl_per_bet = m.mean, lo = m.lo, hi = m.hi, roi_on_risk = sum(g.pnl) / sum(g.risk), median_odds = median(g.odds)))
end
trade = sort(DataFrame(trade), [:model, :side])
r7_show(trade)
r7_write("trade_rule", trade)
# where the rule's bets sit (by HT state and side), to read any single profitable cell
bets.sel = [String(R7_SELS[argmin(abs.([r.odds] .- [mf.o_h[i], mf.o_d[i], mf.o_a[i]]))])
            for (r, i) in zip(eachrow(bets), [findfirst(==(m), mf.match_id) for m in bets.match_id])]
trade_cells = combine(groupby(bets, [:model, :side, :state]), nrow => :n_bets, :pnl => mean => :pnl_per_bet,
                      [:pnl, :risk] => ((a, b) -> sum(a) / sum(b)) => :roi_on_risk)
r7_show(trade_cells)
r7_write("trade_rule_cells", trade_cells)

# 26/27: the order book during the break
live = c7_live_ht_book(ds, windows, kickoffs)
r7_med(x) = (v = filter(isfinite, x); isempty(v) ? NaN : median(v))
lagg = combine(groupby(live, [:match_id, :selection]), nrow => :snaps, :window => first => :window,
               :back => r7_med => :back,
               :lay => r7_med => :lay,
               :back_size => r7_med => :back_size,
               :lay_size => r7_med => :lay_size)
lagg.spread_ticks = c4_spread_ticks.(lagg.back, lagg.lay)
lagg.spread_pp = 100 .* (1 ./ lagg.back .- 1 ./ lagg.lay)
lagg.mid = (lagg.back .+ lagg.lay) ./ 2
complete = Set(g.match_id[1] for g in groupby(lagg, :match_id) if nrow(g) == 3 && all(isfinite, g.mid))
filter!(r -> r.match_id in complete, lagg)
transform!(groupby(lagg, :match_id), :mid => (x -> (1 ./ x) ./ sum(1 ./ x)) => :p_mid,
           :mid => (x -> sum(1 ./ x)) => :mid_overround)
lagg.role = [r.selection == :draw ? "draw" : "" for r in eachrow(lagg)]
for g in groupby(lagg, :match_id)
    hs = [i for i in 1:nrow(g) if g.selection[i] != :draw]
    fav = hs[argmax(g.p_mid[hs])]
    for i in hs
        g.role[i] = i == fav ? "favourite" : "outsider"
    end
end
book_tab = combine(groupby(lagg, :role), nrow => :n, :spread_ticks => r7_med => :median_spread_ticks,
                   :spread_pp => r7_med => :median_spread_pp, :back_size => r7_med => :median_back_size,
                   :lay_size => r7_med => :median_lay_size, :back_size => (x -> mean(x .>= 20)) => :share_back_ge20,
                   :back_size => (x -> mean(x .>= 50)) => :share_back_ge50, :snaps => r7_med => :median_snapshots)
println("26/27 HT order book: ", length(complete), " matches with a complete two-sided 1X2 book in the break (",
        count(g -> first(g.window) === :bbc, groupby(lagg, :match_id)), " BBC-timed)")
r7_show(book_tab)
r7_write("orderbook_ht_2627", book_tab)

# model edges on 26/27 at the executable price
lb = Dict((r.match_id, r.selection) => r for r in eachrow(lagg))
ex_rows = NamedTuple[]
for name in ("(1) state only", "(2) state + pxG/on-target")
    P = Pm(Symbol("P_", name))
    for i in findall((mf.season .== "26/27") .& isfinite.(P[:, 1]))
        r = mf[i, :]
        haskey(lb, (r.match_id, :home)) || continue
        for (k, s) in enumerate(R7_SELS)
            b = lb[(r.match_id, s)]
            e_mid = P[i, k] - b.p_mid
            abs(e_mid) > R7_EDGE || continue
            back = e_mid > 0
            price = back ? b.back : b.lay
            won = r.res == k
            push!(ex_rows, (; model = name, match_id = r.match_id, selection = s, side = back ? "back" : "lay",
                            edge_vs_mid_pp = 100abs(e_mid),
                            edge_vs_exec_pp = 100 * (back ? P[i, k] - 1 / b.back : 1 / b.lay - P[i, k]),
                            size = back ? b.back_size : b.lay_size, price,
                            pnl = back ? (won ? (price - 1) * (1 - R7_COMM) : -1.0) : (won ? -(price - 1) : 1 - R7_COMM)))
        end
    end
end
ex = DataFrame(ex_rows)
ex_tab = isempty(ex) ? DataFrame() :
    combine(groupby(ex, [:model, :side]), nrow => :n_bets, :edge_vs_mid_pp => mean => :mean_edge_vs_mid_pp,
            :edge_vs_exec_pp => mean => :mean_edge_vs_exec_pp, :edge_vs_exec_pp => (x -> mean(x .> 0)) => :share_edge_survives,
            :size => r7_med => :median_size, :pnl => sum => :realised_pnl)
r7_show(ex_tab)
r7_write("orderbook_edges_2627", ex_tab)
r7_write("orderbook_bets_2627", ex)

# %%
# ===================================================================
# 8. Hedging view: hold / close / add a pre-match back at HT (test seasons, HT LTP as the price)
# ===================================================================
#
# Position: 1 unit backed pre-match on the home side and, separately, on the away side, at the
# close TWA odds o_pre. At HT, with price o_ht (median LTP, used as both back and lay price):
#   hold   P&L = won ? (o_pre − 1)(1 − c) : −1
#   close  lay o_pre/o_ht at o_ht: P&L = o_pre/o_ht − 1 whatever happens (commission on a gain)
#   add    back 1 more at o_ht
# Rules: A "close if p_model < 1/o_ht, else hold"; B = A plus "add if p_model > 1/o_ht + 2 pp".

hedge_rows = NamedTuple[]
for name in ("(1) state only", "(2) state + pxG/on-target", "(4) HT market + pxG/on-target")
    P = Pm(Symbol("P_", name))
    for i in findall(common .& isfinite.(P[:, 1]))
        r = mf[i, :]
        for (k, side) in ((1, "home"), (3, "away"))
            o_pre = (r.pre_o_h, r.pre_o_d, r.pre_o_a)[k]; o_ht = (r.o_h, r.o_d, r.o_a)[k]
            (isfinite(o_pre) && isfinite(o_ht)) || continue
            won = r.res == k
            hold = won ? (o_pre - 1) * (1 - R7_COMM) : -1.0
            g = o_pre / o_ht - 1
            closep = g > 0 ? g * (1 - R7_COMM) : g
            addp = won ? (o_ht - 1) * (1 - R7_COMM) : -1.0
            pm = P[i, k]
            ruleA = pm < 1 / o_ht ? closep : hold
            ruleB = pm < 1 / o_ht ? closep : (pm > 1 / o_ht + R7_EDGE ? hold + addp : hold)
            push!(hedge_rows, (; model = name, match_id = r.match_id, date = r.date, side, state = r.state,
                               hold, always_close = closep, ruleA, ruleB, closed = pm < 1 / o_ht,
                               added = pm > 1 / o_ht + R7_EDGE))
        end
    end
end
hedge = DataFrame(hedge_rows)
hedge_tab = NamedTuple[]
for g in groupby(hedge, :model)
    for (lab, col) in (("always close", :always_close), ("A: close if model < lay price", :ruleA),
                       ("B: A + add if model > price + 2 pp", :ruleB))
        p = c7_paired(g[!, col], g.hold, g.date; B = R7_B, seed = R7_SEED)
        push!(hedge_tab, (; model = first(g.model), policy = lab, n_positions = nrow(g), share_closed = mean(g.closed),
                          share_added = startswith(lab, "B") ? mean(g.added) : 0.0, mean_hold = mean(g.hold),
                          mean_policy = mean(g[!, col]), d_vs_hold = p.mean, lo = p.lo, hi = p.hi))
    end
end
hedge_tab = DataFrame(hedge_tab)
r7_show(hedge_tab)
r7_write("hedge", hedge_tab)
println("DONE r07_halftime ", now())
