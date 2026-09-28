# r03_price_movement.jl — brief 03 Part B: is Betfair price movement a signal?
#
# QUESTION
#   A market-only placebo predicts the entry→close move 3–5× better than our models (REPORT §1).
#   Is the price path itself exploitable, alone or combined with a model?
#
# HONESTY CONTRACT (BRIEF_03)
#   discovery    seasons 21/22–23/24. Everything in sections 3–7 runs on discovery only and is
#                explored freely.
#   confirmation seasons 24/25–25/26, used ONCE, only for the hypotheses frozen in REPORT §12.3
#                before the run (section 8, C3_PHASE=confirmation). Holm across them.
#   tradability  the archive is LTP only (no ladder, no volume). A reversion/bounce effect that
#                does not survive the smoothed and drop-last-tick variants of B2 is marked "not
#                tradeable without ladder data".
#
# DATA: 1X2 (and O/U 2.5 where cheap) for tournaments 54–57, from the cached ScottishPyramid
#   DataStore's betfair_odds (LTP ticks). Price-path panel from l03: at each market's earliest
#   complete instant and at T−48h … T−10m and the off: last traded odds, the tick before it, a
#   45-min TWA, tick counts. Close = l01 TWA (−20, 0], de-vigged.
#
# PERSISTENCE: reads out/probs.jls (r01; B4 model prices) and the DataStore. Writes
#   out/r03b_<phase>_*.csv only.
#
# USAGE (mcmc-beast): C3_PHASE=discovery julia --project -t 8 current_development/clv_napkin/r03_price_movement.jl

# %%
# ===================================================================
# 1. Packages, implementation, configuration
# ===================================================================

include(joinpath(@__DIR__, "l03_price_movement.jl"))

const C3B_PHASE   = get(ENV, "C3_PHASE", "discovery")
C3B_PHASE in ("discovery", "confirmation") || error("C3_PHASE must be discovery or confirmation")
const C3B_SEASONS = C3B_PHASE == "discovery" ? C3_DISCOVERY : C3_CONFIRMATION
const C3B_LEAGUES = [("54", [54]), ("55", [55]), ("56", [56]), ("57", [57]), ("all", [54, 55, 56, 57])]
const C3B_B       = 1_000
const C3B_SEED    = 20260928
const C3B_OUT     = joinpath(@__DIR__, "out")
const C3B_W2      = "23/24,24/25,25/26"
c3b_csv(name, df) = CSV.write(joinpath(C3B_OUT, "r03b_$(C3B_PHASE == "discovery" ? "disc" : "conf")_$name.csv"), df)
println("PHASE = ", C3B_PHASE, "  seasons ", sort(collect(C3B_SEASONS)))

# %%
# ===================================================================
# 2. Data snapshot and the price-path panel (54–57, 1X2 + O/U 2.5)
# ===================================================================

ds = Data.load_datastore_cached(Data.ScottishPyramid(); max_age_hours = 10_000)
close_book, _ = cn_close_book(ds)
ids_5457 = Int.(ds.matches.match_id[in.(ds.matches.tournament_id, Ref(Set([54, 55, 56, 57])))])
t_panel = @elapsed P_all = c3_path_panel(ds, close_book; match_ids = ids_5457)
# the ONLY season filter: nothing below sees the other phase
P = c3_subset(P_all; seasons = C3B_SEASONS)
P_all = nothing; GC.gc()
@printf("panel: %d selections, %d markets (%.0fs)\n", nrow(P.keys), length(P.ranges), t_panel)
P1 = c3_subset(P; family = "1X2")
P2 = c3_subset(P; family = "OU2.5")
league(Q, tids) = c3_subset(Q; tournaments = tids)

"p at an instant, with the opening (earliest) price carried back before the market formed."
function c3b_path_p(Q, v, inst)
    p = c3_p(Q, v, inst)
    pe = c3_p(Q, :ltp, "earliest")
    inst == "earliest" && return pe
    pre = .!c3_formed(Q, inst)
    p[pre] .= pe[pre]
    return p
end

# %%
# ===================================================================
# 3. B1 — describe the price path
# ===================================================================

const C3B_PATH = ["T−48h", "T−24h", "T−12h", "T−6h", "T−3h", "T−2h", "T−90m", "T−60m", "T−45m", "T−25m", "T−10m", "off"]
const C3B_SHARE_AT = ["T−48h", "T−24h", "T−12h", "T−6h", "T−3h", "T−60m", "T−25m", "T−10m"]

"Share of the earliest→close squared move resolved by each instant: 1 − Σ(close − p_t)² / Σ(close − p_e)²."
function c3b_share(Q, v, insts)
    pe = c3_p(Q, :ltp, "earliest"); pc = Q.keys.p_close
    out = Float64[]
    for inst in insts
        pt = c3b_path_p(Q, v, inst)
        ok = isfinite.(pe) .& isfinite.(pc) .& isfinite.(pt)
        push!(out, 1 - sum((pc[ok] .- pt[ok]) .^ 2) / sum((pc[ok] .- pe[ok]) .^ 2))
    end
    return out
end

b1_path = DataFrame(); b1_share = DataFrame(); b1_ticks = DataFrame(); b1_class = DataFrame()
for (fam, Qf) in (("1X2", P1), ("OU2.5", P2)), (lg, tids) in C3B_LEAGUES
    Q = league(Qf, tids)
    isempty(Q.ranges) && continue
    nm = length(Q.ranges)
    # formation, abs move per bucket (TWA-smoothed), tick rate per bucket
    prev_p = c3_p(Q, :ltp, "earliest"); prev_lab = "open"; prev_t = NaN
    tick_mkt(inst) = [sum(view(Q.ticks, r, c3_j(inst))) for r in Q.ranges]
    prev_ticks = [sum(view(Q.ticks, r, 1)) for r in Q.ranges]            # ticks up to t0
    for inst in vcat(C3B_PATH, ["close"])
        p = inst == "close" ? Q.keys.p_close : c3b_path_p(Q, :twa, inst)
        ok = isfinite.(p) .& isfinite.(prev_p)
        t = inst == "close" ? 0.0 : C3_TIME[inst]
        hours = isfinite(prev_t) ? (t - prev_t) / 60 : NaN
        tk = inst == "close" ? prev_ticks : tick_mkt(inst)
        push!(b1_path, (; family = fam, league = lg, bucket = "$(prev_lab) → $(inst)", to = inst,
                        n_markets = nm,
                        formed_share = inst == "close" ? 1.0 : mean(Q.keys.t0[first.(Q.ranges)] .<= t),
                        mean_abs_move_pp = 100 * mean(abs.(p[ok] .- prev_p[ok])),
                        pp_per_hour = inst == "close" || !isfinite(hours) ? NaN : 100 * mean(abs.(p[ok] .- prev_p[ok])) / hours,
                        ticks_per_market = inst == "close" ? NaN : mean(tk .- prev_ticks),
                        ticks_per_hour = inst == "close" || !isfinite(hours) ? NaN : mean(tk .- prev_ticks) / hours);
              promote = true)
        prev_p = p; prev_lab = inst; prev_t = t; prev_ticks = tk
    end
    for v in (:ltp, :twa)
        sh = c3b_share(Q, v, C3B_SHARE_AT)
        push!(b1_share, (; family = fam, league = lg, variant = String(v), n_markets = nm,
                         (Symbol(i) => s for (i, s) in zip(C3B_SHARE_AT, sh))...); promote = true)
    end
    # classes: favourite / middle / longshot by the opening price; home / draw / away (1X2)
    pe = c3_p(Q, :ltp, "earliest"); pc = Q.keys.p_close
    cls = fill("", nrow(Q.keys))
    for r in Q.ranges
        o = sortperm(pe[r]; rev = true)
        lab = length(r) == 3 ? ["favourite", "middle", "longshot"] : ["favourite", "longshot"]
        for (k, i) in enumerate(o)
            cls[r[i]] = lab[k]
        end
    end
    groups = [(c, cls .== c) for c in unique(cls)]
    fam == "1X2" && append!(groups, [(String(s), Q.keys.selection .== s) for s in (:home, :draw, :away)])
    p6 = c3b_path_p(Q, :twa, "T−6h"); p60 = c3b_path_p(Q, :twa, "T−60m")
    for (c, m) in groups
        ok = m .& isfinite.(pe) .& isfinite.(pc)
        d = 100 .* (pc .- pe)
        s = c3_mean(d[ok], Q.keys.date[ok]; B = C3B_B, seed = C3B_SEED)
        push!(b1_class, (; family = fam, league = lg, class = c, n = s.n, mean_open_p = mean(pe[ok]),
                         drift_pp = s.mean, drift_lo = s.lo, drift_hi = s.hi,
                         mean_abs_move_pp = mean(abs.(d[ok])),
                         share_by_T6h = let k = ok .& isfinite.(p6); 1 - sum((pc[k] .- p6[k]) .^ 2) / sum((pc[k] .- pe[k]) .^ 2) end,
                         share_by_T60m = let k = ok .& isfinite.(p60); 1 - sum((pc[k] .- p60[k]) .^ 2) / sum((pc[k] .- pe[k]) .^ 2) end);
              promote = true)
    end
end
c3b_csv("b1_path", b1_path); c3b_csv("b1_share", b1_share); c3b_csv("b1_class", b1_class)
println("B1 done")

# %%
# ===================================================================
# 4. B2 — momentum vs reversion, and how much survives de-bouncing
# ===================================================================
#
# y = the later move to the close, x = the earlier move, one row per selection, markets formed
# by the window start and fresh (l02 gate) at the boundary t_m.
#   raw      x = p_ltp(t_m) − p_ltp(t_a)          y = p_close − p_ltp(t_m)        (de-vigged)
#   smooth   x = p_twa(t_m) − p_twa(t_a)          y = p_close − p_twa(t_m)        (45-min TWA, de-vigged)
#   droplast x = c3_bf_move(t_m, t_a)             y = p_close − 1/o_ltp(t_m)      (raw implied; x ends
#            at the tick BEFORE the boundary tick, so x and y share no tick; y is the CLV of a
#            back at LTP; no tick in the window ⇒ x = 0)
#   gap15    as droplast, but x ends at the last tick ≤ t_m − 15 min (still never the boundary
#            tick), for bounce that is autocorrelated across consecutive ticks
#   NB smooth is biased TOWARD momentum on short windows: first differences of a time-averaged
#   random walk are positively autocorrelated (Working 1960), so smooth is not a clean test.

const C3B_WINDOWS = [("earliest", "T−6h"), ("T−24h", "T−6h"), ("T−6h", "T−60m"), ("T−3h", "T−60m"),
                     ("T−2h", "T−25m"), ("T−60m", "T−25m"), ("earliest", "T−60m"), ("earliest", "T−25m")]

function c3b_window(Q, ta, tm)
    keep = c3_formed(Q, ta) .& c3_fresh(Q, tm) .& isfinite.(Q.keys.p_close)
    if ta == "earliest"
        keep .&= Q.keys.t0 .< C3_TIME[tm]
    end
    pc = Q.keys.p_close
    pl_m = c3_p(Q, :ltp, tm); pl_a = c3_p(Q, :ltp, ta)
    pt_m = c3_p(Q, :twa, tm); pt_a = ta == "earliest" ? pl_a : c3_p(Q, :twa, ta)
    om = view(Q.ltp, :, c3_j(tm))
    xs = Dict(:raw => pl_m .- pl_a, :smooth => pt_m .- pt_a,
              :droplast => c3_bf_move(Q, tm, ta), :gap15 => c3_bf_move(Q, tm, ta; gap = 15))
    ys = Dict(:raw => pc .- pl_m, :smooth => pc .- pt_m, :droplast => pc .- 1 ./ om, :gap15 => pc .- 1 ./ om)
    return keep, xs, ys
end

b2 = DataFrame()
for (fam, Qf) in (("1X2", P1), ("OU2.5", P2)), (lg, tids) in C3B_LEAGUES
    fam == "OU2.5" && lg != "all" && continue
    Q = league(Qf, tids)
    for (ta, tm) in C3B_WINDOWS
        keep, xs, ys = c3b_window(Q, ta, tm)
        for v in (:raw, :smooth, :droplast, :gap15)
            x = ifelse.(keep, xs[v], NaN)
            s = c3_ols(x, ys[v], Q.keys.date; B = C3B_B, seed = C3B_SEED)
            push!(b2, (; family = fam, league = lg, window = "$ta → $tm | $tm → close", t_a = ta, t_m = tm,
                       variant = String(v), s..., share_x_zero = mean(abs.(filter(isfinite, x)) .< 1e-9)); promote = true)
        end
    end
end
c3b_csv("b2_slopes", b2)
println("B2 done")

# %%
# ===================================================================
# 5. B3 — does the movement predict RESULTS given the close?
# ===================================================================
#
# Conditional logit per market: P(win_i) = softmax(c·log p_close_i + g·m_i), m = log p_close −
# log p_t (the move from t to the close; t's price is the 45-min TWA, the opening LTP for
# "earliest"). g = 0 ⇔ the close already prices the movement. Markets formed by t.

b3 = DataFrame()
for (lg, tids) in C3B_LEAGUES, t in ("earliest", "T−24h", "T−6h", "T−60m", "T−25m")
    Q = league(P1, tids)
    pt = t == "earliest" ? c3_p(Q, :ltp, t) : c3_p(Q, :twa, t)
    m = log.(Q.keys.p_close) .- log.(pt)
    m[.!c3_formed(Q, t)] .= NaN
    D = c3_clogit_data(Q, m)
    local r = c3_clogit(D; B = 300, seed = C3B_SEED)
    push!(b3, (; league = lg, move_from = t, r...); promote = true)
end
c3b_csv("b3_clogit", b3)
println("B3 done")

# %%
# ===================================================================
# 6. B4 — model edge × price direction
# ===================================================================
#
# Discovery has one held-out season with a model price: 23/24, from the W2 run
# grw_step_a2_carry_jump (56/57 target frames and its 54/55 monitor frames; lineup-free).
# Bets: edge ≥ 2 pp at t_e (LTP, fresh-gated). Direction d = c3_bf_move(t_e, t_a) on the bet
# selection (bounce-free w.r.t. the entry tick; signal "droplast" gap 0 or "gap15"): "with"
# d > +0.1 pp (the price shortened toward the model), "against" d < −0.1 pp, "flat" otherwise.

cached = deserialize(joinpath(C3B_OUT, "probs.jls"))
grw = only(filter(x -> x.label == "grw_step_a2_carry_jump" && x.panel == C3B_W2, cached.runs))
mp = vcat(cached.probs_target[(C3B_W2, grw.run_id)], cached.probs_monitor[(C3B_W2, grw.run_id)])
mp = mp[mp.market_name .== "1X2", :]
b4_models = [("grw_step_a2_carry_jump", mp)]
if C3B_PHASE == "confirmation"
    # secondary B4 models, 56/57 710 panel only (24/25–25/26 = confirmation; never read in discovery).
    # m12_td uses the played XI: its rows before T−60 are not clean.
    for (lab, id) in (("m05_joint_td", nothing), ("m12_td", UUID("132df5c2-c742-4e95-8693-3aeb2b2cbaef")))
        local r = id === nothing ? only(filter(x -> x.label == lab && x.panel == "24/25,25/26", cached.runs)) :
                  only(filter(x -> x.run_id == id && x.panel == "24/25,25/26", cached.runs))
        local q = cached.probs_target[("24/25,25/26", r.run_id)]
        push!(b4_models, (lab, q[q.market_name .== "1X2", :]))
    end
end
cached = nothing; GC.gc()
c3b_pm(df) = Dict(zip(zip(df.match_id, df.market_name, df.market_line, df.selection), df.p_model))
pmodel = c3b_pm(mp)

const C3B_B4_PAIRS = [("T−6h", "earliest"), ("T−6h", "T−12h"), ("T−3h", "earliest"), ("T−3h", "T−6h"),
                      ("T−2h", "T−6h"), ("T−60m", "earliest"), ("T−60m", "T−3h"), ("T−25m", "T−60m")]
const C3B_FLAT = 0.001

"Entry frame at t_e with the model edge and the bounce-free direction since t_a."
function c3b_signal_frame(Q, te, ta; pm = nothing, gap = 0.0)
    keep = c3_formed(Q, ta)
    ta == "earliest" && (keep .&= Q.keys.t0 .< C3_TIME[te])
    f = c3_entry_frame(Q, te; mask = keep)
    d = c3_bf_move(Q, te, ta; gap)
    f.d_pp = 100 .* d[f.row]
    filter!(r -> isfinite(r.d_pp), f)
    if pm !== nothing
        f.p_model = [get(pm, (r.match_id, r.market_name, r.market_line, r.selection), NaN) for r in eachrow(f)]
        filter!(r -> isfinite(r.p_model), f)
        f.edge_pp = 100 .* (f.p_model .- f.p_entry_fair)
    end
    return f
end

const C3B_SIGNALS = [("droplast", 0.0), ("gap15", 15.0)]

b4 = DataFrame()
for (mlab, mdf) in b4_models, (lg, tids) in (("54+55", [54, 55]), ("56+57", [56, 57]), ("all", [54, 55, 56, 57])),
    (te, ta) in C3B_B4_PAIRS, (sig, gap) in C3B_SIGNALS
    mlab != "grw_step_a2_carry_jump" && lg != "56+57" && continue
    Q = league(P1, tids)
    f = c3b_signal_frame(Q, te, ta; pm = c3b_pm(mdf), gap)
    bets = cn_bets(f, 2.0)
    for (grp, sel) in (("model edge ≥ 2 (all)", trues(nrow(bets))),
                       ("model + price WITH model", bets.d_pp .> 100 * C3B_FLAT),
                       ("model + price AGAINST model", bets.d_pp .< -100 * C3B_FLAT),
                       ("model + flat price", abs.(bets.d_pp) .<= 100 * C3B_FLAT))
        s = cn_summary(bets[sel, :]; B = C3B_B, seed = C3B_SEED)
        push!(b4, (; model = mlab, league = lg, entry = te, since = ta, signal = sig, group = grp, s...); promote = true)
    end
    steam = f[f.d_pp .> 100 * C3B_FLAT, :]
    s = cn_summary(steam; B = C3B_B, seed = C3B_SEED)
    push!(b4, (; model = mlab, league = lg, entry = te, since = ta, signal = sig, group = "market only: price shortened (no model)", s...); promote = true)
end
c3b_csv("b4_model_x_direction", b4)
println("B4 done")

# %%
# ===================================================================
# 7. B5 — market-only rules (grid; discovery picks, confirmation tests)
# ===================================================================
#
# Back every 1X2 selection whose bounce-free move d since t_a (droplast or gap15) is ≥ +X pp
# ("steam": back what shortened) or ≤ −X pp ("drift": back what lengthened), at LTP at t_e.

const C3B_B5_PAIRS = [("T−6h", "earliest"), ("T−6h", "T−24h"), ("T−3h", "earliest"), ("T−3h", "T−6h"),
                      ("T−2h", "T−6h"), ("T−60m", "earliest"), ("T−60m", "T−6h"), ("T−60m", "T−3h"),
                      ("T−25m", "T−3h"), ("T−25m", "T−60m")]
const C3B_B5_X = [0.5, 1.0, 2.0, 4.0]

b5 = DataFrame()
for (lg, tids) in C3B_LEAGUES, (te, ta) in C3B_B5_PAIRS, (sig, gap) in C3B_SIGNALS
    Q = league(P1, tids)
    f = c3b_signal_frame(Q, te, ta; gap)
    for X in C3B_B5_X, (dir, sel) in (("steam", f.d_pp .>= X), ("drift", f.d_pp .<= -X))
        s = cn_summary(f[sel, :]; B = C3B_B, seed = C3B_SEED)
        push!(b5, (; league = lg, entry = te, since = ta, signal = sig, dir, X, s...); promote = true)
    end
    s = cn_summary(f; B = C3B_B, seed = C3B_SEED)
    push!(b5, (; league = lg, entry = te, since = ta, signal = sig, dir = "all (null)", X = NaN, s...); promote = true)
end
c3b_csv("b5_rules", b5)
println("B5 done")

# %%
# ===================================================================
# 8. Confirmatory hypotheses (frozen in REPORT §12.3 before the confirmation run)
# ===================================================================
#
# Six one-sided tests, every window/threshold/signal fixed from discovery. They run in both
# phases: on discovery they give the frozen statistics' discovery values (Holm there is not a
# test); on confirmation (24/25–25/26, run once) they are the test.
# p = one-sided slate-bootstrap tail share (B = 4,000, floor 1/B); Holm across the six.

C3B_BH = 4_000
hyp = DataFrame()
begin
    function c3b_hslope(id, text, tids, ta, tm, v, side)
        Q = league(P1, tids)
        keep, xs, ys = c3b_window(Q, ta, tm)
        s = c3_ols(ifelse.(keep, xs[v], NaN), ys[v], Q.keys.date; B = C3B_BH, seed = C3B_SEED)
        return (; id, text, n = s.n, estimate = s.slope, lo = s.lo, hi = s.hi, z = s.z,
                p_one = side == :neg ? s.p_ge0 : s.p_le0)
    end
    push!(hyp, c3b_hslope("H1", "56: slope of T−25m→close on T−2h→T−25m (gap15) < 0", [56], "T−2h", "T−25m", :gap15, :neg))
    push!(hyp, c3b_hslope("H2", "57: slope of T−60m→close on T−6h→T−60m (gap15) < 0", [57], "T−6h", "T−60m", :gap15, :neg))
    push!(hyp, c3b_hslope("H3", "54: slope of T−25m→close on T−2h→T−25m (gap15) > 0", [54], "T−2h", "T−25m", :gap15, :pos))
    # H4: grw_step_a2_carry_jump, 56/57, edge ≥ 2 at T−60m; CLV(against) − CLV(with), gap15 since earliest
    let f = c3b_signal_frame(league(P1, [56, 57]), "T−60m", "earliest"; pm = pmodel, gap = 15.0)
        b = cn_bets(f, 2.0)
        a = b[b.d_pp .< -100 * C3B_FLAT, :]; w = b[b.d_pp .> 100 * C3B_FLAT, :]
        s = c3_diff(a.clv_pp, a.date, w.clv_pp, w.date; B = C3B_BH, seed = C3B_SEED)
        push!(hyp, (; id = "H4", text = "56/57 grw_step edge≥2 at T−60m: CLV(price against model) − CLV(with) > 0",
                    n = s.n_a + s.n_b, estimate = s.diff, lo = s.lo, hi = s.hi, z = s.z, p_one = s.p_le0))
    end
    # H5: 57, back if droplast move since T−6h ≤ −2 pp, at LTP T−60m: EV@close 2% > 0
    let f = c3b_signal_frame(league(P1, [57]), "T−60m", "T−6h"; gap = 0.0)
        b = f[f.d_pp .<= -2.0, :]
        ev = 100 .* (b.p_close .* (b.odds_entry .- 1) .* 0.98 .- (1 .- b.p_close))
        s = c3_mean(ev, b.date; B = C3B_BH, seed = C3B_SEED)
        push!(hyp, (; id = "H5", text = "57: back drift ≤ −2 pp since T−6h (droplast) at T−60m: EV@close 2% > 0",
                    n = s.n, estimate = s.mean, lo = s.lo, hi = s.hi, z = s.z, p_one = s.p_le0))
    end
    # H6: 56, back if gap15 move since T−3h ≤ −2 pp, at LTP T−25m: CLV > 0
    let f = c3b_signal_frame(league(P1, [56]), "T−25m", "T−3h"; gap = 15.0)
        b = f[f.d_pp .<= -2.0, :]
        s = c3_mean(b.clv_pp, b.date; B = C3B_BH, seed = C3B_SEED)
        push!(hyp, (; id = "H6", text = "56: back drift ≤ −2 pp since T−3h (gap15) at T−25m: CLV > 0",
                    n = s.n, estimate = s.mean, lo = s.lo, hi = s.hi, z = s.z, p_one = s.p_le0))
    end
    hyp.p_holm = c3_holm(hyp.p_one)
    hyp.confirmed = hyp.p_holm .<= 0.05
    c3b_csv("hypotheses", hyp)
    println(hyp)
end
println("\nDONE r03_price_movement (", C3B_PHASE, ")")
