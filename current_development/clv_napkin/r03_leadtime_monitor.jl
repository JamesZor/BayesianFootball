# r03_leadtime_monitor.jl — brief 03 Part A: the lead-time curve on the 54/55 monitor.
#
# QUESTION
#   §10 found an early-entry 1X2 edge in League One/Two (EV@close ≈ +2% at the earliest book,
#   crossing 0 between T−45m and T−25m). Does it exist in the more liquid Premiership (54) and
#   Championship (55)? If not, the 56/57 early edge is probably a thin-market artefact.
#
# DESIGN (l02/r02 definitions unchanged)
#   entry   LTP at or before each r02 instant and the earliest complete book; staleness gate
#           max(90 min, 25% of the lead); complete market, overround in [0.90, 1.10].
#   close   TWA (−20, 0], de-vigged.  bets  1X2, flat backs, edge ≥ 2 pp, 2% commission.
#   models  the pooled fits' held-out 54/55 monitor latents already priced by r01
#           (out/probs.jls → probs_monitor): td_spfl_cups_joint, grw_spfl_joint (W1, 24/25–25/26),
#           td_base, grw_step_a2_carry_jump (W2, 23/24–25/26). None uses the played XI.
#   controls NULL (back every priced 1X2 selection, e = −Inf) per panel and league; the per-instant
#           placebo (market TWA over [entry − 155, entry]) for β.
#   56/57   the §10 rows come from out/leadtime_curve_all.csv; the 56/57 null is computed here on
#           the §10 universe (710 fixtures) for the side-by-side.
#   liquidity 1X2 ticks per fixture (all runners) up to each instant and in its last 60 min.
#
# PERSISTENCE: reads out/probs.jls (never re-prices) and the cached ScottishPyramid DataStore.
# Writes out/r03a_*.csv only.
#
# USAGE (mcmc-beast): julia --project -t 8 current_development/clv_napkin/r03_leadtime_monitor.jl

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================

include(joinpath(@__DIR__, "l03_price_movement.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================

const C3A_INSTANTS = [("T−48h", -2880.0), ("T−24h", -1440.0), ("T−12h", -720.0), ("T−6h", -360.0),
                      ("T−3h", -180.0), ("T−2h", -120.0), ("T−90m", -90.0), ("T−60m", -60.0),
                      ("T−45m", -45.0), ("T−25m", -25.0), ("T−10m", -10.0)]
const C3A_ORDER   = vcat(["earliest"], first.(C3A_INSTANTS))
const C3A_E       = 2.0
const C3A_B       = 2_000
const C3A_B_JOINT = 300
const C3A_SEED    = 20260928
const C3A_MODELS  = [   # (label, r01 panel key)
    ("td_spfl_cups_joint",     "24/25,25/26"),
    ("grw_spfl_joint",         "24/25,25/26"),
    ("td_base",                "23/24,24/25,25/26"),
    ("grw_step_a2_carry_jump", "23/24,24/25,25/26")]
const C3A_NULLS   = [("NULL W2 panel (23/24–25/26)", "td_base"), ("NULL W1 panel (24/25–25/26)", "td_spfl_cups_joint")]
const C3A_FLOOR_ID = UUID("de7fa956-87e8-418f-afb4-61ce01cb9f7d")
const C3A_OUT     = joinpath(@__DIR__, "out")
const C3A_PROBS   = joinpath(C3A_OUT, "probs.jls")
isfile(C3A_PROBS) || error("r03 reuses r01's out/probs.jls and never re-prices; run r01 first")

# %%
# ===================================================================
# 3. Data snapshot, close, one entry + placebo book per instant
# ===================================================================

ds = Data.load_datastore_cached(Data.ScottishPyramid(); max_age_hours = 10_000)
close_book, _ = cn_close_book(ds)
a_entry   = Dict{String,DataFrame}()
a_placebo = Dict{String,DataFrame}()
for (lab, t) in C3A_INSTANTS
    a_entry[lab], _ = cn_entry_book(ds, t; max_staleness = cl_max_staleness(t))
    a_placebo[lab], _ = cl_placebo_book_at(ds, t)
end
a_entry["earliest"], _ = cn_earliest_book(ds; latest = -25.0)
a_placebo["earliest"], _ = cl_placebo_earliest(ds, a_entry["earliest"])

# %%
# ===================================================================
# 4. Monitor probabilities from r01's cache
# ===================================================================

cached = deserialize(C3A_PROBS)
a_probs = Dict{String,DataFrame}()
a_outcomes = Dict{String,DataFrame}()
for (label, panel) in C3A_MODELS
    local r = only(filter(x -> x.label == label && x.panel == panel, cached.runs))
    local p = cached.probs_monitor[(panel, r.run_id)]
    a_probs[label] = p
    a_outcomes[label] = cn_outcomes(ds, p)
    println(rpad(label, 26), " monitor fixtures ", length(unique(p.match_id)))
end
# the §10 56/57 universe (24/25–25/26) for the 56/57 null
floor_p = cached.probs_target[("24/25,25/26", C3A_FLOOR_ID)]
floor_oc = cn_outcomes(ds, floor_p)
floor_oc = floor_oc[in.(floor_oc.season, Ref(Set(["24/25", "25/26"]))), :]
cached = nothing; GC.gc()

a_frame(label, inst) = cn_selection_frame(a_probs[label], a_entry[inst], close_book, a_outcomes[label])

# %%
# ===================================================================
# 5. One row per model × league × instant (1X2, e = 2) and the nulls
# ===================================================================

function a_row(label, league, inst, f1; e = C3A_E, placebo = true)
    s = cn_summary(cn_bets(f1, e); B = C3A_B, seed = C3A_SEED)
    jc = placebo ? cn_joint_capture(f1, a_placebo[inst]; B = C3A_B_JOINT, seed = C3A_SEED) : nothing
    stale = filter(isfinite, cl_restrict(a_entry[inst], cl_keys(f1)).staleness_minutes)
    return (; label, league, entry = inst, e,
            median_lead_min = isempty(f1) ? NaN : median(f1.entry_minutes),
            n_selections = nrow(f1), n_fixtures = length(unique(f1.match_id)),
            median_staleness = isempty(stale) ? NaN : median(stale), s...,
            beta_alone = placebo ? cn_move_capture(f1) : NaN,
            beta_given_placebo = jc === nothing ? NaN : jc.beta_model_joint,
            beta_given_placebo_lo = jc === nothing ? NaN : jc.beta_model_joint_lo,
            beta_given_placebo_hi = jc === nothing ? NaN : jc.beta_model_joint_hi)
end

curve = DataFrame()
coverage = DataFrame()
for inst in C3A_ORDER
    for (label, _) in C3A_MODELS
        f = a_frame(label, inst)
        f = f[f.family .== "1X2", :]
        for tid in (54, 55)
            push!(curve, a_row(label, string(tid), inst, f[f.tournament_id .== tid, :]); promote = true)
        end
    end
    for (nlab, from) in C3A_NULLS
        f = a_frame(from, inst)
        f = f[f.family .== "1X2", :]
        for tid in (54, 55)
            sub = f[f.tournament_id .== tid, :]
            push!(curve, a_row(nlab, string(tid), inst, sub; e = -Inf, placebo = false); promote = true)
            n_univ = length(unique(a_outcomes[from].match_id[a_outcomes[from].tournament_id .== tid]))
            push!(coverage, (; panel = nlab, league = string(tid), entry = inst, n_universe = n_univ,
                             n_fixtures = length(unique(sub.match_id)),
                             median_lead_min = isempty(sub) ? NaN : median(sub.entry_minutes),
                             median_staleness = let st = filter(isfinite, cl_restrict(a_entry[inst], cl_keys(sub)).staleness_minutes)
                                 isempty(st) ? NaN : median(st) end); promote = true)
        end
    end
    # 56/57 null on the §10 universe
    f = cn_selection_frame(floor_p, a_entry[inst], close_book, floor_oc)
    f = f[f.family .== "1X2", :]
    push!(curve, a_row("NULL 56/57 §10 universe", "56/57", inst, f; e = -Inf, placebo = false); promote = true)
    println("done ", inst)
end
CSV.write(joinpath(C3A_OUT, "r03a_leadtime_monitor.csv"), curve)
CSV.write(joinpath(C3A_OUT, "r03a_coverage.csv"), coverage)

# %%
# ===================================================================
# 6. Liquidity proxy: 1X2 ticks per fixture up to each instant, by league
# ===================================================================
#
# Universe per league: 54/55 = the W2 monitor fixtures (23/24–25/26); 56/57 = the §10 710
# fixtures. Ticks count every runner's price rows at or before the instant (and in its last
# 60 min). Markets with no tick yet count as 0.

univ = Dict("54" => Set(a_outcomes["td_base"].match_id[a_outcomes["td_base"].tournament_id .== 54]),
            "55" => Set(a_outcomes["td_base"].match_id[a_outcomes["td_base"].tournament_id .== 55]),
            "56/57" => Set(floor_oc.match_id))
bo1 = ds.betfair_odds[(ds.betfair_odds.market_name .== "1X2") .& (ds.betfair_odds.minutes_to_kickoff .<= 0), :]
tick_times = Dict{Int,Vector{Float64}}()
for g in groupby(bo1, :match_id)
    tick_times[Int(first(g.match_id))] = sort(Float64.(g.minutes_to_kickoff))
end
liq = DataFrame()
for (lg, ids) in univ, (lab, t) in C3A_INSTANTS
    cum = [let x = get(tick_times, m, Float64[]); searchsortedlast(x, t) end for m in ids]
    l60 = [let x = get(tick_times, m, Float64[]); searchsortedlast(x, t) - searchsortedlast(x, t - 60) end for m in ids]
    push!(liq, (; league = lg, entry = lab, n_fixtures = length(ids),
                median_ticks_cum = median(cum), mean_ticks_cum = mean(cum),
                median_ticks_last60 = median(l60), mean_ticks_last60 = mean(l60),
                share_any_tick = mean(cum .> 0)); promote = true)
end
for (lg, ids) in univ
    tot = [length(get(tick_times, m, Float64[])) for m in ids]
    push!(liq, (; league = lg, entry = "all pre-off", n_fixtures = length(ids), median_ticks_cum = median(tot),
                mean_ticks_cum = mean(tot), median_ticks_last60 = NaN, mean_ticks_last60 = NaN,
                share_any_tick = mean(tot .> 0)); promote = true)
end
CSV.write(joinpath(C3A_OUT, "r03a_liquidity.csv"), liq)
println(liq)

# %%
# ===================================================================
# 7. ASCII chart: CLV and EV@close vs entry time, per league
# ===================================================================

for lg in ("54", "55"), l in ("td_spfl_cups_joint", "grw_step_a2_carry_jump")
    g = curve[(curve.league .== lg) .& (curve.label .== l), :]
    println("\n", l, "  [league ", lg, "]  (1X2, e = 2)")
    println(cl_signed_bars(g.entry, [("mean clv_pp", g.clv_pp, v -> @sprintf("%+.2f", v)),
                                     ("EV@close 2% (%)", g.ev2, v -> @sprintf("%+.1f", v))]))
end
println("\nDONE r03_leadtime_monitor")
