# r02_clv_leadtime.jl — the lead-time curve: how do CLV and EV@close move with entry time?
#
# QUESTION
#   r01 found CLV ≈ +0.15 pp at T−25 but ≈ +1.0 pp at the earliest complete book (~21 h), with
#   EV@close crossing from negative to about +9.5%. Where between the two does the edge live,
#   and is that window bettable given coverage, staleness and liquidity?
#
# DESIGN
#   entry   last traded price at or before T−48h … T−10m (11 fixed instants) and "earliest"
#           (first instant every runner of the market had traded, ≤ T−25, no staleness gate).
#           Staleness gate: the market's oldest runner tick ≤ max(90 min, 25% of the lead).
#   close   TWA (−20, 0], de-vigged — unchanged from r01. NB T−10 sits INSIDE the close
#           window, so its CLV is mechanically damped.
#   models  six persisted runs on one fixture universe (56/57, 24/25–25/26): the W0/W1 710
#           panel runs as-is, the two W2 co-finalists re-read on their 24/25–25/26 subset.
#           m12_td prices with the played XI (public ≈ T−60): rows before T−60 are NOT clean.
#   placebo per instant, p_model = de-vigged TWA of the market over [entry − 155, entry]
#           (= r01's [−180, −25] at T−25).
#   bets    1X2 headline, flat 1-unit backs at edge ≥ 2 pp, commission 2%.
#   views   (a) every selection priced at the instant; (b) the per-model paired set present
#           at T−24h, T−6h, T−2h, T−60m and T−25m.
#   liquidity  market ticks (and volume, if the odds frame has it) up to the instant, in
#           terciles over the universe's 1X2 markets at T−24h, T−6h, T−25m.
#
# PERSISTENCE: reads out/probs.jls from r01 (no fit is loaded, nothing is re-priced) and the
# cached ScottishPyramid DataStore. Writes out/leadtime_*.csv only.
#
# USAGE (mcmc-beast): julia --project -t 8 current_development/clv_napkin/r02_clv_leadtime.jl

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================

include(joinpath(@__DIR__, "l02_clv_leadtime.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================

const CLR_INSTANTS = [("T−48h", -2880.0), ("T−24h", -1440.0), ("T−12h", -720.0), ("T−6h", -360.0),
                      ("T−3h", -180.0), ("T−2h", -120.0), ("T−90m", -90.0), ("T−60m", -60.0),
                      ("T−45m", -45.0), ("T−25m", -25.0), ("T−10m", -10.0)]
const CLR_ORDER    = vcat(["earliest"], first.(CLR_INSTANTS))
const CLR_PAIRED   = ["T−24h", "T−6h", "T−2h", "T−60m", "T−25m"]
const CLR_LIQ      = ["T−24h", "T−6h", "T−25m"]
const CLR_E        = 2.0
const CLR_B        = 2_000
const CLR_B_JOINT  = 300
const CLR_SEED     = 20260928
const CLR_SEASONS  = Set(["24/25", "25/26"])
const CLR_CLEAN_FROM = -60.0                      # lineup models are clean from T−60 on
const CLR_FLOOR_ID = UUID("de7fa956-87e8-418f-afb4-61ce01cb9f7d")
const CLR_LIVE_ID  = UUID("132df5c2-c742-4e95-8693-3aeb2b2cbaef")
const CLR_MODELS   = [   # (label, r01 panel key, uses the played XI)
    ("m05_joint_td",               "24/25,25/26",       false),
    ("td_spfl_cups_joint",         "24/25,25/26",       false),
    ("m12_td (live)",              "24/25,25/26",       true),
    ("td_lower_a2full_carry_jump", "23/24,24/25,25/26", false),
    ("grw_step_a2_carry_jump",     "23/24,24/25,25/26", false),
    ("FLOOR s12_m01_td_poisson",   "24/25,25/26",       false)]
const CLR_PLACEBO  = "PLACEBO TWA[entry−155, entry]"
const CLR_OUT      = joinpath(@__DIR__, "out")
const CLR_PROBS    = joinpath(CLR_OUT, "probs.jls")
isfile(CLR_PROBS) || error("r02 reuses r01's out/probs.jls and never re-prices; run r01 first")

# %%
# ===================================================================
# 3. Data snapshot, close, and one entry + placebo book per instant
# ===================================================================

ds = Data.load_datastore_cached(Data.ScottishPyramid(); max_age_hours = 10_000)
println("betfair_odds columns: ", names(ds.betfair_odds))

close_book, _ = cn_close_book(ds)
entry_books   = Dict{String,DataFrame}()
placebo_books = Dict{String,DataFrame}()
book_log      = NamedTuple[]
for (lab, t) in CLR_INSTANTS
    entry_books[lab], ref = cn_entry_book(ds, t; max_staleness = cl_max_staleness(t))
    placebo_books[lab], _ = cl_placebo_book_at(ds, t)
    push!(book_log, (; entry = lab, gate = cl_max_staleness(t),
                     (Symbol(k) => v for (k, v) in ref)...))
    println(rpad(lab, 9), " gate ", cl_max_staleness(t), " min  refused ", ref)
end
entry_books["earliest"], ref = cn_earliest_book(ds; latest = -25.0)
placebo_books["earliest"], _ = cl_placebo_earliest(ds, entry_books["earliest"])
println("earliest  refused ", ref)

# %%
# ===================================================================
# 4. Model probabilities from r01's cache, on the common 24/25–25/26 universe
# ===================================================================

cached = deserialize(CLR_PROBS)
runs = copy(cached.runs)
runs.label[runs.run_id .== CLR_FLOOR_ID] .= "FLOOR s12_m01_td_poisson"
runs.label[runs.run_id .== CLR_LIVE_ID] .= "m12_td (live)"

model_probs = Dict{String,DataFrame}()
outcomes    = Dict{String,DataFrame}()
for (label, panel, _) in CLR_MODELS
    local r = only(filter(x -> x.label == label && x.panel == panel, runs))
    p = cached.probs_target[(panel, r.run_id)]
    local oc = cn_outcomes(ds, p)
    oc = oc[in.(oc.season, Ref(CLR_SEASONS)), :]
    ids = Set(oc.match_id)
    model_probs[label] = p[in.(p.match_id, Ref(ids)), :]
    outcomes[label] = oc
    println(rpad(label, 28), " fixtures ", length(ids))
end
universe = sort!(collect(Set(model_probs["FLOOR s12_m01_td_poisson"].match_id)))
placebo_outcomes = outcomes["FLOOR s12_m01_td_poisson"]
cached = nothing; GC.gc()

"Selection frame of one model (or the placebo) at one instant."
function clr_frame(label, inst)
    if label == CLR_PLACEBO
        return cn_selection_frame(cn_placebo_probs(placebo_books[inst], universe),
                                  entry_books[inst], close_book, placebo_outcomes)
    end
    return cn_selection_frame(model_probs[label], entry_books[inst], close_book, outcomes[label])
end

labels = vcat(first.(CLR_MODELS), [CLR_PLACEBO])
uses_xi = Dict(l => x for (l, _, x) in CLR_MODELS)
frames = Dict((l, i) => clr_frame(l, i) for l in labels, i in CLR_ORDER)

# %%
# ===================================================================
# 5. One row per model × instant (1X2, e = 2)
# ===================================================================

function clr_row(label, inst, f; view)
    f1 = f[f.family .== "1X2", :]
    s = cn_summary(cn_bets(f1, CLR_E); B = CLR_B, seed = CLR_SEED)
    lead = isempty(f1) ? NaN : median(f1.entry_minutes)
    jc = label == CLR_PLACEBO ? nothing :
         cn_joint_capture(f1, placebo_books[inst]; B = CLR_B_JOINT, seed = CLR_SEED)
    t = inst == "earliest" ? -Inf : Dict(CLR_INSTANTS)[inst]
    clean = !(get(uses_xi, label, false) && t < CLR_CLEAN_FROM)
    stale = filter(isfinite, cl_restrict(entry_books[inst], cl_keys(f1)).staleness_minutes)
    return (; view, label, entry = inst, median_lead_min = lead, clean,
            n_selections = nrow(f1), n_fixtures = length(unique(f1.match_id)),
            median_staleness = isempty(stale) ? NaN : median(stale), s...,
            beta_alone = cn_move_capture(f1),
            beta_given_placebo = jc === nothing ? NaN : jc.beta_model_joint,
            beta_given_placebo_lo = jc === nothing ? NaN : jc.beta_model_joint_lo,
            beta_given_placebo_hi = jc === nothing ? NaN : jc.beta_model_joint_hi)
end

# (a) every selection priced at the instant
curve_a = DataFrame()
for l in labels, i in CLR_ORDER
    push!(curve_a, clr_row(l, i, frames[(l, i)]; view = "a: all priced"); promote = true)
end

# (b) the per-model paired set present at every CLR_PAIRED instant (earliest shown on that set
# too). The fixtures fresh at T−24h and at T−6h barely overlap, so (b) is thin; b′ and b″ are
# looser pairings (added after the first run) that each drop one of the two sparse instants.
curve_b = DataFrame()
paired_n = Dict{Tuple{String,String},Int}()
for (view, insts) in (("b: paired", CLR_PAIRED),
                      ("b′: paired T−6h…T−25m", ["T−6h", "T−2h", "T−60m", "T−25m"]),
                      ("b″: paired T−24h, T−60m, T−25m", ["T−24h", "T−60m", "T−25m"]))
    for l in labels
        local common = reduce(intersect, (cl_keys(frames[(l, i)][frames[(l, i)].family .== "1X2", :]) for i in insts))
        paired_n[(view, l)] = length(common)
        for i in vcat(["earliest"], insts)
            push!(curve_b, clr_row(l, i, cl_restrict(frames[(l, i)], common); view); promote = true)
        end
    end
end
CSV.write(joinpath(CLR_OUT, "leadtime_curve_all.csv"), curve_a)
CSV.write(joinpath(CLR_OUT, "leadtime_curve_paired.csv"), curve_b)
println("paired 1X2 selections per model: ", paired_n)

# other markets, view (a), kept only where n bets ≥ 100
other = DataFrame()
for l in labels, i in CLR_ORDER
    f = frames[(l, i)]
    for fam in sort(unique(f.family))
        fam == "1X2" && continue
        b = cn_bets(f[f.family .== fam, :], CLR_E)
        nrow(b) >= 100 || continue
        s = cn_summary(b; B = CLR_B, seed = CLR_SEED)
        push!(other, (; label = l, entry = i, family = fam, s...); promote = true)
    end
end
CSV.write(joinpath(CLR_OUT, "leadtime_other_markets.csv"), other)
println("other-market cells with n ≥ 100: ", nrow(other))

# %%
# ===================================================================
# 6. Liquidity terciles at T−24h, T−6h, T−25m (view a, 1X2)
# ===================================================================

const CLR_VOLUME_COL = let c = findfirst(n -> n in names(ds.betfair_odds),
                                         ["traded_volume", "volume", "total_matched", "matched", "size"])
    c === nothing ? nothing : Symbol(["traded_volume", "volume", "total_matched", "matched", "size"][c])
end
println("volume column in the odds frame: ", CLR_VOLUME_COL === nothing ? "NONE (tick count used)" : CLR_VOLUME_COL)
const CLR_LIQ_MEASURE = CLR_VOLUME_COL === nothing ? :ticks : :volume

liquidity = DataFrame()
for inst in CLR_LIQ
    t = Dict(CLR_INSTANTS)[inst]
    liq = cl_market_liquidity(ds, t; volume_col = CLR_VOLUME_COL)
    # cut points over the universe's 1X2 markets that have a valid entry book at this instant
    eb = entry_books[inst]
    mk = unique(eb[(eb.market_name .== "1X2") .& in.(eb.match_id, Ref(Set(universe))), CN_MKEY])
    mk = innerjoin(mk, liq; on = CN_MKEY)
    cuts = quantile(mk[!, CLR_LIQ_MEASURE], [1 / 3, 2 / 3])
    for tt in 1:3
        sel = mk[cl_tercile.(mk[!, CLR_LIQ_MEASURE], Ref(cuts)) .== tt, :]
        @printf("%-6s tercile %d  markets %4d  ticks median %6.0f  volume median %s\n", inst, tt, nrow(sel),
                median(sel.ticks), CLR_VOLUME_COL === nothing ? "—" : @sprintf("%.0f", median(sel.volume)))
    end
    for l in labels
        f = innerjoin(frames[(l, inst)][frames[(l, inst)].family .== "1X2", :], liq; on = CN_MKEY)
        f.tercile = cl_tercile.(f[!, CLR_LIQ_MEASURE], Ref(cuts))
        for tt in 1:3
            sub = f[f.tercile .== tt, :]
            s = cn_summary(cn_bets(sub, CLR_E); B = CLR_B, seed = CLR_SEED)
            push!(liquidity, (; entry = inst, label = l, tercile = tt, measure = String(CLR_LIQ_MEASURE),
                              cut_lo = cuts[1], cut_hi = cuts[2],
                              median_ticks = isempty(sub) ? NaN : median(sub.ticks),
                              median_volume = isempty(sub) ? NaN : median(sub.volume),
                              n_selections = nrow(sub), s...); promote = true)
        end
    end
end
CSV.write(joinpath(CLR_OUT, "leadtime_liquidity.csv"), liquidity)

# %%
# ===================================================================
# 7. ASCII chart: paired CLV and EV@close vs entry time
# ===================================================================

for view in unique(curve_b.view), l in ("m05_joint_td", "grw_step_a2_carry_jump", "FLOOR s12_m01_td_poisson")
    g = curve_b[(curve_b.view .== view) .& (curve_b.label .== l), :]
    println("\n", l, "  [", view, "]  (1X2, e = 2; paired selections ", paired_n[(view, l)], ")")
    println(cl_signed_bars(g.entry, [("mean clv_pp", g.clv_pp, v -> @sprintf("%+.2f", v)),
                                     ("EV@close 2% (%)", g.ev2, v -> @sprintf("%+.1f", v))]))
end
println("\nDONE r02_clv_leadtime")
