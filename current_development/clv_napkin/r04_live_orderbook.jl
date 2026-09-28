# r04_live_orderbook.jl — brief 04 steps 1 and 3: grw_spfl_joint on 26/27 League One/Two,
# entries priced from the betfair_live order book.
#
# QUESTION
#   How would grw_spfl_joint (W1, 82e01a3b) have done this season in League One/Two entering
#   early, at the price you could actually have backed at?
#
# DESIGN (l04 header has the definitions)
#   model    held-out 26/27 latents from r04_extend_grw_spfl_joint.jl (folds 41–44, in memory,
#            out/r04_ext_grw_spfl_joint.jls on the beast)
#   book     betfair_live.order_book_1m, MATCH_ODDS, pre-off snapshots, joined to fixtures by the
#            verified betfair.match_meta crosswalk
#   entries  earliest complete two-sided book, T−24h, T−6h, T−3h, T−60m, T−25m
#            (last snapshot ≤ T−n, ≤ 15 min old, all 3 runners two-sided)
#   bets     1X2 back, edge ≥ 2 pp vs the de-vigged mid, flat 1 unit, 2% commission
#   prices   BACK (executable) — headline; LTP and MID on the same bet set — what the spread costs
#   close    mid-price TWA (−20, 0], de-vigged; LTP TWA as the §2 cross-check
#   also     market-only null (every selection) per instant; §12 AGAINST/WITH split at T−60m
#            (direction of the de-vigged mid since T−3h), descriptive only
#
# PERSISTENCE: reads the two .jls files on the beast; SELECTs from betdb; writes out/r04_*.csv.
#
# USAGE (mcmc-beast): julia --project -t 8 current_development/clv_napkin/r04_live_orderbook.jl

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================

include(joinpath(@__DIR__, "l04_live_orderbook.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================

const R4_INSTANTS = [("earliest", :earliest), ("T−24h", -1440.0), ("T−6h", -360.0),
                     ("T−3h", -180.0), ("T−60m", -60.0), ("T−25m", -25.0)]
const R4_EDGE = 2.0
const R4_B = 2_000
const R4_SEED = 20260928
const R4_OUTDIR = joinpath(@__DIR__, "out")

# %%
# ===================================================================
# 3. Data: fixtures, the order book, the extension's latents
# ===================================================================

ds = deserialize(joinpath(C4_OUT, "r04_datastore_ScottishPyramid.jls"))
ext = deserialize(joinpath(C4_OUT, "r04_ext_grw_spfl_joint.jls"))
fit = Training.load_fit(Training.PostgresStorage(C4_EXPERIMENT), C4_RUN)

played = ds.matches[coalesce.(ds.matches.season .== C4_SEASON, false) .&
                    in.(Int.(ds.matches.tournament_id), Ref(C4_TOURNAMENTS)) .&
                    .!ismissing.(ds.matches.home_score), :]
markets = c4_markets(ds)
snaps = c4_snapshots(markets)
println("played L1/L2 26/27: ", nrow(played), "   with a betfair_live market: ", nrow(markets),
        "   pre-off snapshot rows: ", nrow(snaps))

# %%
# ===================================================================
# 4. Coverage and the order book itself (step 1, reported in §13)
# ===================================================================

# Per fixture: first pre-off snapshot, first complete two-sided book, snapshot gaps.
cov_rows = NamedTuple[]
for r in eachrow(sort(played, :match_date))
    id = Int(r.match_id)
    g = snaps[snaps.match_id .== id, :]
    has_mkt = id in Set(markets.match_id)
    if isempty(g)
        push!(cov_rows, (; match_id = id, date = Date(r.match_date), tournament_id = Int(r.tournament_id),
                         has_market = has_mkt, n_snapshots = 0, first_min = NaN, first_book_min = NaN,
                         max_gap_min = NaN, cadence_min = NaN))
        continue
    end
    ts = sort!(unique(g.minutes))
    two = combine(groupby(g[isfinite.(g.mid), :], :minutes), nrow => :n)
    fbm = two[two.n .== 3, :minutes]
    push!(cov_rows, (; match_id = id, date = Date(r.match_date), tournament_id = Int(r.tournament_id),
                     has_market = has_mkt, n_snapshots = length(ts), first_min = first(ts),
                     first_book_min = isempty(fbm) ? NaN : minimum(fbm),
                     max_gap_min = length(ts) > 1 ? maximum(diff(ts)) : NaN,
                     cadence_min = length(ts) > 1 ? median(diff(ts)) : NaN))
end
cov = DataFrame(cov_rows)
entries = Dict(lab => c4_entry_book(snaps, t) for (lab, t) in R4_INSTANTS)
for (lab, _) in R4_INSTANTS
    ids = Set(entries[lab].match_id)
    cov[!, Symbol("book_", lab)] = in.(cov.match_id, Ref(ids))
end
CSV.write(joinpath(R4_OUTDIR, "r04_coverage_fixtures.csv"), cov)

bydate = combine(groupby(cov, :date), nrow => :played, :has_market => sum => :market,
                 :n_snapshots => (x -> count(>(0), x)) => :pre_off,
                 :first_book_min => (x -> (y = filter(isfinite, x); isempty(y) ? NaN : median(y))) => :first_book_median,
                 [Symbol("book_", l) => sum => Symbol(l) for (l, _) in R4_INSTANTS]...)
show(stdout, MIME"text/plain"(), bydate; allrows = true, allcols = true); println()
CSV.write(joinpath(R4_OUTDIR, "r04_coverage_by_date.csv"), bydate)

fb = filter(isfinite, cov.first_book_min)
println("first complete book: n ", length(fb), "  median ", median(fb), " min  range ", extrema(fb))
cad = filter(isfinite, cov.cadence_min)
println("snapshot cadence median ", median(cad), " min;  max gap median ", median(filter(isfinite, cov.max_gap_min)),
        "  fixtures with a gap > 15 min: ", count(>(15), filter(isfinite, cov.max_gap_min)))

# Book quality by minutes to the off (all runners, two-sided rows).
bands = [(-Inf, -1440.0, ">24h"), (-1440.0, -360.0, "6–24h"), (-360.0, -180.0, "3–6h"),
         (-180.0, -60.0, "1–3h"), (-60.0, -25.0, "25–60m"), (-25.0, 0.0, "0–25m")]
bq = NamedTuple[]
tw = snaps[isfinite.(snaps.mid), :]
for (lo, hi, lab) in bands
    x = tw[(tw.minutes .> lo) .& (tw.minutes .<= hi), :]
    isempty(x) && continue
    sp = c4_spread_ticks.(x.back, x.lay)
    push!(bq, (; band = lab, rows = nrow(x), fixtures = length(unique(x.match_id)),
               spread_ticks_med = median(sp), spread_pct_med = median(100 .* (x.lay .- x.back) ./ x.back),
               back_size_med = median(x.back_size), back_size_p25 = quantile(x.back_size, 0.25),
               depth3_med = median(x.back_depth3),
               market_matched_med = (y = filter(isfinite, x.market_matched); isempty(y) ? NaN : median(y)),
               ltp_present_pct = 100 * mean(isfinite.(x.ltp))))
end
bq = DataFrame(bq)
show(stdout, MIME"text/plain"(), bq; allrows = true, allcols = true); println()
CSV.write(joinpath(R4_OUTDIR, "r04_book_quality.csv"), bq)

# %%
# ===================================================================
# 5. Model probabilities, closes, outcomes
# ===================================================================

l12_ids = sort!(intersect(Int.(played.match_id), Int.(ext.latents.match_ids)))
println("held-out L1/L2 fixtures with latents: ", length(l12_ids), " of ", nrow(played))
probs = cn_model_probs((; config = (; model = fit.config.model), latents = ext.latents), l12_ids)
filter!(r -> r.market_name == "1X2", probs)
close_mid = c4_close(snaps, :mid)
close_ltp = c4_close(snaps, :ltp)
outcomes = cn_outcomes(ds, select(probs, CN_KEY))
println("closes: mid ", length(unique(close_mid.match_id)), " fixtures, LTP ", length(unique(close_ltp.match_id)))

# Model sanity on all 79 (no book needed): 1X2 log-loss vs the mid close where both exist.
ll = combine(groupby(innerjoin(probs, outcomes, on = CN_KEY), :match_id),
             [:p_model, :won] => ((p, w) -> -log(sum(p .* w))) => :ll_model)
llm = combine(groupby(innerjoin(close_mid, outcomes, on = CN_KEY), :match_id),
              [:p_close, :won] => ((p, w) -> -log(sum(p .* w))) => :ll_close)
llj = innerjoin(ll, llm, on = :match_id)
println("1X2 log-loss on ", nrow(llj), " fixtures with a mid close: model ", round(mean(llj.ll_model), digits = 4),
        "  mid close ", round(mean(llj.ll_close), digits = 4), "   (model on all ", nrow(ll), ": ",
        round(mean(ll.ll_model), digits = 4), ")")

# %%
# ===================================================================
# 6. Per instant: bets at the back price, at LTP and at the mid (same bet set)
# ===================================================================

size_stats(b) = nrow(b) == 0 ? (; back_size_med = NaN, ge20 = NaN, ge50 = NaN, ge100 = NaN, spread_med = NaN, entry_age_med = NaN) :
    (; back_size_med = median(b.back_size), ge20 = 100 * mean(b.back_size .>= 20),
       ge50 = 100 * mean(b.back_size .>= 50), ge100 = 100 * mean(b.back_size .>= 100),
       spread_med = median(b.spread_ticks), entry_age_med = median(-b.entry_minutes))

main = NamedTuple[]
for (lab, _) in R4_INSTANTS
    e = entries[lab]
    isempty(e) && continue
    for (price, close, clab) in ((:back, close_mid, "mid"), (:ltp, close_mid, "mid"), (:mid, close_mid, "mid"),
                                 (:back, close_ltp, "ltp"))
        f = c4_frame(probs, e, close, outcomes; price)
        for (grp, bets) in (("model edge ≥ 2", cn_bets(f, R4_EDGE)), ("null: every selection", f))
            s = cn_summary(bets; B = R4_B, seed = R4_SEED)
            push!(main, (; instant = lab, group = grp, price = String(price), close = clab,
                         n_fixtures_book = length(unique(e.match_id)), s..., size_stats(bets)...))
        end
    end
end
main = DataFrame(main)
CSV.write(joinpath(R4_OUTDIR, "r04_per_instant.csv"), main)
hdr = main[(main.close .== "mid") .& (main.price .== "back"), :]
show(stdout, MIME"text/plain"(), select(hdr, :instant, :group, :n_fixtures_book, :n_bets, :n_slates, :mean_edge,
     :clv_pp, :clv_pp_lo, :clv_pp_hi, :ev2, :ev2_lo, :ev2_hi, :roi2, :roi2_lo, :roi2_hi,
     :back_size_med, :ge20, :ge50, :ge100, :spread_med, :mean_odds); allrows = true, allcols = true); println()

# The spread's price, paired on the SAME bets: back vs LTP vs mid at entry.
pair_rows = NamedTuple[]
for (lab, _) in R4_INSTANTS
    e = entries[lab]
    isempty(e) && continue
    f = c4_frame(probs, e, close_mid, outcomes; price = :back)
    b = cn_bets(f, R4_EDGE)
    b = b[isfinite.(b.ltp) .& (b.ltp .> 1.0), :]
    nrow(b) == 0 && continue
    push!(pair_rows, (; instant = lab, n_bets_with_ltp = nrow(b),
                      pp_back_vs_mid = 100 * mean(1 ./ b.back .- 1 ./ b.mid),
                      pp_back_vs_ltp = 100 * mean(1 ./ b.back .- 1 ./ b.ltp),
                      ev_back = 100 * mean(b.p_close .* (b.back .- 1) .* 0.98 .- (1 .- b.p_close)),
                      ev_ltp = 100 * mean(b.p_close .* (b.ltp .- 1) .* 0.98 .- (1 .- b.p_close)),
                      ev_mid = 100 * mean(b.p_close .* (b.mid .- 1) .* 0.98 .- (1 .- b.p_close)),
                      ltp_below_back_pct = 100 * mean(b.ltp .< b.back .- 1e-9)))
end
pairs = DataFrame(pair_rows)
show(stdout, MIME"text/plain"(), pairs; allrows = true, allcols = true); println()
CSV.write(joinpath(R4_OUTDIR, "r04_spread_cost.csv"), pairs)

# %%
# ===================================================================
# 7. §12 AGAINST/WITH at T−60m (direction of the de-vigged mid since T−3h), descriptive
# ===================================================================

aw_rows = NamedTuple[]
for since in ("T−3h", "earliest")
    a = select(entries[since], CN_KEY..., :p_mid_fair => :p_since, :entry_minutes => :since_minutes)
    f = c4_frame(probs, entries["T−60m"], close_mid, outcomes; price = :back)
    f = innerjoin(f, a, on = CN_KEY)
    since == "earliest" && filter!(r -> r.since_minutes < -60.0, f)
    f.d_pp = 100 .* (f.p_entry_fair .- f.p_since)
    b = cn_bets(f, R4_EDGE)
    for (grp, sel) in (("model edge ≥ 2 (all)", trues(nrow(b))),
                       ("price WITH model", b.d_pp .> 0.1), ("price AGAINST model", b.d_pp .< -0.1),
                       ("flat price", abs.(b.d_pp) .<= 0.1))
        s = cn_summary(b[sel, :]; B = R4_B, seed = R4_SEED)
        push!(aw_rows, (; entry = "T−60m", since, group = grp, s...))
    end
end
aw = DataFrame(aw_rows)
show(stdout, MIME"text/plain"(), select(aw, :since, :group, :n_bets, :n_slates, :clv_pp, :clv_pp_lo, :clv_pp_hi,
     :ev2, :ev2_lo, :ev2_hi, :roi2); allrows = true, allcols = true); println()
CSV.write(joinpath(R4_OUTDIR, "r04_against_with.csv"), aw)

# %%
# ===================================================================
# 8. What the sample can detect
# ===================================================================

pw = select(hdr[hdr.group .== "model edge ≥ 2", :], :instant, :n_bets, :n_slates,
            [:clv_pp_lo, :clv_pp_hi] => ByRow((l, h) -> h - l) => :clv_ci_width,
            [:ev2_lo, :ev2_hi] => ByRow((l, h) -> h - l) => :ev2_ci_width,
            [:roi2_lo, :roi2_hi] => ByRow((l, h) -> h - l) => :roi2_ci_width)
# 80% power, two-sided 5%: MDE ≈ (1.96 + 0.84)·SE ≈ 1.43 × CI half-width
pw.mde_clv_pp = 1.43 .* pw.clv_ci_width ./ 2
pw.mde_ev2 = 1.43 .* pw.ev2_ci_width ./ 2
pw.mde_roi2 = 1.43 .* pw.roi2_ci_width ./ 2
show(stdout, MIME"text/plain"(), pw; allrows = true, allcols = true); println()
CSV.write(joinpath(R4_OUTDIR, "r04_power.csv"), pw)

# %%
# ===================================================================
# 9. Where the disagreement lives: fixtures with a club that changed tier this summer
# ===================================================================

tiers = Data.club_season_tiers(ds.matches)
moved(c) = get(tiers, (c, "25/26"), 0) != get(tiers, (c, C4_SEASON), -1)
mrow = Dict(Int(r.match_id) => r for r in eachrow(ds.matches))
is_trans = Dict(id => moved(String(mrow[id].home_team)) || moved(String(mrow[id].away_team)) for id in l12_ids)
println("clubs that changed tier: ", sort(unique([c for id in l12_ids
        for c in (String(mrow[id].home_team), String(mrow[id].away_team)) if moved(c)])),
        "   fixtures: ", count(values(is_trans)), " of ", length(is_trans))
tr_rows = NamedTuple[]
for (lab, _) in R4_INSTANTS
    isempty(entries[lab]) && continue
    b = cn_bets(c4_frame(probs, entries[lab], close_mid, outcomes; price = :back), R4_EDGE)
    b.trans = [is_trans[i] for i in b.match_id]
    for (g, sel) in (("transition fixture", b.trans), ("other", .!b.trans))
        s = cn_summary(b[sel, :]; B = R4_B, seed = R4_SEED)
        push!(tr_rows, (; instant = lab, group = g, s...))
    end
end
tr = DataFrame(tr_rows)
show(stdout, MIME"text/plain"(), select(tr, :instant, :group, :n_bets, :mean_edge, :clv_pp, :clv_pp_lo, :clv_pp_hi,
     :ev2, :ev2_lo, :ev2_hi, :roi2); allrows = true, allcols = true); println()
CSV.write(joinpath(R4_OUTDIR, "r04_transition_split.csv"), tr)
llt = innerjoin(ll, llm, on = :match_id)
llt.transition = [is_trans[i] for i in llt.match_id]
llt = combine(groupby(llt, :transition), nrow => :n, :ll_model => mean => :ll_model, :ll_close => mean => :ll_mid_close)
show(stdout, MIME"text/plain"(), llt); println()
CSV.write(joinpath(R4_OUTDIR, "r04_logloss.csv"), llt)

# Selection-level frame for audit (T−60m and T−25m bets at the back price).
aud = vcat([insertcols!(cn_bets(c4_frame(probs, entries[l], close_mid, outcomes; price = :back), R4_EDGE),
                        1, :instant => l) for (l, _) in R4_INSTANTS if !isempty(entries[l])]...; cols = :union)
CSV.write(joinpath(R4_OUTDIR, "r04_bets.csv"), select(aud, :instant, :date, :match_id, :tournament_id, :selection,
          :p_model, :p_entry_fair, :edge_pp, :back, :back_size, :lay, :mid, :ltp, :spread_ticks, :entry_minutes,
          :p_close, :clv_pp, :won))
println("DONE r04_live_orderbook ", now())
