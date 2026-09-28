# r01_clv_napkin.jl — do any of our models beat the closing line, and where?
#
# WHAT THIS IS AND IS NOT
#   A napkin CLV / flat-stake ROI study over PERSISTED held-out latents. No sampling, no
#   new fits, no calibration layer, no Kelly: every selection with model edge ≥ e pp at
#   T−25 is backed for one unit. It is evidence about edge (CLV) plus a noisy P&L check.
#   It is not a staking study and it does not re-score LogLoss.
#
# QUESTION (BRIEF.md)
#   1. CLV and ROI by model, flat stakes, pre-kick-off entry.
#   2. CLV by market, for every market we can price and Betfair archives.
#   3. Compression vs CLV/ROI across models: does less compression buy or cost CLV/ROI?
#   Control: the market-only null (back everything, e = −Inf), the pure team-strength
#   Poisson floor s12_m01_td_poisson (de7fa956), and a market-only PLACEBO whose "model" is
#   the de-vigged TWA of the market's own prices over [T−180, T−25] (prices LTP noise
#   reversion; added after the first pass showed every model, floor included, beat the null).
#
# FILTRATION / COMPARABILITY CONTRACT
#   entry  last traded price at or before T−25 (also T−60 and the earliest complete book),
#          freshest tick ≤ 90 min old, complete market, overround in band.
#   close  TWA over (−20, 0], de-vigged within market; used ONLY for CLV, never selection.
#   model  posterior-mean probability from the fit's walk-forward held-out latents.
#   panels each run is scored on its own panel (W2 = 56/57 23/24–25/26, 1,070 fixtures;
#          W1/W0 = 56/57 24/25–25/26, 710) and runs are never pooled across panels.
#          "W2@710" re-reads the W2 frames on the 24/25–25/26 subset for a like-for-like
#          line against W0/W1; it is a view, not a new panel.
#
# PERSISTENCE CAVEAT
#   Reads mcmc_experiments (fits, harness_scores) and the cached ScottishPyramid DataStore
#   (betdb snapshot). Writes only to current_development/clv_napkin/out/ (replaceable).
#   Prototype model types are included via `cn_include_prototype_loaders!` before any fit
#   is deserialised.
#
# USAGE (mcmc-beast, see BRIEF.md)
#   set -a; source .env; set +a
#   nice -n 19 /root/.juliaup/bin/julia --project -t 8 current_development/clv_napkin/r01_clv_napkin.jl

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================

const CNR_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(@__DIR__, "l01_clv_napkin.jl"))
cn_include_prototype_loaders!(CNR_ROOT)

# %%
# ===================================================================
# 2. Configuration
# ===================================================================

const CNR_ENTRY_MINUTES   = -25.0              # primary entry instant
const CNR_LEAD_MINUTES    = -60.0              # lead-time sensitivity
const CNR_MAX_STALENESS   = 90.0               # PointInTimeBookConfig default
const CNR_THRESHOLDS      = [0.0, 2.0, 4.0, 6.0]
const CNR_HEADLINE_E      = 2.0
const CNR_BOOT_B          = 2_000
const CNR_SEED            = 20260928
const CNR_FLOOR_ID        = UUID("de7fa956-87e8-418f-afb4-61ce01cb9f7d")
const CNR_LIVE_ID         = UUID("132df5c2-c742-4e95-8693-3aeb2b2cbaef")   # m12_td
const CNR_EDGE_BUCKETS    = [-Inf, -6, -4, -2, 0, 2, 4, 6, 10, Inf]
const CNR_OUT             = joinpath(@__DIR__, "out")
mkpath(CNR_OUT)

# %%
# ===================================================================
# 3. Data snapshot and books
# ===================================================================

ds = Data.load_datastore_cached(Data.ScottishPyramid(); max_age_hours = 10_000)

close_book, close_refused = cn_close_book(ds)
entry_book, entry_refused = cn_entry_book(ds, CNR_ENTRY_MINUTES; max_staleness = CNR_MAX_STALENESS)
lead_book, lead_refused   = cn_entry_book(ds, CNR_LEAD_MINUTES; max_staleness = CNR_MAX_STALENESS)
early_book, early_refused = cn_earliest_book(ds; latest = CNR_ENTRY_MINUTES)

println("close    markets refused: ", close_refused)
println("T−25     markets refused: ", entry_refused)
println("T−60     markets refused: ", lead_refused)
println("earliest markets refused: ", early_refused)

# %%
# ===================================================================
# 4. Runs and compression slopes
# ===================================================================

runs = cn_run_table(CNR_ROOT)
slopes = cn_compression_slopes(Set(runs.run_id))
runs.slope = [get(slopes, id, NaN) for id in runs.run_id]
runs.label[runs.run_id .== CNR_FLOOR_ID] .= "FLOOR s12_m01_td_poisson"
runs.label[runs.run_id .== CNR_LIVE_ID] .= "m12_td (live)"
println("runs: ", nrow(runs), " (panel,run) rows; missing slopes: ", count(isnan, runs.slope))

# %%
# ===================================================================
# 5. Held-out model probabilities, one fit at a time
# ===================================================================
#
# For each run: the target panel (56/57) and, for pooled fits, the 54/55 monitor panel.
# Only posterior-mean probabilities are kept; the fit is dropped before the next load.

probs_target  = Dict{Tuple{String,UUID},DataFrame}()
probs_monitor = Dict{Tuple{String,UUID},DataFrame}()
panel_sizes   = Dict{Tuple{String,UUID},Tuple{Int,Int}}()
load_failures = NamedTuple[]
const CNR_PROBS_CACHE = joinpath(CNR_OUT, "probs.jls")
const CNR_REUSE_PROBS = get(ENV, "CNR_REUSE_PROBS", "0") == "1" && isfile(CNR_PROBS_CACHE)
if CNR_REUSE_PROBS
    cached = deserialize(CNR_PROBS_CACHE)
    merge!(probs_target, cached.probs_target)
    merge!(probs_monitor, cached.probs_monitor)
    merge!(panel_sizes, cached.panel_sizes)
    println("reused ", length(probs_target), " cached probability frames")
end

for r in eachrow(runs)
    CNR_REUSE_PROBS && haskey(probs_target, (r.panel, r.run_id)) && continue
    key = (r.panel, r.run_id)
    t0 = time()
    try
        fit = cn_load_fit(r.experiment, r.run_id)
        target = cn_panel(ds, fit, r.seasons)
        monitor = cn_panel(ds, fit, r.seasons; tournaments = (54, 55))
        probs_target[key] = cn_model_probs(fit, target)
        isempty(monitor) || (probs_monitor[key] = cn_model_probs(fit, monitor))
        panel_sizes[key] = (length(target), length(monitor))
        fit = nothing
    catch err
        push!(load_failures, (; label = r.label, run_id = r.run_id, err = sprint(showerror, err)))
    end
    GC.gc()
    @printf("%-32s %-22s target %4d  monitor %4d  %.0fs\n", r.label, r.panel,
            get(panel_sizes, key, (0, 0))..., time() - t0)
end
isempty(load_failures) || foreach(println, load_failures)
serialize(CNR_PROBS_CACHE, (; probs_target, probs_monitor, panel_sizes, runs))

# %%
# ===================================================================
# 6. Selection frames and the per-run summary (Table A + appendix)
# ===================================================================

function cnr_frame(probs, book)
    oc = cn_outcomes(ds, probs)
    return cn_selection_frame(probs, book, close_book, oc)
end

frames = Dict(k => cnr_frame(p, entry_book) for (k, p) in probs_target)
runs_ok = filter(r -> haskey(frames, (r.panel, r.run_id)), runs)

"The W2 frames re-read on the 24/25–25/26 subset (a view, for like-for-like with W0/W1)."
w2_710(f) = f[in.(f.season, Ref(Set(["24/25", "25/26"]))), :]

function cnr_row(r, f, e; view = r.panel)
    s = cn_summary(cn_bets(f, e); B = CNR_BOOT_B, seed = CNR_SEED)
    return (; view, wave = r.wave, label = r.label, run_id = string(r.run_id)[1:8],
            slope = r.slope, e, s..., beta = cn_move_capture(f))
end

table_a = DataFrame()
for r in eachrow(runs_ok), e in CNR_THRESHOLDS
    f = frames[(r.panel, r.run_id)]
    push!(table_a, cnr_row(r, f, e); promote = true)
    r.wave == "W2" && push!(table_a, cnr_row(r, w2_710(f), e; view = "W2@24/25,25/26"); promote = true)
end
# market-only null per view: every selection with a model price (same for every run in a view)
for (view, f) in (("23/24,24/25,25/26", frames[(runs_ok.panel[findfirst(==("W2"), runs_ok.wave)], runs_ok.run_id[findfirst(==("W2"), runs_ok.wave)])]),
                  ("24/25,25/26", frames[("24/25,25/26", CNR_FLOOR_ID)]))
    s = cn_summary(cn_bets(f, -Inf); B = CNR_BOOT_B, seed = CNR_SEED)
    push!(table_a, (; view, wave = "null", label = "MARKET NULL (back all)", run_id = "—",
                    slope = NaN, e = -Inf, s..., beta = NaN); promote = true)
end
# market-only placebo per view: p_model = de-vigged TWA of the market's own prices over
# [T−180, T−25]. It prices LTP noise reversion, not football; see the loader §5b.
placebo_book, placebo_refused = cn_placebo_book(ds; window = (-180.0, CNR_ENTRY_MINUTES))
placebo_frames = Dict{String,DataFrame}()
for (view, key) in (("23/24,24/25,25/26", ("23/24,24/25,25/26", runs_ok.run_id[findfirst(==("W2"), runs_ok.wave)])),
                    ("24/25,25/26", ("24/25,25/26", CNR_FLOOR_ID)))
    pp = cn_placebo_probs(placebo_book, unique(probs_target[key].match_id))
    placebo_frames[view] = cnr_frame(pp, entry_book)
    for e in CNR_THRESHOLDS
        s = cn_summary(cn_bets(placebo_frames[view], e); B = CNR_BOOT_B, seed = CNR_SEED)
        push!(table_a, (; view, wave = "placebo", label = "PLACEBO market TWA[−180,−25]", run_id = "—",
                        slope = NaN, e, s..., beta = cn_move_capture(placebo_frames[view])); promote = true)
    end
end
CSV.write(joinpath(CNR_OUT, "table_a_all_thresholds.csv"), table_a)

# %%
# ===================================================================
# 6b. Does the model know more than the market's own recent prices?
# ===================================================================
#
# OLS of the entry→close fair move on the model edge AND the placebo edge, per run, with a
# slate bootstrap on the model coefficient (B = 300 to keep this cheap).

table_joint = DataFrame()
for r in eachrow(runs_ok)
    f = frames[(r.panel, r.run_id)]
    jc = cn_joint_capture(f, placebo_book; B = 300, seed = CNR_SEED)
    push!(table_joint, (; view = r.panel, wave = r.wave, label = r.label, slope = r.slope, jc...); promote = true)
end
CSV.write(joinpath(CNR_OUT, "table_joint_capture.csv"), table_joint)

# 1X2-only and O/U 2.5-only headline, the two markets we have actually bet
table_a_core = DataFrame()
for r in eachrow(runs_ok), fam in ("1X2", "OU2.5")
    f = frames[(r.panel, r.run_id)]
    push!(table_a_core, (; family = fam, cnr_row(r, f[f.family .== fam, :], CNR_HEADLINE_E)...); promote = true)
end
CSV.write(joinpath(CNR_OUT, "table_a_core_markets.csv"), table_a_core)

# %%
# ===================================================================
# 7. Table B — by market, for the best three models plus the floor
# ===================================================================
#
# "Best" = highest mean clv_pp at e = 2 on its own panel (selection-biased by construction;
# the report says so).

head = filter(r -> r.e == CNR_HEADLINE_E && r.wave ∉ ("null", "placebo") && r.view != "W2@24/25,25/26", table_a)
best3 = first(sort(head, :clv_pp; rev = true), 3)
picked = vcat([(best3.label[i], best3.view[i]) for i in 1:3],
              [("FLOOR s12_m01_td_poisson", "24/25,25/26"), ("m12_td (live)", "24/25,25/26")])
table_b = DataFrame()
for (label, view) in unique(picked)
    r = only(filter(x -> x.label == label && x.panel == view, runs_ok))
    f = frames[(r.panel, r.run_id)]
    for fam in sort(unique(f.family))
        sub = f[f.family .== fam, :]
        for e in (-Inf, CNR_HEADLINE_E)
            s = cn_summary(cn_bets(sub, e); B = CNR_BOOT_B, seed = CNR_SEED)
            push!(table_b, (; label, view, family = fam, e, s..., beta = cn_move_capture(sub)); promote = true)
        end
    end
end
for (view, f) in placebo_frames
    for fam in sort(unique(f.family))
        sub = f[f.family .== fam, :]
        s = cn_summary(cn_bets(sub, CNR_HEADLINE_E); B = CNR_BOOT_B, seed = CNR_SEED)
        push!(table_b, (; label = "PLACEBO market TWA[−180,−25]", view, family = fam, e = CNR_HEADLINE_E,
                        s..., beta = cn_move_capture(sub)); promote = true)
    end
end
CSV.write(joinpath(CNR_OUT, "table_b_by_market.csv"), table_b)

# %%
# ===================================================================
# 8. Table C — edge buckets, league, lead time
# ===================================================================

bucket_label(x) = begin
    i = searchsortedlast(CNR_EDGE_BUCKETS, x)
    lo = CNR_EDGE_BUCKETS[i]; hi = CNR_EDGE_BUCKETS[i + 1]
    isinf(lo) ? "< $(Int(hi))" : isinf(hi) ? "≥ $(Int(lo))" : "[$(Int(lo)), $(Int(hi)))"
end
bucket_order(x) = searchsortedlast(CNR_EDGE_BUCKETS, x)

table_c_bucket = DataFrame()
table_c_league = DataFrame()
table_c_lead = DataFrame()
for (label, view) in unique(picked)
    r = only(filter(x -> x.label == label && x.panel == view, runs_ok))
    f = frames[(r.panel, r.run_id)]
    f.bucket = bucket_label.(f.edge_pp)
    f.border = bucket_order.(f.edge_pp)
    for g in groupby(sort(f, :border), :bucket; sort = false)
        s = cn_summary(g; B = CNR_BOOT_B, seed = CNR_SEED)
        push!(table_c_bucket, (; label, bucket = first(g.bucket), order = first(g.border),
                               n = s.n_bets, clv_pp = s.clv_pp, clv_pp_lo = s.clv_pp_lo,
                               clv_pp_hi = s.clv_pp_hi, beat = s.beat, roi2 = s.roi2); promote = true)
    end
    for tid in (56, 57)
        s = cn_summary(cn_bets(f[f.tournament_id .== tid, :], CNR_HEADLINE_E); B = CNR_BOOT_B, seed = CNR_SEED)
        push!(table_c_league, (; label, tournament = tid, s...); promote = true)
    end
    # lead time on the PAIRED set: selections present in all three entry books
    p = probs_target[(r.panel, r.run_id)]
    fl = cnr_frame(p, lead_book)
    fe = cnr_frame(p, early_book)
    common = intersect(Set(eachrow(select(f, CN_KEY)) .|> Tuple),
                       Set(eachrow(select(fl, CN_KEY)) .|> Tuple),
                       Set(eachrow(select(fe, CN_KEY)) .|> Tuple))
    for (inst, ff) in (("earliest", fe), ("T−60", fl), ("T−25", f))
        sub = ff[[Tuple(x) in common for x in eachrow(select(ff, CN_KEY))], :]
        s = cn_summary(cn_bets(sub, CNR_HEADLINE_E); B = CNR_BOOT_B, seed = CNR_SEED)
        push!(table_c_lead, (; label, entry = inst, median_lead_min = median(sub.entry_minutes),
                             n_selections = nrow(sub), s..., beta = cn_move_capture(sub)); promote = true)
    end
end
CSV.write(joinpath(CNR_OUT, "table_c_edge_buckets.csv"), table_c_bucket)
CSV.write(joinpath(CNR_OUT, "table_c_league.csv"), table_c_league)
CSV.write(joinpath(CNR_OUT, "table_c_lead_time.csv"), table_c_lead)

# %%
# ===================================================================
# 9. Compression vs CLV / ROI, one line per panel
# ===================================================================

spearman(x, y) = cor(invperm(sortperm(x)), invperm(sortperm(y)))
compression = DataFrame()
for g in groupby(filter(r -> r.e == CNR_HEADLINE_E && r.wave ∉ ("null", "placebo") && isfinite(r.slope), table_a), :view)
    length(g.slope) < 4 && continue
    push!(compression, (; view = first(g.view), n_runs = nrow(g),
                        rho_slope_clv = spearman(g.slope, g.clv_pp),
                        rho_slope_roi2 = spearman(g.slope, g.roi2),
                        rho_slope_nbets = spearman(g.slope, Float64.(g.n_bets)),
                        rho_slope_beta = spearman(g.slope, g.beta),
                        rho_slope_edge = spearman(g.slope, g.mean_edge),
                        rho_slope_ev2 = spearman(g.slope, g.ev2)))
    println("\n", first(g.view), "  (x = 1X2 compression slope, y = mean clv_pp at e = 2)")
    println(cn_ascii_scatter(g.slope, g.clv_pp, [Char('a' + i - 1) for i in 1:nrow(g)];
                             xlab = "slope", ylab = "clv_pp"))
    for (i, r) in enumerate(eachrow(g))
        @printf("   %c  %-32s slope %.2f  clv %+.3f  roi2 %+.1f  n %d\n", 'a' + i - 1, r.label,
                r.slope, r.clv_pp, r.roi2, r.n_bets)
    end
end
CSV.write(joinpath(CNR_OUT, "compression_vs_clv.csv"), compression)

# %%
# ===================================================================
# 10. 54/55 monitor panel (pooled fits only)
# ===================================================================

table_monitor = DataFrame()
for r in eachrow(runs_ok)
    key = (r.panel, r.run_id)
    haskey(probs_monitor, key) || continue
    f = cnr_frame(probs_monitor[key], entry_book)
    for tid in (54, 55, 0)
        sub = tid == 0 ? f : f[f.tournament_id .== tid, :]
        s = cn_summary(cn_bets(sub, CNR_HEADLINE_E); B = CNR_BOOT_B, seed = CNR_SEED)
        null = cn_summary(cn_bets(sub, -Inf); B = 200, seed = CNR_SEED)
        push!(table_monitor, (; view = r.panel, label = r.label, tournament = tid == 0 ? "54+55" : string(tid),
                              n_fixtures = length(unique(sub.match_id)), s...,
                              null_clv_pp = null.clv_pp, beta = cn_move_capture(sub)); promote = true)
    end
end
CSV.write(joinpath(CNR_OUT, "table_monitor.csv"), table_monitor)

# %%
# ===================================================================
# 11. Coverage record
# ===================================================================

coverage = DataFrame()
for (view, key) in (("23/24,24/25,25/26", (runs_ok.panel[findfirst(==("W2"), runs_ok.wave)], runs_ok.run_id[findfirst(==("W2"), runs_ok.wave)])),
                    ("24/25,25/26", ("24/25,25/26", CNR_FLOOR_ID)))
    f = frames[key]
    n_panel = panel_sizes[key][1]
    for fam in sort(unique(probs_target[key].market_name .* "|" .* string.(probs_target[key].market_line)))
        name, line = split(fam, "|")
        famlab = cn_family(name, parse(Float64, line))
        push!(coverage, (; view, family = famlab, n_panel,
                         n_fixtures = length(unique(f.match_id[f.family .== famlab])),
                         n_selections = count(==(famlab), f.family)); promote = true)
    end
end
CSV.write(joinpath(CNR_OUT, "coverage.csv"), coverage)
println("\nDONE r01_clv_napkin")
