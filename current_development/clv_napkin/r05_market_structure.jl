# r05_market_structure.jl — how does the market price matches? (BRIEF_05)
#
# QUESTION
#   Part B  Which score grid does the Betfair market use? Fit six grid families per match to the
#           de-vigged 1X2 + O/U 2.5 close and compare the implied correct-score grid with the
#           market's CS book; the global shape parameter (ρ, λ₃, κ, ν, δ) is chosen by CS KL.
#           Then: the residual pattern, our models' grids, thin-market consistency.
#   Part A  Invert every close into (λ_home, λ_away) with the chosen grid, fit market ratings
#           (attack/defence random walks per league-season), compare with m12_td and
#           grw_spfl_joint, and ask what explains the market's season-start ratings and how fast
#           it re-rates clubs that changed tier.
#
# DATA / FILTRATION
#   archive  betfair.odds_history via the ScottishPyramid DataStore rebuilt for §13
#            (/root/BF_runs/clv_napkin_dev/out/r04_datastore_ScottishPyramid.jls); 54–57,
#            21/22–25/26. 1X2/O-U/BTTS close = TWA (−20, 0], de-vigged (l01 `cn_close_book`).
#            CS = partial book: each runner's last trade at or before kick-off, ≤ 120 min stale,
#            ≥ 16 of 19 runners including 0-0…1-1, compared on the runners present.
#   live     betfair_live.order_book_1m for 26/27: mid of best back/lay, TWA (−20, 0].
#   models   posterior-mean probabilities from out/probs.jls (r01; m12_td 132df5c2 and
#            grw_spfl_joint 82e01a3b, 24/25–25/26) and the §13 extension for 26/27.
#   Nothing here is a betting test; no sampling, no DB writes.
#
# USAGE (mcmc-beast): julia --project -t 16 current_development/clv_napkin/r05_market_structure.jl

# %%
# ===================================================================
# 1. Packages and configuration
# ===================================================================

include(joinpath(@__DIR__, "l05_market_structure.jl"))
LinearAlgebra.BLAS.set_num_threads(1)

const R5_OUT = joinpath(@__DIR__, "out")
const R5_SEASONS = ["21/22", "22/23", "23/24", "24/25", "25/26"]
const R5_TIDS = (54, 55, 56, 57)
const R5_LEAGUE = Dict(54 => "Prem", 55 => "Champ", 56 => "L1", 57 => "L2", 1 => "EPL")
const R5_F25 = (2.5,)                        # fit set B: 1X2 + O/U 2.5
const R5_FALL = (1.5, 2.5, 3.5)              # fit set A / θ_OU: 1X2 + O/U 1.5, 2.5, 3.5
const R5_B = 2_000
const R5_SEED = 20260928
const R5_M12 = UUID("132df5c2-c742-4e95-8693-3aeb2b2cbaef")
const R5_GRW = UUID("82e01a3b-d686-4e29-a269-3702b96fc5ba")
const R5_TRANSITION_2627 = ["ross-county", "airdrieonians", "east-kilbride", "kelty-hearts-fc"]
const R5_GRIDS = (C5_KINDS..., C5_COMBOS...)
println("threads: ", Threads.nthreads(), "  started ", now())

r5_write(name, df) = CSV.write(joinpath(R5_OUT, "r05_" * name * ".csv"), df)
r5_show(df) = (show(stdout, MIME"text/plain"(), df; allrows = true, allcols = true, truncate = 0); println())

# %%
# ===================================================================
# 2. Books and targets
# ===================================================================

ds = deserialize(joinpath(C4_OUT, "r04_datastore_ScottishPyramid.jls"))
meta = Dict(Int(r.match_id) => r for r in eachrow(ds.matches))
tid_of(m) = Int(meta[m].tournament_id)
close_book, _ = cn_close_book(ds)
cs_pit = c5_cs_pit(ds.betfair_odds; stale = 120.0)
in_scope(m) = haskey(meta, m) && tid_of(m) in R5_TIDS && String(meta[m].season) in R5_SEASONS
tg_all = c5_targets(close_book[in_scope.(close_book.match_id), :]; pcol = :p_close, cs = cs_pit)
live_book, live_cs = c5_live_close(ds)
tg_live = c5_targets(live_book; pcol = :p_close, cs = live_cs)

cov_rows = NamedTuple[]
for tid in R5_TIDS
    ids = [m for m in keys(meta) if tid_of(m) == tid && String(meta[m].season) in R5_SEASONS &&
                                     !ismissing(meta[m].home_score)]
    t = [tg_all[m] for m in ids if haskey(tg_all, m)]
    lt = [v for v in values(tg_live) if tid_of(v.match_id) == tid]
    push!(cov_rows, (; league = R5_LEAGUE[tid], played = length(ids), core_1x2_ou25 = length(t),
                     with_ou15_35 = count(x -> haskey(x.under, 1.5) && haskey(x.under, 3.5), t),
                     with_btts = count(x -> !isnan(x.btts), t), with_cs = count(c5_has_cs, t),
                     cs_all19 = count(x -> length(x.cs_idx) == 19, t),
                     live2627_core = length(lt), live2627_cs = count(c5_has_cs, lt)))
end
coverage = DataFrame(cov_rows)
r5_show(coverage)
r5_write("coverage", coverage)

# %%
# ===================================================================
# 3. Part B.2: six grids, profiled over the shape parameter
# ===================================================================
#
# Sample: archive matches with 1X2 + O/U 2.5 AND a partial CS book. θ̂_CS = argmin of the mean CS
# KL (per-match fit to 1X2 + O/U 2.5); the reported KL is in-sample descriptive, not an
# out-of-sample model-selection score. θ̂_OU = argmin of the mean 1X2 + O/U 1.5/2.5/3.5 misfit
# (per-match fit to the same set) — what the goal-line markets alone say about the shape.

tb = sort([t for t in values(tg_all) if c5_has_cs(t)]; by = t -> t.match_id)
tb_date = [Date(meta[t.match_id].match_date) for t in tb]
tb_tid = [tid_of(t.match_id) for t in tb]
tou = sort([t for t in values(tg_all) if haskey(t.under, 1.5) && haskey(t.under, 3.5)]; by = t -> t.match_id)
tou_date = [Date(meta[t.match_id].match_date) for t in tou]
println("Part B sample: ", length(tb), " matches with CS; θ_OU sample ", length(tou))

prof = Dict{Symbol,Any}()
prof_ou = Dict{Symbol,Any}()
for k in R5_GRIDS
    t0 = time()
    prof[k] = c5_profile(tb, k, R5_F25)
    prof_ou[k] = c5_profile(tou, k, R5_FALL)
    @printf("%-8s profiled in %.0fs\n", k, time() - t0)
end

"Grid-comparison table for one sample (rows of `pr` restricted by `keep`)."
function r5_grid_table(prof, prof_ou, keep, clusters, keep_ou, clusters_ou; label = "all")
    kl0 = prof[:poisson].KL[keep, 1]
    rows = NamedTuple[]
    for k in R5_GRIDS
        pr = prof[k]
        a = c5_argmin_ci(pr.KL[keep, :], clusters[keep], pr.θs; B = R5_B, seed = R5_SEED)
        d = c5_mean_ci(1000 .* (pr.KL[keep, a.j] .- kl0), clusters[keep]; B = R5_B, seed = R5_SEED)
        o = keep_ou === nothing ? nothing :
            c5_argmin_ci(prof_ou[k].L[keep_ou, :], clusters_ou[keep_ou], pr.θs; B = R5_B, seed = R5_SEED)
        push!(rows, (; sample = label, grid = C5_KIND_LABEL[k], kind = String(k), n = count(keep),
                     theta_cs = a.θ, theta_cs_lo = a.lo, theta_cs_hi = a.hi, at_edge = a.at_edge,
                     kl_x1000 = 1000 * a.curve[a.j], dkl_vs_poisson_x1000 = d.mean,
                     dkl_lo = d.lo, dkl_hi = d.hi,
                     theta_ou = o === nothing ? NaN : o.θ, theta_ou_lo = o === nothing ? NaN : o.lo,
                     theta_ou_hi = o === nothing ? NaN : o.hi,
                     ou_misfit_x1000 = o === nothing ? NaN : 1000 * o.curve[o.j]))
    end
    return DataFrame(rows)
end

grid_all = r5_grid_table(prof, prof_ou, trues(length(tb)), tb_date, trues(length(tou)), tou_date)
r5_show(grid_all)
by_league = vcat([r5_grid_table(prof, prof_ou, tb_tid .== tid, tb_date,
                                [tid_of(t.match_id) == tid for t in tou], tou_date; label = R5_LEAGUE[tid])
                  for tid in R5_TIDS]...)
r5_show(select(by_league, :sample, :kind, :n, :theta_cs, :theta_cs_lo, :theta_cs_hi, :dkl_vs_poisson_x1000,
               :dkl_lo, :dkl_hi, :theta_ou, :theta_ou_lo, :theta_ou_hi))
tot_pois = prof[:poisson].MH[:, 1] .+ prof[:poisson].MA[:, 1]
tbins = [(0.0, 2.4, "<2.4"), (2.4, 2.8, "2.4–2.8"), (2.8, 3.2, "2.8–3.2"), (3.2, Inf, "≥3.2")]
by_total = vcat([r5_grid_table(prof, prof_ou, lo .<= tot_pois .< hi, tb_date, nothing, nothing; label = "total " * lab)
                 for (lo, hi, lab) in tbins]...)
r5_show(select(by_total, :sample, :kind, :n, :theta_cs, :theta_cs_lo, :theta_cs_hi, :dkl_vs_poisson_x1000, :dkl_lo, :dkl_hi))
r5_write("grid_fit", vcat(grid_all, by_league, by_total))

best = grid_all[argmin(grid_all.kl_x1000), :]
R5_BEST = (Symbol(best.kind), best.theta_cs)
println("best grid: ", R5_BEST)

# %%
# ===================================================================
# 4. Part B.3: the residual pattern (market CS minus the fitted grid)
# ===================================================================
#
# Per match: Poisson and the best grid fitted to 1X2 + O/U 2.5; residual = market − grid on the
# CS runners present (both renormalised over that set), in percentage points. `draw (CS)` sums
# the draw runners present; `draw (1X2)` is the 1X2 draw the 2-parameter fit cannot reproduce.

const R5_CELLS = [("0-0", :cs_00), ("1-0", :cs_10), ("0-1", :cs_01), ("1-1", :cs_11), ("2-2", :cs_22)]
const R5_DRAWS = [C5_CS_INDEX[s] for s in (:cs_00, :cs_11, :cs_22, :cs_33, :cs_any_other_draw)]

"Per-match residual rows (pp) of market-vs-grid for targets `ts` against fits `fits`."
function r5_residual_frame(ts, fits; qfun = c5_cs_q)
    rows = NamedTuple[]
    for (t, f) in zip(ts, fits)
        q = qfun(t)
        p = c5_cs_on(t, f.cs)
        present = [k for k in R5_DRAWS if !isnan(q[k])]
        push!(rows, (; match_id = t.match_id,
                     (Symbol("r_", replace(lab, "-" => "")) => 100 * (q[C5_CS_INDEX[s]] - p[C5_CS_INDEX[s]])
                      for (lab, s) in R5_CELLS)...,
                     r_draw_cs = 100 * (sum(q[present]) - sum(p[present])),
                     r_draw_1x2 = 100 * (t.x12[2] - f.x12[2]),
                     total = f.mh + f.ma, lam_h = f.mh, lam_a = f.ma))
    end
    return DataFrame(rows)
end

fits_pois = [c5_fit(t, :poisson, 0.0, R5_F25) for t in tb]
fits_best = [c5_fit(t, R5_BEST..., R5_F25) for t in tb]
res_pois = r5_residual_frame(tb, fits_pois)
res_best = r5_residual_frame(tb, fits_best)
res_pois.tid = tb_tid; res_best.tid = tb_tid
res_pois.date = tb_date; res_best.date = tb_date
const R5_RCOLS = [:r_00, :r_10, :r_01, :r_11, :r_22, :r_draw_cs, :r_draw_1x2]

"Mean residual [cluster CI] per cell for every group of `res`."
function r5_residual_table(res, gridlabel)
    rows = NamedTuple[]
    groups = vcat([("all", trues(nrow(res)))],
                  [(R5_LEAGUE[tid], res.tid .== tid) for tid in R5_TIDS],
                  [("total " * lab, lo .<= res.total .< hi) for (lo, hi, lab) in tbins])
    for (g, keep) in groups
        for c in R5_RCOLS
            m = c5_mean_ci(res[keep, c], res.date[keep]; B = R5_B, seed = R5_SEED)
            push!(rows, (; grid = gridlabel, group = g, cell = String(c)[3:end], n = m.n,
                         mean_pp = m.mean, lo = m.lo, hi = m.hi))
        end
    end
    return DataFrame(rows)
end
residuals = vcat(r5_residual_table(res_pois, "poisson"), r5_residual_table(res_best, String(R5_BEST[1])))
r5_show(unstack(select(residuals[residuals.group .== "all", :], :grid, :cell, :mean_pp), :cell, :mean_pp))
r5_show(unstack(select(residuals[residuals.grid .== "poisson", :], :group, :cell, :mean_pp), :cell, :mean_pp))
r5_write("residuals", residuals)

"ASCII heatmap of the mean residual (pp) over the 4×4 box plus the three any-other runners."
function r5_heatmap(ts, fits; title = "")
    S = zeros(19); N = zeros(19)
    for (t, f) in zip(ts, fits)
        q = c5_cs_q(t); p = c5_cs_on(t, f.cs)
        for k in 1:19
            isnan(q[k]) && continue
            S[k] += 100 * (q[k] - p[k]); N[k] += 1
        end
    end
    m = S ./ N
    shade(x) = abs(x) < 0.1 ? "  " : abs(x) < 0.25 ? (x > 0 ? "+░" : "-░") :
               abs(x) < 0.5 ? (x > 0 ? "+▒" : "-▒") : (x > 0 ? "+▓" : "-▓")
    io = IOBuffer()
    println(io, title)
    println(io, "            away 0      away 1      away 2      away 3")
    for h in 0:3
        print(io, "home ", h, "  ")
        for a in 0:3
            k = C5_CS_INDEX[Symbol("cs_$(h)$(a)")]
            print(io, @sprintf("  %+6.2f %s", m[k], shade(m[k])))
        end
        println(io)
    end
    println(io, @sprintf("any other: home win %+6.2f   draw %+6.2f   away win %+6.2f   (pp; n = %d)",
                         m[17], m[18], m[19], maximum(N)))
    return join(rstrip.(split(String(take!(io)), '\n'; keepempty = true)), '\n')
end
hm = r5_heatmap(tb, fits_pois; title = "Market CS − independent Poisson fitted to 1X2 + O/U 2.5 (mean pp, Scottish 54–57)") * "\n" *
     r5_heatmap(tb, fits_best; title = "Market CS − best grid $(R5_BEST) fitted to 1X2 + O/U 2.5 (mean pp)")
println(hm)
write(joinpath(R5_OUT, "r05_heatmap.txt"), hm)

# %%
# ===================================================================
# 5. Part B.4: our models' grids
# ===================================================================
#
# probs.jls (r01): m12_td target panel (56/57, 24/25–25/26) and grw_spfl_joint target + monitor
# (54/55) panels. For each model: its own CS shape (model CS − Poisson fitted to the model's own
# 1X2 + O/U 2.5) over all its fixtures, and on the market-CS sample the KL of the market's CS from
# (i) the model's CS, (ii) Poisson and (iii) the best grid, each fitted to the MODEL's 1X2 + O/U 2.5.

pc = deserialize(joinpath(R5_OUT, "probs.jls"))
r5_probs(id) = vcat([vcat(v, get(pc.probs_monitor, k, DataFrame())) for (k, v) in pc.probs_target if k[2] == id]...)
model_probs = Dict("m12_td" => r5_probs(R5_M12), "grw_spfl_joint" => r5_probs(R5_GRW))
ext = deserialize(joinpath(C4_OUT, "r04_ext_grw_spfl_joint.jls"))
fit_grw = Training.load_fit(Training.PostgresStorage(C4_EXPERIMENT), C4_RUN)
ext_ids = sort!(collect(Int.(ext.latents.match_ids)))
probs_2627 = cn_model_probs((; config = (; model = fit_grw.config.model), latents = ext.latents), ext_ids)
model_probs["grw_spfl_joint"] = vcat(model_probs["grw_spfl_joint"], probs_2627)

tb_index = Dict(t.match_id => i for (i, t) in enumerate(tb))
mg_rows = NamedTuple[]
shape_rows = NamedTuple[]
model_targets = Dict{String,Dict{Int,C5Target}}()
for (name, pr) in model_probs
    mt = c5_model_targets(pr)
    model_targets[name] = mt
    ts = sort(collect(values(mt)); by = t -> t.match_id)
    mfits = [c5_fit(t, :poisson, 0.0, R5_F25) for t in ts]
    own = r5_residual_frame(ts, mfits)                       # model CS − Poisson(model)
    own.date = [Date(meta[m].match_date) for m in own.match_id]
    for c in R5_RCOLS
        m = c5_mean_ci(own[!, c], own.date; B = R5_B, seed = R5_SEED)
        push!(shape_rows, (; source = name * " (own CS vs own Poisson, all fixtures)", cell = String(c)[3:end],
                           n = m.n, mean_pp = m.mean, lo = m.lo, hi = m.hi))
    end
    both = [t for t in ts if haskey(tb_index, t.match_id)]
    for t in both
        tm = tb[tb_index[t.match_id]]
        fp = c5_fit(t, :poisson, 0.0, R5_F25); fb = c5_fit(t, R5_BEST..., R5_F25)
        mcs = zeros(19); for (k, q) in zip(t.cs_idx, t.cs_q); mcs[k] = q; end
        push!(mg_rows, (; model = name, match_id = t.match_id, date = Date(meta[t.match_id].match_date),
                        kl_model_cs = c5_kl_cs(tm, mcs), kl_pois_on_model = c5_kl_cs(tm, fp.cs),
                        kl_best_on_model = c5_kl_cs(tm, fb.cs),
                        kl_pois_on_market = c5_kl_cs(tm, fits_pois[tb_index[t.match_id]].cs),
                        kl_best_on_market = c5_kl_cs(tm, fits_best[tb_index[t.match_id]].cs)))
    end
end
mkt_shape = res_pois
for c in R5_RCOLS
    m = c5_mean_ci(mkt_shape[!, c], mkt_shape.date; B = R5_B, seed = R5_SEED)
    push!(shape_rows, (; source = "market (CS vs Poisson fitted to its 1X2 + O/U 2.5)", cell = String(c)[3:end],
                       n = m.n, mean_pp = m.mean, lo = m.lo, hi = m.hi))
end
shapes = DataFrame(shape_rows)
r5_show(unstack(select(shapes, :source, :cell, :mean_pp), :cell, :mean_pp))
mg = DataFrame(mg_rows)
mg_sum = combine(groupby(mg, :model), nrow => :n,
                 [c => (x -> 1000 * mean(x)) => Symbol(c, "_x1000") for c in
                  (:kl_model_cs, :kl_pois_on_model, :kl_best_on_model, :kl_pois_on_market, :kl_best_on_market)]...)
r5_show(mg_sum)
r5_write("model_grid_shape", shapes)
r5_write("model_grid_kl", mg_sum)

# %%
# ===================================================================
# 6. Part B.5: thin markets against the grid implied by 1X2 + O/U 2.5
# ===================================================================

"Deviation table (pp): market − grid for BTTS, O/U 0.5/1.5/3.5/4.5 and CS cells."
function r5_consistency(targets, date_of; label)
    ts = sort(collect(values(targets)); by = t -> t.match_id)
    rows = NamedTuple[]
    fb = [c5_fit(t, R5_BEST..., R5_F25) for t in ts]
    fp = [c5_fit(t, :poisson, 0.0, R5_F25) for t in ts]
    specs = vcat([("BTTS yes", t -> t.btts, f -> c5_btts(f.P))],
                 [("O/U $(L) under", t -> get(t.under, L, NaN), f -> c5_under(f.P, L)) for L in (0.5, 1.5, 3.5, 4.5)],
                 [("CS $(lab)", t -> c5_cs_q(t)[C5_CS_INDEX[s]], nothing) for (lab, s) in R5_CELLS])
    for (name, qf, pf) in specs
        for (glab, fits) in (("best", fb), ("poisson", fp))
            d = Float64[]; dates = Date[]
            for (t, f) in zip(ts, fits)
                q = qf(t)
                isnan(q) && continue
                p = pf === nothing ? c5_cs_on(t, f.cs)[C5_CS_INDEX[Dict(R5_CELLS)[name[4:end]]]] : pf(f)
                push!(d, 100 * (q - p)); push!(dates, date_of(t.match_id))
            end
            isempty(d) && continue
            m = c5_mean_ci(d, dates; B = R5_B, seed = R5_SEED)
            push!(rows, (; source = label, market = name, grid = glab, n = length(d), bias_pp = m.mean,
                         lo = m.lo, hi = m.hi, mad_pp = mean(abs.(d)), p90_abs_pp = quantile(abs.(d), 0.9),
                         share_gt2pp = mean(abs.(d) .> 2)))
        end
    end
    return DataFrame(rows)
end
date_of(m) = Date(meta[m].match_date)
consistency = vcat(r5_consistency(tg_all, date_of; label = "archive LTP close 21/22–25/26"),
                   r5_consistency(tg_live, date_of; label = "26/27 order-book mid close"))
r5_show(consistency)
r5_write("consistency", consistency)

# %%
# ===================================================================
# 7. Reference: the English Premier League archive (Part B.2 only)
# ===================================================================

epl_odds, epl_matches = c5_other_league_odds(1)
epl_close, _ = cn_close_book((; betfair_odds = epl_odds))
epl_cs = c5_cs_pit(epl_odds; stale = 120.0)
epl_tg = c5_targets(epl_close; pcol = :p_close, cs = epl_cs)
r5_utc_date(x) = x isa Real ? Date(unix2datetime(x)) :
                 Date(DateTime(Data.TimeZones.astimezone(x, Data.TimeZones.TimeZone("UTC"))))
epl_date = Dict(Int(r.match_id) => r5_utc_date(r.start_timestamp) for r in eachrow(epl_matches))
etb = sort([t for t in values(epl_tg) if c5_has_cs(t) && haskey(epl_date, t.match_id)]; by = t -> t.match_id)
etb_date = [epl_date[t.match_id] for t in etb]
println("EPL: ", length(epl_tg), " matches with 1X2 + O/U 2.5; ", length(etb), " with CS")
eprof = Dict(k => c5_profile(etb, k, R5_F25) for k in R5_GRIDS)
grid_epl = r5_grid_table(eprof, nothing, trues(length(etb)), etb_date, nothing, nothing; label = "EPL")
r5_show(select(grid_epl, :sample, :kind, :n, :theta_cs, :theta_cs_lo, :theta_cs_hi, :kl_x1000, :dkl_vs_poisson_x1000, :dkl_lo, :dkl_hi))
efits_pois = [c5_fit(t, :poisson, 0.0, R5_F25) for t in etb]
eres = r5_residual_frame(etb, efits_pois)
eres.date = etb_date
epl_resid = DataFrame([(m -> (; cell = String(c)[3:end], n = m.n, mean_pp = m.mean, lo = m.lo, hi = m.hi))(
                           c5_mean_ci(eres[!, c], eres.date; B = R5_B, seed = R5_SEED)) for c in R5_RCOLS])
r5_show(epl_resid)
hm_epl = r5_heatmap(etb, efits_pois; title = "EPL: market CS − independent Poisson (mean pp)")
println(hm_epl)
write(joinpath(R5_OUT, "r05_heatmap.txt"), hm * "\n" * hm_epl)
r5_write("grid_fit_epl", grid_epl)
r5_write("residuals_epl", epl_resid)

# %%
# ===================================================================
# 8. Part A.1: invert every close (and every model price) into (λ_home, λ_away)
# ===================================================================
#
# Grid: the best grid of §3 (R5_BEST), fitted to 1X2 + O/U 1.5/2.5/3.5 (those present); λ = the
# grid's goal means. The same inversion is applied to the models' own probabilities, so model and
# market λ are on the same footing. Independent-Poisson λ from 1X2 + O/U 2.5 kept for comparison.

"(match_id, λ_home, λ_away) for every target, inverted with `kind, θ` over `lines`."
function r5_invert(targets; kind = R5_BEST[1], θ = R5_BEST[2], lines = R5_FALL)
    ts = sort(collect(values(targets)); by = t -> t.match_id)
    out = Vector{NamedTuple}(undef, length(ts))
    Threads.@threads for i in eachindex(ts)
        f = c5_fit(ts[i], kind, θ, lines)
        p = c5_fit(ts[i], :poisson, 0.0, R5_F25)
        out[i] = (; match_id = ts[i].match_id, lam_h = f.mh, lam_a = f.ma, lam_h_pois = p.mh, lam_a_pois = p.ma,
                  fit_loss = f.loss)
    end
    return DataFrame(out)
end

mkt_lam = vcat(insertcols!(r5_invert(tg_all), :source => "archive"),
               insertcols!(r5_invert(tg_live), :source => "live 26/27"))
mkt_lam.tid = [tid_of(m) for m in mkt_lam.match_id]
mkt_lam.season = [String(meta[m].season) for m in mkt_lam.match_id]
mkt_lam.date = [Date(meta[m].match_date) for m in mkt_lam.match_id]
mkt_lam.home = [String(meta[m].home_team) for m in mkt_lam.match_id]
mkt_lam.away = [String(meta[m].away_team) for m in mkt_lam.match_id]
filter!(r -> r.tid in R5_TIDS, mkt_lam)
@printf("market λ: %d matches; best-grid vs Poisson: mean Δ total %.3f, sd Δ log-ratio %.3f\n", nrow(mkt_lam),
        mean(mkt_lam.lam_h .+ mkt_lam.lam_a .- mkt_lam.lam_h_pois .- mkt_lam.lam_a_pois),
        std(log.(mkt_lam.lam_h ./ mkt_lam.lam_a) .- log.(mkt_lam.lam_h_pois ./ mkt_lam.lam_a_pois)))
model_lam = Dict(name => r5_invert(mt) for (name, mt) in model_targets)
r5_write("market_lambda", mkt_lam)

# %%
# ===================================================================
# 9. Part A.2: market ratings per league-season (random walk over each team's matches)
# ===================================================================

"True league match number of each (team, match) — the team's k-th league match of the season."
function r5_match_numbers(matches)
    out = Dict{Tuple{String,Int},Int}()
    m = matches[in.(matches.tournament_id, Ref(R5_TIDS)), :]
    for g in groupby(sort(m, [:match_date, :match_id]), [:tournament_id, :season])
        k = Dict{String,Int}()
        for r in eachrow(g), t in (String(r.home_team), String(r.away_team))
            k[t] = get(k, t, 0) + 1
            out[(t, Int(r.match_id))] = k[t]
        end
    end
    return out
end
mno = r5_match_numbers(ds.matches)

lsg = Dict{Tuple{Int,String},DataFrame}()
for g in groupby(mkt_lam, [:tid, :season])
    o = sort(DataFrame(match_id = g.match_id, date = g.date, home = g.home, away = g.away,
                       yh = log.(g.lam_h), ya = log.(g.lam_a)), [:date, :match_id])
    lsg[(first(g.tid), first(g.season))] = o
end
const R5_W = [1.0, 4.0, 16.0, 64.0, 256.0, 1024.0]
ks = collect(keys(lsg))
one_step = zeros(length(ks), length(R5_W)); one_n = zeros(Int, length(ks))
Threads.@threads for i in eachindex(ks)
    for (j, w) in enumerate(R5_W)
        e = c5_one_step(lsg[ks[i]]; w)
        one_step[i, j] = sum(e); one_n[i] = length(e)
    end
end
wsel = DataFrame(w = R5_W, tau_over_sigma = 1 ./ sqrt.(R5_W),
                 one_step_rmse = sqrt.(vec(sum(one_step; dims = 1)) ./ sum(one_n)))
r5_show(wsel)
R5_WSEL = wsel.w[argmin(wsel.one_step_rmse)]
println("chosen w = ", R5_WSEL, " (τ/σ = ", round(1 / sqrt(R5_WSEL), digits = 3), ")")
r5_write("rating_rw_choice", wsel)

rfits = Dict(k => c5_ratings(o; w = R5_WSEL) for (k, o) in lsg)
imp = vcat([insertcols!(c5_implied(lsg[k], rfits[k]), :tid => k[1], :season => k[2]) for k in ks]...)
imp.mno = [mno[(r.team, r.match_id)] for r in eachrow(imp)]
imp.net_s = imp.att_s .+ imp.def_s
fit_rmse = sqrt(mean(vcat([begin
    o, f = lsg[k], rfits[k]
    vcat(o.yh .- (f.μ .+ f.home .+ f.att_h .- f.def_a), o.ya .- (f.μ .+ f.att_a .- f.def_h))
end for k in ks]...) .^ 2))
@printf("smoothed ratings: log-λ RMSE %.4f over %d team-matches\n", fit_rmse, nrow(imp))
tiers = Data.club_season_tiers(ds.matches)
function r5_status(team, season)
    now_t = get(tiers, (team, season), 0)
    prev = get(tiers, (team, c5_prev_season(season)), 0)
    prev == 0 && return "new to SPFL"
    C5_TIER[prev] > C5_TIER[now_t] && return "promoted"
    C5_TIER[prev] < C5_TIER[now_t] && return "relegated"
    return "stayed"
end
imp.status = [r5_status(r.team, r.season) for r in eachrow(imp)]
r5_write("rating_paths", select(imp, :tid, :season, :team, :opp, :venue, :date, :match_id, :mno,
                                :lam_for, :lam_against, :att, :def, :net, :att_s, :def_s, :net_s, :status))

# %%
# ===================================================================
# 10. Part A.3: our models' λ against the market's
# ===================================================================
#
# Per team-match: Δ = model − market for log λ_for, log λ_against and the opponent-adjusted
# net rating (both measured with the MARKET's smoothed opponent rating). By status, stage (league
# match 1–5 vs later) and team.

cmp_rows = DataFrame[]
for (name, ml) in model_lam
    mlam = Dict(r.match_id => (r.lam_h, r.lam_a) for r in eachrow(ml))
    for k in ks
        o = lsg[k]
        any(m -> haskey(mlam, m), o.match_id) || continue
        om = copy(o)
        om.yh = [haskey(mlam, m) ? log(mlam[m][1]) : NaN for m in o.match_id]
        om.ya = [haskey(mlam, m) ? log(mlam[m][2]) : NaN for m in o.match_id]
        a = c5_implied(o, rfits[k]); b = c5_implied(om, rfits[k])
        d = DataFrame(model = name, tid = k[1], season = k[2], team = a.team, match_id = a.match_id, date = a.date,
                      d_log_for = log.(b.lam_for) .- log.(a.lam_for), d_log_against = log.(b.lam_against) .- log.(a.lam_against),
                      d_net = b.net .- a.net, mkt_net = a.net, model_net = b.net)
        push!(cmp_rows, d[.!isnan.(d.d_net), :])
    end
end
cmp = vcat(cmp_rows...)
cmp.mno = [mno[(r.team, r.match_id)] for r in eachrow(cmp)]
cmp.stage = ifelse.(cmp.mno .<= 5, "matches 1–5", "match 6+")
cmp.status = [r5_status(r.team, r.season) for r in eachrow(cmp)]
cmp.league = [R5_LEAGUE[t] for t in cmp.tid]
cmp_tab = NamedTuple[]
for g in groupby(cmp, [:model, :status, :stage]; sort = true)
    m = c5_mean_ci(g.d_net, g.date; B = R5_B, seed = R5_SEED)
    push!(cmp_tab, (; model = first(g.model), status = first(g.status), stage = first(g.stage), n = nrow(g),
                    d_net = m.mean, lo = m.lo, hi = m.hi, mean_abs_d_net = mean(abs.(g.d_net)),
                    d_log_for = mean(g.d_log_for), d_log_against = mean(g.d_log_against)))
end
cmp_tab = DataFrame(cmp_tab)
r5_show(cmp_tab)
cmp_all = combine(groupby(cmp, [:model, :league]), nrow => :n, :d_net => (x -> mean(abs.(x))) => :mean_abs_d_net,
                  :d_log_for => (x -> sqrt(mean(abs2, x))) => :rms_d_log_for)
r5_show(cmp_all)
cmp_team = combine(groupby(cmp, [:model, :team, :season]), nrow => :n, :d_net => mean => :d_net,
                   :status => first => :status, :league => first => :league)
filter!(r -> r.n >= 8, cmp_team)
sort!(cmp_team, :d_net; by = abs, rev = true)
r5_show(first(cmp_team, 20))
r5_write("model_vs_market", cmp_tab)
r5_write("model_vs_market_league", cmp_all)
r5_write("model_vs_market_team", cmp_team)

# %%
# ===================================================================
# 11. Part A.4: what explains the market's season-start ratings; how fast it re-rates movers
# ===================================================================

ts_rows = NamedTuple[]
for g in groupby(imp, [:tid, :season, :team])
    s = g[g.mno .<= 3, :]; w = g[6 .<= g.mno .<= 10, :]; l = g[g.mno .>= 16, :]
    push!(ts_rows, (; tid = first(g.tid), season = first(g.season), team = first(g.team), status = first(g.status),
                    n_obs = nrow(g), start = nrow(s) >= 2 ? mean(s.net) : NaN,
                    wk6_10 = nrow(w) >= 3 ? mean(w.net) : NaN,
                    settled = nrow(l) >= 5 ? mean(l.net) : NaN))
end
tsd = DataFrame(ts_rows)
r5_centre(x) = (v = filter(!isnan, x); isempty(v) ? fill(NaN, length(x)) : x .- mean(v))
for c in (:start, :wk6_10, :settled)       # centre within league-season
    transform!(groupby(tsd, [:tid, :season]), c => r5_centre => Symbol(c, "_c"))
end
tab = c5_tables(ds.matches)
tabd = Dict((r.team, r.season) => r for r in eachrow(tab))
settled_d = Dict((r.team, r.season) => r.settled_c for r in eachrow(tsd))
tsd.tier = [C5_TIER[t] for t in tsd.tid]
pr(r, f) = (x = get(tabd, (r.team, c5_prev_season(r.season)), nothing); x === nothing ? NaN : Float64(x[f]))
tsd.prev_tier = [pr(r, :tier) for r in eachrow(tsd)]
tsd.prev_gdpg = [pr(r, :gdpg) for r in eachrow(tsd)]
tsd.prev_rank = [pr(r, :rank_frac) for r in eachrow(tsd)]
tsd.prev_settled = [get(settled_d, (r.team, c5_prev_season(r.season)), NaN) for r in eachrow(tsd)]
tsd.promoted = Float64.(tsd.status .== "promoted")
tsd.relegated = Float64.(tsd.status .== "relegated")
tsd.new_spfl = Float64.(tsd.status .== "new to SPFL")
tm = select(imp, :team, :season, :match_id, :mno => :j)
lf = c5_team_lineup_features(ds, tm)
tsd = leftjoin(tsd, lf, on = [:team, :season])
tsd.log_wealth = coalesce.(tsd.log_wealth, NaN); tsd.turnover = coalesce.(tsd.turnover, NaN)
transform!(groupby(tsd, [:tid, :season]), :log_wealth => r5_centre => :log_wealth_c)
r5_write("team_seasons", tsd)

"OLS of `y` on `feats` over rows where all are finite; features with a stayed×/moved× split as given."
function r5_reg(df, y, feats; label)
    X0 = hcat(ones(nrow(df)), [Float64.(f[2](df)) for f in feats]...)
    yy = Float64.(df[!, y])
    ok = isfinite.(yy) .& vec(all(isfinite.(X0); dims = 2))
    names = vcat(["intercept"], [f[1] for f in feats])
    keepc = [j == 1 || std(X0[ok, j]) > 0 for j in axes(X0, 2)]      # drop columns constant on the sample
    r = c5_ols(yy[ok], X0[ok, keepc], names[keepc])
    return (; label, y = String(y), r.r2, r.n, r.rmse, coef = insertcols!(r.coef, 1, :model => label))
end
nz(x) = coalesce.(ifelse.(isnan.(x), 0.0, x), 0.0)
st(df) = Float64.(df.status .== "stayed")
F_TIER = [("promoted", d -> d.promoted), ("relegated", d -> d.relegated), ("new to SPFL", d -> d.new_spfl)]
F_PREV = [("prev GD/game × stayed", d -> nz(d.prev_gdpg) .* st(d)),
          ("prev GD/game × promoted", d -> nz(d.prev_gdpg) .* d.promoted),
          ("prev GD/game × relegated", d -> nz(d.prev_gdpg) .* d.relegated),
          ("prev rank (0 top–1 bottom) × stayed", d -> nz(d.prev_rank) .* st(d))]
F_MKT = [("prev market rating × stayed", d -> nz(d.prev_settled) .* st(d)),
         ("prev market rating × moved", d -> nz(d.prev_settled) .* (1 .- st(d)))]
F_LU = [("log wealth (league-centred)", d -> d.log_wealth_c), ("squad turnover", d -> d.turnover)]
sub_mkt(d) = d[.!isnan.(d.prev_settled) .| (d.status .== "new to SPFL"), :]
regs = [r5_reg(tsd, :start_c, F_TIER; label = "A tier change"),
        r5_reg(tsd, :start_c, vcat(F_TIER, F_PREV); label = "B + last season's table"),
        r5_reg(sub_mkt(tsd), :start_c, vcat(F_TIER, F_PREV); label = "B on C's sample"),
        r5_reg(sub_mkt(tsd), :start_c, vcat(F_TIER, F_PREV, F_MKT); label = "C + last season's market rating"),
        r5_reg(tsd[in.(tsd.tid, Ref((56, 57))), :], :start_c, vcat(F_TIER, F_PREV); label = "B, L1/L2 with lineup data"),
        r5_reg(tsd[in.(tsd.tid, Ref((56, 57))) .& isfinite.(tsd.log_wealth_c) .& isfinite.(tsd.turnover), :],
               :start_c, vcat(F_TIER, F_PREV); label = "B, L1/L2 matched wealth sample"),
        r5_reg(tsd[in.(tsd.tid, Ref((56, 57))), :], :start_c, vcat(F_TIER, F_PREV, F_LU); label = "D + wealth, turnover (L1/L2)"),
        r5_reg(tsd, :settled_c, vcat(F_TIER, F_PREV); label = "B for the SETTLED rating"),
        r5_reg(transform(tsd, [:wk6_10, :start] => ((a, b) -> a .- b) => :chg), :chg, vcat(F_TIER, F_PREV);
               label = "B for the change, matches 1–3 → 6–10")]
reg_sum = DataFrame([(; model = r.label, y = r.y, n = r.n, r2 = r.r2, rmse = r.rmse) for r in regs])
reg_coef = vcat([r.coef for r in regs]...)
r5_show(reg_sum)
r5_show(reg_coef)
r5_write("start_regression", reg_sum)
r5_write("start_regression_coef", reg_coef)

# Convergence: gap between the per-match implied rating and the club's settled level (matches 16+)
conv_rows = NamedTuple[]
setl = Dict((r.team, r.season) => r.settled for r in eachrow(tsd) if isfinite(r.settled))
imp.gap = [get(setl, (r.team, r.season), NaN) for r in eachrow(imp)]
imp.gap = imp.net .- imp.gap
imp.ts_id = imp.team .* "|" .* imp.season
for grp in ("promoted", "relegated", "new to SPFL", "stayed"), j in 1:15
    g = imp[(imp.status .== grp) .& (imp.mno .== j) .& isfinite.(imp.gap), :]
    nrow(g) < 3 && continue
    m = c5_mean_ci(g.gap, g.ts_id; B = R5_B, seed = R5_SEED)
    push!(conv_rows, (; status = grp, mno = j, n = m.n, gap = m.mean, lo = m.lo, hi = m.hi,
                      mean_abs_gap = mean(abs.(g.gap))))
end
convergence = DataFrame(conv_rows)
r5_show(unstack(select(convergence, :status, :mno, :gap), :status, :gap))
r5_write("convergence", convergence)
# the same for the models: model implied rating − MARKET settled level
cmp.gap = [get(setl, (r.team, r.season), NaN) for r in eachrow(cmp)]
cmp.gap = cmp.model_net .- cmp.gap
cmp.mkt_gap = cmp.mkt_net .- [get(setl, (r.team, r.season), NaN) for r in eachrow(cmp)]
mconv = combine(groupby(cmp[isfinite.(cmp.gap) .& (cmp.mno .<= 15), :], [:model, :status, :stage]), nrow => :n,
                :gap => mean => :model_gap, :mkt_gap => mean => :market_gap)
r5_show(mconv)
r5_write("convergence_models", mconv)

# %%
# ===================================================================
# 12. Worked examples: 26/27's four transition clubs, market vs grw_spfl_joint
# ===================================================================

k2627 = [k for k in ks if k[2] == "26/27"]
wex = NamedTuple[]
glam = Dict(r.match_id => (r.lam_h, r.lam_a) for r in eachrow(model_lam["grw_spfl_joint"]))
for club in R5_TRANSITION_2627, k in k2627
    o = lsg[k]; f = rfits[k]
    a = c5_implied(o, f)
    om = copy(o)
    om.yh = [haskey(glam, m) ? log(glam[m][1]) : NaN for m in o.match_id]
    om.ya = [haskey(glam, m) ? log(glam[m][2]) : NaN for m in o.match_id]
    b = c5_implied(om, f)
    for i in findall(a.team .== club)
        push!(wex, (; club, league = R5_LEAGUE[k[1]], date = a.date[i], mno = mno[(club, a.match_id[i])],
                    opp = a.opp[i], venue = a.venue[i], status = r5_status(club, "26/27"),
                    mkt_lam_for = a.lam_for[i], mkt_lam_against = a.lam_against[i],
                    grw_lam_for = b.lam_for[i], grw_lam_against = b.lam_against[i],
                    mkt_net = a.net[i], grw_net = b.net[i], mkt_net_smoothed = a.att_s[i] + a.def_s[i]))
    end
end
worked = sort(DataFrame(wex), [:club, :date])
r5_show(worked)
r5_write("transition_2627", worked)
# historical analogues: mean start and settled (league-centred) by status, with the prior-season tier
ana = combine(groupby(tsd[tsd.season .!= "26/27", :], :status), nrow => :n,
              :start_c => (x -> mean(filter(isfinite, x))) => :start_c,
              :settled_c => (x -> mean(filter(isfinite, x))) => :settled_c,
              [:start, :settled] => ((a, b) -> mean(filter(isfinite, b .- a))) => :settled_minus_start)
r5_show(ana)
r5_write("transition_analogues", ana)
println("DONE r05_market_structure ", now())
