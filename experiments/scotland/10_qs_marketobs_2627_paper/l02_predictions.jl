module QSPaper2627Predictions

# Phase 1 (TODO 045): per-fixture held-out probabilities of the in-memory extensions, the m12
# DB-fold check on the shared fixtures, and 1X2 log loss against the order-book mid close.
# Read-only: serialised extension files, the pinned inputs and m12's saved fit by UUID.
# Comments only below; the static checker scans every non-comment line for write verbs.

import BayesianFootball as BF
import CSV
import DataFrames as DF
import Dates
import Random
import Serialization
import Statistics
import UUIDs

const Q = Main.QSPaper2627
const S = Main.QSPaper2627Sampling
const CAL = BF.Calibration
const P = BF.Predictions
const MARKETS = vcat([BF.Data.Market1X2()], [BF.Data.MarketOverUnder(l) for l in (0.5, 1.5, 2.5, 3.5)],
                     [BF.Data.MarketBTTS()])
const SELS = [:home, :draw, :away, :over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
              :over_35, :under_35, :btts_yes, :btts_no]
const BOOT_B, BOOT_SEED = 2_000, 20260928    # CLV napkin cn_summary defaults

load_extension(label) = Serialization.deserialize(joinpath(Q.OUT_DIR, "ext_$(label)_2627.jls"))

# Posterior-mean market probabilities through the production double-Poisson grid (QSMarketArm
# probabilities, widened to every canonical line).
function probabilities(latents, label; fold_of = Dict{Int,Int}(), cutoff_of = Dict{Int,Dates.Date}())
    rows = NamedTuple[]
    for (i, id) in enumerate(latents.match_ids)
        grid = P.compute_score_grid(latents, i)
        p = Dict{Symbol,Float64}()
        for m in MARKETS, (k, v) in P.price_market(grid, m)
            p[k] = Statistics.mean(v)
        end
        push!(rows, (; model = label, match_id = Int(id), fold = get(fold_of, Int(id), missing),
                     train_cutoff = get(cutoff_of, Int(id), missing),
                     lambda_home = Statistics.mean(view(latents.λ_home, i, :)),
                     lambda_away = Statistics.mean(view(latents.λ_away, i, :)),
                     (Symbol("p_", s) => p[s] for s in SELS)...))
    end
    return DF.sort!(DF.DataFrame(rows), :match_id)
end

function fold_maps(ext)
    fold_of = Dict{Int,Int}()
    for (p, l) in ext.per_fold_latents, id in l.match_ids
        fold_of[Int(id)] = p
    end
    cut = Dict(r.fold => r.train_last for r in eachrow(ext.filtration))
    return fold_of, Dict(id => cut[f] for (id, f) in fold_of)
end

# m12 DB folds vs the in-memory extension on the fixtures both hold (a check only, ruling 4).
function m12_db_check(ext, mem_probs, fx)
    fit = BF.Training.load_fit(BF.Training.PostgresStorage(Q.M12.ns), UUIDs.UUID(Q.M12.run))
    shared = intersect(Set(Int.(fit.latents.match_ids)), Set(fx.match_id))
    db = CAL.restrict_latents(fit.latents, shared)
    dbp = probabilities(db, "m12_db")
    j = DF.innerjoin(DF.select(mem_probs, :match_id, :p_home => :mem_home, :p_draw => :mem_draw, :p_away => :mem_away,
                               :lambda_home => :mem_lh, :lambda_away => :mem_la),
                     DF.select(dbp, :match_id, :p_home => :db_home, :p_draw => :db_draw, :p_away => :db_away,
                               :lambda_home => :db_lh, :lambda_away => :db_la), on = :match_id)
    shapes = DF.DataFrame([(; fold = f.fold, mem_params = size(f.chain, 2), mem_draws = size(f.chain, 1) * size(f.chain, 3),
                             db_params = size(only(filter(d -> d.fold == f.fold, fit.folds)).chain, 2),
                             db_draws = size(only(filter(d -> d.fold == f.fold, fit.folds)).chain, 1) *
                                        size(only(filter(d -> d.fold == f.fold, fit.folds)).chain, 3)) for f in ext.folds])
    d = abs.(vcat(j.mem_home .- j.db_home, j.mem_draw .- j.db_draw, j.mem_away .- j.db_away))
    summary = (; n_shared = DF.nrow(j), mean_abs_dp_1x2 = Statistics.mean(d), max_abs_dp_1x2 = maximum(d),
               cor_p_home = Statistics.cor(j.mem_home, j.db_home),
               mean_abs_dlog_lambda = Statistics.mean(abs.(vcat(log.(j.mem_lh ./ j.db_lh), log.(j.mem_la ./ j.db_la)))))
    println("M12_DB_CHECK ", summary)
    for r in eachrow(shapes)
        println("M12_CHAIN_SHAPE fold=", r.fold, " mem=", r.mem_params, "x", r.mem_draws, " db=", r.db_params, "x", r.db_draws)
    end
    return j, shapes, DF.DataFrame([summary])
end

# Per-selection binary log loss and 3-way log loss for 1X2.
binll(p, y) = -(y * log(p) + (1 - y) * log(1 - p))

function logloss_rows(fx, probs, closebook)
    close1 = filter(r -> r.market_name == "1X2", closebook)
    cp = Dict((Int(r.match_id), Symbol(r.selection)) => Float64(r.prob_fair_close) for r in eachrow(close1))
    rows = NamedTuple[]
    for r in eachrow(fx)
        y = r.home_score > r.away_score ? :home : r.home_score == r.away_score ? :draw : :away
        has_close = haskey(cp, (r.match_id, :home))
        for (label, pr) in probs
            row = only(filter(x -> x.match_id == r.match_id, eachrow(pr)))
            ps = Dict(:home => row.p_home, :draw => row.p_draw, :away => row.p_away)
            push!(rows, (; match_id = r.match_id, date = r.match_date, transition = r.transition, has_close,
                         source = label, sel_ll = Statistics.mean(binll(ps[s], s == y ? 1.0 : 0.0) for s in (:home, :draw, :away)),
                         ll3 = -log(ps[y])))
        end
        if has_close
            ps = Dict(s => cp[(r.match_id, s)] for s in (:home, :draw, :away))
            push!(rows, (; match_id = r.match_id, date = r.match_date, transition = r.transition, has_close,
                         source = "mid_close", sel_ll = Statistics.mean(binll(ps[s], s == y ? 1.0 : 0.0) for s in (:home, :draw, :away)),
                         ll3 = -log(ps[y])))
        end
    end
    return DF.DataFrame(rows)
end

# Slate-clustered bootstrap (match days resampled with replacement), as cn_summary.
function slate_ci(dates, values; B = BOOT_B, seed = BOOT_SEED)
    days = sort!(unique(dates))
    idx = Dict(d => i for (i, d) in enumerate(days))
    S_ = length(days)
    s_sum, s_n = zeros(S_), zeros(S_)
    for (d, v) in zip(dates, values)
        s_sum[idx[d]] += v
        s_n[idx[d]] += 1
    end
    rng = Random.MersenneTwister(seed)
    reps = zeros(B)
    w = zeros(S_)
    for b in 1:B
        fill!(w, 0.0)
        for _ in 1:S_
            w[rand(rng, 1:S_)] += 1.0
        end
        reps[b] = sum(w .* s_sum) / sum(w .* s_n)
    end
    return (; point = sum(s_sum) / sum(s_n), lo95 = Statistics.quantile(reps, 0.025),
            hi95 = Statistics.quantile(reps, 0.975), n = length(values), n_days = S_)
end

# Log loss by group on matched fixtures; model minus close with slate-clustered 95% intervals.
function logloss_table(ll)
    out = NamedTuple[]
    for (panel, keep) in (("all_79", r -> true), ("close_60", r -> r.has_close))
        for (group, gsel) in (("all", r -> true), ("transition", r -> r.transition), ("other", r -> !r.transition))
            sub = filter(r -> keep(r) && gsel(r), ll)
            for src in sort!(unique(sub.source))
                x = filter(r -> r.source == src, sub)
                DF.nrow(x) == 0 && continue
                a = slate_ci(x.date, x.sel_ll)
                b = slate_ci(x.date, x.ll3)
                diff = (; d_sel_ll = missing, d_lo95 = missing, d_hi95 = missing)
                if panel == "close_60" && src != "mid_close"
                    c = Dict(r.match_id => r.sel_ll for r in eachrow(filter(r -> r.source == "mid_close", sub)))
                    dd = slate_ci(x.date, [r.sel_ll - c[r.match_id] for r in eachrow(x)])
                    diff = (; d_sel_ll = dd.point, d_lo95 = dd.lo95, d_hi95 = dd.hi95)
                end
                push!(out, (; panel, group, source = src, n_fixtures = DF.nrow(x), n_days = a.n_days,
                            sel_ll = a.point, sel_ll_lo95 = a.lo95, sel_ll_hi95 = a.hi95, ll3 = b.point,
                            diff..., ))
            end
        end
    end
    return DF.DataFrame(out)
end

end # module
