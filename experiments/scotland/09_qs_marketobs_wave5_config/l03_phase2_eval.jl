# Phase 2 (included into QSWave5): evaluation after the FREEZE release.
# Scores the frozen config, the fallback, m12 canonical and every Phase-1 basket at its own lambda
# on both halves (each half from bankroll 1.0), T-6h rows (ruling 9), bootstrap columns, paired d
# vs m12, per-direction tables, and the ruling-8 decision. The only caller of release!.

import Random

const T6H = P3.SNAPSHOTS[2]
const PHASE1_DIR = joinpath(@__DIR__, "results", "phase1")
const PHASE0_SEASON_CSV = joinpath(@__DIR__, "results", "phase0", "phase0_season_mdd.csv")
const FROZEN_BASKET, FROZEN_LAMBDA, FALLBACK_LAMBDA = "B1_canonical", 12.0, 23.0
const HALF_SEASON = Dict("selection" => "24/25", "evaluation" => "25/26")

# Wave 4's 8-week moving-block bootstrap (P3.block_bootstrap) with wrap-around block starts: the
# circular-block column wave 4 ran as a numpy sensitivity (x04_phase4_tables.py). Copied because
# the start rule changes; the week index, seed and quantiles are as P3.block_bootstrap.
function block_bootstrap_circular(dates, values, season_of; B = P3.B_REPS, block = P3.BLOCK_WEEKS,
                                  seed = P3.BOOT_SEED)
    weeks = Dict{Tuple{String,Dates.Date},Vector{Int}}()
    for (i, d) in enumerate(dates)
        push!(get!(weeks, (season_of[d], Dates.firstdayofweek(d)), Int[]), i)
    end
    seasons = Dict{String,Vector{Vector{Int}}}()
    for key in sort!(collect(keys(weeks)))
        push!(get!(seasons, key[1], Vector{Int}[]), weeks[key])
    end
    rng = Random.Xoshiro(seed)
    stats = zeros(B)
    for b in 1:B
        total, n = 0.0, 0
        for season in sort!(collect(keys(seasons)))
            ws = seasons[season]
            W = length(ws)
            L = min(block, W)
            chosen = Int[]
            while length(chosen) < W
                s = rand(rng, 0:(W - 1))
                append!(chosen, [mod(s + k, W) + 1 for k in 0:(L - 1)])
            end
            for w in chosen[1:W], i in ws[w]
                total += values[i]
                n += 1
            end
        end
        stats[b] = total / n
    end
    return (; point = Statistics.mean(values), boot_mean = Statistics.mean(stats),
            lo90 = Statistics.quantile(stats, 0.05), hi90 = Statistics.quantile(stats, 0.95),
            n_days = length(values), n_weeks = length(weeks))
end

# Phase-1 baskets and their matched lambdas, rebuilt from the committed selection table.
function phase1_baskets(dirs)
    sel = CSV.read(joinpath(PHASE1_DIR, "phase1_selection.csv"), DF.DataFrame)
    lab2key = Dict(direction_label(k) => k for k in dirs)
    rows = NamedTuple[]
    for r in eachrow(sel)
        r.feasible || continue
        bk = basket(r.basket, [lab2key[l] for l in split(r.directions, ",")])
        push!(rows, (; bk, lambda = Float64(r.lambda), g1 = Float64(r.g), selected = r.selected))
    end
    fr = only(filter(r -> r.selected, rows))
    fr.bk.name == FROZEN_BASKET && fr.lambda == FROZEN_LAMBDA || error("phase1_selection.csv disagrees with FREEZE.md")
    return rows
end

function half_cells(acc, inputs, snap, half::Symbol, baskets, target)
    ctx = inputs.ctx
    h = String(half)
    close = P3.close_index(ctx.ds, half_ids(ctx, half))
    qb, qr = half_books(inputs, "qs_marketobs", snap.label, half)
    mb, mr = half_books(inputs, "m12", snap.label, half)
    days = day_problems(inputs.models["qs_marketobs"], qb)
    all(d -> half_of(d.slate.window) === half, days) || error("SEAL: a day lies outside the $h half")
    println("HALF_INPUTS snapshot=", snap.label, " half=", h, " qs_books=", length(qb), " m12_books=", length(mb),
            " days=", length(days))
    flush(stdout)
    t25 = snap.label == T25.label
    score = (cell, role, model, bname, lam, result, books, report, diag) -> begin
        key = (; snapshot = snap.label, half = h, cell, role, model, basket = bname, lambda = lam)
        t = P3.cell_tables(key, result, books, report, close, ctx.season_of)
        cr = curve_row((;), result, diag)
        s = (; t.summary..., mean_exposure = cr.mean_exposure, max_exposure = cr.max_exposure,
             budget_share = cr.budget_share, cap_share = cr.cap_share, risk_active_share = cr.risk_active_share)
        push!(acc.summary, s)
        append!(acc.markets, t.markets)
        append!(acc.daily, t.daily)
        push!(acc.bets, t.bets)
        append!(acc.directions, direction_table(key, result, books, close; half = half))
        diag === nothing || append!(acc.diag, [(; key..., r...) for r in eachrow(diag)])
        println("CELL snapshot=", snap.label, " half=", h, " cell=", cell, " wealth=", s.final_wealth,
                " g=", s.mean_daily_log_growth, " mdd=", s.mdd_pct, " bets=", s.n_bets)
        flush(stdout)
        return s
    end

    canon = MD.canonical_scottish_lower_policy()
    m = score("m12_canonical_l23", "benchmark", "m12", "canonical", 23.0, simulate_books(canon, mb, mr), mb, mr, nothing)
    half === :selection && t25 && m.mdd_pct != target && error("m12 canonical 24/25 MDD no longer equals the target")

    for r in (t25 ? baskets : filter(r -> r.selected, baskets))
        slates, diag = solve_days(days, s -> in_basket(r.bk, s))
        DF.nrow(diag) == length(days) || error("basket $(r.bk.name) dropped a day")
        books = PF.MatchBook[b for sl in slates for b in sl.books]
        cell, role = r.selected ? ("frozen_B1_l12", "frozen") : ("cand_$(r.bk.name)_l$(Int(r.lambda))", "candidate")
        s = score(cell, role, "qs_marketobs", r.bk.name, r.lambda,
                  simulate_slates(basket_policy(r.bk.trust, r.lambda), slates, qr), books, qr, diag)
        if half === :selection && t25
            s.mean_daily_log_growth == r.g1 || error("$(r.bk.name): 24/25 g differs from Phase 1")
        end
        if r.selected
            score("fallback_B1_l23", "fallback", "qs_marketobs", r.bk.name, FALLBACK_LAMBDA,
                  simulate_slates(basket_policy(r.bk.trust, FALLBACK_LAMBDA), slates, qr), books, qr, diag)
        end
    end

    if t25
        # Wave-4 scenario Kelly (all legs, canonical overlay): reproduces Phase 0's per-season numbers.
        us, ud = solve_days(days, all_legs)
        ubooks = PF.MatchBook[b for sl in us for b in sl.books]
        w = score("wave4_sk_all_legs_l23", "check", "qs_marketobs", "all_legs_canonical_overlay", 23.0,
                  simulate_slates(canon, us, qr), ubooks, qr, ud)
        ref = CSV.read(PHASE0_SEASON_CSV, DF.DataFrame)
        for (s, model, policy) in ((m, "m12", "canonical"), (w, "qs_marketobs", "scenario_kelly"))
            p0 = only(filter(x -> x.model == model && x.policy == policy && x.season == HALF_SEASON[h], ref))
            ok = isapprox(s.final_wealth, p0.wealth; rtol = 1e-9) && abs(s.mdd_pct - p0.mdd_pct) < 1e-7 &&
                 s.n_bets == p0.bets
            println("PHASE0_SEASON_CHECK half=", h, " model=", model, " wealth=", s.final_wealth, " ref=", p0.wealth,
                    " mdd=", s.mdd_pct, " ref=", p0.mdd_pct, " pass=", ok)
            ok || error("half run differs from the Phase 0 per-season numbers")
        end
    end
    return nothing
end

function phase2_uncertainty(daily::DF.DataFrame, season_of)
    single, paired = NamedTuple[], NamedTuple[]
    for g in DF.groupby(daily, [:snapshot, :half, :cell]; sort = true)
        nc = P3.block_bootstrap(g.date, g.log_growth, season_of)
        ci = block_bootstrap_circular(g.date, g.log_growth, season_of)
        push!(single, (; snapshot = first(g.snapshot), half = first(g.half), cell = first(g.cell), g = nc.point,
                       g_lo90 = nc.lo90, g_hi90 = nc.hi90, g_lo90_circ = ci.lo90, g_hi90_circ = ci.hi90,
                       n_days = nc.n_days, n_weeks = nc.n_weeks))
    end
    for g in DF.groupby(daily, [:snapshot, :half]; sort = true)
        ref = filter(r -> r.cell == "m12_canonical_l23", g)
        refmap = Dict(zip(ref.date, ref.log_growth))
        for c in setdiff(unique(g.cell), ["m12_canonical_l23"])
            x = filter(r -> r.cell == c, g)
            xmap = Dict(zip(x.date, x.log_growth))
            days = sort!(collect(union(keys(refmap), keys(xmap))))
            unmatched = count(d -> !(haskey(refmap, d) && haskey(xmap, d)), days)
            diff = [get(xmap, d, 0.0) - get(refmap, d, 0.0) for d in days]
            nc = P3.block_bootstrap(days, diff, season_of)
            ci = block_bootstrap_circular(days, diff, season_of)
            push!(paired, (; snapshot = first(g.snapshot), half = first(g.half), cell = c, reference = "m12_canonical_l23",
                           d = nc.point, d_lo90 = nc.lo90, d_hi90 = nc.hi90, d_lo90_circ = ci.lo90,
                           d_hi90_circ = ci.hi90, n_days = nc.n_days, unmatched_days = unmatched))
        end
    end
    return DF.DataFrame(single), DF.DataFrame(paired)
end

function phase2(out, target, freeze_path)
    freeze_commit = release!(freeze_path)
    inputs = load_inputs([T25, T6H])
    qb, _ = half_books(inputs, "qs_marketobs", T25.label, :selection)
    dirs = sort!(unique(direction_key(s) for b in qb for s in b.sels))
    baskets = phase1_baskets(dirs)
    println("PHASE2_BASKETS ", join(["$(r.bk.name)@$(r.lambda)" for r in baskets], " "))
    acc = (; summary = NamedTuple[], markets = NamedTuple[], daily = NamedTuple[], bets = DF.DataFrame[],
           directions = NamedTuple[], diag = NamedTuple[])
    for snap in (T25, T6H), half in (:selection, :evaluation)
        half_cells(acc, inputs, snap, half, baskets, target)
    end
    daily = DF.DataFrame(acc.daily)
    single, paired = phase2_uncertainty(daily, inputs.ctx.season_of)
    summary = DF.DataFrame(acc.summary)
    summary = DF.leftjoin(summary, single, on = [:snapshot, :half, :cell])
    summary = DF.leftjoin(summary, DF.select(paired, DF.Not([:reference, :n_days])), on = [:snapshot, :half, :cell])
    CSV.write(joinpath(out, "phase2_summary.csv"), summary)
    CSV.write(joinpath(out, "phase2_uncertainty.csv"), single)
    CSV.write(joinpath(out, "phase2_paired.csv"), paired)
    CSV.write(joinpath(out, "phase2_markets.csv"), DF.DataFrame(acc.markets))
    CSV.write(joinpath(out, "phase2_daily.csv"), daily)
    CSV.write(joinpath(out, "phase2_bets.csv"), vcat(acc.bets...; cols = :union))
    dtab = DF.DataFrame(acc.directions)
    dtab.verdict = eda_verdict.(eachrow(dtab))
    CSV.write(joinpath(out, "phase2_directions.csv"), dtab)
    CSV.write(joinpath(out, "phase2_solve_diag.csv"), DF.DataFrame(acc.diag))

    # Ruling 8 on the frozen config at T-25m, 25/26 only.
    fr = only(filter(r -> r.snapshot == T25.label && r.half == "evaluation" && r.role == "frozen", eachrow(summary)))
    bound = 1.5 * target
    g_fail, mdd_fail = fr.mean_daily_log_growth <= 0.0, fr.mdd_pct < bound
    decision = g_fail || mdd_fail ? "PAPER_TRADE_FALLBACK" : "PAPER_TRADE_FROZEN"
    dec = DF.DataFrame([(; freeze_commit, cell = fr.cell, g = fr.mean_daily_log_growth, mdd_pct = fr.mdd_pct,
                          mdd_bound_pct = bound, g_fail, mdd_fail, decision)])
    CSV.write(joinpath(out, "phase2_decision.csv"), dec)
    println("DECISION ", decision, " g=", fr.mean_daily_log_growth, " mdd=", fr.mdd_pct, " bound=", bound,
            " g_fail=", g_fail, " mdd_fail=", mdd_fail)
    println("PHASE2_DONE")
    return nothing
end
