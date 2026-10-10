# Phase 1 (included into QSWave5): selection half only (slates dated <= 2025-05-03).
# Per-direction table, lambda curve per basket, ruling-4 sensitivity, ruling-6 selection.
# The evaluation half is never built here: every book, day and trajectory date is asserted.

# Basket definitions, written before any basket is scored. B4 is fixed by a rule (below),
# applied mechanically to the ungated per-direction table.
const H, D, A = ("1x2", 0.0, :home), ("1x2", 0.0, :draw), ("1x2", 0.0, :away)
const U25, O25 = ("over_under", 2.5, :under), ("over_under", 2.5, :over)
const FIXED_BASKETS = [
    ("B1_canonical", [H, D, A, U25]),
    ("B2_1x2", [H, D, A]),
    ("B3_1x2_ou25", [H, D, A, O25, U25]),
    ("B5_all13", :all),              # explored: every canonical-markets direction
    ("B6_home_u25", [H, U25]),       # explored: the two 0.35 tiers only
]
const SELECTION_TOL = 0.5e-3

function assert_selection_dates(dates)
    all(d -> half_of(d) === :selection, dates) || error("SEAL: a phase 1 date lies in the evaluation half")
    return nothing
end

# Bet ledger (wave 4) plus the direction label of each leg.
function direction_ledger(result, books, close)
    bets = P3.bet_ledger(result, books, close)
    lab = Dict((b.m_id, s.family) => direction_label(direction_key(s)) for b in books for s in b.sels)
    bets.direction = [lab[(r.match_id, r.family)] for r in eachrow(bets)]
    return bets
end

# EDA section 2 columns per direction; money figures rescaled by the slate's opening bankroll.
function direction_table(key, result, books, close)
    bets = direction_ledger(result, books, close)
    dates = result.trajectory.dates
    assert_selection_dates(dates)
    tot_stake, tot_pnl = sum(bets.stake_unit), sum(bets.pnl_unit)
    book_roi = 100 * tot_pnl / tot_stake
    rows = NamedTuple[]
    for (dname, g) in vcat([("ALL", bets)], [(first(g.direction), g) for g in DF.groupby(bets, :direction; sort = true)])
        st, pl = sum(g.stake_unit), sum(g.pnl_unit)
        kroi = 100 * pl / st
        perday = Dict(d => 0.0 for d in dates)
        units = Dict(d => 0.0 for d in dates)
        for r in eachrow(g)
            perday[r.date] += r.pnl_frac
            units[r.date] += r.pnl_unit
        end
        bank = vcat(1.0, cumprod([1.0 + perday[d] for d in dates]))
        wu = vcat(0.0, cumsum([units[d] for d in dates]))
        push!(rows, (; key..., direction = dname, n_bets = DF.nrow(g), win_pct = 100 * Statistics.mean(g.pnl_frac .> 0),
                     avg_odds = Statistics.mean(g.odds_quoted), edge = Statistics.mean(g.p_model .- g.p_market),
                     flat_roi_pct = 100 * Statistics.mean(g.pnl_frac ./ g.stake_frac), kelly_roi_pct = kroi,
                     cap_share_pct = 100 * st / tot_stake, efficiency = kroi / book_roi,
                     turnover_unit = st, pnl_unit = pl,
                     standalone_mdd_pct = drawdown_pct(bank),
                     standalone_dd_unit = minimum(wu .- accumulate(max, wu)),
                     P3.clv_stats(g)...))
    end
    return rows
end

# EDA section 2 rule, applied per direction to the ungated run (B4).
eda_verdict(r) = r.kelly_roi_pct > 0 && r.efficiency >= 0.5 && r.n_bets >= 100 ? "KEEP" :
                 r.kelly_roi_pct <= 0 || (r.efficiency < 0.25 && r.n_bets >= 100) ? "PRUNE" : "CONDITIONAL"

# Ruling 3: smallest grid lambda whose MDD is within target there and at every larger lambda.
function matched_lambda(curve, target)
    ok = [r.mdd_pct >= target for r in curve]
    ok[end] || return nothing
    i = length(ok)
    while i > 1 && ok[i - 1]
        i -= 1
    end
    return curve[i]
end

function phase1(out, target)
    RELEASED[] && error("SEAL: phase 1 must run before release")
    inputs = load_inputs([T25])
    ctx = inputs.ctx
    close = P3.close_index(ctx.ds, half_ids(ctx, :selection))
    qb, qr = half_books(inputs, "qs_marketobs", T25.label, :selection)
    days = day_problems(inputs.models["qs_marketobs"], qb)
    assert_selection_dates([d.slate.window for d in days])
    dirs = sort!(unique(direction_key(s) for b in qb for s in b.sels))
    length(dirs) == 13 || error("expected 13 canonical-markets directions, found $(length(dirs))")
    println("PHASE1_INPUTS books=", length(qb), " days=", length(days), " directions=", length(dirs))
    flush(stdout)

    # 1. Per-direction table: ungated SK (every direction at 0.25, lambda 23), canonical SK, m12.
    dir_rows, diags = NamedTuple[], NamedTuple[]
    ungated = PF.TieredTrust(Dict(k => 0.25 for k in dirs); default = 0.0)
    us, ud = solve_days(days, all_legs)
    ures = simulate_slates(basket_policy(ungated, 23.0), us, qr)
    ubooks = PF.MatchBook[b for sl in us for b in sl.books]
    append!(dir_rows, direction_table((; run = "qs_sk_ungated_t025_l23"), ures, ubooks, close))
    canon = MD.canonical_scottish_lower_policy()
    cres = simulate_slates(canon, us, qr)
    append!(dir_rows, direction_table((; run = "qs_sk_canonical_wave4_l23"), cres, ubooks, close))
    mb, mr = half_books(inputs, "m12", T25.label, :selection)
    mres = simulate_books(canon, mb, mr)
    append!(dir_rows, direction_table((; run = "m12_canonical_l23"), mres, mb, close))
    dtab = DF.DataFrame(dir_rows)
    dtab.verdict = eda_verdict.(eachrow(dtab))
    CSV.write(joinpath(out, "phase1_directions.csv"), dtab)
    mrow = curve_row((; basket = "m12_canonical", lambda = 23.0), mres, nothing)
    println("M12_CONTEXT g=", mrow.g, " mdd=", mrow.mdd_pct, " target=", target)
    mrow.mdd_pct == target || error("m12 canonical 24/25 MDD no longer equals the Phase 0 target")

    keep = dtab.direction[(dtab.run .== "qs_sk_ungated_t025_l23") .& (dtab.direction .!= "ALL") .&
                          (dtab.verdict .== "KEEP")]
    lab2key = Dict(direction_label(k) => k for k in dirs)
    b4 = Tuple{String,Float64,Symbol}[lab2key[d] for d in keep]
    println("B4_RULE keep=", join(keep, ","))
    flush(stdout)

    # 2. Lambda curve per basket (one solve per basket; lambda does not enter the solve).
    specs = [(n, d === :all ? dirs : d) for (n, d) in FIXED_BASKETS]
    specs = vcat(specs[1:3], [("B4_eda_keep", b4)], specs[4:end])
    curve_rows, chosen, daily = NamedTuple[], NamedTuple[], NamedTuple[]
    for (name, ds_) in specs
        if isempty(ds_)
            println("BASKET name=", name, " empty: no direction passed the rule; not scored")
            continue
        end
        bk = basket(name, ds_)
        slates, diag = solve_days(days, s -> in_basket(bk, s))
        DF.nrow(diag) == length(days) || error("basket $name dropped a day")
        append!(diags, [(; basket = name, r...) for r in eachrow(diag)])
        curve = NamedTuple[]
        for lam in LAMBDA_GRID
            res = simulate_slates(basket_policy(bk.trust, lam), slates, qr)
            assert_selection_dates(res.trajectory.dates)
            length(res.trajectory.dates) == length(days) || error("basket $name lost a slate")
            row = curve_row((; basket = name, lambda = lam), res, diag)
            push!(curve, (; row..., mean_raw_joint = Statistics.mean(diag.raw_total_joint)))
        end
        append!(curve_rows, curve)
        pick = matched_lambda(curve, target)
        at23 = only(filter(r -> r.lambda == 23.0, curve))
        f(k) = pick === nothing ? missing : getfield(pick, k)
        push!(chosen, (; basket = name, n_directions = length(bk.directions),
                       directions = join(direction_label.(bk.directions), ","), feasible = pick !== nothing,
                       lambda = f(:lambda), g = f(:g), mdd_pct = f(:mdd_pct), budget_share = f(:budget_share),
                       cap_share = f(:cap_share), risk_active_share = f(:risk_active_share),
                       mean_exposure = f(:mean_exposure), final_wealth = f(:final_wealth), n_bets = f(:n_bets),
                       g_l23 = at23.g, mdd_l23 = at23.mdd_pct))
        if pick !== nothing
            res = simulate_slates(basket_policy(bk.trust, pick.lambda), slates, qr)
            t = res.trajectory
            append!(daily, [(; basket = name, lambda = pick.lambda, date = d, pnl_frac = t.slate_pl[k],
                               log_growth = log1p(t.slate_pl[k]), bank_close = t.bankroll[k + 1])
                            for (k, d) in enumerate(t.dates)])
        end
        c = last(chosen)
        println("BASKET name=", name, " dirs=", c.directions, " lambda=", c.lambda, " g=", c.g, " mdd=", c.mdd_pct,
                " g_l23=", c.g_l23, " mdd_l23=", c.mdd_l23)
        flush(stdout)
    end
    CSV.write(joinpath(out, "phase1_lambda_curve.csv"), DF.DataFrame(curve_rows))
    CSV.write(joinpath(out, "phase1_basket_diag.csv"), DF.DataFrame(diags))
    CSV.write(joinpath(out, "phase1_daily.csv"), DF.DataFrame(daily))

    # 3. Ruling-4 sensitivity: canonical basket at lambda 23, all legs (wave 4) vs basket-only legs.
    sens = NamedTuple[]
    push!(sens, (; curve_row((; variant = "wave4_all_legs_then_trust", lambda = 23.0), cres, ud)...,
                 mean_raw_joint = Statistics.mean(ud.raw_total_joint)))
    b1 = only(filter(r -> r.basket == "B1_canonical" && r.lambda == 23.0, curve_rows))
    push!(sens, (; variant = "basket_only_legs", lambda = 23.0,
                 Base.structdiff(b1, NamedTuple{(:basket, :lambda)})...))
    push!(sens, (; curve_row((; variant = "m12_canonical_context", lambda = 23.0), mres, nothing)...,
                 mean_raw_joint = missing))
    CSV.write(joinpath(out, "phase1_sensitivity.csv"), DF.DataFrame(sens))
    for r in sens
        println("SENSITIVITY variant=", r.variant, " g=", r.g, " mdd=", r.mdd_pct, " wealth=", r.final_wealth,
                " bets=", r.n_bets, " budget_share=", r.budget_share, " raw_joint=", r.mean_raw_joint)
    end

    # 4. Ruling 6: best g at the matched lambda; a smaller basket within tolerance wins.
    feas = filter(r -> r.feasible, chosen)
    isempty(feas) && error("no basket meets the MDD target at any grid lambda")
    best = feas[argmax([r.g for r in feas])]
    near = filter(r -> r.n_directions < best.n_directions && r.g >= best.g - SELECTION_TOL, feas)
    pick = isempty(near) ? best :
           sort(near, by = r -> (r.n_directions, -r.g))[1]
    sel = DF.DataFrame(chosen)
    sel.best_g = fill(best.basket, DF.nrow(sel))
    sel.selected = sel.basket .== pick.basket
    CSV.write(joinpath(out, "phase1_selection.csv"), sel)
    println("SELECTION best_g=", best.basket, " selected=", pick.basket, " lambda=", pick.lambda, " g=", pick.g,
            " mdd=", pick.mdd_pct, " budget_share=", pick.budget_share, " cap_share=", pick.cap_share)
    println("PHASE1_DONE")
    return nothing
end
