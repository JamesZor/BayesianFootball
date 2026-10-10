# Phase 0 (included into QSWave5): reproduce the two wave-4 T-25m cells bit-identically,
# recompute the per-season MDDs, and prove the selection-half and basket machinery.
# The full period appears here only for the two already-public wave-4 configs (gate = true).

const GATE_CELLS = [
    (model = "m12", policy = "canonical", wealth = 2.773493294562897, bets = 1132),
    (model = "qs_marketobs", policy = "scenario_kelly", wealth = 2.154029744590488, bets = 952),
]
# Brief: per-season MDD, bankroll reset to 1 at each season start; must match to 0.01 pp.
const SEASON_MDD_TARGETS = Dict(("m12", "canonical", "24/25") => -13.60, ("m12", "canonical", "25/26") => -16.15,
                                ("qs_marketobs", "scenario_kelly", "24/25") => -7.81,
                                ("qs_marketobs", "scenario_kelly", "25/26") => -10.47)
const T25 = P3.SNAPSHOTS[1]

# One wave-4 config on one half: canonical uses the stock per-match path; scenario_kelly uses
# the wave-5 joint solve with every canonical leg, then the canonical overlay (wave-4 style).
function wave4_cell(inputs, model_label, policy_label, half; gate = false)
    books, report = half_books(inputs, model_label, T25.label, half; gate = gate)
    policy = MD.canonical_scottish_lower_policy()
    if policy_label == "canonical"
        return (; result = simulate_books(policy, books, report), books, report, diag = nothing)
    end
    days = day_problems(inputs.models[model_label], books)
    slates, diag = solve_days(days, all_legs)
    result = simulate_slates(policy, slates, report)
    return (; result, books = PF.MatchBook[b for sl in slates for b in sl.books], report, diag, days)
end

function phase0(out)
    inputs = load_inputs([T25])
    ctx = inputs.ctx
    gate_rows, season_rows, half_rows, daily = NamedTuple[], NamedTuple[], NamedTuple[], NamedTuple[]
    full = Dict{String,Any}()
    for c in GATE_CELLS
        cell = wave4_cell(inputs, c.model, c.policy, :full; gate = true)
        w, n = cell.result.trajectory.bankroll[end], cell.result.summary.n_bets
        ok = w == c.wealth && n == c.bets
        push!(gate_rows, (; model = c.model, snapshot = T25.label, policy = c.policy, final_wealth = w,
                          n_bets = n, n_slates = cell.result.summary.n_slates, expected_wealth = c.wealth,
                          expected_bets = c.bets, abs_gap = abs(w - c.wealth), bit_identical = ok,
                          status = ok ? "PASS" : "FAIL"))
        println("GATE model=", c.model, " policy=", c.policy, " wealth=", repr(w), " bets=", n,
                " status=", ok ? "PASS" : "FAIL")
        flush(stdout)
        full[c.model] = cell
        for r in season_mdd(cell.result, ctx.season_of)
            target = SEASON_MDD_TARGETS[(c.model, c.policy, r.season)]
            push!(season_rows, (; model = c.model, policy = c.policy, r..., target_pct = target,
                                gap_pp = r.mdd_pct - target, match = abs(r.mdd_pct - target) <= 0.01))
        end
        t = cell.result.trajectory
        for (k, d) in enumerate(t.dates)
            push!(daily, (; model = c.model, policy = c.policy, date = d, season = ctx.season_of[d],
                          pnl_frac = t.slate_pl[k], log_growth = log1p(t.slate_pl[k]),
                          bank_open = t.bankroll[k], bank_close = t.bankroll[k + 1]))
        end
    end
    CSV.write(joinpath(out, "phase0_gate.csv"), DF.DataFrame(gate_rows))
    CSV.write(joinpath(out, "phase0_season_mdd.csv"), DF.DataFrame(season_rows))
    CSV.write(joinpath(out, "phase0_gate_daily.csv"), DF.DataFrame(daily))
    for r in season_rows
        println("SEASON_MDD model=", r.model, " policy=", r.policy, " season=", r.season, " mdd=",
                round(r.mdd_pct; digits = 4), " target=", r.target_pct, " match=", r.match)
    end
    all(r -> r.bit_identical, gate_rows) || error("PHASE0_GATE_FAILED: a wave-4 cell did not reproduce")
    all(r -> r.match, season_rows) || error("PHASE0_GATE_FAILED: a per-season MDD differs by > 0.01 pp")

    # Selection-half machinery: a 24/25-only run must equal the first season of the full run.
    for c in GATE_CELLS
        cell = wave4_cell(inputs, c.model, c.policy, :selection)
        ft = full[c.model].result.trajectory
        ht = cell.result.trajectory
        n = length(ht.dates)
        same = ht.dates == ft.dates[1:n] && ht.bankroll == ft.bankroll[1:(n + 1)] &&
               all(d -> half_of(d) === :selection, ht.dates) && half_of(ft.dates[n + 1]) === :evaluation
        sm = only(filter(r -> r.model == c.model && r.season == "24/25", season_rows))
        push!(half_rows, (; model = c.model, policy = c.policy, half = "selection", n_slates = n,
                          wealth = ht.bankroll[end], n_bets = cell.result.summary.n_bets,
                          mdd_pct = cell.result.summary.mdd, full_slice_identical = same,
                          season_mdd_pct = sm.mdd_pct, season_bets = sm.bets,
                          mdd_equal = cell.result.summary.mdd == sm.mdd_pct,
                          bets_equal = cell.result.summary.n_bets == sm.bets))
        println("HALF_CHECK model=", c.model, " slates=", n, " identical=", same, " mdd=",
                round(cell.result.summary.mdd; digits = 4))
        if c.policy == "scenario_kelly"
            # Basket equivalence on the first selection days, for two filtered baskets.
            eq = NamedTuple[]
            for bk in (basket("B1_canonical", [("1X2", 0.0, :home), ("1X2", 0.0, :draw),
                                               ("1X2", 0.0, :away), ("OverUnder", 2.5, :under)]),
                       basket("B2_1x2", [("1X2", 0.0, :home), ("1X2", 0.0, :draw), ("1X2", 0.0, :away)]))
                append!(eq, subset_equivalence(cell.days, bk; n_days = 6))
            end
            CSV.write(joinpath(out, "phase0_basket_equivalence.csv"), DF.DataFrame(eq))
            println("BASKET_EQUIVALENCE days=", length(eq), " all_equal=", all(r -> r.equal, eq))
            all(r -> r.equal, eq) || error("PHASE0_GATE_FAILED: column subset differs from rebuilt problem")
        end
    end
    CSV.write(joinpath(out, "phase0_half_check.csv"), DF.DataFrame(half_rows))
    all(r -> r.full_slice_identical && r.mdd_equal && r.bets_equal, half_rows) ||
        error("PHASE0_GATE_FAILED: the selection-half run is not the full run's first season")

    # The seal must refuse 25/26 now.
    sealed = try
        half_ids(ctx, :evaluation)
        false
    catch err
        occursin("SEAL", sprint(showerror, err))
    end
    println("SEAL_REFUSES_EVALUATION=", sealed)
    sealed || error("PHASE0_GATE_FAILED: the seal did not refuse the evaluation half")
    println("PHASE0_GATE_PASS")
    return nothing
end
