# ==============================================================================
# r06 — Closing-line portfolio under Option B: market-anchored GRW vs the GRW baseline
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# Every arm staked through ONE contract — `MatchDay.option_b_system()`, the audited
# production book and policy (1X2 home/draw/away, Under 2.5, Over 1.5; TieredTrust,
# SlateDrawdown, FixedCap 0.25, daily slates, 2% commission) — against the de-vigged
# Betfair TWA(−20, 0] close, over the identical buildable subset of the 710-fixture
# 24/25 + 25/26 walk-forward panel. Only the posterior differs between rows.
#
#   m05_joint_grw_baseline               Task 013 b0961bc4 (pinned control)
#   m05_joint_grw_supremacy_w040         0ee58d18
#   m05_joint_grw_smile_supremacy_w020   fcd5e974
#   m05_joint_grw_smile_supremacy_w040   30620d3e
#   m05_joint_grw_smile_supremacy_w070   32d588f1
#
# It is a closing-line simulation: prices are the close, not a tradeable T−25 book, and a
# model pulled toward the close is partly being scored against its own anchor. r04 found no
# proper-score difference; at n ≈ 630 fixtures neighbouring arms' bootstrap growth intervals
# overlap, so a bankroll ranking here is descriptive, not a result.
#
# GATES
#
#   P1  reproduction: the baseline, on Task 014's 632-fixture buildable panel, reproduces its
#       published Option B row (+385.8% return, ROI 11.68%, 1,247 bets). If the buildable panel
#       here differs, the gate reports that and does not claim reproduction.
#   P2  smile routing: every staked totals bet of a smile arm carries p_model equal to
#       mean cdf(Poisson(λ_tot·φ_K), K) to ≤ 1e-9 — the books priced the smile, not the grid.
#   P3  persistence: the four candidate portfolios are written to `portfolio_runs` /
#       `portfolio_bets` / `portfolio_artifacts` under `scottish_lower_grw_market_smile` and
#       each reloads with an identical bet ledger. The baseline belongs to Task 013's namespace
#       and is not re-written there.
#
# USAGE (mcmc-beast, from /root/BF_grw_market_smile, after r02)
#
#   julia --project -t 16 current_development/grw_market_smile/r06_portfolio.jl
# ==============================================================================

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball
using CSV
using DataFrames
using Dates
using Printf
using Statistics

include(joinpath(@__DIR__, "l01_loader.jl"))
include(joinpath(@__DIR__, "l02_evaluation.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R06_CONFIG = GMSConfig()
const R06_BOOTSTRAP_B = 4000
const R06_SEED = 1
const R06_OUT_DIR = joinpath(R06_CONFIG.save_root, "portfolio")
const R06_LATENT_DIR = joinpath(R06_CONFIG.save_root, "latents")

# Task 014 README §6 — the same run under the same contract on its own buildable panel.
const R06_BASELINE_PUBLISHED = (n_panel = 632, total_return_pct = 385.8, roi_pct = 11.68, n_bets = 1247)

mkpath(R06_OUT_DIR)
println("\n" * "="^96)
println("  r06 CLOSING-LINE PORTFOLIO — Option B, market-anchored GRW vs GRW baseline")
println("="^96)

# %%
# ===================================================================
# 3. Data, contract, arms, panel
# ===================================================================
r06_ds = gph_load_data()
r06_splitter = gph_splitter(R06_CONFIG.extension_seasons)
r06_odds = gms_betfair_closing_odds(r06_ds)
r06_book, r06_policy = gms_option_b()
println("  book   : ", r06_book)
println("  policy : ", r06_policy)

r06_arms = gms_arms(R06_CONFIG)
r06_raw = Dict{String,Any}()
for arm in r06_arms
    r06_raw[arm.label] = gms_load_arm(arm, r06_ds; splitter = r06_splitter, latent_dir = R06_LATENT_DIR)
end
r06_panel_ids = gms_common_panel(r06_ds, r06_raw, R06_CONFIG.target_seasons)
length(r06_panel_ids) == R06_CONFIG.expected_oos || error(
    "common panel is $(length(r06_panel_ids)) fixtures; expected $(R06_CONFIG.expected_oos)")
r06_fits = Dict(label => gms_restrict(fit, r06_panel_ids) for (label, fit) in r06_raw)
r06_raw = nothing
GC.gc()

r06_quoted = sort!(collect(intersect(Set(r06_panel_ids), Set(r06_odds.match_id))))
r06_panel, r06_dropped = gms_buildable_panel(r06_book, r06_fits, r06_odds, r06_ds, r06_quoted)
CSV.write(joinpath(R06_OUT_DIR, "r06_dropped_fixtures.csv"), r06_dropped)
println("  panel  : ", length(r06_panel_ids), " walk-forward → ", length(r06_quoted),
        " quoted → ", length(r06_panel), " buildable by every arm (", nrow(r06_dropped), " dropped)")

# %%
# ===================================================================
# 4. Simulation — identical contract, identical panel
# ===================================================================
r06_results = Dict{String,Any}()
r06_rows = NamedTuple[]
r06_breakdown_rows = NamedTuple[]
r06_gate_rows = NamedTuple[]
for arm in r06_arms
    result, restricted = gms_simulate(r06_book, r06_policy, r06_fits[arm.label], r06_odds, r06_ds,
                                      r06_panel; label = arm.label, B = R06_BOOTSTRAP_B, seed = R06_SEED)
    r06_results[arm.label] = result
    bets = result.trajectory.bets
    row = gms_portfolio_row(arm.label, result; n_panel = length(r06_panel))
    edge = gms_edge_summary(bets)
    breakdown = gms_breakdown(arm.label, bets)
    append!(r06_breakdown_rows, breakdown)
    n_1x2 = sum((r.n_bets for r in breakdown if r.level == "market" && r.group == "1X2"); init = 0)
    n_tot = sum((r.n_bets for r in breakdown if r.level == "market" && r.group == "totals"); init = 0)
    push!(r06_rows, (; row..., n_bets_1x2 = n_1x2, n_bets_totals = n_tot,
                       capture_ratio = edge.capture_ratio, run_id = string(arm.run_id)))

    # --- P2: the smile books priced the smile -----------------------------------
    if restricted.latents isa SmileLatents
        g = gms_smile_book_gate(bets, restricted.latents)
        push!(r06_gate_rows, (; model = arm.label, g...))
        g.n_totals_bets > 0 || error("P2: $(arm.label) staked no totals bet to check")
        g.max_abs_vs_smile <= 1e-9 || error(
            "P2 FAILED: $(arm.label) staked totals at p_model up to $(g.max_abs_vs_smile) away from " *
            "cdf(Poisson(λ_tot·φ)) — the books were not priced through the smile")
        @printf("  P2 %-34s %d totals bets | max |p − smile| %.1e | min |p − grid| %.4f  PASS\n",
                arm.label, g.n_totals_bets, g.max_abs_vs_smile, g.min_abs_vs_grid)
    end

    @printf("  %-34s return %+8.2f%%  ROI %+6.2f%%  Sharpe %6.3f  MDD %7.2f%%  bets %4d (1X2 %d / totals %d)  WR %.1f%%\n",
            arm.label, row.total_return_pct, row.roi_pct, row.sharpe_ann, row.max_drawdown_pct,
            row.n_bets, n_1x2, n_tot, row.win_rate_pct)
end
r06_summary = DataFrame(r06_rows)
r06_breakdown = DataFrame(r06_breakdown_rows)
r06_gates = DataFrame(r06_gate_rows)

# --- P1: reproduction of the pinned baseline ------------------------------------
r06_base = only(filter(r -> r.model == GMS_BASELINE_CONTROL.label, eachrow(r06_summary)))
r06_repro_basis = length(r06_panel) == R06_BASELINE_PUBLISHED.n_panel
r06_repro_pass = r06_repro_basis &&
    abs(r06_base.total_return_pct - R06_BASELINE_PUBLISHED.total_return_pct) < 0.1 &&
    abs(r06_base.roi_pct - R06_BASELINE_PUBLISHED.roi_pct) < 0.01 &&
    r06_base.n_bets == R06_BASELINE_PUBLISHED.n_bets
@printf("\n  P1 reproduction: baseline %+.2f%% / ROI %.2f%% / %d bets on %d fixtures vs published %+.1f%% / %.2f%% / %d on %d — %s\n",
        r06_base.total_return_pct, r06_base.roi_pct, r06_base.n_bets, length(r06_panel),
        R06_BASELINE_PUBLISHED.total_return_pct, R06_BASELINE_PUBLISHED.roi_pct,
        R06_BASELINE_PUBLISHED.n_bets, R06_BASELINE_PUBLISHED.n_panel,
        r06_repro_pass ? "REPRODUCED" : r06_repro_basis ? "FAILED" : "NOT COMPARABLE (different panel)")
r06_repro_basis && !r06_repro_pass && error(
    "P1 FAILED: on the published panel the baseline does not reproduce its Option B row")

# %%
# ===================================================================
# 5. Pairwise attribution against the baseline
# ===================================================================
# Shared bets (same fixture, family, selection → same price and outcome) against exclusive
# ones: separates "priced the same bets differently" from "took different bets".
r06_pair_rows = NamedTuple[]
r06_base_bets = DataFrame(r06_results[GMS_BASELINE_CONTROL.label].trajectory.bets)
r06_key(r) = (Int(r.match_id), String(r.family), String(Symbol(r.selection)))
for arm in r06_arms
    arm.label == GMS_BASELINE_CONTROL.label && continue
    bets = DataFrame(r06_results[arm.label].trajectory.bets)
    base_keys = Set(r06_key(r) for r in eachrow(r06_base_bets))
    cand_keys = Set(r06_key(r) for r in eachrow(bets))
    shared = intersect(base_keys, cand_keys)
    for (set, owner, frame) in (
            ("shared", arm.label, filter(r -> r06_key(r) in shared, bets)),
            ("shared", "baseline", filter(r -> r06_key(r) in shared, r06_base_bets)),
            ("exclusive", arm.label, filter(r -> !(r06_key(r) in base_keys), bets)),
            ("exclusive", "baseline", filter(r -> !(r06_key(r) in cand_keys), r06_base_bets)))
        s = gms_edge_summary(frame)
        push!(r06_pair_rows, (; pair = "$(arm.label) vs baseline", bet_set = set, owner,
                                n_bets = s.n_bets, win_rate_pct = 100 * s.win_rate, roi_pct = s.roi,
                                edge_mean_pp = s.edge_mean, capture_ratio = s.capture_ratio))
    end
end
r06_pairs = DataFrame(r06_pair_rows)

# %%
# ===================================================================
# 6. Persistence — the four candidate portfolios, with ledger round-trip (P3)
# ===================================================================
r06_persisted = NamedTuple[]
r06_db = PostgresStorage(R06_CONFIG.experiment)
for arm in r06_arms
    arm.role == "candidate" || continue
    result = r06_results[arm.label]
    portfolio_id = save_portfolio_db(result, arm.run_id, r06_db;
        book_spec = r06_book, policy_spec = r06_policy,
        metadata = (; runner = "r06_portfolio", contract = "option_b_system",
                      odds = "betfair_twa_-20_0", n_panel = length(r06_panel), task = "015"))
    reloaded = load_portfolio_db(portfolio_id, r06_db)
    reloaded.trajectory.bets == result.trajectory.bets || error(
        "P3 FAILED: $(arm.label) reloaded portfolio ledger differs from the simulated one")
    push!(r06_persisted, (; model = arm.label, model_run_id = string(arm.run_id),
                            portfolio_run_id = string(portfolio_id)))
    println("  P3 persisted ", arm.label, " → portfolio ", portfolio_id, " (ledger reloads identically)")
end

# %%
# ===================================================================
# 7. Final report
# ===================================================================
CSV.write(joinpath(R06_OUT_DIR, "r06_portfolio_summary.csv"), r06_summary)
CSV.write(joinpath(R06_OUT_DIR, "r06_market_breakdown.csv"), r06_breakdown)
CSV.write(joinpath(R06_OUT_DIR, "r06_smile_book_gate.csv"), r06_gates)
CSV.write(joinpath(R06_OUT_DIR, "r06_pair_attribution.csv"), r06_pairs)
CSV.write(joinpath(R06_OUT_DIR, "r06_persisted.csv"), DataFrame(r06_persisted))

r06_f2 = v -> gph_num(v; digits = 2)
r06_f3 = v -> gph_num(v; digits = 3)
open(joinpath(R06_OUT_DIR, "r06_portfolio_report.md"), "w") do io
    println(io, "# r06 closing-line portfolio — Task 015 (market-anchored MultiScaleGRW)\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"),
            ". Contract: `MatchDay.option_b_system()`. Book: de-vigged Betfair TWA(−20, 0] close. Panel: ",
            length(r06_panel_ids), " walk-forward → ", length(r06_quoted), " quoted → ",
            length(r06_panel), " buildable by every arm. Bootstrap B = ", R06_BOOTSTRAP_B, ".\n")
    @printf(io, "P1 reproduction — baseline %+.2f%% / ROI %.2f%% / %d bets on %d fixtures; published %+.1f%% / %.2f%% / %d on %d: **%s**.\n\n",
            r06_base.total_return_pct, r06_base.roi_pct, r06_base.n_bets, length(r06_panel),
            R06_BASELINE_PUBLISHED.total_return_pct, R06_BASELINE_PUBLISHED.roi_pct,
            R06_BASELINE_PUBLISHED.n_bets, R06_BASELINE_PUBLISHED.n_panel,
            r06_repro_pass ? "reproduced" : r06_repro_basis ? "failed" : "not comparable (different panel)")

    println(io, "## Headline\n")
    print(io, gph_markdown_table(select(r06_summary,
        :model, :n_bets, :n_bets_1x2, :n_bets_totals, :total_return_pct, :roi_pct, :sharpe_ann,
        :calmar, :max_drawdown_pct, :win_rate_pct, :growth_lo, :growth_hi, :p_roi_positive,
        :capture_ratio, :mean_edge_pp);
        formats = Dict(:total_return_pct => r06_f2, :roi_pct => r06_f2, :sharpe_ann => r06_f3,
                       :calmar => r06_f3, :max_drawdown_pct => r06_f2, :win_rate_pct => r06_f2,
                       :growth_lo => v -> gph_signed(v; digits = 4),
                       :growth_hi => v -> gph_signed(v; digits = 4),
                       :p_roi_positive => r06_f3, :capture_ratio => r06_f3, :mean_edge_pp => r06_f2)))

    println(io, "\n## Market breakdown\n")
    print(io, gph_markdown_table(select(sort(r06_breakdown, [:level, :group, :model]),
        :level, :group, :model, :n_bets, :win_rate_pct, :roi_pct, :stake_share_pct,
        :edge_mean_pp, :odds_mean, :capture_ratio);
        formats = Dict(:win_rate_pct => r06_f2, :roi_pct => r06_f2, :stake_share_pct => r06_f2,
                       :edge_mean_pp => r06_f2, :odds_mean => r06_f2, :capture_ratio => r06_f3)))

    if nrow(r06_gates) > 0
        println(io, "\n## P2 smile routing on the staked ledger\n")
        print(io, gph_markdown_table(r06_gates;
            formats = Dict(:max_abs_vs_smile => v -> @sprintf("%.1e", v),
                           :min_abs_vs_grid => v -> gph_num(v; digits = 5))))
    end

    println(io, "\n## Shared vs exclusive bets against the baseline\n")
    print(io, gph_markdown_table(r06_pairs;
        formats = Dict(:win_rate_pct => r06_f2, :roi_pct => r06_f2, :edge_mean_pp => r06_f2,
                       :capture_ratio => r06_f3)))

    println(io, "\n## Persisted portfolios\n")
    print(io, gph_markdown_table(DataFrame(r06_persisted)))
end

println("\nR06_DONE report=", joinpath(R06_OUT_DIR, "r06_portfolio_report.md"))
