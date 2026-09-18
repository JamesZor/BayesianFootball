# ==============================================================================
# r05 — Counterfactual re-pricing of the 2026-09-12 Scottish Lower card
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# The live account `live_scottish_m12_500` staked the nine-fixture 15:00 BST card at
# T−25 from Run 67 (`m12_joint_hybrid_synergy`, flat `GlobalHomeAdvantage`, Fold 43) with
# the Option B calibrator, took six away legs, lost all six, and closed −£45.89. This
# runner asks one question: would the SAME pipeline, at the SAME instant, against the SAME
# book, have staked differently had the hybrid carried `HierarchicalTeamHomeAdvantage`?
#
# It is one Saturday. Nine fixtures cannot validate a model, and a counterfactual P&L
# on them is an anecdote with a sign; `r04` is the evidence. What this runner can show is
# mechanism — whether the per-club γ_i moves P(home) on the turf grounds and whether that
# movement removes or shrinks the away legs.
#
# FOUR ARMS, ONE PIPELINE
#
#   flat_raw     Run 67                                  no calibrator
#   flat_optB    Run 67                                  Option B T−25 calibrator  ← the live arm
#   hier_raw     m12_joint_hybrid_synergy_hier_ha F43    no calibrator
#   hier_optB    m12_joint_hybrid_synergy_hier_ha F43    Option B T−25 calibrator
#
# FILTRATION CONTRACT
#
# * as_of = 2026-09-12 13:35 UTC (T−25), the live decision instant.
# * Book: `PreloadedBook`, served by `searchsortedlast(ts, as_of)` — no later tick reachable.
# * Lineups: `PreloadedLineups`, `scraped_at <= as_of`, no historical fallback, and the
#   hybrid's `:player_lineup_ratings_map` materialised from that XI (`slot_latents`) — so
#   the played teamsheet in today's DataStore is unreachable. The live runner chained a
#   BBC confirmed-XI source first; that network source has no point-in-time twin, which is
#   the main reason `flat_optB` may not reproduce the live orders exactly. The
#   reproduction diff is printed, not assumed.
# * Both arms condition on Fold 43, asserted.
#
# SETTLEMENT
#
# Every sheet is settled at PLANNED risk (full fill) with the account's commission rate.
# The live account filled through `LadderSweep`, partially on some legs, so the realised
# −£45.89 is printed beside the full-fill `flat_optB` figure and never subtracted from a
# hierarchical figure directly.
#
# NOTHING IS WRITTEN. `paper_runbook` is read; `paper_replay` is never executed into.
#
# USAGE (mcmc-beast, from /root/BF_hier_ha_slate, after r03)
#
#   /root/.juliaup/bin/julia --project -t 16 current_development/hierarchical_home_advantage/r05_slate_repricing.jl
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
using UUIDs

include(joinpath(@__DIR__, "l01_loader.jl"))
include(joinpath(@__DIR__, "l03_slate.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R05_CONFIG = HHAConfig()
const R05_DAY = Date(2026, 9, 12)
const R05_T_MINUTES = -25
const R05_BANKROLL = 500.0
const R05_LIVE_ACCOUNT = "live_scottish_m12_500"
const R05_LIVE_SCHEMA = "paper_runbook"
const R05_FLAT_EXPERIMENT = "scottish_lower_joint_player_2426"
const R05_FLAT_RUN = 67
const R05_HIER_NAME = "m12_joint_hybrid_synergy_hier_ha"
const R05_EXPECTED_FOLD = 43
const R05_REALISED_QUOTED = -45.89
const R05_OUT_DIR = joinpath(R05_CONFIG.save_root, "slate_20260912")

mkpath(R05_OUT_DIR)
println("\n" * "="^96)
println("  r05 COUNTERFACTUAL RE-PRICING — Scottish Lower card ", R05_DAY, " at T", R05_T_MINUTES)
println("  flat : ", R05_FLAT_EXPERIMENT, " / Run ", R05_FLAT_RUN)
println("  hier : ", R05_CONFIG.experiment, " / ", R05_HIER_NAME)
println("="^96)

# %%
# ===================================================================
# 3. Data snapshot, the card, and the live ledger
# ===================================================================
r05_ds = gph_load_data()
r05_conn = MD.paper_connection()
r05_full_card = load_replay_card(r05_conn, R05_DAY)
r05_card, r05_dropped = hha_played_card(r05_full_card)
r05_as_of = as_of_at(r05_card, R05_T_MINUTES)

println("  store      : ", nrow(r05_ds.matches), " matches, latest ", maximum(r05_ds.matches.match_date))
println("  kickoff    : ", r05_card.kickoff, " UTC   as_of: ", r05_as_of, " UTC")
println("  fixtures   : ", length(r05_card.fixtures), " played",
        isempty(r05_dropped) ? "" : " | removed (no score): " *
        join(["$(f.home) v $(f.away)" for f in r05_dropped], ", "))
println("  XI visible : ", count(f -> get(r05_card.lineup_drop, f.m_id, DateTime(9999)) <= r05_as_of,
                               r05_card.fixtures), " of ", length(r05_card.fixtures), " at as_of")

r05_live = hha_live_ledger(r05_conn, R05_LIVE_ACCOUNT, R05_DAY; schema = R05_LIVE_SCHEMA)
@printf("  live slate : %d legs, fold %d, run %s | realised net £%+.2f (work package quotes £%+.2f) | commission %.2f%%\n",
        nrow(r05_live.legs), r05_live.slate.fold_idx, r05_live.slate.run_name,
        r05_live.realised_net, R05_REALISED_QUOTED, 100 * r05_live.commission_rate)

# %%
# ===================================================================
# 4. The four arms
# ===================================================================
r05_hier_run = gph_run_by_name(PostgresStorage(R05_CONFIG.experiment), R05_HIER_NAME)
r05_hier_run === nothing && error("no completed $R05_HIER_NAME run — run r02 and r03 first")

r05_slots = [
    (ModelSlot("flat_raw", "Run 67 flat HA, raw", R05_FLAT_EXPERIMENT, ""), R05_FLAT_RUN),
    (ModelSlot("flat_optB", "Run 67 flat HA, Option B", R05_FLAT_EXPERIMENT, "",
               MD.option_b_calibrator()), R05_FLAT_RUN),
    (ModelSlot("hier_raw", "hierarchical HA, raw", R05_CONFIG.experiment, ""), r05_hier_run),
    (ModelSlot("hier_optB", "hierarchical HA, Option B", R05_CONFIG.experiment, "",
               MD.option_b_calibrator()), r05_hier_run),
]
for (slot, key) in r05_slots
    hha_load_pinned_slot!(slot, key, r05_ds, r05_card)
    println("  ", rpad(slot.key, 10), " run ", slot.run_name, " | fold ", slot.fold_idx,
            " | covered ", length(slot.covered), "/", length(r05_card.fixtures),
            isempty(slot.fold_warning) ? "" : " | WARNING " * slot.fold_warning,
            @sprintf(" | %.0f s", slot.load_seconds))
    slot.fold_idx == R05_EXPECTED_FOLD || error("$(slot.key) selected fold $(slot.fold_idx); expected $R05_EXPECTED_FOLD")
    isempty(slot.refused) || error("$(slot.key) refuses fixtures: $(slot.refused)")
end

r05_state = ReplayState(r05_ds, r05_conn, r05_card;
                        system = MD.option_b_system(),
                        models = [s for (s, _) in r05_slots],
                        active = "flat_optB",
                        bankroll = R05_BANKROLL,
                        account_id = "hha_counterfactual_20260912")
r05_state.clock.t = R05_T_MINUTES

# %%
# ===================================================================
# 5. Pricing at T−25
# ===================================================================
r05_slates = Dict{String,Any}()
r05_probs = Dict{String,Any}()
for (slot, _) in r05_slots
    r05_state.active = slot.key
    r05_state.slate = nothing
    slate = reprice!(r05_state)
    isempty(r05_state.tick_error) || error("$(slot.key) failed to price: $(r05_state.tick_error)")
    slate === nothing && error("$(slot.key) priced nothing: $(r05_state.tick_note)")
    isempty(slate.blocked) || @warn "gated fixtures" arm = slot.key n = length(slate.blocked)
    r05_slates[slot.key] = slate
    r05_probs[slot.key] = model_probs_at(r05_state, r05_as_of; slot)
    @printf("  %-10s %2d legs | risk £%.2f | k_risk %.3f | capped %s\n",
            slot.key, nrow(slate.sheet), slate.total_risk, slate.k_risk, slate.capped)
end

# %%
# ===================================================================
# 6. What moved: 1X2 probabilities per fixture
# ===================================================================
r05_fixture_of = Dict(f.m_id => f for f in r05_card.fixtures)
r05_surface = Dict(zip(unique(vcat([f.home for f in r05_card.fixtures], [f.away for f in r05_card.fixtures])),
                       hha_classify_surface(unique(vcat([f.home for f in r05_card.fixtures],
                                                        [f.away for f in r05_card.fixtures])))))
r05_prob_rows = NamedTuple[]
for f in r05_card.fixtures
    hg, ag = r05_card.results[f.m_id]
    row = Dict{Symbol,Any}(:match_id => f.m_id, :home => f.home, :away => f.away,
                           :home_surface => r05_surface[f.home] ? "turf" : "grass",
                           :score => "$hg-$ag")
    for arm in ("flat_raw", "hier_raw", "flat_optB", "hier_optB"), sel in (:home, :draw, :away)
        p = get(get(r05_probs[arm], f.m_id, Dict()), (group = "1X2", line = 0.0, selection = sel), NaN)
        row[Symbol("$(arm)_p_$(sel)")] = p
    end
    row[:raw_delta_p_home] = row[:hier_raw_p_home] - row[:flat_raw_p_home]
    row[:optB_delta_p_home] = row[:hier_optB_p_home] - row[:flat_optB_p_home]
    push!(r05_prob_rows, NamedTuple(row))
end
r05_prob_frame = DataFrame(r05_prob_rows)
all(isfinite, r05_prob_frame.flat_raw_p_home) || error(
    "1X2 home probability missing for some fixture — check the SelectionKey line for 1X2")
println("\n=== P(HOME), flat vs hierarchical ===")
show(stdout, MIME"text/plain"(),
     select(r05_prob_frame, :home, :away, :home_surface, :score,
            :flat_raw_p_home, :hier_raw_p_home, :raw_delta_p_home,
            :flat_optB_p_home, :hier_optB_p_home, :optB_delta_p_home);
     allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 7. Stake sheets, settlement and the reproduction diff
# ===================================================================
r05_settled = Dict(arm => hha_settle_sheet(r05_slates[arm], r05_card.results,
                                           r05_live.commission_rate; label = arm)
                   for arm in keys(r05_slates))
r05_all_legs = vcat([r05_settled[a] for a in ("flat_raw", "flat_optB", "hier_raw", "hier_optB")]...)
r05_all_legs.fixture = ["$(r05_fixture_of[m].home) v $(r05_fixture_of[m].away)" for m in r05_all_legs.match_id]

r05_repro = hha_reproduction(r05_settled["flat_optB"], r05_live.legs)
println("\n=== REPRODUCTION: flat_optB re-price vs the live orders ===")
@printf("  live %d legs | re-priced %d | shared %d | max |Δrisk| £%.2f | max |Δp_model| %.4f\n",
        r05_repro.n_live, r05_repro.n_repriced, r05_repro.n_shared,
        r05_repro.max_risk_gap, r05_repro.max_p_model_gap)
isempty(r05_repro.only_live) || println("  only live     : ", r05_repro.only_live)
isempty(r05_repro.only_repriced) || println("  only re-priced: ", r05_repro.only_repriced)

r05_summary_rows = NamedTuple[]
for arm in ("flat_raw", "flat_optB", "hier_raw", "hier_optB")
    legs = r05_settled[arm]
    away = legs[(legs.group .== "1X2") .& (legs.selection .== :away), :]
    home = legs[(legs.group .== "1X2") .& (legs.selection .== :home), :]
    turf_away = away[[r05_surface[r05_fixture_of[m].home] for m in away.match_id], :]
    push!(r05_summary_rows, (; arm, n_legs = nrow(legs),
                               risk = sum(legs.risk; init = 0.0),
                               net_pnl = sum(legs.net_pnl; init = 0.0),
                               n_away = nrow(away), away_risk = sum(away.risk; init = 0.0),
                               away_net = sum(away.net_pnl; init = 0.0),
                               n_away_on_turf = nrow(turf_away),
                               n_home = nrow(home), home_net = sum(home.net_pnl; init = 0.0),
                               n_under25 = count(==(:under_25), legs.selection),
                               n_wins = count(==(:win), legs.outcome)))
end
push!(r05_summary_rows, (; arm = "live_ledger (realised fills)", n_legs = nrow(r05_live.legs),
                           risk = sum(r05_live.legs.risk), net_pnl = r05_live.realised_net,
                           n_away = count((r05_live.legs.group .== "1X2") .& (r05_live.legs.selection .== :away)),
                           away_risk = NaN, away_net = NaN, n_away_on_turf = -1,
                           n_home = count((r05_live.legs.group .== "1X2") .& (r05_live.legs.selection .== :home)),
                           home_net = NaN,
                           n_under25 = count(==(:under_25), r05_live.legs.selection), n_wins = -1))
r05_summary = DataFrame(r05_summary_rows)

println("\n=== COUNTERFACTUAL TEARSHEET (full-fill settlement; live row is realised fills) ===")
show(stdout, MIME"text/plain"(), r05_summary; allrows = true, allcols = true)
println("\n\n=== LEGS ===")
show(stdout, MIME"text/plain"(),
     select(r05_all_legs, :arm, :fixture, :group, :line, :selection, :side, :effective_odds,
            :p_model, :p_market, :edge, :risk, :score, :outcome, :net_pnl);
     allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 8. Final report
# ===================================================================
CSV.write(joinpath(R05_OUT_DIR, "r05_probabilities_1x2.csv"), r05_prob_frame)
CSV.write(joinpath(R05_OUT_DIR, "r05_counterfactual_legs.csv"), r05_all_legs)
CSV.write(joinpath(R05_OUT_DIR, "r05_counterfactual_summary.csv"), r05_summary)
CSV.write(joinpath(R05_OUT_DIR, "r05_live_legs.csv"), r05_live.legs)

open(joinpath(R05_OUT_DIR, "r05_slate_report.md"), "w") do io
    println(io, "# r05 counterfactual re-pricing — 2026-09-12, T", R05_T_MINUTES, "\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), ". as_of ", r05_as_of,
            " UTC. Flat arm Run ", R05_FLAT_RUN, "; hierarchical arm `", R05_HIER_NAME, "` run ",
            r05_hier_run, ". Both on Fold ", R05_EXPECTED_FOLD, ".\n")
    @printf(io, "Reproduction of the live orders by `flat_optB`: %d live legs, %d re-priced, %d shared; max |Δrisk| £%.2f, max |Δp_model| %.4f.\n\n",
            r05_repro.n_live, r05_repro.n_repriced, r05_repro.n_shared,
            r05_repro.max_risk_gap, r05_repro.max_p_model_gap)
    println(io, "## Tearsheet\n")
    print(io, gph_markdown_table(r05_summary))
    println(io, "\n## 1X2 home probability\n")
    print(io, gph_markdown_table(select(r05_prob_frame, :home, :away, :home_surface, :score,
        :flat_raw_p_home, :hier_raw_p_home, :raw_delta_p_home,
        :flat_optB_p_home, :hier_optB_p_home, :optB_delta_p_home);
        formats = Dict(:raw_delta_p_home => v -> gph_signed(v; digits = 4),
                       :optB_delta_p_home => v -> gph_signed(v; digits = 4))))
    println(io, "\n## Legs\n")
    print(io, gph_markdown_table(select(r05_all_legs, :arm, :fixture, :group, :line, :selection,
        :side, :effective_odds, :p_model, :p_market, :risk, :score, :outcome, :net_pnl)))
end
close(r05_conn)
println("\nR05_DONE report=", joinpath(R05_OUT_DIR, "r05_slate_report.md"))
