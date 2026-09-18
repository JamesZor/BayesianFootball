# ==============================================================================
# r05 — Counterfactual re-pricing of the 2026-09-12 Scottish Lower card
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# The live account `live_scottish_m12_500` staked the 15:00 card at T−25 from Run 67
# (`m12_joint_hybrid_synergy`, TimeDecay, Fold 43) with the Option B calibrator, priced every
# home side into a 0.40–0.43 band against a market at 0.55–0.60 on the favourites, backed the
# away sides, and closed −£45.89. This runner asks: at the SAME instant, against the SAME
# book, through the SAME pipeline, what would the market-anchored GRW rungs have priced and
# staked?
#
# It is one Saturday. Nine fixtures cannot validate a model and a counterfactual P&L on them
# is an anecdote with a sign; `r04` is the evidence. What this runner can show is MECHANISM:
# whether the supremacy pillar lifts P(home) on the favourites, and whether that removes or
# shrinks the away legs.
#
# ARMS — one pipeline, every arm on Fold 43
#
#   live_optB     Run 67 (TimeDecay hybrid), Option B     the live arm, for reproduction only
#   base_*        m05_joint_grw_baseline (Task 013 b0961bc4)
#   sup040_*      m05_joint_grw_supremacy_w040
#   smile020_*    m05_joint_grw_smile_supremacy_w020
#   smile040_*    m05_joint_grw_smile_supremacy_w040
#   smile070_*    m05_joint_grw_smile_supremacy_w070
#   *_raw no calibrator · *_optB the Option B T−25 calibrator (see the smile caveat in l03)
#
# The clean attribution is `base_raw` → `sup040_raw` → `smile040_raw`: identical dynamics,
# covariate, observation, fold and book, differing only in the pillars. Run 67 is a different
# model (TimeDecay + lineup) and is carried to tie the counterfactual to what was staked.
#
# FILTRATION CONTRACT
#
# * as_of = 2026-09-12 13:35 UTC (T−25); `PreloadedBook` makes later ticks unreachable.
# * Team-level rungs read no lineup, so the provisional-XI gap that stopped Task 008's re-price
#   reproducing the live orders exactly does not touch them — it touches only `live_optB`.
# * Every arm asserts Fold 43 and that it refuses no fixture.
#
# SETTLEMENT at planned risk (full fill), account commission. The realised −£45.89 is printed
# beside the full-fill `live_optB` figure, never subtracted from a candidate directly.
#
# NOTHING IS WRITTEN.
#
# USAGE (mcmc-beast, from /root/BF_grw_market_smile, after r02)
#
#   julia --project -t 16 current_development/grw_market_smile/r05_slate_repricing.jl
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
const R05_CONFIG = GMSConfig()
const R05_DAY = Date(2026, 9, 12)
const R05_T_MINUTES = -25
const R05_BANKROLL = 500.0
const R05_LIVE_ACCOUNT = "live_scottish_m12_500"
const R05_LIVE_SCHEMA = "paper_runbook"
const R05_LIVE_EXPERIMENT = "scottish_lower_joint_player_2426"
const R05_LIVE_RUN = 67
const R05_EXPECTED_FOLD = 43
const R05_REALISED_QUOTED = -45.89
const R05_OUT_DIR = joinpath(R05_CONFIG.save_root, "slate_20260912")

"(arm prefix, run name in the Task 015 namespace, or the pinned baseline)."
const R05_RUNGS = [
    ("base",     "m05_joint_grw_baseline"),
    ("sup040",   "m05_joint_grw_supremacy_w040"),
    ("smile020", "m05_joint_grw_smile_supremacy_w020"),
    ("smile040", "m05_joint_grw_smile_supremacy_w040"),
    ("smile070", "m05_joint_grw_smile_supremacy_w070"),
]

mkpath(R05_OUT_DIR)
println("\n" * "="^96)
println("  r05 COUNTERFACTUAL RE-PRICING — Scottish Lower card ", R05_DAY, " at T", R05_T_MINUTES)
println("="^96)

# %%
# ===================================================================
# 3. Data snapshot, the card, and the live ledger
# ===================================================================
r05_ds = gph_load_data()
r05_conn = MD.paper_connection()
r05_card, r05_dropped = gms_played_card(load_replay_card(r05_conn, R05_DAY))
r05_as_of = as_of_at(r05_card, R05_T_MINUTES)

println("  store      : ", nrow(r05_ds.matches), " matches, latest ", maximum(r05_ds.matches.match_date))
println("  kickoff    : ", r05_card.kickoff, " UTC   as_of: ", r05_as_of, " UTC")
println("  fixtures   : ", length(r05_card.fixtures), " played",
        isempty(r05_dropped) ? "" : " | removed (no score): " * join(["$(f.home) v $(f.away)" for f in r05_dropped], ", "))

r05_live = gms_live_ledger(r05_conn, R05_LIVE_ACCOUNT, R05_DAY; schema = R05_LIVE_SCHEMA)
@printf("  live slate : %d legs, fold %d, run %s | realised net £%+.2f (work package quotes £%+.2f) | commission %.2f%%\n",
        nrow(r05_live.legs), r05_live.slate.fold_idx, r05_live.slate.run_name,
        r05_live.realised_net, R05_REALISED_QUOTED, 100 * r05_live.commission_rate)

# %%
# ===================================================================
# 4. The arms
# ===================================================================
r05_task_db = PostgresStorage(R05_CONFIG.experiment)
r05_slots = ModelSlot[]
r05_live_slot = ModelSlot("live_optB", "Run 67 TD hybrid, Option B", R05_LIVE_EXPERIMENT, "",
                          MD.option_b_calibrator())
gms_load_pinned_slot!(r05_live_slot, R05_LIVE_RUN, r05_ds, r05_card)
push!(r05_slots, r05_live_slot)

r05_run_of = Dict{String,String}()
for (prefix, name) in R05_RUNGS
    experiment, key = if name == GMS_BASELINE_CONTROL.label
        (GMS_BASELINE_CONTROL.experiment, GMS_BASELINE_CONTROL.run_id)
    else
        run_id = gms_run_by_name(r05_task_db, name)
        run_id === nothing && error("no completed $name run — run r02 first")
        (R05_CONFIG.experiment, run_id)
    end
    r05_run_of[prefix] = string(key)
    raw = ModelSlot("$(prefix)_raw", "$name raw", experiment, "")
    gms_load_pinned_slot!(raw, key, r05_ds, r05_card)
    calibrated = ModelSlot("$(prefix)_optB", "$name Option B", experiment, "", MD.option_b_calibrator())
    gms_clone_slot!(calibrated, raw, r05_ds, r05_card)
    push!(r05_slots, raw, calibrated)
end

for slot in r05_slots
    println("  ", rpad(slot.key, 14), " run ", slot.run_name, " | fold ", slot.fold_idx,
            " | covered ", length(slot.covered), "/", length(r05_card.fixtures),
            isempty(slot.fold_warning) ? "" : " | WARNING " * slot.fold_warning,
            @sprintf(" | %.0f s", slot.load_seconds))
    slot.fold_idx == R05_EXPECTED_FOLD || error("$(slot.key) selected fold $(slot.fold_idx); expected $R05_EXPECTED_FOLD")
    isempty(slot.refused) || error("$(slot.key) refuses fixtures: $(slot.refused)")
end

r05_state = ReplayState(r05_ds, r05_conn, r05_card;
                        system = MD.option_b_system(),
                        models = r05_slots,
                        active = "live_optB",
                        bankroll = R05_BANKROLL,
                        account_id = "gms_counterfactual_20260912")
r05_state.clock.t = R05_T_MINUTES

# %%
# ===================================================================
# 5. Pricing at T−25
# ===================================================================
r05_slates = Dict{String,Any}()
r05_probs = Dict{String,Any}()
for slot in r05_slots
    r05_state.active = slot.key
    r05_state.slate = nothing
    slate = reprice!(r05_state)
    isempty(r05_state.tick_error) || error("$(slot.key) failed to price: $(r05_state.tick_error)")
    r05_probs[slot.key] = model_probs_at(r05_state, r05_as_of; slot)
    if slate === nothing
        # A sheet with no leg is a legitimate answer ("stake nothing"), not a failure.
        println("  ", rpad(slot.key, 14), " no legs: ", r05_state.tick_note)
        continue
    end
    isempty(slate.blocked) || @warn "gated fixtures" arm = slot.key n = length(slate.blocked)
    r05_slates[slot.key] = slate
    @printf("  %-14s %2d legs | risk £%.2f | k_risk %.3f | capped %s\n",
            slot.key, nrow(slate.sheet), slate.total_risk, slate.k_risk, slate.capped)
end

# %%
# ===================================================================
# 6. What moved: 1X2 home probability against the book at T−25
# ===================================================================
r05_book = r05_slates["live_optB"].odds
# The quoted book carries the executable price per runner (`odds_close`) and no fair
# probability, so the three 1X2 quotes are de-vigged here by proportional normalisation.
function r05_market_p(match_id, selection)
    rows = r05_book[(r05_book.match_id .== match_id) .& (String.(r05_book.market_name) .== "1X2"), :]
    nrow(rows) == 3 || return NaN
    implied = 1.0 ./ Float64.(rows.odds_close)
    i = findfirst(==(selection), Symbol.(rows.selection))
    return i === nothing ? NaN : implied[i] / sum(implied)
end

r05_prob_rows = NamedTuple[]
for f in r05_card.fixtures
    hg, ag = r05_card.results[f.m_id]
    row = Dict{Symbol,Any}(:match_id => f.m_id, :home => f.home, :away => f.away, :score => "$hg-$ag",
                           :mkt_p_home => r05_market_p(f.m_id, :home))
    for slot in r05_slots, sel in (:home, :away)
        row[Symbol("$(slot.key)_p_$(sel)")] = get(get(r05_probs[slot.key], f.m_id, Dict()),
                                                  (group = "1X2", line = 0.0, selection = sel), NaN)
    end
    push!(r05_prob_rows, NamedTuple(row))
end
r05_prob_frame = DataFrame(r05_prob_rows)
all(isfinite, r05_prob_frame.base_raw_p_home) || error("1X2 home probability missing for a fixture")
println("\n=== P(HOME) at T−25, raw arms ===")
show(stdout, MIME"text/plain"(),
     select(r05_prob_frame, :home, :away, :score, :mkt_p_home, :live_optB_p_home,
            :base_raw_p_home, :sup040_raw_p_home, :smile020_raw_p_home, :smile040_raw_p_home,
            :smile070_raw_p_home); allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 7. Stake sheets, settlement, reproduction
# ===================================================================
r05_settled = Dict(k => gms_settle_sheet(s, r05_card.results, r05_live.commission_rate; label = k)
                   for (k, s) in r05_slates)
r05_fixture_of = Dict(f.m_id => f for f in r05_card.fixtures)
r05_all_legs = vcat([r05_settled[s.key] for s in r05_slots if haskey(r05_settled, s.key)]...)
r05_all_legs.fixture = ["$(r05_fixture_of[m].home) v $(r05_fixture_of[m].away)" for m in r05_all_legs.match_id]

r05_repro = gms_reproduction(r05_settled["live_optB"], r05_live.legs)
@printf("\n  reproduction (live_optB vs live orders): live %d | re-priced %d | shared %d | max |Δrisk| £%.2f | max |Δp_model| %.4f\n",
        r05_repro.n_live, r05_repro.n_repriced, r05_repro.n_shared, r05_repro.max_risk_gap,
        r05_repro.max_p_model_gap)

r05_summary = DataFrame([gms_arm_summary(s.key, get(r05_settled, s.key, gms_empty_legs()))
                         for s in r05_slots])
push!(r05_summary, (; arm = "live ledger (realised fills)", n_legs = nrow(r05_live.legs),
                      risk = sum(r05_live.legs.risk), net_pnl = r05_live.realised_net,
                      n_away = count((r05_live.legs.group .== "1X2") .& (r05_live.legs.selection .== :away)),
                      away_risk = NaN, away_net = NaN,
                      n_home = count((r05_live.legs.group .== "1X2") .& (r05_live.legs.selection .== :home)),
                      home_risk = NaN, home_net = NaN,
                      n_under25 = count(==(:under_25), r05_live.legs.selection), n_wins = -1))
println("\n=== COUNTERFACTUAL TEARSHEET (full fill; last row realised) ===")
show(stdout, MIME"text/plain"(), r05_summary; allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 8. Final report
# ===================================================================
CSV.write(joinpath(R05_OUT_DIR, "r05_probabilities_1x2.csv"), r05_prob_frame)
CSV.write(joinpath(R05_OUT_DIR, "r05_counterfactual_legs.csv"), r05_all_legs)
CSV.write(joinpath(R05_OUT_DIR, "r05_counterfactual_summary.csv"), r05_summary)

open(joinpath(R05_OUT_DIR, "r05_slate_report.md"), "w") do io
    println(io, "# r05 counterfactual re-pricing — 2026-09-12, T", R05_T_MINUTES, "\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), ". as_of ", r05_as_of,
            " UTC. All arms Fold ", R05_EXPECTED_FOLD, ". Runs: ",
            join(["$k `$(v)`" for (k, v) in sort(collect(r05_run_of))], "; "), ".\n")
    @printf(io, "Reproduction of the live orders by `live_optB`: %d live legs, %d re-priced, %d shared; max |Δrisk| £%.2f, max |Δp_model| %.4f.\n\n",
            r05_repro.n_live, r05_repro.n_repriced, r05_repro.n_shared, r05_repro.max_risk_gap,
            r05_repro.max_p_model_gap)
    println(io, "## Tearsheet (full-fill settlement)\n")
    print(io, gph_markdown_table(r05_summary;
        formats = Dict(c => (v -> gph_num(v; digits = 2)) for c in
                       (:risk, :net_pnl, :away_risk, :away_net, :home_risk, :home_net))))
    println(io, "\n## 1X2 P(home) at T−25\n")
    println(io, "`mkt_p_home` is the de-vigged book the pipeline quoted at as_of.\n")
    print(io, gph_markdown_table(select(r05_prob_frame, :home, :away, :score, :mkt_p_home,
        :live_optB_p_home, :base_raw_p_home, :base_optB_p_home, :sup040_raw_p_home,
        :smile020_raw_p_home, :smile040_raw_p_home, :smile070_raw_p_home, :smile040_optB_p_home);
        formats = Dict(c => (v -> gph_num(v; digits = 3)) for c in
                       (:mkt_p_home, :live_optB_p_home, :base_raw_p_home, :base_optB_p_home,
                        :sup040_raw_p_home, :smile020_raw_p_home, :smile040_raw_p_home,
                        :smile070_raw_p_home, :smile040_optB_p_home))))
    println(io, "\n## Legs\n")
    print(io, gph_markdown_table(select(r05_all_legs, :arm, :fixture, :group, :line, :selection,
        :side, :effective_odds, :p_model, :p_market, :risk, :score, :outcome, :net_pnl)))
end
close(r05_conn)
println("\nR05_DONE report=", joinpath(R05_OUT_DIR, "r05_slate_report.md"))
