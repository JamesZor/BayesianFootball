# ==============================================================================
# r09 — Rung 5 (m12 hybrid + contextual HA) and counterfactual re-pricing of 2026-09-12
# ==============================================================================
#
# WHAT THIS IS AND IS NOT
#
# Two steps in one runner, because the second needs the first's Fold 43:
#
#   A. Fit `m12_joint_hybrid_contextual` — the Gen 4 TimeDecay hybrid (Run 67's recipe)
#      with the Phase 2 HA slot and the contextual terms named by R09_TERMS — over the
#      24/25 → 26/27 extension splitter (43 folds), persisted to `scottish_lower_contextual_ha`.
#      Folds 1–40 hold the same 710 walk-forward fixtures r08 scores, so the same run serves
#      as ladder rung 5 in r08 (restricted to the panel) and as the slate arm here.
#   B. Re-price the 2026-09-12 card through the replay engine at T−25, exactly as Phase 1's
#      r05 did, with the contextual arm in place of the hierarchical one.
#
# One Saturday. Nine fixtures show mechanism — whether turf / timing terms move P(home)
# on the turf grounds and whether the away legs shrink — not evidence. r08 is the evidence.
#
# FOUR ARMS, ONE PIPELINE (Phase 1 r05's, unchanged)
#   flat_raw   Run 67 (flat HA)                       no calibrator
#   flat_optB  Run 67                                 Option B T−25   ← the live arm
#   ctx_raw    m12_joint_hybrid_contextual  Fold 43   no calibrator
#   ctx_optB   m12_joint_hybrid_contextual  Fold 43   Option B T−25
#
# FILTRATION: r05's contract — as_of = T−25, `PreloadedBook` by `searchsortedlast`,
# `PreloadedLineups` with `scraped_at <= as_of`, both arms on Fold 43 (asserted). The
# contextual design reads only the registry (dated: Dumbarton turf from 2026), the
# kickoff date, and kickoffs on earlier days. NOTHING IS WRITTEN to any paper schema.
#
# USAGE (mcmc-beast, from /root/BF_hier_ha_slate — the 2026-09-12 DataStore cache)
#
#   R09_TERMS=turf_asym,turf_gen,turf_pace julia --project -t 16 .../r09_slate_repricing.jl
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

include(joinpath(@__DIR__, "l04_contextual_loader.jl"))
include(joinpath(@__DIR__, "l03_slate.jl"))

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R09_CONFIG = CtxConfig()
const R09_NAME = "m12_joint_hybrid_contextual"
const R09_TERM_CTORS = Dict(:turf_asym => ctx_turf_asym, :turf_gen => ctx_turf_gen,
                            :turf_pace => ctx_turf_pace, :midweek => ctx_midweek,
                            :rest_diff => ctx_rest_diff)
const R09_TERMS = let raw = strip(get(ENV, "R09_TERMS", "turf_asym,turf_gen,turf_pace,midweek,rest_diff"))
    Symbol.(strip.(split(raw, ",")))
end
all(in(keys(R09_TERM_CTORS)), R09_TERMS) || error("R09_TERMS: unknown term in $(R09_TERMS)")
const R09_DAY = Date(2026, 9, 12)
const R09_T_MINUTES = -25
const R09_BANKROLL = 500.0
const R09_LIVE_ACCOUNT = "live_scottish_m12_500"
const R09_FLAT_EXPERIMENT = "scottish_lower_joint_player_2426"
const R09_FLAT_RUN = 67
const R09_EXPECTED_FOLDS = 43
const R09_OUT_DIR = joinpath(R09_CONFIG.save_root, "slate_20260912")

mkpath(R09_OUT_DIR)
println("\n" * "="^96)
println("  r09 RUNG 5 + COUNTERFACTUAL — ", R09_NAME, " with terms ", join(R09_TERMS, ", "))
println("  card: Scottish Lower ", R09_DAY, " at T", R09_T_MINUTES, "  | flat: Run ", R09_FLAT_RUN)
println("="^96)

# %%
# ===================================================================
# 3. Step A — fit rung 5 over 43 folds (or load it)
# ===================================================================
r09_ds = gph_load_data()
r09_db = gph_database(R09_CONFIG.experiment)
r09_splitter = gph_splitter(R09_CONFIG.extension_seasons)
r09_model = ctx_m12(R09_NAME, Tuple(R09_TERM_CTORS[t]() for t in R09_TERMS))
r09_fit_config = FitConfig(
    name = R09_NAME, model = r09_model, splitter = r09_splitter,
    sampler = ctx_production_sampler(R09_CONFIG),
    execution = hha_execution(ctx_as_hha(R09_CONFIG)),
    tags = vcat(CTX_TAGS, ["26/27", "rung5"]),
    description = CTX_DESCRIPTIONS[R09_NAME] * " Terms: " * join(R09_TERMS, ", ") * ".",
    save_dir = joinpath(R09_CONFIG.save_root, R09_NAME))
println("  store: ", nrow(r09_ds.matches), " matches, latest ", maximum(r09_ds.matches.match_date))
println("  covariates: ", join(string.(predictor_name.(r09_model.covariates)), ", "))

r09_run = gph_completed_run(r09_db, r09_fit_config)
r09_fit_row = nothing
if r09_run === nothing
    save_model(r09_db, R09_NAME, r09_model; description = r09_fit_config.description,
               tags = r09_fit_config.tags)
    inputs = gph_fold_inputs(r09_ds, r09_splitter, r09_model)
    length(inputs.feature_sets) == R09_EXPECTED_FOLDS || error(
        "extension splitter gave $(length(inputs.feature_sets)) folds; expected $R09_EXPECTED_FOLDS")
    filtration = gph_filtration_report(r09_ds, inputs)
    all(filtration.ordered) || error("training/OOS kickoff ordering violated")
    unmapped = hha_unmapped_home_report(inputs)
    CSV.write(joinpath(R09_CONFIG.save_root, "r09_unmapped_home_$(R09_NAME).csv"), unmapped)
    CSV.write(joinpath(R09_CONFIG.save_root, "r09_oos_design_$(R09_NAME).csv"), ctx_oos_design(inputs))
    fit = ctx_sample(r09_fit_config, inputs, R09_CONFIG;
                     checkpoint_dir = joinpath(R09_CONFIG.save_root, R09_NAME, "checkpoints"))
    d = fit.diagnostics
    sites = ctx_site_report(fit)
    CSV.write(joinpath(R09_CONFIG.save_root, "r09_sites_$(R09_NAME).csv"), sites)
    coefs = vcat([ctx_coefficients(fit, f) for f in eachindex(fit.folds)]...)
    CSV.write(joinpath(R09_CONFIG.save_root, "r09_coefficients_$(R09_NAME).csv"), coefs)
    @printf("  audit: R̂ %.4f | site R̂ %.4f | ESS bulk %.0f tail %.0f | div %d | BFMI %.3f | %s\n",
            d.max_rhat, maximum(sites.max_rhat), d.min_ess_bulk, d.min_ess_tail, d.n_divergent,
            d.min_bfmi, d.passed ? "PASS" : "FAIL: " * join(d.failures, "; "))
    d.passed || error("$R09_NAME failed its convergence gate — not persisted, not priced")
    persisted = gph_thin_for_persistence(fit, inputs, R09_CONFIG.persist_stride)
    global r09_run = gph_save_and_verify(r09_db, persisted)
    global r09_fit_row = (; hha_convergence_row(R09_NAME, persisted, ctx_as_hha(R09_CONFIG); run_id = r09_run)...,
                            max_site_rhat = maximum(sites.max_rhat), n_unmapped_home = nrow(unmapped))
    CSV.write(joinpath(R09_CONFIG.save_root, "r09_rung5_run.csv"), DataFrame([r09_fit_row]))
    println("R09_FIT_DONE ", r09_run)
    fit = nothing; persisted = nothing; inputs = nothing; GC.gc()
else
    println("  rung 5 already persisted as run ", r09_run, " — loading, not sampling")
end

# %%
# ===================================================================
# 4. Step B — the card, the live ledger, the arms
# ===================================================================
r09_conn = MD.paper_connection()
r09_card, r09_dropped = hha_played_card(load_replay_card(r09_conn, R09_DAY))
r09_as_of = as_of_at(r09_card, R09_T_MINUTES)
r09_live = hha_live_ledger(r09_conn, R09_LIVE_ACCOUNT, R09_DAY)
@printf("  card: %d fixtures | as_of %s UTC | live: %d legs, realised £%+.2f\n",
        length(r09_card.fixtures), r09_as_of, nrow(r09_live.legs), r09_live.realised_net)

r09_slots = [
    (ModelSlot("flat_raw", "Run 67 flat HA, raw", R09_FLAT_EXPERIMENT, ""), R09_FLAT_RUN),
    (ModelSlot("flat_optB", "Run 67 flat HA, Option B", R09_FLAT_EXPERIMENT, "",
               MD.option_b_calibrator()), R09_FLAT_RUN),
    (ModelSlot("ctx_raw", "contextual HA, raw", R09_CONFIG.experiment, ""), r09_run),
    (ModelSlot("ctx_optB", "contextual HA, Option B", R09_CONFIG.experiment, "",
               MD.option_b_calibrator()), r09_run),
]
for (slot, key) in r09_slots
    hha_load_pinned_slot!(slot, key, r09_ds, r09_card)
    println("  ", rpad(slot.key, 10), " run ", slot.run_name, " | fold ", slot.fold_idx,
            " | covered ", length(slot.covered), "/", length(r09_card.fixtures))
    slot.fold_idx == R09_EXPECTED_FOLDS || error("$(slot.key) selected fold $(slot.fold_idx)")
    isempty(slot.refused) || error("$(slot.key) refuses fixtures: $(slot.refused)")
end

r09_state = ReplayState(r09_ds, r09_conn, r09_card; system = MD.option_b_system(),
                        models = [s for (s, _) in r09_slots], active = "flat_optB",
                        bankroll = R09_BANKROLL, account_id = "ctx_counterfactual_20260912")
r09_state.clock.t = R09_T_MINUTES

# %%
# ===================================================================
# 5. Pricing at T−25
# ===================================================================
r09_slates = Dict{String,Any}()
r09_probs = Dict{String,Any}()
for (slot, _) in r09_slots
    r09_state.active = slot.key
    r09_state.slate = nothing
    slate = reprice!(r09_state)
    isempty(r09_state.tick_error) || error("$(slot.key) failed to price: $(r09_state.tick_error)")
    slate === nothing && error("$(slot.key) priced nothing: $(r09_state.tick_note)")
    r09_slates[slot.key] = slate
    r09_probs[slot.key] = model_probs_at(r09_state, r09_as_of; slot)
    @printf("  %-10s %2d legs | risk £%.2f\n", slot.key, nrow(slate.sheet), slate.total_risk)
end

# %%
# ===================================================================
# 6. What moved, by the contextual design of each fixture
# ===================================================================
r09_bridge = ctx_bridge(r09_ds, ContextualMatchFeature())
r09_fixture_of = Dict(f.m_id => f for f in r09_card.fixtures)
r09_design = Dict(f.m_id => ctx_design(r09_bridge, f.home, f.away, R09_DAY, nothing; strict = false)
                  for f in r09_card.fixtures)
r09_prob_rows = NamedTuple[]
for f in r09_card.fixtures
    hg, ag = r09_card.results[f.m_id]
    d = r09_design[f.m_id]
    row = Dict{Symbol,Any}(:match_id => f.m_id, :home => f.home, :away => f.away,
                           :turf_home => d.turf_home, :turf_away => d.turf_away,
                           :rest_diff => d.rest_diff, :score => "$hg-$ag")
    for arm in ("flat_raw", "ctx_raw", "flat_optB", "ctx_optB"), sel in (:home, :draw, :away)
        row[Symbol("$(arm)_p_$(sel)")] =
            get(get(r09_probs[arm], f.m_id, Dict()), (group = "1X2", line = 0.0, selection = sel), NaN)
    end
    for arm in ("flat_raw", "ctx_raw")
        row[Symbol("$(arm)_p_over25")] =
            get(get(r09_probs[arm], f.m_id, Dict()), (group = "OverUnder", line = 2.5, selection = :over_25), NaN)
    end
    row[:raw_delta_p_home] = row[:ctx_raw_p_home] - row[:flat_raw_p_home]
    row[:optB_delta_p_home] = row[:ctx_optB_p_home] - row[:flat_optB_p_home]
    row[:raw_delta_p_away] = row[:ctx_raw_p_away] - row[:flat_raw_p_away]
    push!(r09_prob_rows, NamedTuple(row))
end
r09_prob_frame = DataFrame(r09_prob_rows)
println("\n=== 1X2 BY FIXTURE, flat vs contextual ===")
show(stdout, MIME"text/plain"(),
     select(r09_prob_frame, :home, :away, :turf_home, :turf_away, :score, :flat_raw_p_home,
            :ctx_raw_p_home, :raw_delta_p_home, :raw_delta_p_away, :flat_optB_p_home,
            :ctx_optB_p_home, :optB_delta_p_home); allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 7. Sheets, settlement, reproduction
# ===================================================================
r09_arms = ("flat_raw", "flat_optB", "ctx_raw", "ctx_optB")
r09_settled = Dict(a => hha_settle_sheet(r09_slates[a], r09_card.results, r09_live.commission_rate;
                                         label = a) for a in r09_arms)
r09_legs = vcat([r09_settled[a] for a in r09_arms]...)
r09_legs.fixture = ["$(r09_fixture_of[m].home) v $(r09_fixture_of[m].away)" for m in r09_legs.match_id]
r09_repro = hha_reproduction(r09_settled["flat_optB"], r09_live.legs)
@printf("\n  reproduction flat_optB vs live: live %d | re-priced %d | shared %d | max |Δrisk| £%.2f\n",
        r09_repro.n_live, r09_repro.n_repriced, r09_repro.n_shared, r09_repro.max_risk_gap)

r09_summary = DataFrame([begin
    legs = r09_settled[a]
    away = legs[(legs.group .== "1X2") .& (legs.selection .== :away), :]
    turf_away = away[[r09_design[m].turf_home == 1.0 for m in away.match_id], :]
    (; arm = a, n_legs = nrow(legs), risk = sum(legs.risk; init = 0.0),
       net_pnl = sum(legs.net_pnl; init = 0.0), n_away = nrow(away),
       away_risk = sum(away.risk; init = 0.0), away_net = sum(away.net_pnl; init = 0.0),
       n_away_on_turf = nrow(turf_away), away_risk_on_turf = sum(turf_away.risk; init = 0.0))
end for a in r09_arms])
println("\n=== COUNTERFACTUAL TEARSHEET (full-fill settlement) ===")
show(stdout, MIME"text/plain"(), r09_summary; allrows = true, allcols = true)
println()

# %%
# ===================================================================
# 8. Report
# ===================================================================
CSV.write(joinpath(R09_OUT_DIR, "r09_probabilities.csv"), r09_prob_frame)
CSV.write(joinpath(R09_OUT_DIR, "r09_counterfactual_legs.csv"), r09_legs)
CSV.write(joinpath(R09_OUT_DIR, "r09_counterfactual_summary.csv"), r09_summary)
open(joinpath(R09_OUT_DIR, "r09_slate_report.md"), "w") do io
    println(io, "# r09 counterfactual re-pricing — 2026-09-12, T", R09_T_MINUTES, " — Task 008 Phase 2\n")
    println(io, "Generated ", Dates.format(now(), "yyyy-mm-dd HH:MM"), ". as_of ", r09_as_of,
            " UTC. Flat arm Run ", R09_FLAT_RUN, "; contextual arm `", R09_NAME, "` run ", r09_run,
            " (terms: ", join(R09_TERMS, ", "), "). Both on Fold ", R09_EXPECTED_FOLDS, ".\n")
    @printf(io, "Reproduction by `flat_optB`: %d live legs, %d re-priced, %d shared; max |Δrisk| £%.2f. Live realised £%+.2f.\n\n",
            r09_repro.n_live, r09_repro.n_repriced, r09_repro.n_shared, r09_repro.max_risk_gap,
            r09_live.realised_net)
    r09_fit_row === nothing || (println(io, "## Rung 5 fit\n"); print(io, gph_markdown_table(DataFrame([r09_fit_row]))); println(io))
    println(io, "## Tearsheet\n")
    print(io, gph_markdown_table(r09_summary))
    println(io, "\n## 1X2 by fixture\n")
    print(io, gph_markdown_table(select(r09_prob_frame, :home, :away, :turf_home, :turf_away, :score,
        :flat_raw_p_home, :ctx_raw_p_home, :raw_delta_p_home, :raw_delta_p_away,
        :flat_optB_p_home, :ctx_optB_p_home, :optB_delta_p_home, :flat_raw_p_over25, :ctx_raw_p_over25);
        formats = Dict(:raw_delta_p_home => v -> gph_signed(v; digits = 4),
                       :raw_delta_p_away => v -> gph_signed(v; digits = 4),
                       :optB_delta_p_home => v -> gph_signed(v; digits = 4))))
    println(io, "\n## Legs\n")
    print(io, gph_markdown_table(select(r09_legs, :arm, :fixture, :group, :line, :selection,
        :effective_odds, :p_model, :p_market, :risk, :score, :outcome, :net_pnl)))
end
close(r09_conn)
println("\nR09_DONE report=", joinpath(R09_OUT_DIR, "r09_slate_report.md"))
