# ==============================================================================
# r01 — Cross-paradigm market trust pruning sweep (Task 018)
# ==============================================================================
#
# WHAT THIS IS
#
# A portfolio-allocation study over three already-fitted Scottish Lower models. It asks which
# directional market additions improve the exact Option B slate portfolio, and whether those
# answers survive a change of model paradigm (un-smiled team state, lineup hybrid, smile spine).
# Every posterior prediction is fold-held-out. No model is sampled here.
#
# WHAT IS HELD FIXED
#
# Betfair TWA(-20, 0] close; the common 24/25 + 25/26 fixture panel; Option B price, allocator,
# fractional-Kelly shrinkage, drawdown risk, cap, filter and daily grouping. Phase 1 changes one
# directional trust weight at a time. Phase 2 changes only the three conviction-tier weights.
#
# T012 CONTRACT
#
# `:excise_pruned` (the research default) omits a WHOLE market when all its directions have zero
# trust. `:retain_zero_trust` keeps the seven-market catalog and diagnoses the legacy payoff-matrix
# reprice. BookSpec cannot admit one side of a two-sided market, so this runner does not falsely
# claim selection-level coordinate excision: activating Under 3.5 also admits zero-trust Over 3.5
# to the Kelly geometry.
#
# FILTRATION / COMPARABILITY
#
# Fits are loaded by immutable UUID (the spine by its canonical name, then UUID recorded in the
# output) and restricted to the intersection of target-season latent IDs. The final panel is the
# fixtures every model can build under the excised P0 book. A candidate must preserve that panel.
# Survivor selection and tier scoring use the same settlement period, so Phase 2 is descriptive,
# not a fresh out-of-sample promotion test.
#
# OUTPUTS
#
# current_development/market_pruning_harness/results/
#   sweep_summary.csv, line_screening.csv, conviction_tiers.csv,
#   t012_contrast.csv, gates.csv, MARKET_PRUNING_REPORT.md
#
# USAGE (mcmc-beast; no MCMC is launched)
#
#   julia --project -t 16 current_development/market_pruning_harness/r01_sweep.jl
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
using LibPQ
using Printf
using UUIDs

# The spine fit persists chains with a prototype model type and a detached SmileLatents panel.
# This evaluation loader defines that type and deterministically rebuilds the panel (T010).
include(joinpath(@__DIR__, "..", "grw_smile_spine", "l02_evaluation.jl"))
include(joinpath(@__DIR__, "l01_pruning.jl"))
using .MarketPruningHarness

# %%
# ===================================================================
# 2. Configuration
# ===================================================================
const R01_OUTPUT_DIR = joinpath(@__DIR__, "results")
const R01_TARGET_SEASONS = ["24/25", "25/26"]
const R01_EXPECTED_OOS = 710
const R01_MODE = :excise_pruned
const R01_CANDIDATE_TRUST = 1.0 / 1.4
const R01_GATE_TOL = 1.0e-9

# The work-package prompt spells the spine UUID `...883a4c`; the completed Task-016 artefact and
# r06/r08 outputs record `...88e79a`. Loading by canonical name refuses either a typo or a stale
# address, and the resolved UUID is recorded below.
const R01_REFERENCES = [
    (model = "m05_joint_production_wealth", experiment = "scottish_lower_joint_2426",
     run_id = UUID("5eff755c-3591-48d1-a2cc-5fc2744ddf88"), kind = :standard),
    (model = "m12_joint_hybrid_synergy", experiment = "scottish_lower_joint_player_2426",
     run_id = UUID("132df5c2-c742-4e95-8693-3aeb2b2cbaef"), kind = :standard),
    (model = "m05_joint_grw_smile_spine_w040", experiment = "scottish_lower_grw_smile_spine",
     run_id = UUID("582035c0-e145-44f7-9f40-89e25388e79a"), kind = :smile),
]

const R01_PRUNING_CONFIG = PruningConfig(
    t012_mode = R01_MODE,
    candidate_trust = R01_CANDIDATE_TRUST,
    initial_bankroll = 1_000.0,
    return_tolerance_pp = 1.0e-9,
    bootstrap = false,
    seed = 18,
)

const R01_TIER_GRID = TierGrid(
    tier1 = collect(0.30:0.05:0.45),
    tier2 = collect(0.15:0.05:0.25),
    tier3 = collect(0.00:0.05:0.10),
)

# Task 016 is the only requested model with a published result under this exact Option B contract
# and common 632-fixture close panel. The other paradigms have published portfolios under different
# books/policies; calling those an Option B reproduction would be false.
const R01_S1_PUBLISHED = Dict(
    "m05_joint_grw_smile_spine_w040" => (n_panel = 632, n_bets = 1232,
                                          total_return_pct = 469.353509772741),
)

# %%
# ===================================================================
# 3. Runtime, data snapshot, and immutable model addresses
# ===================================================================
function r01_inventory_row(ref)
    storage = PostgresStorage(ref.experiment)
    conn = BayesianFootball.Training.Inference._db_connect(storage)
    try
        query = LibPQ.execute(conn, """
            SELECT r.experiment_name, r.name, r.status,
                   COUNT(fr.*) AS n_folds,
                   COUNT(*) FILTER (WHERE fr.converged) AS converged_folds
            FROM runs r
            LEFT JOIN fold_results fr ON fr.run_id = r.run_id
            WHERE r.run_id = \$1::uuid
            GROUP BY r.experiment_name, r.name, r.status;
        """, (string(ref.run_id),))
        frame = DataFrame(query)
        close(query)
        nrow(frame) == 1 || error("run $(ref.run_id) did not resolve exactly once")
        row = frame[1, :]
        String(row.experiment_name) == ref.experiment || error(
            "run $(ref.run_id) resolved in $(row.experiment_name), expected $(ref.experiment)")
        String(row.name) == ref.model || error(
            "run $(ref.run_id) is named $(row.name), expected $(ref.model)")
        String(row.status) == "completed" || error("$(ref.model) status is $(row.status)")
        n_folds = Int(row.n_folds)
        converged_folds = Int(row.converged_folds)
        n_folds > 0 && converged_folds == n_folds || error(
            "$(ref.model) converged $converged_folds/$n_folds folds")
        return (; model = ref.model, experiment = ref.experiment,
                  run_id = string(ref.run_id), n_folds, converged_folds)
    finally
        close(conn)
    end
end

# Old standard-fit blobs contain a pre-builder JointGammaPoissonObservation type and cannot be
# deserialised at current HEAD. Their exact CountLatents panel is relationally reconstructed from
# match_latents instead. This is the same function load_fit calls immediately after deserialising
# the blob; immutable UUID/name/status and every fold convergence flag are checked above first.
function r01_load_count_latents(ref)
    storage = PostgresStorage(ref.experiment)
    conn = BayesianFootball.Training.Inference._db_connect(storage)
    try
        latents = BayesianFootball.Training.Inference._db_load_count_latents(conn, ref.run_id)
        latents === nothing && error("$(ref.model) has no relational match_latents")
        return latents
    finally
        close(conn)
    end
end

function r01_load_models(ds)
    config = gss_config()
    splitter = gph_splitter(config.extension_seasons)
    latent_dir = joinpath(config.save_root, "latents")

    sources = Dict{String,Any}()
    resolved = NamedTuple[]
    for ref in R01_REFERENCES
        inventory = r01_inventory_row(ref)
        latents = if ref.kind === :smile
            arm = only(filter(a -> a.label == ref.model, gss_arms(config)))
            arm.run_id == ref.run_id || error(
                "$(ref.model) resolved to $(arm.run_id), expected $(ref.run_id)")
            gms_load_arm(arm, ds; splitter, latent_dir).latents
        else
            r01_load_count_latents(ref)
        end
        sources[ref.model] = latents
        push!(resolved, (; inventory...,
                         latent_type = string(nameof(typeof(latents))),
                         n_latents = BayesianFootball.Models.n_matches(latents)))
    end
    return sources, DataFrame(resolved)
end

function r01_target_panel(ds, sources)
    season_of = Dict(Int(row.match_id) => String(row.season) for row in eachrow(ds.matches))
    panels = [Set(Int(match_id) for match_id in BayesianFootball.Models.latent_match_ids(latents)
                  if get(season_of, Int(match_id), "") in R01_TARGET_SEASONS)
              for latents in values(sources)]
    return sort!(collect(reduce(intersect, panels)))
end

function r01_common_panel(ds, sources, odds, catalog, base_book)
    panel = r01_target_panel(ds, sources)
    length(panel) == R01_EXPECTED_OOS || error(
        "latent intersection has $(length(panel)) fixtures; expected $R01_EXPECTED_OOS")

    weights = baseline_weights(catalog)
    spec = pruning_book_spec(base_book, catalog, weights; mode = :excise_pruned)
    buildable = Set(panel)
    for (model, latents) in sources
        restricted = gms_restrict_latents(latents, panel)
        books, report = Portfolio.build_books_reported(
            spec, restricted, odds, ds; converged = true, gated = true, quiet = true)
        isempty(report.errored) || error("$model P0 book errors: $(report.errored)")
        intersect!(buildable, Set(book.m_id for book in books))
    end
    isempty(buildable) && error("no common buildable P0 fixtures")
    return sort!(collect(buildable))
end

# %%
# ===================================================================
# 4. Execute the two-phase pruning protocol
# ===================================================================
function r01_execute()
    mkpath(R01_OUTPUT_DIR)
    catalog = default_catalog()
    candidates = default_candidates()
    system = MatchDay.option_b_system()
    base_book = system.book
    base_policy = system.policy

    println("\n", "="^104)
    println("  TASK 018 — CROSS-PARADIGM MARKET PRUNING")
    println("  mode: ", R01_MODE, "  candidate trust: ", round(R01_CANDIDATE_TRUST; digits = 6))
    println("  host: ", gethostname(), "  Julia threads: ", Threads.nthreads())
    println("="^104)

    println("\n[1/7] Loading cached Scottish Lower data and immutable fits...")
    ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
    odds = gms_betfair_closing_odds(ds)
    sources, inventory = r01_load_models(ds)
    CSV.write(joinpath(R01_OUTPUT_DIR, "model_inventory.csv"), inventory)
    for row in eachrow(inventory)
        @printf("  %-36s folds=%2d  %-13s  %s\n",
                row.model, row.n_folds, row.latent_type, row.run_id)
    end

    println("\n[2/7] Freezing the common, buildable P0 panel...")
    panel = r01_common_panel(ds, sources, odds, catalog, base_book)
    panel_set = Set(panel)
    panel_odds = filter(:match_id => in(panel_set), odds)
    restricted = Dict(model => gms_restrict_latents(latents, panel)
                      for (model, latents) in sources)
    println("  latent intersection: ", R01_EXPECTED_OOS,
            "  common P0 books: ", length(panel), "  excluded: ", R01_EXPECTED_OOS - length(panel))

    line_frames = DataFrame[]
    tier_frames = DataFrame[]
    t012_frames = DataFrame[]
    gate_rows = NamedTuple[]
    s1_results = Dict{String,Any}()

    # S1 is intentionally evaluated on the published Option B geometry and its own buildable
    # target-season panel. The excised sweep panel is smaller (a fixture with no active-market
    # quote cannot be rescued by an all-zero fringe market), so comparing its P0 directly with
    # r06's retained canonical book would conflate T012 with fixture coverage.
    target_panel = r01_target_panel(ds, sources)
    target_odds = filter(:match_id => in(Set(target_panel)), odds)
    s1_outputs = Vector{Any}(undef, length(R01_REFERENCES))
    Threads.@threads for index in eachindex(R01_REFERENCES)
        ref = R01_REFERENCES[index]
        model = ref.model
        target_latents = gms_restrict_latents(sources[model], target_panel)
        books, report = Portfolio.build_books_reported(
            base_book, target_latents, target_odds, ds;
            converged = true, gated = true, quiet = true)
        policy = pruning_policy(base_policy, catalog, baseline_weights(catalog))
        result = Portfolio.simulate_portfolio(
            policy, books, report; initial_bankroll = R01_PRUNING_CONFIG.initial_bankroll,
            bootstrap = false)
        s1_outputs[index] = (; result, books, report)
    end
    for (index, ref) in enumerate(R01_REFERENCES)
        s1_results[ref.model] = s1_outputs[index]
    end

    println("\n[3/7] Gate S0 — quantifying zero-trust market repricing...")
    t012_outputs = Vector{Any}(undef, length(R01_REFERENCES))
    Threads.@threads for index in eachindex(R01_REFERENCES)
        ref = R01_REFERENCES[index]
        t012_outputs[index] = run_t012_contrast(
            ref.model, restricted[ref.model], panel_odds, ds, base_book, base_policy;
            catalog, config = R01_PRUNING_CONFIG)
    end
    t012_by_model = Dict{String,Any}()
    for (index, ref) in enumerate(R01_REFERENCES)
        model = ref.model
        contrast = t012_outputs[index]
        t012_by_model[model] = contrast
        push!(t012_frames, contrast.row)
        row = only(eachrow(contrast.row))
        @printf("  %-36s retained − excised return = %+9.3f pp  ledgers identical=%s\n",
                model, row.delta_return_pp, row.ledgers_identical)
    end

    println("\n[4/7] Phase 1 — one directional line at a time...")
    screening_outputs = Vector{Any}(undef, length(R01_REFERENCES))
    Threads.@threads for index in eachindex(R01_REFERENCES)
        ref = R01_REFERENCES[index]
        screening_outputs[index] = run_line_screening(
            ref.model, restricted[ref.model], panel_odds, ds, base_book, base_policy;
            catalog, candidates, config = R01_PRUNING_CONFIG)
    end
    screening_by_model = Dict{String,Any}()
    for (index, ref) in enumerate(R01_REFERENCES)
        model = ref.model
        screening = screening_outputs[index]
        screening_by_model[model] = screening
        push!(line_frames, screening.rows)
        for row in eachrow(screening.rows[screening.rows.classification .!= "baseline", :])
            @printf("  %-36s %-11s %-31s added ROI=%+8.2f%% Δreturn=%+9.2f pp core=%.3f\n",
                    model, row.candidate_label, row.classification,
                    row.added_roi_pct, row.delta_return_pp, row.core_stake_vs_p0)
        end
    end
    line_screening = vcat(line_frames...; cols = :union)

    println("\n[5/7] Gates S1/S2 — published Option B reproduction and score-grid coherence...")
    for ref in R01_REFERENCES
        model = ref.model
        published = s1_results[model]
        if haskey(R01_S1_PUBLISHED, model)
            expected = R01_S1_PUBLISHED[model]
            observed = published.result.summary
            pass = length(published.books) == expected.n_panel && observed.n_bets == expected.n_bets &&
                   abs(observed.total_return_pct - expected.total_return_pct) <= 1.0e-6
            detail = @sprintf("panel %d/%d, bets %d/%d, return %.6f/%.6f",
                              length(published.books), expected.n_panel,
                              observed.n_bets, expected.n_bets,
                              observed.total_return_pct, expected.total_return_pct)
            push!(gate_rows, (gate = "S1 published Option B reproduction", model,
                              pass, detail))
            pass || error("S1 failed for $model: $detail")
        else
            observed = published.result.summary
            detail = @sprintf(
                "no published exact-contract reference; recorded panel %d, bets %d, return %.6f",
                length(published.books), observed.n_bets, observed.total_return_pct)
            push!(gate_rows, (gate = "S1 published Option B reproduction", model,
                              pass = missing, detail))
        end

        broad_books = t012_by_model[model].books_retained
        coherence = smile_coherence_gate(broad_books, restricted[model];
                                          tolerance = R01_GATE_TOL)
        detail = @sprintf("%d books, %d checks, max gap %.3e (tol %.1e)",
                          coherence.n_books, coherence.n_checks,
                          coherence.max_abs_gap, coherence.tolerance)
        push!(gate_rows, (gate = coherence.gate, model,
                          pass = coherence.pass, detail))
        coherence.pass || error("S2 failed for $model: $detail")
    end

    println("\n[6/7] Phase 2 — conviction tiers on each model's accretive survivors...")
    tier_outputs = Vector{Any}(undef, length(R01_REFERENCES))
    Threads.@threads for index in eachindex(R01_REFERENCES)
        ref = R01_REFERENCES[index]
        model = ref.model
        survivors = accretive_candidates(screening_by_model[model].rows)
        tier_outputs[index] = (;
            survivors,
            result = run_conviction_sweep(
                model, restricted[model], panel_odds, ds, base_book, base_policy, survivors;
                catalog, grid = R01_TIER_GRID, config = R01_PRUNING_CONFIG),
        )
    end
    for (index, ref) in enumerate(R01_REFERENCES)
        output = tier_outputs[index]
        println("  ", ref.model, " survivors: ",
                isempty(output.survivors) ? "none" : join(String.(output.survivors), ", "))
        push!(tier_frames, output.result.rows)
    end
    conviction_tiers = vcat(tier_frames...; cols = :union)

    println("\n[7/7] Validating and writing reproducibility artefacts...")
    t012_contrast = vcat(t012_frames...; cols = :union)
    gates = DataFrame(gate_rows)
    all(coalesce.(gates.pass, true)) || error("at least one measured verification gate failed")
    expected_lines = length(R01_REFERENCES) * (1 + length(candidates))
    nrow(line_screening) == expected_lines || error(
        "line screen has $(nrow(line_screening)) rows; expected $expected_lines")
    expected_tiers = length(R01_REFERENCES) * length(R01_TIER_GRID.tier1) *
                     length(R01_TIER_GRID.tier2) * length(R01_TIER_GRID.tier3)
    nrow(conviction_tiers) == expected_tiers || error(
        "tier sweep has $(nrow(conviction_tiers)) rows; expected $expected_tiers")

    summary = summary_frame(line_screening, conviction_tiers, t012_contrast)
    written = write_pruning_outputs(
        R01_OUTPUT_DIR;
        summary,
        line_screening,
        conviction_tiers,
        t012_contrast,
        gates,
    )
    println("  wrote ", join(written.outputs, ", "))
    println("  report ", written.report)
    return (; summary, line_screening, conviction_tiers, t012_contrast, gates, inventory, panel)
end

# %%
# ===================================================================
# 5. Final report
# ===================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    R01_RESULT = r01_execute()
    println("\nR01_MARKET_PRUNING_DONE")
end
