# ==============================================================================
# Experiment 08 · l08 — paired evaluation and portfolio helpers
# ==============================================================================
#
# Loader only.  This file keeps common-fixture selection, proper-score aggregation,
# production-manifest discovery, and portfolio persistence checks out of the two
# research runners.  It does not sample models and it never manufactures latents.
# ==============================================================================

import BayesianFootball
import CSV
import DataFrames
import Dates
import Statistics
import TOML
import UUIDs

const L08_EVAL = BayesianFootball.Evaluation
const L08_PORTFOLIO_EVAL = BayesianFootball.Portfolio
const L08_INFERENCE = BayesianFootball.Training.Inference

const L08_BASELINE_RUNS = (
    (name = "m00_poisson_control",
     run_id = UUIDs.UUID("d63f8877-b825-40ae-9ae5-d829e8b8a7f7"),
     source_experiment = "scottish_lower_joint_2426",
     expected_oos = 749),
    (name = "m05_joint_production_wealth",
     run_id = UUIDs.UUID("5eff755c-3591-48d1-a2cc-5fc2744ddf88"),
     source_experiment = "scottish_lower_joint_2426",
     expected_oos = 710),
    (name = "m08_joint_composite",
     run_id = UUIDs.UUID("61fc5d87-1bd6-46d1-bb2b-c2aaad39e348"),
     source_experiment = "scottish_lower_joint_2426",
     expected_oos = 710),
)

const L08_PRIMARY_FAMILIES = ("1X2", "OU2.5", "BTTS")
const L08_ECE_BINS = 10

"The market family used in the Experiment 08 paired-score contract."
function l08_market_family(market_name::AbstractString, line::Real)
    market = lowercase(strip(market_name))
    market == "1x2" && return "1X2"
    occursin("btts", market) && return "BTTS"
    if occursin("over", market) || occursin("under", market) || occursin("total", market)
        return "OU" * string(Float64(line))
    end
    return String(market_name)
end

"Map the package's typed evaluation rows to a narrow, auditable score panel."
function l08_scoring_panel(context; families = L08_PRIMARY_FAMILIES)
    accepted = Set(String.(families))
    rows = L08_EVAL.evaluation_rows(context)
    panel = DataFrames.DataFrame(
        match_id = Int[],
        selection = Symbol[],
        family = String[],
        p_model = Float64[],
        p_market = Float64[],
        is_winner = Int8[],
    )
    for row in rows
        row.outcome < 0 && continue
        family = row.selection in (:home, :draw, :away) ? "1X2" :
                 row.selection in (:btts_yes, :btts_no) ? "BTTS" :
                 startswith(String(row.selection), "over_") || startswith(String(row.selection), "under_") ?
                 "OU" * string(parse(Float64, String(row.selection)[end-1:end-1] * "." * String(row.selection)[end:end])) :
                 "unknown"
        family in accepted || continue
        isfinite(row.model_prob) && isfinite(row.market_prob) || continue
        push!(panel, (row.match_id, row.selection, family, row.model_prob,
                      row.market_prob, row.outcome))
    end
    DataFrames.nrow(panel) > 0 || error("no scored Betfair rows in requested families $(collect(families))")
    return panel
end

"Return model and market scores over one explicitly selected panel."
function l08_binary_scores(panel::DataFrames.AbstractDataFrame; n_bins::Integer = L08_ECE_BINS)
    n = DataFrames.nrow(panel)
    n > 0 || error("cannot score an empty panel")
    y = Float64.(panel.is_winner)
    function side(p)
        ll = mean(-((y .* log.(clamp.(p, 1e-15, 1.0 - 1e-15))) .+
                     ((1.0 .- y) .* log.(clamp.(1.0 .- p, 1e-15, 1.0 - 1e-15)))))
        br = mean((p .- y) .^ 2)
        edges = range(0.0, 1.0; length = Int(n_bins) + 1)
        counts = zeros(Int, n_bins)
        predicted = zeros(Float64, n_bins)
        observed = zeros(Float64, n_bins)
        for i in eachindex(p)
            bin = clamp(floor(Int, p[i] * n_bins) + 1, 1, n_bins)
            counts[bin] += 1
            predicted[bin] += p[i]
            observed[bin] += y[i]
        end
        for bin in eachindex(counts)
            counts[bin] == 0 && continue
            predicted[bin] /= counts[bin]
            observed[bin] /= counts[bin]
        end
        curve = L08_EVAL.CalibrationCurve(collect(edges), counts, predicted, observed)
        return (logloss = ll, brier = br, ece = L08_EVAL.expected_calibration_error(curve))
    end
    return (model = side(Float64.(panel.p_model)), market = side(Float64.(panel.p_market)), n = n)
end

"Compute 1X2 RPS through the package implementation, never by summing binary rows."
function l08_rps(context, family::AbstractString; source::Symbol = :model)
    family == "1X2" || return (score = NaN, n = 0)
    score, n = L08_EVAL.ranked_probability_score(context; source = source)
    return (score = score, n = n)
end

"The documented package CRPS: plug-in marginal home/away goal-count CRPS, not scoreline CRPS."
function l08_marginal_crps(context)
    result = L08_EVAL.compute_metric(L08_EVAL.CRPS(), context)
    return (home = result.home.mean, away = result.away.mean, all = result.all.mean)
end

"One long-form score row per market family plus an all-family aggregate."
function l08_score_summary(model::AbstractString, context)
    panel = l08_scoring_panel(context)
    crps = l08_marginal_crps(context)
    rows = NamedTuple[]
    for family in (L08_PRIMARY_FAMILIES..., "ALL")
        part = family == "ALL" ? panel : DataFrames.filter(:family => ==(family), panel)
        DataFrames.nrow(part) == 0 && error("$model has no scored rows for $family")
        scores = l08_binary_scores(part)
        model_rps = l08_rps(context, family; source = :model)
        market_rps = l08_rps(context, family; source = :market)
        push!(rows, (
            model = String(model), family = family, n_selections = scores.n,
            model_logloss = scores.model.logloss, market_logloss = scores.market.logloss,
            model_brier = scores.model.brier, market_brier = scores.market.brier,
            model_ece_equal_width_10 = scores.model.ece,
            market_ece_equal_width_10 = scores.market.ece,
            model_rps_1x2 = model_rps.score, market_rps_1x2 = market_rps.score,
            n_rps_1x2 = model_rps.n,
            marginal_crps_home = crps.home, marginal_crps_away = crps.away,
            marginal_crps_match_mean = crps.all,
        ))
    end
    return DataFrames.DataFrame(rows), panel
end

"The m05 run, rather than an intersection, defines the exact 710-fixture target."
function l08_canonical_match_ids(fits::AbstractDict)
    haskey(fits, "m05_joint_production_wealth") || error("m05 baseline is required to define canonical coverage")
    ids = Int.(fits["m05_joint_production_wealth"].latents.match_ids)
    length(ids) == 710 || error("m05 canonical baseline has $(length(ids)) latent fixtures, expected 710")
    length(unique(ids)) == 710 || error("m05 canonical baseline contains duplicate match IDs")
    return sort!(ids)
end

function l08_require_coverage(name::AbstractString, fit, canonical_ids::Vector{Int})
    fit.latents isa BayesianFootball.CountLatents || error("$name has $(typeof(fit.latents)), not CountLatents")
    ids = Int.(fit.latents.match_ids)
    length(unique(ids)) == length(ids) || error("$name has duplicate latent match IDs")
    Set(canonical_ids) ⊆ Set(ids) || error("$name does not cover every canonical m05 fixture")
    return nothing
end

"Read four persisted candidate UUIDs from the immutable production manifest."
function l08_production_candidates(output_dir::AbstractString)
    manifests = filter(path -> startswith(basename(path), "manifest_production") && endswith(path, ".toml"),
                       readdir(output_dir; join = true))
    isempty(manifests) && return nothing
    manifest = TOML.parsefile(last(sort(manifests)))
    extra = get(manifest, "extra", Dict{String,Any}())
    raw = get(extra, "candidate_run_ids", nothing)
    raw isa AbstractDict || error("production manifest lacks [extra].candidate_run_ids; evaluation refuses manual UUID input")
    names = sort!(collect(String.(keys(raw))))
    expected = sort!(collect(L08_CANDIDATE_NAMES))
    names == expected || error("production manifest candidate names $(names) do not equal $(expected)")
    return [(name = name, run_id = UUIDs.UUID(String(raw[name]),),
             source_experiment = L08_EXPERIMENT, expected_oos = 710) for name in names]
end

"Load a fit by immutable UUID, checking only typed latent and coverage facts."
function l08_load_fit_checked(entry)
    db = BayesianFootball.PostgresStorage(entry.source_experiment)
    fit = BayesianFootball.load_fit(db, entry.run_id)
    fit isa BayesianFootball.Fit || error("$(entry.name)/$(entry.run_id) did not deserialize to Fit")
    fit.latents isa BayesianFootball.CountLatents || error("$(entry.name)/$(entry.run_id) has no CountLatents")
    BayesianFootball.n_matches(fit.latents) == entry.expected_oos || error(
        "$(entry.name)/$(entry.run_id) has $(BayesianFootball.n_matches(fit.latents)) OOS fixtures; expected $(entry.expected_oos)")
    return db, fit
end

"Exact portfolio persistence is part of the portfolio contract, not a headline-only check."
function l08_assert_portfolio_roundtrip(result, portfolio_id, db)
    reloaded = L08_PORTFOLIO_EVAL.load_portfolio_db(portfolio_id, db)
    reloaded.summary.total_return_pct == result.summary.total_return_pct || error(
        "portfolio return changed during persistence round-trip")
    isequal(reloaded.trajectory.bets, result.trajectory.bets) || error(
        "portfolio bet ledger changed during persistence round-trip")
    return reloaded
end
