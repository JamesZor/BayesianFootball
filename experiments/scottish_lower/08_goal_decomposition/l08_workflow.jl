# ==============================================================================
# Experiment 08 · l08 — goal-decomposition workflow infrastructure
# ==============================================================================
#
# Loader only.  It owns immutable-run provenance, the canonical walk-forward
# splitter, strict promotion gates, and the common Betfair closing book.  The
# model loader owns `l08_models(registry; snapshot_hash=...)`, its structural
# parameter contract, and its ReverseDiff checks; the incident loader owns the
# typed component-count features and their filtration audit.
# ==============================================================================

import BayesianFootball
import DotEnv
import DataFrames
import Dates
import LinearAlgebra
import SHA
import Serialization
import TOML
import UUIDs

const L08_DATA = BayesianFootball.Data
const L08_FEATURES = BayesianFootball.Features
const L08_PORTFOLIO = BayesianFootball.Portfolio
const L08_TRAINING = BayesianFootball.Training

# Historical Gen-3 baseline artefacts predate the fourth (`SharedKappa`) type
# parameter on `JointGammaPoissonObservation`. Keep their established reader
# shim local to the experiment workflow that loads those immutable runs.
function Serialization.deserialize(
        serializer::Serialization.AbstractSerializer,
        observation_type::Type{<:BayesianFootball.Models.PreGame.JointGammaPoissonObservation})
    observation_type isa DataType && return invoke(
        Serialization.deserialize,
        Tuple{Serialization.AbstractSerializer,DataType}, serializer, observation_type)

    fields = Any[]
    for _ in 1:3
        tag = Int32(read(serializer.io, UInt8)::UInt8)
        push!(fields, Serialization.handle_deserialize(serializer, tag))
    end
    pregame = BayesianFootball.Models.PreGame
    return pregame.JointGammaPoissonObservation(
        fields[1], fields[2], fields[3], pregame.SharedKappa())
end

"Load the git-ignored operational DB environment at runtime; package precompilation cannot retain it."
function l08_load_runtime_env!()
    env_file = joinpath(pkgdir(BayesianFootball), ".env")
    isfile(env_file) && DotEnv.load!(ENV, env_file)
    return nothing
end

const L08_EXPERIMENT = "scottish_lower_goal_decomposition_2426"
const L08_CANDIDATE_NAMES = (
    "m00_recombined_control",
    "m01_decomposed_baseline",
    "m02_decomposed_team_penalties",
    "m03_decomposed_pressure_own_goals",
)
const L08_TARGET_SEASONS = ["24/25", "25/26"]
const L08_EXPECTED_FOLDS = 40
const L08_EXPECTED_OOS = 710
const L08_CHAINS = 4
# Fixed before the first real smoke: zero-divergence promotion is incompatible
# with the prior 0.90/800 historical budget, which observed 2–6 divergences.
const L08_SAMPLER = BayesianFootball.QueuedNUTSConfig(
    n_samples = 1_000,
    n_warmup = 1_000,
    n_chains = L08_CHAINS,
    accept_rate = 0.95,
)
const L08_THRESHOLDS = BayesianFootball.ConvergenceThresholds(
    max_rhat = 1.05,
    min_ess = 200.0,
    max_divergence_rate = eps(Float64),
    min_bfmi = 0.30,
    max_treedepth_rate = 0.05,
)

"""
Immutable inputs shared by every Experiment 08 runner.

There is deliberately no global datastore, database, or model registry.  A runner
constructs this value once and passes it explicitly to `l08_models`; consequently a
model cannot silently price a different cache snapshot than its manifest records.
"""
struct L08Registry{D,S}
    datastore::D
    database::S
    snapshot_hash::String
    source_hashes::Dict{String,String}
    output_dir::String
    created_at::Dates.DateTime
end

"The canonical grouped 40-boundary Scottish Lower 24/25–25/26 walk-forward splitter."
function l08_splitter()
    return L08_DATA.GroupedCVConfig(
        tournament_groups = [[56, 57]],
        target_seasons = copy(L08_TARGET_SEASONS),
        history_seasons = 2,
        dynamics_col = :match_biweek,
        warmup_period = 0,
        end_dynamics = nothing,
        stop_early = true,
    )
end

"The canonical portfolio book, restricted to the requested 1X2 and O/U 2.5 markets."
function l08_book_spec()
    return BayesianFootball.BookSpec(
        markets = L08_DATA.MarketConfig(L08_DATA.AbstractMarket[
            L08_DATA.Market1X2(),
            L08_DATA.MarketOverUnder(2.5),
        ]),
        price = BayesianFootball.DeArb(),
        allocator = BayesianFootball.KellyLogUtility(),
        shrink = L08_PORTFOLIO.BakerMcHale(),
        exec = BayesianFootball.ExecutionConfig(
            commission = BayesianFootball.PerBetCommission(0.02),
            budget = 0.99,
            min_selection_stake = 0.001,
        ),
    )
end

"The historical Gen-3-comparable capped daily-slate policy used in Experiment 08."
function l08_policy_spec()
    return BayesianFootball.PolicySpec(
        trust = BayesianFootball.FlatTrust(0.30),
        risk = BayesianFootball.SlateDrawdown(23.0),
        cap = BayesianFootball.FixedCap(0.20),
        grouping = BayesianFootball.DailySlate(),
    )
end

function l08_file_hash(path::AbstractString)
    isfile(path) || error("cannot manifest absent source file: $path")
    return bytes2hex(SHA.sha256(read(path)))
end

function l08_datastore_hash(ds)
    matches = ds.matches
    required = (:match_id, :match_date, :season, :home_score, :away_score)
    available = Set(Symbol.(DataFrames.names(matches)))
    all(column -> column in available, required) || error(
        "datastore matches lacks one of $(collect(required)); cannot freeze snapshot")
    rows = DataFrames.sort(DataFrames.select(matches, collect(required)), :match_id)
    payload = join((string(row.match_id, "|", row.match_date, "|", row.season,
                           "|", row.home_score, "|", row.away_score)
                    for row in eachrow(rows)), "\n")
    return bytes2hex(SHA.sha256(payload))
end

function l08_repo_relative(path::AbstractString)
    root = pkgdir(BayesianFootball)
    absolute = abspath(path)
    startswith(absolute, root * "/") || error("source manifest file is outside repository: $absolute")
    return relpath(absolute, root)
end

function l08_registry(ds, db;
                      output_dir::AbstractString = joinpath(@__DIR__, "results"),
                      source_files::Vector{String} = String[])
    mkpath(output_dir)
    source_hashes = Dict(l08_repo_relative(path) => l08_file_hash(path) for path in source_files)
    return L08Registry(ds, db, l08_datastore_hash(ds), source_hashes,
                       abspath(output_dir), Dates.now())
end

"""
    l08_write_manifest!(registry; stage, extra = Dict()) -> path

Write an immutable, machine-readable TOML manifest before a prepare, smoke, grid,
evaluation, or portfolio stage.  Existing manifests are never overwritten: a mismatch
means code, data, or stage inputs drifted and must be resolved explicitly.
"""
function l08_write_manifest!(registry::L08Registry;
                             stage::AbstractString,
                             extra::AbstractDict = Dict{String,Any}())
    manifest = Dict{String,Any}(
        "stage" => String(stage),
        "experiment" => L08_EXPERIMENT,
        "julia_version" => string(VERSION),
        "datastore_snapshot_hash" => registry.snapshot_hash,
        "source_hashes" => registry.source_hashes,
        "target_seasons" => copy(L08_TARGET_SEASONS),
        "expected_folds" => L08_EXPECTED_FOLDS,
        "expected_oos" => L08_EXPECTED_OOS,
        "chains" => L08_CHAINS,
        "extra" => Dict(string(k) => v for (k, v) in extra),
    )
    io = IOBuffer()
    TOML.print(io, manifest)
    text = String(take!(io))
    hash_prefix = first(bytes2hex(SHA.sha256(text)), 16)
    path = joinpath(registry.output_dir, "manifest_$(stage)_$(hash_prefix).toml")
    if isfile(path)
        read(path, String) == text || error(
            "immutable manifest already exists with different contents: $path")
    else
        write(path, text)
    end
    return path
end

"The TWA Betfair close used for every candidate and baseline comparison."
function l08_betfair_closing_odds(ds)
    raw = L08_DATA.summarize_odds(ds.betfair_odds, L08_DATA.TWAEstimator(); window = (-20.0, 0.0))
    odds = DataFrames.DataFrame(
        match_id = Int.(raw.match_id),
        market_name = String.(raw.market_name),
        market_line = Float64.(raw.market_line),
        selection = Symbol.(raw.selection),
        odds_close = Float64.(raw.odds),
    )
    DataFrames.filter!(row -> isfinite(row.odds_close) && row.odds_close > 1.0, odds)
    odds.prob_implied_close = 1.0 ./ odds.odds_close
    DataFrames.transform!(DataFrames.groupby(odds, [:match_id, :market_name, :market_line]),
        :prob_implied_close => (p -> p ./ sum(p)) => :prob_fair_close)
    winners = DataFrames.unique(DataFrames.select(
        ds.odds, :match_id, :market_name, :market_line, :selection, :is_winner))
    odds = DataFrames.leftjoin(odds, winners;
        on = [:match_id, :market_name, :market_line, :selection])
    DataFrames.sort!(odds, [:match_id, :market_name, :market_line, :selection])
    return odds
end

"Restrict a fit's OOS fixtures to a frozen common intersection, failing on duplication."
function l08_common_match_ids(fits::AbstractDict)
    isempty(fits) && error("cannot freeze a common fixture intersection from no fits")
    id_sets = [Set(Int.(fit.latents.match_ids)) for fit in values(fits)]
    common = reduce(intersect, id_sets)
    isempty(common) && error("candidate/baseline fits have no common OOS fixtures")
    return sort!(collect(common))
end

function l08_assert_fit_coverage(name::AbstractString, fit, common_ids::Vector{Int})
    fit.latents isa BayesianFootball.CountLatents || error(
        "$name returned $(typeof(fit.latents)); Experiment 08 requires CountLatents")
    ids = Int.(fit.latents.match_ids)
    length(unique(ids)) == length(ids) || error("$name has duplicate latent match IDs")
    Set(common_ids) ⊆ Set(ids) || error("$name does not cover the frozen common fixture set")
    return nothing
end

"The strict six-part promotion gate; no unavailable diagnostic is treated as a pass."
function l08_assert_promotion(name::AbstractString, diagnostics)
    diagnostics.passed || error("$name failed native convergence audit: $(join(diagnostics.failures, "; "))")
    diagnostics.max_rhat <= 1.05 || error("$name max R-hat $(diagnostics.max_rhat) exceeds 1.05")
    diagnostics.min_ess_bulk >= 200 || error("$name bulk ESS $(diagnostics.min_ess_bulk) is below 200")
    diagnostics.min_ess_tail >= 200 || error("$name tail ESS $(diagnostics.min_ess_tail) is below 200")
    diagnostics.n_divergent == 0 || error("$name has $(diagnostics.n_divergent) divergences; require zero")
    diagnostics.min_bfmi >= 0.30 || error("$name BFMI $(diagnostics.min_bfmi) is below 0.30")
    diagnostics.treedepth_rate < 0.05 || error("$name tree-depth rate $(diagnostics.treedepth_rate) is not below 0.05")
    return nothing
end

function l08_fit_config(name::String, model, splitter, sampler = L08_SAMPLER;
                        save_root::AbstractString = joinpath(@__DIR__, "results"))
    return BayesianFootball.FitConfig(
        name = name,
        model = model,
        splitter = splitter,
        sampler = sampler,
        execution = BayesianFootball.QueuedExecution(),
        tags = ["scottish-lower", "24/25", "25/26", "goal-decomposition"],
        description = "Experiment 08 $name decomposed goal-intensity candidate.",
        save_dir = joinpath(save_root, name),
    )
end

"Register every mutable recipe before sampling; runners then preflight the immutable run hash."
function l08_register!(registry::L08Registry, models, splitter, sampler, configs, book, policy)
    db = registry.database
    model_ids = Dict{String,Int}()
    fit_hashes = Dict{String,String}()
    for (name, model) in models
        model_ids[name] = BayesianFootball.save_model(db, name, model;
            description = "Experiment 08 $name decomposed goal-intensity candidate.",
            tags = ["scottish-lower", "goal-decomposition"])
        fit_hashes[name] = BayesianFootball.save_config(db, name * "_fit", configs[name];
            description = "Experiment 08 immutable inference recipe for $name.",
            tags = ["scottish-lower", "goal-decomposition"])
    end
    splitter_id = BayesianFootball.save_splitter(db, "goal_decomposition_40fold", splitter;
        description = "Canonical grouped 56/57 walk-forward 24/25 and 25/26 splitter.",
        tags = ["scottish-lower", "goal-decomposition"])
    sampler_id = BayesianFootball.save_sampler(db, "goal_decomposition_queued_nuts_4x800", sampler;
        description = "Four-chain queued NUTS, 1,000 warmup and retained draws, target acceptance 0.95; fixed before first real smoke for zero-divergence promotion.",
        tags = ["scottish-lower", "goal-decomposition"])
    book_id = BayesianFootball.save_book_spec(db, "goal_decomposition_betfair_1x2_ou25", book;
        description = "Betfair TWA-close 1X2 and O/U 2.5, DeArb, Baker-McHale.",
        tags = ["scottish-lower", "goal-decomposition", "betfair"])
    policy_id = BayesianFootball.save_policy_spec(db, "goal_decomposition_daily_slate", policy;
        description = "30% flat trust, daily slate drawdown, 20% fixed cap (Gen-3-comparable).",
        tags = ["scottish-lower", "goal-decomposition"])
    return (; model_ids, splitter_id, sampler_id, book_id, policy_id, fit_hashes)
end

"Return the completed persisted run UUID for a config hash, or `nothing` before sampling."
function l08_completed_run_id(db, config_hash::AbstractString)
    inference = BayesianFootball.Training.Inference
    conn = inference._db_connect(db)
    try
        rows = inference._db_rows(conn, """
            SELECT r.run_id
            FROM configs AS c JOIN runs AS r ON r.run_id = c.config_id
            WHERE c.config_hash = \$1 AND r.status = 'completed'
            LIMIT 1;
        """, (config_hash,))
        return isempty(rows) ? nothing : UUIDs.UUID(string(rows.run_id[1]))
    finally
        close(conn)
    end
end

function l08_assert_prepare!(name::AbstractString, model, feature_sets, oos, splitter)
    length(feature_sets) == L08_EXPECTED_FOLDS || error(
        "$name built $(length(feature_sets)) FeatureSets; expected $L08_EXPECTED_FOLDS")
    length(oos) == L08_EXPECTED_FOLDS || error("$name has $(length(oos)) OOS frames")
    all(frame -> DataFrames.nrow(frame) > 0, oos) || error("$name has an empty OOS fold")
    total = sum(DataFrames.nrow, oos)
    total > 0 || error("$name has no OOS fixtures")
    for (idx, feature_set) in enumerate(feature_sets)
        training_ids = Set(Int.(first(feature_set).data[:goal_decomposition_training_ids]))
        held_out = Set(Int.(oos[idx].match_id))
        isempty(intersect(training_ids, held_out)) || error(
            "$name fold $idx leaks held-out IDs into training features")
    end
    return total
end
