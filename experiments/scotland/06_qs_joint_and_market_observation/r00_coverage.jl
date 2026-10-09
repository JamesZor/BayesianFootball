# Phase 0: does the pinned snapshot support the unchanged joint proxy-xG arm?
# Descriptive coverage only; no sampling, SQL writes, market fitting or scoring.
# Include in a fresh owned mcmc-beast REPL (-t 16).

# %% 1. Packages and wave-1 snapshot/splitter
using BayesianFootball, CSV, DataFrames, Dates, LinearAlgebra, ThreadPinning
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "..", "05_quality_style_goal_vs_market", "market_arm.jl"))

# %% 2. Fixed observation feature and output
const W2_COVERAGE_FEATURE = Features.MatchProxyXGFeature(k = 25.0, fallback = :none)
const W2_COVERAGE_OUT = "/root/BF_runs/qs_experiment_w2_out/phase0"
mkpath(W2_COVERAGE_OUT)
w2_ds = QSMarketArm.checked_datastore()
w2_folds = QSMarketArm.goal_folds(w2_ds)
w2_boundaries = Data.create_id_boundaries(w2_folds.scoped,
    Data.ScopedWalkForwardCV(QSMarketArm.goal_scope()))
w2_matches = Dict(Int(r.match_id) => r for r in eachrow(w2_folds.scoped.matches))

# %% 3. Exact extractor, history-only cell fit; coverage by fold/tournament/partition
w2_coverage_rows = NamedTuple[]
for (fold, (boundary, _)) in zip(w2_folds.folds, w2_boundaries)
    ids = vcat(fold.train_ids, fold.ids)
    values = Dict{Symbol,Any}(:history_match_ids => Set(Int.(boundary.history_match_ids)))
    Features.add_feature!(values, W2_COVERAGE_FEATURE, ids, Dict(), w2_folds.scoped)
    mask = Dict(id => values[:flat_pxg_obs_available][i] for (i, id) in enumerate(ids))
    all(x -> x == 0.0 || x == 1.0, values[:flat_pxg_obs_available]) || error("nonbinary mask")
    for (partition, partition_ids) in (("training", fold.train_ids), ("heldout", fold.ids))
        for tournament in (56, 57)
            selected = filter(id -> w2_matches[id].tournament_id == tournament, partition_ids)
            n = length(selected)
            covered = count(id -> mask[id] == 1.0, selected)
            push!(w2_coverage_rows, (; fold = fold.number, cutoff = fold.cutoff,
                tournament, partition, matches = n, covered, missing = n - covered,
                fraction = n == 0 ? 0.0 : covered / n))
            partition == "training" && covered == 0 && error(
                "no proxy-xG in fold $(fold.number), tournament $tournament training window")
        end
    end
end
w2_coverage = DataFrame(w2_coverage_rows)
CSV.write(joinpath(W2_COVERAGE_OUT, "proxy_xg_coverage.csv"), w2_coverage)
for tournament in (56, 57)
    training = w2_coverage[(w2_coverage.partition .== "training") .&
        (w2_coverage.tournament .== tournament), :]
    heldout = w2_coverage[(w2_coverage.partition .== "heldout") .&
        (w2_coverage.tournament .== tournament), :]
    println("COVERAGE tournament=", tournament, " training_covered=", extrema(training.covered),
        " training_fraction=", extrema(training.fraction),
        " heldout=", sum(heldout.covered), "/", sum(heldout.matches))
end
println("PHASE0_COVERAGE_DONE folds=", length(w2_folds.folds))
