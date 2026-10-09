# Phase 2 checks: market panel provenance, 40-fold leakage audit, and adapter proof.
# Include in an owned persistent mcmc-beast REPL (-t 16). Reads mcmc_experiments; no writes.

# %% 1. Packages and loaders
using BayesianFootball, CSV, DataFrames, Dates, LinearAlgebra, ThreadPinning, UUIDs
pinthreads(:cores)
BLAS.set_num_threads(1)
const QSX = joinpath(@__DIR__)
isdefined(Main, :QSMarketArm) || include(joinpath(QSX, "market_arm.jl"))
isdefined(Main, :QSScoringAdapter) || include(joinpath(QSX, "scoring_adapter.jl"))
const MA = QSMarketArm
const SA = QSScoringAdapter
const OUT = get(ENV, "QSX_OUT", "/root/BF_runs/qs_experiment_out")

# %% 2. Pinned snapshot, B2 market panel on four seasons
p2_ds = MA.checked_datastore()
p2_market = MA.prepare_panel(p2_ds, joinpath(OUT, "market_panel"))
println("PANEL counts=", p2_market.counts, " fixtures=", MA.MID.n_fixtures(p2_market.panel),
        " weeks=", p2_market.panel.n_weeks, " teams=", MA.MID.n_teams(p2_market.panel))

# %% 3. Every fold: no target-week close, no held-out fixture, all targets forecastable
p2_folds = MA.goal_folds(p2_ds)
p2_rows = NamedTuple[]
for fold in p2_folds.folds
    t = MA.training_panel(p2_market.panel, fold)
    trained = Set(t.panel.obs_match)
    isempty(intersect(trained, Set(fold.ids))) || error("fold $(fold.number) trains on a target")
    issubset(trained, Set(fold.train_ids)) || error("fold $(fold.number) uses a non-goal-training close")
    push!(p2_rows, (; fold = fold.number, goal_train = length(fold.train_ids),
        market_train = MA.MID.n_fixtures(t.panel), weeks = t.panel.n_weeks,
        last_train_week_start = t.panel.week_start[end], cutoff = fold.cutoff,
        heldout = length(fold.ids),
        heldout_quoted = count(in(Set(p2_market.panel.matches.match_id)), fold.ids)))
end
p2_audit = DataFrame(p2_rows)
mkpath(joinpath(OUT, "phase2"))
CSV.write(joinpath(OUT, "phase2", "fold_audit.csv"), p2_audit)
println("FOLDS market_train=", extrema(p2_audit.market_train), " heldout_total=", sum(p2_audit.heldout),
        " heldout_quoted=", sum(p2_audit.heldout_quoted))

# %% 4. Adapter proof on a completed 40-fold lower Poisson goal run (W1 grw_lower_poisson)
p2_ref = BayesianFootball.Harness.RunRef("grw_lower_poisson", "scottish_lower_multiscale_grw_2426",
    UUID("f64a00a2-34a0-4f31-8c58-c093c92d54b7"), :control)
p2_fit = BayesianFootball.Training.load_fit(
    BayesianFootball.Training.PostgresStorage(p2_ref.experiment), p2_ref.run_id)
p2_tiers = BayesianFootball.Harness.club_season_tiers(p2_ds)
p2_harness = BayesianFootball.Harness.score_fits([p2_ref => p2_fit]; ds = p2_ds, tiers = p2_tiers,
    control = p2_ref, target_seasons = ["24/25", "25/26"], expected_fixtures = 710,
    bootstrap_B = 10_000)
p2_adapted = SA.score_latents([p2_ref => SA.permuted(p2_fit.latents, 20261009)], p2_fit;
    ds = p2_ds, control_label = p2_ref, bootstrap_B = 10_000)
p2_gap = SA.max_difference(p2_harness, p2_adapted)
CSV.write(joinpath(OUT, "phase2", "adapter_harness_scores.csv"), p2_harness)
println("ADAPTER rows=", nrow(p2_harness), " max_abs_difference=", p2_gap)
p2_gap <= 1e-10 || error("adapter disagrees with harness: $p2_gap")
println("PHASE2_CHECKS_DONE")
