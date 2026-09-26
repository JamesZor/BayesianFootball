# experiments/scotland/00_harness_validation/candidates.jl

using BayesianFootball
using UUIDs

const EXPERIMENT = "harness_validation"
const TODO = 30
const QUESTION = "Reproduce g2_grw_all_spfl_cups through the shared harness"
const DIMENSION = "validation"
const STATUS = "active"
const DECISION = "pending"
const README = joinpath(@__DIR__, "README.md")

# g2's original run from scottish_pyramid_grw_cups as the comparison control
const CONTROL = Harness.RunRef(
    "g2_grw_all_spfl_cups",
    "scottish_pyramid_grw_cups",
    UUID("a6f62436-ec8a-461d-8bd5-dc1861a2daaa"),
    :control
)

const g2_model = CountModelBuilder(:g2_harness_repro) |>
    add(GlobalInterception()) |>
    add(MultiScaleGRW()) |>
    add(GlobalHomeAdvantage()) |>
    add(PoissonObservation()) |>
    build

const g2_scope = Data.DataScope(
    name = "spfl_cups",
    train_tournaments = [54, 55, 56, 57],
    cups = :senior_spfl_ties,
    target_tournaments = [56, 57],
    monitor_tournaments = Int[],
    clock_tournaments = [56, 57],
    target_seasons = ["24/25", "25/26"],
    history_seasons = 2,
    dynamics_col = :match_biweek
)

const g2_sampler = Samplers.QueuedNUTSConfig(
    n_samples = 1_000,
    n_warmup = 500,
    n_chains = 4,
    accept_rate = 0.80,
    max_depth = 10,
    initialisation = nothing,
    show_progress = false,
    silence_initial_stepsize = true
)

const CANDIDATES = [
    Harness.Candidate(
        name = "g2_harness_repro",
        model = g2_model,
        scope = g2_scope,
        sampler = g2_sampler,
        role = :candidate,
        hypothesis = "Exact reproduction of g2_grw_all_spfl_cups under the unified harness"
    )
]
