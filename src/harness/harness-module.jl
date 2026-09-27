module Harness

using DataFrames
using Dates
using Distributions
using DynamicPPL
using ForwardDiff
using JSON3
using LibPQ
using LinearAlgebra: norm
import Logging
using LogDensityProblems
using MCMCChains
import Profile
using Random
using ReverseDiff
using SHA
using Statistics
using UUIDs
import Turing

using ..Data
using ..Features
using ..Models
using ..Samplers
using ..Training
using ..Predictions
using ..Evaluation
using ..Portfolio
using ..BackTesting
using ..Calibration
using ..MatchDay

include("candidate.jl")
include("tape_profile.jl")
include("perf_monitor.jl")
include("checks.jl")
include("cohorts.jl")
include("scoring.jl")
include("store.jl")
include("portfolio.jl")
include("stages.jl")

export Candidate, recipe_hash, fit_config, HarnessCheckError,
       screen, smoke, grid,
       write_checks!, read_checks, has_passing_smoke, find_completed_run, InMemoryCheckStore,
       RunRef, score_runs, score_fits, leaderboard, portfolio_runs,
       club_season_tiers, transition_cohort,
       write_experiment!, read_experiments

end
