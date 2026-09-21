# ==============================================================================
# experiments/scottish_lower/10_momentum_multiscale_grw/l10_momentum_grw_loader.jl
# ==============================================================================
#
# Definitions only. `r10_momentum_smoke.jl`, `r20_momentum_production_grid.jl`
# and `r30_momentum_evaluation.jl` execute.
#
# THREE BENCHMARK ARMS (Scottish Lower, Tournaments 56 & 57, 40-fold walk-forward):
#   m01_poisson_time_decay       TimeDecayDynamics(180.0)      (Control 1)
#   m02_poisson_grw_1st_order    MultiScaleGRW() 1st-order     (Control 2)
#   m03_poisson_momentum_grw     MomentumMultiScaleGRW         (Candidate: 2nd-order)
#
# STAGE 0 RESEARCH RESPONSIBILITY (Pi / GPT-6 Astra):
# Formulate and implement the 2nd-order state-space momentum dynamics:
#   α_t = α_{t-1} + v_{t-1} + σ_α ε_t
#   v_t = φ v_{t-1} + σ_v η_t
# where v_t represents team directional momentum/velocity and φ ∈ [0, 1) governs
# persistence. Ensure compatibility with ReverseDiff compiled tapes (zero allocs).
# ==============================================================================

include(joinpath(@__DIR__, "..", "..", "..", "current_development", "grw_player_hybrid", "l01_loader.jl"))

const MMG_CAL = BayesianFootball.Calibration

# ==============================================================================
# 1. Configuration Struct
# ==============================================================================

Base.@kwdef struct MomentumGRWConfig
    experiment::String = "scottish_lower_momentum_grw"
    smoke_experiment::String = "smoke_scottish_lower_momentum_grw"
    save_root::String = joinpath(@__DIR__, "results")

    target_seasons::Vector{String} = ["24/25", "25/26"]
    expected_folds::Int = 40
    expected_oos::Int = 710

    # Smoke folds: fold 1 has no target steps (tests edge cases),
    # folds 20 and 40 test deep within each season.
    smoke_folds::Vector{Int} = [1, 20, 40]
end

# ==============================================================================
# 2. Control Model Builders
# ==============================================================================

"""
    build_m01_time_decay()

Control 1: Standard exponential time decay Poisson model.
"""
function build_m01_time_decay()
    return CountModelBuilder(:m01_poisson_time_decay) |>
        add(GlobalInterception()) |>
        add(GlobalHomeAdvantage()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(PoissonObservation()) |>
        build
end

"""
    build_m02_grw_1st_order()

Control 2: Standard 1st-order MultiScaleGRW Poisson model.
"""
function build_m02_grw_1st_order()
    return CountModelBuilder(:m02_poisson_grw_1st_order) |>
        add(GlobalInterception()) |>
        add(GlobalHomeAdvantage()) |>
        add(MultiScaleGRW()) |>
        add(PoissonObservation()) |>
        build
end

# ==============================================================================
# 3. Stage 0: Momentum MultiScale GRW Formulation
# ==============================================================================
# Pi (GPT-6 Astra): Define your MomentumMultiScaleGRW component, Turing submodel,
# and reconstruction logic below. Ensure ReverseDiff compiled tape safety.
