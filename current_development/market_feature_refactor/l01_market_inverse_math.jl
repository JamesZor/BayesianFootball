# current_development/market_feature_refactor/l01_market_inverse_math.jl

using Optim
using Distributions

# Include the abstract types defined in l00
include("l00_abstract_market_types.jl")

# Include the FrankCopulaNegBin distribution from the project
# Note: Adjust path if needed when moving to production
include("../../src/MyDistributions/frank_copula_negbin.jl")

# ==============================================================================
# Helper Math (Probability Matrix Builders)
# ==============================================================================

# Mock Dixon-Coles builder (You already have a full version in `market_inverse_utils.jl`)
function _build_probability_matrix_dixon(λ_h, λ_a, ρ; max_goals=10)
    P = zeros(max_goals, max_goals)
    for i in 0:max_goals-1
        for j in 0:max_goals-1
            p = pdf(Poisson(λ_h), i) * pdf(Poisson(λ_a), j)
            # Apply tau correction
            if i == 0 && j == 0; p *= max(0.0, 1 - λ_h * λ_a * ρ)
            elseif i == 0 && j == 1; p *= max(0.0, 1 + λ_h * ρ)
            elseif i == 1 && j == 0; p *= max(0.0, 1 + λ_a * ρ)
            elseif i == 1 && j == 1; p *= max(0.0, 1 - ρ)
            end
            P[i+1, j+1] = p
        end
    end
    return P ./ sum(P) # Normalize
end

# Frank Copula Negative Binomial Builder
function _build_probability_matrix_frank_negbin(λ_h, λ_a, r_h, r_a, κ; max_goals=10)
    dist = MyDistributions.FrankCopulaNegBin(r_h, λ_h, r_a, λ_a, κ)
    P = zeros(max_goals, max_goals)
    for i in 0:max_goals-1
        for j in 0:max_goals-1
            # We use exp(logpdf) to get the raw probability
            P[i+1, j+1] = exp(logpdf(dist, i, j))
        end
    end
    return P ./ sum(P) # Normalize
end

# ==============================================================================
# 1. Multiple Dispatch: Initial Guesses
# ==============================================================================
# Returns the starting vector θ for the Optim.jl NelderMead solver
get_initial_guess(::DixonColesMarketFeature) = [log(1.5), log(1.0), 0.05]
# Frank Copula NegBin needs [log(λ_h), log(λ_a), log(r_h), log(r_a), κ]
get_initial_guess(::FrankCopulaMarketFeature) = [log(1.5), log(1.0), log(3.0), log(3.0), 0.1]

# ==============================================================================
# 2. Multiple Dispatch: Matrix Construction
# ==============================================================================
# Unpacks θ into the specific parameters for the chosen inverse model
function build_probability_matrix(::DixonColesMarketFeature, θ::Vector{Float64}, max_goals::Int)
    λh, λa, ρ = exp(θ[1]), exp(θ[2]), θ[3]
    return _build_probability_matrix_dixon(λh, λa, ρ, max_goals=max_goals)
end

function build_probability_matrix(::FrankCopulaMarketFeature, θ::Vector{Float64}, max_goals::Int)
    λh, λa, rh, ra, κ = exp(θ[1]), exp(θ[2]), exp(θ[3]), exp(θ[4]), θ[5]
    return _build_probability_matrix_frank_negbin(λh, λa, rh, ra, κ, max_goals=max_goals)
end

# ==============================================================================
# 3. Multiple Dispatch: Extract Parameters
# ==============================================================================
# Maps the optimized θ back to a NamedTuple. 
# This dictates EXACTLY what keys will be flattened into the F_data dictionary!
extract_parameters(::DixonColesMarketFeature, θ) = (λ_h=exp(θ[1]), λ_a=exp(θ[2]), ρ=θ[3])
extract_parameters(::FrankCopulaMarketFeature, θ) = (λ_h=exp(θ[1]), λ_a=exp(θ[2]), r_h=exp(θ[3]), r_a=exp(θ[4]), κ=θ[5])

# ==============================================================================
# Target Error Calculators (Mocked for prototyping)
# ==============================================================================
# You will replace these with the actual logic from `market_inverse_utils.jl`
function _calculate_error(::Val{:result_1x2}, P, targets)
    home_prob = sum(tril(P, -1))
    away_prob = sum(triu(P, 1))
    draw_prob = sum(diag(P))
    
    err = 0.0
    err += haskey(targets, :home) ? (targets[:home] - home_prob)^2 : 0.0
    err += haskey(targets, :away) ? (targets[:away] - away_prob)^2 : 0.0
    err += haskey(targets, :draw) ? (targets[:draw] - draw_prob)^2 : 0.0
    return err
end

function _calculate_error(::Val{:btts}, P, targets)
    # Prob both teams > 0 goals
    btts_yes = sum(P[2:end, 2:end])
    btts_no = 1.0 - btts_yes
    
    err = 0.0
    err += haskey(targets, :btts_yes) ? (targets[:btts_yes] - btts_yes)^2 : 0.0
    err += haskey(targets, :btts_no) ? (targets[:btts_no] - btts_no)^2 : 0.0
    return err
end

function _calculate_error(::Val{:uo_25}, P, targets)
    under_mask = [(i-1) + (j-1) < 2.5 for i in 1:size(P,1), j in 1:size(P,2)]
    u25_prob = sum(P[under_mask])
    o25_prob = 1.0 - u25_prob
    
    err = 0.0
    err += haskey(targets, :under_25) ? (targets[:under_25] - u25_prob)^2 : 0.0
    err += haskey(targets, :over_25)  ? (targets[:over_25] - o25_prob)^2 : 0.0
    return err
end

# Fallback
_calculate_error(::Val{T}, P, targets) where T = 0.0

# ==============================================================================
# The Master Optimizer
# ==============================================================================
"""
    fit_market_implied_parameters(targets::Dict{Symbol, Float64}, config::AbstractMarketFeatureConfig; max_goals=10)

Dynamically fits market parameters based on the specific rules dictated by the `config`.
Note: Modified from `match_df` to `targets::Dict` for easier prototyping in REPL.
"""
function fit_market_implied_parameters(targets::Dict{Symbol, Float64}, config::AbstractMarketFeatureConfig; max_goals=10)
    
    function loss(θ::Vector{Float64})
        P = build_probability_matrix(config, θ, max_goals)
        sse = 0.0
        # Iterate over the target lines requested by the Layer 1 model configuration!
        for line in config.lines
            sse += _calculate_error(Val(line), P, targets)
        end
        return sse
    end

    # Run the NelderMead optimizer
    result = optimize(loss, get_initial_guess(config), NelderMead())
    
    # Return the extracted NamedTuple directly!
    return extract_parameters(config, Optim.minimizer(result))
end
