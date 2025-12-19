# src/predictions/methods/poisson.jl

using Distributions
using DataFrames
using LinearAlgebra
using ..Markets
using ..TypesInterfaces: AbstractPoissonModel

# ==============================================================================
# 1. The Kernel (Row-Wise Abstraction)
# ==============================================================================

"""
    predict_match_kernel(model, match_id, λ_h, λ_a, markets)

The atomic unit of work. 
1. Builds the ScoreGrid for this match (Compute Once).
2. Solves all requested markets using that Grid (Solve Many).
3. Returns a DataFrame for this specific match.
"""
function predict_match_kernel(
    model::AbstractPoissonModel, 
    match_id::Int, 
    λ_h::Any, 
    λ_a::Any, 
    markets::Vector
)
    # A. HEAVY MATH: Build the Joint Distribution ONCE
    # This handles scalar params or vector chains (MCMC integration) internally
    grid = build_score_grid(λ_h, λ_a)

    # B. LIGHT SOLVING: Iterate markets on the pre-computed grid
    # This is fast because it's just summing matrix cells
    results = DataFrame[]
    for market in markets
        push!(results, solve_market(grid, market, match_id))
    end

    return vcat(results...)
end

# ==============================================================================
# 2. The Distribution Builder (Compute Once)
# ==============================================================================

struct ScoreGrid
    probs::Matrix{Float64} # 11x11 matrix (0-10 goals)
end

"""
    build_score_grid(λ_h, λ_a)
Computes the P(home_goals, away_goals) matrix.
Handles both Point Estimates (Scalars) and MCMC Chains (Vectors).
"""
function build_score_grid(model::AbstractPoissonModel, λ_h::Number, λ_a::Number, max_goals::Int=10)
    # Point Estimate
    grid = zeros(Float64, max_goals+1, max_goals+1)
    d_h = Poisson(λ_h)
    d_a = Poisson(λ_a)

    # Outer Product: P(h,a) = P(h) * P(a)
    # (Broadcasting is faster here)
    p_h_vec = pdf.(d_h, 0:max_goals)
    p_a_vec = pdf.(d_a, 0:max_goals)
    
    grid = p_h_vec * p_a_vec' 
    
    return ScoreGrid(grid)
end

function build_score_grid(λ_h::AbstractVector, λ_a::AbstractVector, max_goals::Int=10)
    # MCMC Chains: Average the grids produced by each sample
    n_samples = length(λ_h)
    avg_grid = zeros(Float64, max_goals+1, max_goals+1)
    
    # We iterate the chain to integrate out the parameters
    for k in 1:n_samples
        d_h = Poisson(λ_h[k])
        d_a = Poisson(λ_a[k])
        
        p_h_vec = pdf.(d_h, 0:max_goals)
        p_a_vec = pdf.(d_a, 0:max_goals)
        
        # Accumulate: grid += new_grid
        # BLAS rank-1 update could be used, but loop is clear
        avg_grid .+= (p_h_vec * p_a_vec')
    end
    
    return ScoreGrid(avg_grid ./ n_samples)
end

# ==============================================================================
# 3. The Solvers (Solve Many)
# ==============================================================================

function solve_market(grid::ScoreGrid, ::Markets.Market1X2, match_id::Int)
    P = grid.probs
    
    # Fast LinearAlgebra sums
    p_home = sum(tril(P, -1)) # Sum lower triangle (excl diagonal)
    p_draw = sum(diag(P))     # Sum diagonal
    p_away = sum(triu(P, 1))  # Sum upper triangle (excl diagonal)
    
    # Normalize to ensure sum=1.0 (handling 10-goal truncation)
    total = p_home + p_draw + p_away
    
    return DataFrame(
        match_id = match_id,
        market_name = "1X2",
        market_line = 0.0,
        selection = [:home, :draw, :away],
        prob_model = [p_home/total, p_draw/total, p_away/total]
    )
end

function solve_market(grid::ScoreGrid, m::Markets.MarketOverUnder, match_id::Int)
    P = grid.probs
    threshold = m.line
    
    # Logic: Sum all cells where (row_idx-1 + col_idx-1) > threshold
    # Note: Matrix indices are 1-based, goals are 0-based.
    
    p_over = 0.0
    rows, cols = size(P)
    
    for r in 1:rows
        for c in 1:cols
            goals = (r-1) + (c-1)
            if goals > threshold
                p_over += P[r,c]
            end
        end
    end
    
    # Renormalize based on total grid mass (truncation handling)
    total_mass = sum(P)
    p_over_norm = p_over / total_mass
    p_under_norm = 1.0 - p_over_norm
    
    return DataFrame(
        match_id = match_id,
        market_name = "OverUnder",
        market_line = m.line,
        selection = [:over, :under],
        prob_model = [p_over_norm, p_under_norm]
    )
end

function solve_market(grid::ScoreGrid, ::Markets.MarketBTTS, match_id::Int)
    P = grid.probs
    
    # Logic: Sum all cells where h > 0 AND a > 0
    # This corresponds to P[2:end, 2:end]
    
    p_yes = sum(view(P, 2:size(P,1), 2:size(P,2)))
    
    total_mass = sum(P)
    p_yes_norm = p_yes / total_mass
    p_no_norm = 1.0 - p_yes_norm
    
    return DataFrame(
        match_id = match_id,
        market_name = "BTTS",
        market_line = 0.0,
        selection = [:btts_yes, :btts_no],
        prob_model = [p_yes_norm, p_no_norm]
    )
end
