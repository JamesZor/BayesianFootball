# src/predictions/pregame/predict-dixoncoles.jl

using Distributions
using ...MyDistributions: DixonColes 

# We need to extend the predict_market dispatch for the extra parameter (ρ)
import ..Predictions: predict_market

# ==============================================================================
# 1. HELPER: FAST GRID CALCULATOR
# ==============================================================================
"""
    _calculate_dc_grid(λ, μ, ρ, max_goals)

Efficiently computes the full probability grid (PDF) for a single match sample.
Optimized to avoid re-instantiating Poisson distributions inside the loops.
Returns a (max_goals+1) x (max_goals+1) matrix of probabilities.
"""
function _calculate_dc_grid(λ::Float64, μ::Float64, ρ::Float64, max_goals::Int=10)
    # 1. Pre-compute independent Poisson marginals
    # We use valid truncation: we assume probability > max_goals is negligible for 1x2 calc
    home_dist = Poisson(λ)
    away_dist = Poisson(μ)
    
    p_home = [pdf(home_dist, i) for i in 0:max_goals]
    p_away = [pdf(away_dist, i) for i in 0:max_goals]

    # 2. Compute the Correction Matrix (Tau)
    # Default is 1.0 (Independence)
    # Only 0-0, 0-1, 1-0, 1-1 are affected
    
    grid = zeros(Float64, max_goals+1, max_goals+1)
    
    for h in 0:max_goals
        for a in 0:max_goals
            # Base Independence
            prob = p_home[h+1] * p_away[a+1]
            
            # Apply Correction
            correction = 1.0
            if h == 0 && a == 0
                correction = 1.0 - (λ * μ * ρ)
            elseif h == 0 && a == 1
                correction = 1.0 + (λ * ρ)
            elseif h == 1 && a == 0
                correction = 1.0 + (μ * ρ)
            elseif h == 1 && a == 1
                correction = 1.0 - ρ
            end
            
            # Safety clamp (probabilities cannot be negative)
            if correction < 0.0; correction = 0.0; end
            
            grid[h+1, a+1] = prob * correction
        end
    end

    return grid
end

# ==============================================================================
# 2. MARKET: 1X2
# ==============================================================================

function predict_market(
    model::TypesInterfaces.AbstractDixonColesModel,
    market::Markets.Market1X2, 
    λ_h::AbstractVector, 
    λ_a::AbstractVector,
    ρ::AbstractVector
)
    n_samples = length(λ_h)
    
    p_home_vec = zeros(n_samples)
    p_draw_vec = zeros(n_samples)
    p_away_vec = zeros(n_samples)

    Threads.@threads for i in 1:n_samples
        grid = _calculate_dc_grid(λ_h[i], λ_a[i], ρ[i])
        
        ph, pd, pa = 0.0, 0.0, 0.0
        
        rows, cols = size(grid)
        for h in 1:rows
            for a in 1:cols
                prob = grid[h, a]
                if (h-1) > (a-1)     # h > a (Index 1 is 0 goals)
                    ph += prob
                elseif (h-1) == (a-1) # Draw
                    pd += prob
                else                 # Away
                    pa += prob
                end
            end
        end
        
        # Normalize (Crucial for DC as corrections might shift sum slightly off 1.0)
        total = ph + pd + pa
        p_home_vec[i] = ph / total
        p_draw_vec[i] = pd / total
        p_away_vec[i] = pa / total
    end

    return NamedTuple{(:home, :draw, :away)}((p_home_vec, p_draw_vec, p_away_vec))
end

# ==============================================================================
# 3. MARKET: OVER/UNDER
# ==============================================================================

function predict_market(
    model::TypesInterfaces.AbstractDixonColesModel,
    market::Markets.MarketOverUnder, 
    λ_h::AbstractVector, 
    λ_a::AbstractVector,
    ρ::AbstractVector
)
    threshold = market.line # e.g. 2.5
    n_samples = length(λ_h)
    
    p_under_vec = zeros(n_samples)

    Threads.@threads for i in 1:n_samples
        grid = _calculate_dc_grid(λ_h[i], λ_a[i], ρ[i])
        
        prob_under = 0.0
        rows, cols = size(grid)
        
        for h in 1:rows
            for a in 1:cols
                goals = (h-1) + (a-1)
                if goals < threshold
                    prob_under += grid[h, a]
                end
            end
        end
        
        # We don't necessarily normalize against total grid sum here to keep "true" probabilities,
        # but normalizing against sum(grid) is safer if ρ is extreme.
        # Let's normalize for consistency.
        prob_under /= sum(grid)
        
        p_under_vec[i] = prob_under
    end

    p_over_vec = 1.0 .- p_under_vec
    
    line_str = replace(string(market.line), "." => "")
    over_key = Symbol("over_", line_str)
    under_key = Symbol("under_", line_str)
    
    return NamedTuple{(over_key, under_key)}((p_over_vec, p_under_vec))
end

# ==============================================================================
# 4. MARKET: BTTS
# ==============================================================================

function predict_market(
    model::TypesInterfaces.AbstractDixonColesModel,
    market::Markets.MarketBTTS, 
    λ_h::AbstractVector, 
    λ_a::AbstractVector,
    ρ::AbstractVector
)
    n_samples = length(λ_h)
    p_yes_vec = zeros(n_samples)

    Threads.@threads for i in 1:n_samples
        grid = _calculate_dc_grid(λ_h[i], λ_a[i], ρ[i])
        
        prob_yes = 0.0
        rows, cols = size(grid)
        
        # BTTS Yes: h > 0 AND a > 0
        # Indices start at 1 (which is 0 goals), so we start loop at 2 (1 goal)
        for h in 2:rows
            for a in 2:cols
                prob_yes += grid[h, a]
            end
        end
        
        prob_yes /= sum(grid)
        p_yes_vec[i] = prob_yes
    end

    p_no_vec = 1.0 .- p_yes_vec
    return NamedTuple{(:btts_yes, :btts_no)}((p_yes_vec, p_no_vec))
end

# ==============================================================================
# 5. ORCHESTRATION (DISPATCH FOR DIXONCOLES)
# ==============================================================================

"""
    predict_market(model::DixonColesModel, config, λ_h, λ_a, ρ)

Orchestrator that accepts the extra ρ parameter and broadcasts it to specific markets.
"""
function predict_market(
    model::TypesInterfaces.AbstractDixonColesModel,
    predict_config::PredictionConfig,
    λ_h::AbstractVector{Float64},
    λ_a::AbstractVector{Float64},
    ρ::AbstractVector{Float64}
)
    # We iterate over the markets in the config and call the specific methods above
    market_results_generator = (
        predict_market(model, market, λ_h, λ_a, ρ) for market in predict_config.markets
    )
    
    match_predict = reduce(merge, market_results_generator; init = (;) )
    return match_predict
end
