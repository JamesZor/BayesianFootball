# src/feature/extractors.jl
using Distributions
using Optim
using LinearAlgebra # For diag, tril, triu

# ---------------------------------------------------------
# 1. Core Mathematical Helpers
# ---------------------------------------------------------

function dixon_coles_tau(i::Int, j::Int, λ_h::Float64, λ_a::Float64, ρ::Float64)::Float64
    if i == 0 && j == 0 
        return 1.0 - (λ_h * λ_a * ρ)
    elseif i == 0 && j == 1 
        return 1.0 + (λ_h * ρ)
    elseif i == 1 && j == 0 
        return 1.0 + (λ_a * ρ)
    elseif i == 1 && j == 1 
        return 1.0 - ρ
    else 
        return 1.0 
    end
end

function _build_probability_matrix_P(λh::Float64, λa::Float64, ρ::Float64, max_goals::Integer)::Matrix{Float64}
    P = zeros(Float64, max_goals + 1, max_goals + 1)
    dist_h = Poisson(λh)
    dist_a = Poisson(λa)
    
    # Column-major loop order
    for j in 0:max_goals
        for i in 0:max_goals
            P[i+1, j+1] = pdf(dist_h, i) * pdf(dist_a, j) * dixon_coles_tau(i, j, λh, λa, ρ)
        end
    end
    return P
end

# ---------------------------------------------------------
# 2. Market Error Calculators (Using Multiple Dispatch)
# ---------------------------------------------------------
# Notice we return the calculated error instead of mutating an 'sse' argument

function _calculate_error(::Val{:result_1x2}, P::Matrix{Float64}, targets::Dict{Symbol,Float64})
    err = 0.0
    if haskey(targets, :home) err += (sum(tril(P, -1)) - targets[:home])^2 end
    if haskey(targets, :draw) err += (sum(diag(P)) - targets[:draw])^2 end
    if haskey(targets, :away) err += (sum(triu(P, 1)) - targets[:away])^2 end
    return err
end

function _calculate_error(::Val{:btts}, P::Matrix{Float64}, targets::Dict{Symbol,Float64})
    err = 0.0
    if haskey(targets, :btts_yes)
        # Use @views to prevent memory allocation when slicing the matrix
        prob_btts = sum(@views P[2:end, 2:end]) 
        err += (prob_btts - targets[:btts_yes])^2
        if haskey(targets, :btts_no) 
            err += ((1.0 - prob_btts) - targets[:btts_no])^2 
        end
    end 
    return err
end

function _calculate_error_underover(k::Integer, P::Matrix{Float64}, targets::Dict{Symbol,Float64})
    err = 0.0
    over_key = Symbol("over_$(k)5")
    under_key = Symbol("under_$(k)5")

    if haskey(targets, over_key) || haskey(targets, under_key)
        prob_under = 0.0
        max_goals = size(P, 1) - 1 # Dynamically get max_goals from matrix size
        
        for j in 0:max_goals
            for i in 0:max_goals
                if (i + j) <= k
                    prob_under += P[i+1, j+1]
                end
            end
        end
        
        prob_over = 1.0 - prob_under
        
        if haskey(targets, over_key)  err += (prob_over - targets[over_key])^2 end
        if haskey(targets, under_key) err += (prob_under - targets[under_key])^2 end
    end
    return err
end

function _calculate_error(::Val{:uo}, P::Matrix{Float64}, targets::Dict{Symbol,Float64}; min_k::Integer=0, max_k::Integer=8)
    err = 0.0
    for k in min_k:max_k
        err += _calculate_error_underover(k, P, targets)
    end
    return err
end

# ---------------------------------------------------------
# 3. The Main Wrapper & Optimizer
# ---------------------------------------------------------

function fit_market_implied_parameters(match_df; max_goals=10)
    
    # 1. Build the target dictionary once
    targets = Dict{Symbol, Float64}(row.selection => row.prob_fair_close for row in eachrow(match_df))

    # 2. Define the loss function as a closure
    # This allows Optim to call loss(θ), but the function still has access to `targets` and `max_goals`
    function loss(θ::Vector{Float64})
        λh, λa = exp(θ[1]), exp(θ[2])
        ρ = θ[3]

        P = _build_probability_matrix_P(λh, λa, ρ, max_goals)
        
        # Sum up the errors by passing the Type Values to trigger dispatch
        sse = 0.0
        sse += _calculate_error(Val(:result_1x2), P, targets)
        sse += _calculate_error(Val(:btts), P, targets)
        sse += _calculate_error(Val(:uo), P, targets)

        return sse
    end

    # 3. Run Optimization
    initial_guess = [log(1.5), log(1.0), 0.05]
    result = optimize(loss, initial_guess, NelderMead())
    
    # 4. Return formatted results
    # Using first() is safer than [1] for extracting a single value from a column
    return (
        match_id = first(match_df.match_id),
        λ_home = exp(Optim.minimizer(result)[1]),
        λ_away = exp(Optim.minimizer(result)[2]),
        ρ = Optim.minimizer(result)[3],
        fit_error = Optim.minimum(result) 
    )
end

