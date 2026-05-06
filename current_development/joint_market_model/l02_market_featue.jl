# current_development/joint_market_model/l02_market_feature.jl 

# example 
# function add_feature!(F_data::Dict, ::Val{:is_plastic}, ordered_ids, team_map::Dict, ds)
#     # Map match_id to home_team to check for plastic pitches
#     home_team_map = Dict(row.match_id => row.home_team for row in eachrow(ds.matches))
#
#     # PLASTIC_TEAMS is already defined in your file
#     F_data[:flat_is_plastic] = [home_team_map[id] in PLASTIC_TEAMS ? 1 : 0 for id in ordered_ids]
# end
#

using Distributions
using Optim
using LinearAlgebra # For diag, tril, triu


function BayesianFootball.Features.add_feature!(F_data::Dict, ::Val{:market_odds}, ordered_ids, team_map::Dict, ds)


end

# helper functions

function dixon_coles_tau(i::Int, j::Int, λ_h::Float64, λ_a::Float64, ρ::Float64)
    if i == 0 && j == 0 return 1.0 - (λ_h * λ_a * ρ)
    elseif i == 0 && j == 1 return 1.0 + (λ_h * ρ)
    elseif i == 1 && j == 0 return 1.0 + (λ_a * ρ)
    elseif i == 1 && j == 1 return 1.0 - ρ
    else return 1.0 end
end



#### 
# -----
# 1. Isolate the 
matches_2025 = subset(ds.matches, :season => ByRow(isequal("2025")))
odds_2025 = subset(ds.odds, :match_id => ByRow(in(matches_2025.match_id)))

# 2. Extract Market Parameters for all 2025 Matches
# Group the odds dataframe by match_id
grouped_odds = groupby(odds_2025, :match_id)

# Apply your fit_market_implied_parameters function to each group 
# and collect the NamedTuples directly into a new DataFrame
println("Fitting market parameters for $(length(grouped_odds)) matches...")
market_df = DataFrame([fit_market_implied_parameters(sub_df) for sub_df in grouped_odds])

####
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

# matches = subset(ds.matches, :season => ByRow(isequal("2025")))
# odds = subset(ds.odds, :match_id => ByRow(in(matches.match_id)))
#
#
# rand_matchid = rand( matches.match_id)
# rand_odds = subset(odds, :match_id  => ByRow(isequal(rand_matchid)))
# # Assuming 'rand_odds' is the 21-row DataFrame for match 13250794
# market_params = fit_market_implied_parameters(rand_odds)
#

############################################################
# --- Feature add 
############################################################

using Base.Threads

function BayesianFootball.Features.add_feature!(F_data::Dict, ::Val{:market_odds}, ordered_ids, team_map::Dict, ds)
    # 1. Filter the odds data FIRST
    # Converting to a Set is critical for performance so the 'in' check is instant
    id_set = Set(ordered_ids)
    
    # Keep only the rows where the match_id is in our target Set
    filtered_odds = subset(ds.odds, :match_id => ByRow(in(id_set)))
    
    # 2. Group ONLY the necessary matches
    odds_by_match = groupby(filtered_odds, :match_id)
    n_matches = length(odds_by_match)
    
    # 3. Pre-allocate an array to hold the thread results safely
    # Storing Tuple: (match_id, λ_home, λ_away, ρ)
    thread_results = Vector{Tuple{Int, Float64, Float64, Float64}}(undef, n_matches)
    
    # 4. Multi-threaded loop (Now only running for the matches we actually need!)
    @threads for i in 1:n_matches
        match_df = odds_by_match[i]
        res = fit_market_implied_parameters(match_df)
        
        # Save to the pre-allocated array safely
        thread_results[i] = (res.match_id, res.λ_home, res.λ_away, res.ρ)
    end
    
    # 5. Build the Dictionary from the array
    market_map = Dict{Int, NTuple{3, Float64}}(
        r[1] => (r[2], r[3], r[4]) for r in thread_results
    )

    # 6. Align to ordered_ids (Fallback to missings if a match had no odds data)
    F_data[:flat_market_λ_home] = [get(market_map, id, (missing, missing, missing))[1] for id in ordered_ids]
    F_data[:flat_market_λ_away] = [get(market_map, id, (missing, missing, missing))[2] for id in ordered_ids]
    F_data[:flat_market_ρ]      = [get(market_map, id, (missing, missing, missing))[3] for id in ordered_ids]
end

############################################################
# --- Model 
############################################################

Base.@kwdef struct DynamicMarketGoalsModel{
  I<:BayesianFootball.Models.PreGame.AbstractInterceptionConfig,
  T<:BayesianFootball.Models.PreGame.AbstractDynamicsConfig, 
  D<:BayesianFootball.Models.PreGame.AbstractDispersionConfig, 
  H<:BayesianFootball.Models.PreGame.AbstractHomeAdvantageConfig
  } <: BayesianFootball.AbstractNegBinModel
    interception_config::I
    dynamics_config::T
    dispersion_config::D
    homeadvantage_config::H
end


# src/Models/PreGame/engines/market_goals_engine.jl

using Turing
using Distributions

# ==========================================
# 1. THE TURING ENGINE
# =========================================
@model function build_market_goals_engine(
    # --- Base Data ---
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    time_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    # --- Market Data ---
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    idx_market::Vector{Int},
    idx_no_market::Vector{Int},
    # --- Dimensions ---
    n_teams::Int,
    n_history::Int,
    n_target::Int,
    config::DynamicMarketGoalsModel # Assume you create this config struct
)
    # ==========================================
    # 1. LOAD COMPONENTS
    # ==========================================
    inter ~ to_submodel(BayesianFootball.Models.PreGame.build_interception(config.interception_config))
    disp  ~ to_submodel(BayesianFootball.Models.PreGame.build_dispersion(config.dispersion_config))
    ha    ~ to_submodel(BayesianFootball.Models.PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(BayesianFootball.Models.PreGame.build_dynamics(config.dynamics_config, n_teams, n_history, n_target))

    # Market Variance: How much should we trust the market?
    # If σ is small, the model tightly hugs the market. If large, it relies more on goals.
    σ_market ~ truncated(Normal(0.1, 0.2), lower=0.01) 

    # ==========================================
    # 2. VECTORIZED INDEXING
    # ==========================================
    idx_h = CartesianIndex.(home_team_indices, time_indices)
    idx_a = CartesianIndex.(away_team_indices, time_indices)

    att_h = view(dyn.α, idx_h)
    def_h = view(dyn.β, idx_h)
    att_a = view(dyn.α, idx_a)
    def_a = view(dyn.β, idx_a)

    home_adv = view(ha, home_team_indices)

    # ==========================================
    # 3. RATE GENERATION (Log Scale)
    # ==========================================
    log_λ_h = clamp.(inter .+ home_adv .+ att_h .+ def_a, -10.0, 10.0) 
    log_λ_a = clamp.(inter .+             att_a .+ def_h, -10.0, 10.0)

    λ_h = exp.(log_λ_h) .+ 1e-6
    λ_a = exp.(log_λ_a) .+ 1e-6

    # AD-Safe Rejection
    if any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
        Turing.@addlogprob! -Inf
        return
    end

    # ==========================================
    # 4. LIKELIHOOD PIPELINE
    # ==========================================
    
    # A. Goal Likelihood (Always runs)
    home_goals ~ arraydist(BayesianFootball.MyDistributions.RobustNegativeBinomial.(disp.h, λ_h))
    away_goals ~ arraydist(BayesianFootball.MyDistributions.RobustNegativeBinomial.(disp.a, λ_a))

    # B. Market Likelihood (Only for matches where we scraped/solved the odds)
    if !isempty(idx_market)
        # We model the market's implied log-lambda as a noisy observation of the true model log-lambda
        market_log_λ_h[idx_market] ~ arraydist(Normal.(log_λ_h[idx_market], σ_market))
        market_log_λ_a[idx_market] ~ arraydist(Normal.(log_λ_a[idx_market], σ_market))
    end
end

# ==========================================
# 2. THE BUILDER
# ==========================================
function BayesianFootball.Features.required_features(model::DynamicMarketGoalsModel)
    return [:team_ids, :goals, :market_odds] 
end

function BayesianFootball.Models.PreGame.build_turing_model(config::DynamicMarketGoalsModel, feature_set::BayesianFootball.Features.FeatureSet)
    data = feature_set.data
    
    n_teams   = Int(data[:n_teams])
    n_rounds  = Int(data[:n_rounds])
    n_history = Int(data[:n_history_steps])
    n_target  = Int(data[:n_target_steps])
    
    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    time_idxs  = Vector{Int}(data[:time_indices])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    # 1. Extract Market Lambdas (assume you pre-calculated these via Optim and saved to db)
    # We take the log() immediately because our engine expects log_lambdas
    market_log_h = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_home]), NaN))
    market_log_a = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_away]), NaN))

    # 2. Split logic for missing market data
    idx_market    = findall(x -> !isnan(x), market_log_h)
    idx_no_market = findall(isnan, market_log_h)

    return build_market_goals_engine(
        home_ids, away_ids, time_idxs, 
        home_goals, away_goals, 
        market_log_h, market_log_a, 
        idx_market, idx_no_market,
        n_teams, n_history, n_target, config
    )
end

function BayesianFootball.Models.PreGame.extract_parameters(
    model::DynamicMarketGoalsModel, 
    df::AbstractDataFrame, 
    feature_set::FeatureSet,
    chain::Chains
)
    # 1. Unpack Metadata
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_rounds  = Int(data[:n_rounds])
    n_history = Int(data[:n_history_steps])
    n_target  = Int(data[:n_target_steps])
    team_map  = data[:team_map]

    # ==========================================
    # 2. DELEGATE TO COMPONENTS
    # ==========================================
    inter_v = extract_interception(chain, model.interception_config)
    disp_nt = extract_dispersion(chain, model.dispersion_config)
    ha_mat  = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn_nt  = extract_dynamics(chain, model.dynamics_config, "dyn", n_teams, n_history, n_target)


    n_samples = length(inter_v)
    results = Dict{Int, NamedTuple}()

    # ==========================================
    # 3. FIXTURE LOOP
    # ==========================================
    for row in eachrow(df)
        mid = Int(row.match_id)
        t_idx = hasproperty(row, :time_index) ? Int(row.time_index) : n_rounds

        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)

        α_h = h_idx > 0 ? dyn_nt.α[h_idx, t_idx, :] : zeros(n_samples)
        β_h = h_idx > 0 ? dyn_nt.β[h_idx, t_idx, :] : zeros(n_samples)
        α_a = a_idx > 0 ? dyn_nt.α[a_idx, t_idx, :] : zeros(n_samples)
        β_a = a_idx > 0 ? dyn_nt.β[a_idx, t_idx, :] : zeros(n_samples)

        γ_h = h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples)

        # ==========================================
        # 4. FINAL LIKELIHOOD MATH (Mirrors Turing Engine)
        # ==========================================
        # We add clamping and the 1e-6 offset to perfectly match what Turing saw
        log_λ_h = clamp.(inter_v .+ γ_h .+ α_h .+ β_a, -10.0, 10.0)
        log_λ_a = clamp.(inter_v .+        α_a .+ β_h, -10.0, 10.0)

        λ_goals_h = exp.(log_λ_h) .+ 1e-6
        λ_goals_a = exp.(log_λ_a) .+ 1e-6

        results[mid] = (;
            λ_h = λ_goals_h,
            λ_a = λ_goals_a,
            r_h = disp_nt.h,  
            r_a = disp_nt.a,
            true_xg_h = λ_goals_h, 
            true_xg_a = λ_goals_a,
        )
    end
    
    return results
end
