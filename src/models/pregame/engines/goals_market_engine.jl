# src/Models/PreGame/engines/goals_market_engine.jl

using Turing
using Distributions

Base.@kwdef struct DynamicMarketGoalsModel{
  I<:AbstractInterceptionConfig,
  T<:AbstractDynamicsConfig, 
  D<:AbstractDispersionConfig, 
  H<:AbstractHomeAdvantageConfig
    } <: AbstractDynamicNegBinModel
    interception_config::I
    dynamics_config::T
    dispersion_config::D
    homeadvantage_config::H
    market_σ::Distribution = truncated(Normal(0.1, 0.2), lower=0.01) 
end

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
    season_indices::Vector{Int}, # <--- NEW
    # --- Market Data ---
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    idx_market::Vector{Int},
    idx_no_market::Vector{Int},
    # --- Dimensions ---
    n_teams::Int,
    n_history::Int,
    n_target::Int,
    n_seasons::Int,              # <--- NEW
    config::DynamicMarketGoalsModel # Assume you create this config struct
)
    # ==========================================
    # 1. LOAD COMPONENTS
    # ==========================================
    inter ~ to_submodel(build_interception(config.interception_config, n_seasons))
    disp  ~ to_submodel(build_dispersion(config.dispersion_config))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams, n_history, n_target))

    # Market Variance: How much should we trust the market?
    # If σ is small, the model tightly hugs the market. If large, it relies more on goals.
    σ_market ~ config.market_σ

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

    inter_match = view(inter, season_indices) # <--- NEW

    # ==========================================
    # 3. RATE GENERATION (Log Scale)
    # ==========================================
    log_λ_h = clamp.(inter_match .+ home_adv .+ att_h .+ def_a, -10.0, 10.0) 
    log_λ_a = clamp.(inter_match .+             att_a .+ def_h, -10.0, 10.0)

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
    home_goals ~ arraydist(MyDistributions.RobustNegativeBinomial.(disp.h, λ_h))
    away_goals ~ arraydist(MyDistributions.RobustNegativeBinomial.(disp.a, λ_a))

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
function Features.required_features(model::DynamicMarketGoalsModel)
    return [:team_ids, :goals, :market_lambda] 
end

function build_turing_model(config::DynamicMarketGoalsModel, feature_set::Features.FeatureSet)
    data = feature_set.data
    
    n_teams   = Int(data[:n_teams])
    n_rounds  = Int(data[:n_rounds])
    n_history = Int(data[:n_history_steps])
    n_target  = Int(data[:n_target_steps])
    n_seasons  = Int(data[:n_seasons])
    
    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    time_idxs  = Vector{Int}(data[:time_indices])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])
    season_ids = Vector{Int}(data[:season_indices])

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
        season_ids,
        market_log_h, market_log_a, 
        idx_market, idx_no_market,
        n_teams, n_history, n_target, n_seasons,
        config
    )
end

function extract_parameters(
    model::DynamicMarketGoalsModel, 
    df::AbstractDataFrame, 
    feature_set::Features.FeatureSet,
    chain::Chains
)
    # 1. Unpack Metadata
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_rounds  = Int(data[:n_rounds])
    n_history = Int(data[:n_history_steps])
    n_target  = Int(data[:n_target_steps])
    n_seasons = Int(data[:n_seasons])
    team_map  = data[:team_map]

    # ==========================================
    # 2. DELEGATE TO COMPONENTS
    # ==========================================
    inter_mat = extract_interception(chain, model.interception_config, n_seasons)
    disp_nt = extract_dispersion(chain, model.dispersion_config)
    ha_mat  = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn_nt  = extract_dynamics(chain, model.dynamics_config, "dyn", n_teams, n_history, n_target)


    n_samples = size(chain, 1) * size(chain, 3) # total draws across all chains
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

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        inter_match = inter_mat[:, s_idx] 

        # ==========================================
        # 4. FINAL LIKELIHOOD MATH (Mirrors Turing Engine)
        # ==========================================
        # We add clamping and the 1e-6 offset to perfectly match what Turing saw
        log_λ_h = clamp.(inter_match .+ γ_h .+ α_h .+ β_a, -10.0, 10.0)
        log_λ_a = clamp.(inter_match .+        α_a .+ β_h, -10.0, 10.0)

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
