# src/models/pregame/engines/player_level/time_decay/outfield_xg_double_poisson.jl

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel{
    I<:AbstractInterceptionConfig,
    P<:OutfieldPlayerDynamicsConfig, 
    D<:AbstractDispersionConfig, # Unused mathematically in Poisson, but kept for interface consistency
    H<:AbstractHomeAdvantageConfig,
    K<:AbstractKappaConfig,
    R<:Features.AbstractFeatureConfig,
    M<:Features.AbstractMarketFeatureConfig
  } <: AbstractTimeDecayPlayerModel
      interception_config::I
      player_dynamics_config::P 
      dispersion_config::D
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      ν_xg::Distribution = truncated(Normal(3.0, 0.5), lower=0.5) 
      market_σ::Distribution = truncated(Normal(0.1, 0.2), lower=0.01) 
      market_weight::Float64 = 1.0
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_double_poisson_xg_market_player_engine(
    # --- Base Data ---
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    month_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    match_weights::Vector{Float64},
    # --- Player Positional Ratings ---
    home_G_ratings::Vector{Float64},
    home_D_ratings::Vector{Float64},
    home_M_ratings::Vector{Float64},
    home_F_ratings::Vector{Float64},
    away_G_ratings::Vector{Float64},
    away_D_ratings::Vector{Float64},
    away_M_ratings::Vector{Float64},
    away_F_ratings::Vector{Float64},
    # --- Expected Goals Data ---
    home_xg::Vector{Float64},
    away_xg::Vector{Float64},
    idx_xg::Vector{Int},
    idx_no_xg::Vector{Int},
    # --- Market Data ---
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    idx_market::Vector{Int},
    # --- Dimensions ---
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel
)
    # ==========================================
    # 1. LOAD COMPONENTS
    # ==========================================
    ν_xg     ~ config.ν_xg
    σ_market ~ config.market_σ
    
    inter ~ to_submodel(build_interception(config.interception_config, n_seasons))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(build_kappa(config.kappa_config, n_teams))
    p_dyn ~ to_submodel(build_dynamics(config.player_dynamics_config, n_teams))

    # ==========================================
    # 2. VECTORIZED INDEXING & MATH
    # ==========================================
    home_Outfield = home_D_ratings .+ home_M_ratings .+ home_F_ratings
    away_Outfield = away_D_ratings .+ away_M_ratings .+ away_F_ratings

    att_h = (p_dyn.w_G_att .* home_G_ratings) .+ (p_dyn.w_Outfield_att .* home_Outfield)
    def_h = (p_dyn.w_G_def .* home_G_ratings) .+ (p_dyn.w_Outfield_def .* home_Outfield)
    att_a = (p_dyn.w_G_att .* away_G_ratings) .+ (p_dyn.w_Outfield_att .* away_Outfield)
    def_a = (p_dyn.w_G_def .* away_G_ratings) .+ (p_dyn.w_Outfield_def .* away_Outfield)

    home_adv    = view(ha, home_team_indices)
    inter_match = view(inter, season_indices)
    κ_h_flat    = view(kap, home_team_indices)
    κ_a_flat    = view(kap, away_team_indices)

    # ==========================================
    # 3. STABLE RATE GENERATION (True xG)
    # ==========================================
    log_λₕ = clamp.(inter_match .+ home_adv .+ att_h .+ def_a, -20.0, 20.0) 
    log_λₐ = clamp.(inter_match .+             att_a .+ def_h, -20.0, 20.0)

    λₕ = exp.(log_λₕ) .+ 1e-6
    λₐ = exp.(log_λₐ) .+ 1e-6

    if any(isnan, λₕ) || any(isnan, λₐ) || any(isinf, λₕ) || any(isinf, λₐ)
        Turing.@addlogprob! -Inf
        return
    end

    # ==========================================
    # 4. TIME-DECAYED LIKELIHOOD PIPELINE
    # ==========================================
    
    # --- Pillar A: xG (Gamma) ---
    if !isempty(idx_xg)
        λₕ_xg = λₕ[idx_xg]
        λₐ_xg = λₐ[idx_xg]
        
        log_lik_xg_h = logpdf.(Gamma.(ν_xg, λₕ_xg ./ ν_xg), home_xg[idx_xg])
        log_lik_xg_a = logpdf.(Gamma.(ν_xg, λₐ_xg ./ ν_xg), away_xg[idx_xg])

        Turing.@addlogprob! sum(log_lik_xg_h .* match_weights[idx_xg])
        Turing.@addlogprob! sum(log_lik_xg_a .* match_weights[idx_xg])
    end

    # --- Pillar B: Actual Goals (Double Poisson) ---
    λ_goals_h = κ_h_flat .* λₕ
    λ_goals_a = κ_a_flat .* λₐ

    # Independent Poisson component
    log_lik_indep_h = logpdf.(Poisson.(λ_goals_h), home_goals)
    log_lik_indep_a = logpdf.(Poisson.(λ_goals_a), away_goals)

    # Combine into final likelihood vector for all matches
    log_lik_goals = log_lik_indep_h .+ log_lik_indep_a
    Turing.@addlogprob! sum(log_lik_goals .* match_weights)

    # --- Pillar C: The Market (Normal) ---
    if !isempty(idx_market)
        market_rate_h = log_λₕ[idx_market] .+ log.(κ_h_flat[idx_market])
        market_rate_a = log_λₐ[idx_market] .+ log.(κ_a_flat[idx_market])

        log_lik_market_h = logpdf.(Normal.(market_rate_h, σ_market), market_log_λ_h[idx_market])
        log_lik_market_a = logpdf.(Normal.(market_rate_a, σ_market), market_log_λ_a[idx_market])
        
        Turing.@addlogprob! config.market_weight * (sum(log_lik_market_h .* match_weights[idx_market]) + sum(log_lik_market_a .* match_weights[idx_market]))
    end
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel)
    return Features.AbstractFeatureConfig[
       Features.TeamIDsFeature(), 
       Features.GoalsFeature(), 
       Features.DatesFeature(), 
       Features.MonthFeature(), 
       Features.XGFeature(), 
       model.market_feature_config,
       model.player_ratings_feature,
       Features.TimeIndicesFeature()
    ] 
end

function build_turing_model(
    config::DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel, 
    feature_set::FeatureSet
)
    data = feature_set.data
    home_ids = Vector{Int}(data[:flat_home_team_id])
    away_ids = Vector{Int}(data[:flat_away_team_id])
    season_ids = Vector{Int}(data[:flat_season_id])
    month_indices = Vector{Int}(data[:flat_time_indices])
    home_goals = Vector{Int}(data[:flat_home_score])
    away_goals = Vector{Int}(data[:flat_away_score])
    match_weights = Vector{Float64}(data[:flat_match_weight])
    
    n_teams = data[:n_teams]
    n_seasons = data[:n_seasons]
    n_months = data[:n_months]

    # Player Ratings
    h_G = Vector{Float64}(data[:flat_home_G_rating])
    h_D = Vector{Float64}(data[:flat_home_D_rating])
    h_M = Vector{Float64}(data[:flat_home_M_rating])
    h_F = Vector{Float64}(data[:flat_home_F_rating])
    a_G = Vector{Float64}(data[:flat_away_G_rating])
    a_D = Vector{Float64}(data[:flat_away_D_rating])
    a_M = Vector{Float64}(data[:flat_away_M_rating])
    a_F = Vector{Float64}(data[:flat_away_F_rating])

    # xG
    home_xg = Vector{Float64}(coalesce.(data[:flat_home_xg], NaN))
    away_xg = Vector{Float64}(coalesce.(data[:flat_away_xg], NaN))
    idx_xg    = findall(x -> !isnan(x), home_xg)
    idx_no_xg = findall(isnan, home_xg)

    # Market
    market_log_h = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_home]), NaN))
    market_log_a = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_away]), NaN))
    idx_market   = findall(x -> !isnan(x), market_log_h)

    return build_double_poisson_xg_market_player_engine(
        home_ids, away_ids, season_ids, month_indices,
        home_goals, away_goals, match_weights,
        h_G, h_D, h_M, h_F, a_G, a_D, a_M, a_F,
        home_xg, away_xg, idx_xg, idx_no_xg,
        market_log_h, market_log_a, idx_market,
        n_teams, n_seasons, n_months,
        config
    )
end

# ==========================================
# 4. THE EXTRACTOR
# ==========================================
function extract_parameters(
    model::DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel, 
    df::AbstractDataFrame, 
    feature_set::FeatureSet, 
    chain::Chains
)
    data = feature_set.data
    n_matches = nrow(df)
    results = Vector{NamedTuple}(undef, n_matches)

    # 1. Base Rates
    inter_μ = mean(chain["inter.μ"])

    # 2. Team Ratings
    ha_team = [mean(chain["ha.γ_team_raw[$i]"]) for i in 1:data[:n_teams]]
    ha_base = mean(chain["ha.γ_base"])
    ha_σ = mean(chain["ha.σ_γ"])

    kap_team = [mean(chain["kap.κ_team_raw[$i]"]) for i in 1:data[:n_teams]]
    kap_base = mean(chain["kap.κ_base"])
    kap_σ = mean(chain["kap.σ_κ"])

    # 3. Time Dynamics (Weights)
    w_G_att = mean(chain["p_dyn.w_G_att"])
    w_G_def = mean(chain["p_dyn.w_G_def"])
    w_O_att = mean(chain["p_dyn.w_Outfield_att"])
    w_O_def = mean(chain["p_dyn.w_Outfield_def"])

    # Loop over matches to construct final parameters
    @inbounds for mid in 1:n_matches
        h_id = data[:flat_home_team_id][mid]
        a_id = data[:flat_away_team_id][mid]
        
        # Player specific ratings
        h_att = (w_G_att * data[:flat_home_G_rating][mid]) + 
                (w_O_att * (data[:flat_home_D_rating][mid] + data[:flat_home_M_rating][mid] + data[:flat_home_F_rating][mid]))
                
        h_def = (w_G_def * data[:flat_home_G_rating][mid]) + 
                (w_O_def * (data[:flat_home_D_rating][mid] + data[:flat_home_M_rating][mid] + data[:flat_home_F_rating][mid]))

        a_att = (w_G_att * data[:flat_away_G_rating][mid]) + 
                (w_O_att * (data[:flat_away_D_rating][mid] + data[:flat_away_M_rating][mid] + data[:flat_away_F_rating][mid]))
                
        a_def = (w_G_def * data[:flat_away_G_rating][mid]) + 
                (w_O_def * (data[:flat_away_D_rating][mid] + data[:flat_away_M_rating][mid] + data[:flat_away_F_rating][mid]))

        # Hierarchical Team adjustments
        ha_val = ha_base + (ha_team[h_id] * ha_σ)
        κ_h = exp(kap_base + (kap_team[h_id] * kap_σ))
        κ_a = exp(kap_base + (kap_team[a_id] * kap_σ))

        # True log rates (no kappa)
        log_λ_h = inter_μ + ha_val + h_att + a_def
        log_λ_a = inter_μ + a_att + h_def

        # Expected Goals
        λ_goals_h = κ_h * exp(log_λ_h) + 1e-6
        λ_goals_a = κ_a * exp(log_λ_a) + 1e-6

        results[mid] = (;
            λ_h = λ_goals_h,
            λ_a = λ_goals_a,
            θ_1 = log.(λ_goals_h),
            θ_2 = log.(λ_goals_a),
            ρ = 0.0, 
            true_xg_h = exp(log_λ_h), 
            true_xg_a = exp(log_λ_a),
        )
    end

    return results
end
