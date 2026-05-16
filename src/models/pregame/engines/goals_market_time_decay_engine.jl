# src/models/pregame/engines/goals_market_time_decay_engine.jl

function calculate_market_match_weights(deltas::Vector{<:Real}, half_life_days::Real)
    weights = 0.5 .^ (deltas ./ half_life_days)
    return weights
end

@model function build_weighted_market_goals_engine(
    # --- Base Data ---
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    time_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    match_weights::Vector{Float64},
    # --- Market Data ---
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    idx_market::Vector{Int},
    idx_no_market::Vector{Int},
    # --- Dimensions ---
    n_teams::Int,
    n_seasons::Int,
    config::DynamicMarketGoalsTimeDecayModel
)
    # 1. LOAD COMPONENTS
    inter ~ to_submodel(build_interception(config.interception_config, n_seasons))
    disp  ~ to_submodel(build_dispersion(config.dispersion_config))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams))

    # Market Variance
    σ_market ~ config.market_σ

    # 2. VECTORIZED INDEXING 
    att_h = view(dyn.α, home_team_indices)
    def_h = view(dyn.β, home_team_indices)
    att_a = view(dyn.α, away_team_indices)
    def_a = view(dyn.β, away_team_indices)
    inter_match = view(inter, season_indices)
    home_adv = view(ha, home_team_indices)

    # 3. VECTORIZED RATES (Log Scale)
    log_λ_h = clamp.(inter_match .+ home_adv .+ att_h .+ def_a, -10.0, 10.0)
    log_λ_a = clamp.(inter_match .+             att_a .+ def_h, -10.0, 10.0)

    λ_h = exp.(log_λ_h) .+ 1e-6
    λ_a = exp.(log_λ_a) .+ 1e-6

    # AD-Safe Rejection
    if any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
        Turing.@addlogprob! -Inf
        return
    end

    # 4. TIME-DECAYED LIKELIHOOD PIPELINE
    
    # A. Goal Likelihood (Always runs)
    log_lik_goals_h = logpdf.(RobustNegativeBinomial.(disp.h, λ_h), home_goals)
    log_lik_goals_a = logpdf.(RobustNegativeBinomial.(disp.a, λ_a), away_goals)

    Turing.@addlogprob! sum(log_lik_goals_h .* match_weights)
    Turing.@addlogprob! sum(log_lik_goals_a .* match_weights)

    # B. Market Likelihood (Only for matches with odds)
    if !isempty(idx_market)
        log_lik_market_h = logpdf.(Normal.(log_λ_h[idx_market], σ_market), market_log_λ_h[idx_market])
        log_lik_market_a = logpdf.(Normal.(log_λ_a[idx_market], σ_market), market_log_λ_a[idx_market])
        
        # Apply time decay weights AND the custom mixing weight (default 1.0)
        Turing.@addlogprob! sum(log_lik_market_h .* match_weights[idx_market]) * config.market_weight
        Turing.@addlogprob! sum(log_lik_market_a .* match_weights[idx_market]) * config.market_weight
    end
end

function Features.required_features(model::DynamicMarketGoalsTimeDecayModel)
    return [:team_ids, :goals, :dates, :market_lambda] 
end

function build_turing_model(config::DynamicMarketGoalsTimeDecayModel, feature_set::FeatureSet)
    data = feature_set.data
    
    n_teams    = Int(data[:n_teams])
    n_seasons  = Int(data[:n_seasons])
    
    date_deltas = Vector{Int}(data[:dates])
    # Note: re-using the weight calculation logic from calculate_match_weights in core time decay
    match_weights = 0.5 .^ (date_deltas ./ config.dynamics_config.days_half_life)
    
    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices])
    time_idxs  = Vector{Int}(data[:time_indices])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    # Extract Market Lambdas
    market_log_h = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_home]), NaN))
    market_log_a = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_away]), NaN))

    # Split logic for missing market data
    idx_market    = findall(x -> !isnan(x), market_log_h)
    idx_no_market = findall(isnan, market_log_h)

    return build_weighted_market_goals_engine(
        home_ids, away_ids,
        season_ids, time_idxs,
        home_goals, away_goals,
        match_weights,
        market_log_h, market_log_a, 
        idx_market, idx_no_market,
        n_teams, n_seasons,
        config
    )
end

function extract_parameters(
    model::DynamicMarketGoalsTimeDecayModel, 
    df::AbstractDataFrame, 
    feature_set::FeatureSet,
    chain::Chains
)
    # 1. Unpack Metadata
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_rounds  = Int(data[:n_rounds])
    n_seasons = Int(data[:n_seasons])
    team_map  = data[:team_map]

    # 2. DELEGATE TO COMPONENTS
    inter_mat = extract_interception(chain, model.interception_config, n_seasons)
    disp_nt   = extract_dispersion(chain, model.dispersion_config)
    ha_mat    = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn_nt    = extract_dynamics(chain, model.dynamics_config, "dyn", n_teams)

    n_samples = size(chain, 1) * size(chain, 3) 
    results = Dict{Int, NamedTuple}()

    # 3. FIXTURE LOOP
    for row in eachrow(df)
        mid = Int(row.match_id)
        
        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)

        α_h = h_idx > 0 ? dyn_nt.α[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? dyn_nt.β[:, h_idx] : zeros(n_samples)
        α_a = a_idx > 0 ? dyn_nt.α[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? dyn_nt.β[:, a_idx] : zeros(n_samples)

        γ_h = h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        inter_match = inter_mat[:, s_idx] 

        # 4. FINAL LIKELIHOOD MATH
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
