# src/models/pregame/engines/goals_time_decay_engine.jl

function calculate_match_weights(deltas::Vector{<:Real}, half_life_days::Real)
    weights = 0.5 .^ (deltas ./ half_life_days)
    return weights
end



function Features.required_features(model::DynamicGoalsTimeDecayModel)
    return [:team_ids, :goals, :dates] 
end


@model function build_weighted_goals_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    time_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    match_weights::Vector{Float64},
    n_teams::Int,
    n_seasons::Int,
    config::DynamicGoalsTimeDecayModel
)
    # 1. LOAD COMPONENTS
    inter ~ to_submodel(build_interception(config.interception_config, n_seasons))
    disp  ~ to_submodel(build_dispersion(config.dispersion_config))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams))

    # 2. VECTORIZED INDEXING 
    att_h = view(dyn.α, home_team_indices)
    def_h = view(dyn.β, home_team_indices)
    att_a = view(dyn.α, away_team_indices)
    def_a = view(dyn.β, away_team_indices)
    inter_match = view(inter, season_indices)
    home_adv = view(ha, home_team_indices)

    # 3. VECTORIZED RATES (λ)
    λ_h = exp.(inter_match .+ home_adv .+ att_h .+ def_a)
    λ_a = exp.(inter_match .+             att_a .+ def_h)

    # 4. TIME-DECAYED LIKELIHOOD
    log_lik_h = logpdf.(RobustNegativeBinomial.(disp.h, λ_h), home_goals)
    log_lik_a = logpdf.(RobustNegativeBinomial.(disp.a, λ_a), away_goals)

    Turing.@addlogprob! sum(log_lik_h .* match_weights)
    Turing.@addlogprob! sum(log_lik_a .* match_weights)
end

function build_turing_model(model::DynamicGoalsTimeDecayModel, feature_set::FeatureSet)
    data = feature_set.data
    
    n_teams    = Int(data[:n_teams])
    n_seasons  = Int(data[:n_seasons])
    
    date_deltas = Vector{Int}(data[:dates])
    match_weights = calculate_match_weights(date_deltas, model.dynamics_config.days_half_life)
    
    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices])
    time_idxs  = Vector{Int}(data[:time_indices])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    return build_weighted_goals_engine(
        home_ids,
        away_ids,
        season_ids,
        time_idxs,
        home_goals,
        away_goals,
        match_weights,
        n_teams,
        n_seasons,
        model
    )
end

function extract_parameters(
    model::DynamicGoalsTimeDecayModel, 
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

    # inter_mat is [Samples, Seasons]
    inter_mat = extract_interception(chain, model.interception_config, n_seasons)
    disp_nt   = extract_dispersion(chain, model.dispersion_config)
    ha_mat    = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn_nt    = extract_dynamics(chain, model.dynamics_config, "dyn", n_teams)

    n_samples = size(chain, 1) * size(chain, 3) 
    results = Dict{Int, NamedTuple}()

    for row in eachrow(df)
        mid = Int(row.match_id)

        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)

        # dyn_nt.α is [Samples, Teams]
        α_h = h_idx > 0 ? dyn_nt.α[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? dyn_nt.β[:, h_idx] : zeros(n_samples)
        α_a = a_idx > 0 ? dyn_nt.α[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? dyn_nt.β[:, a_idx] : zeros(n_samples)

        γ_h = h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        inter_match = inter_mat[:, s_idx] 

        λ_goals_h = exp.(inter_match .+ γ_h .+ α_h .+ β_a)
        λ_goals_a = exp.(inter_match .+        α_a .+ β_h)

        results[mid] = (;
            λ_h = λ_goals_h,
            λ_a = λ_goals_a,
            r_h = disp_nt.h,
            r_a = disp_nt.a,
            true_xg_h = λ_goals_h, 
            true_xg_a = λ_goals_a
        )
    end

    return results
end
