# src/Models/PreGame/engines/goals_engine.jl

using Turing
using Distributions

Base.@kwdef struct DynamicGoalsModel{
  I<:AbstractInterceptionConfig,
  T<:AbstractDynamicsConfig, 
  D<:AbstractDispersionConfig, 
  H<:AbstractHomeAdvantageConfig
    } <: AbstractDynamicNegBinModel
    interception_config::I
    dynamics_config::T
    dispersion_config::D
    homeadvantage_config::H
end

@model function build_goals_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int}, # <--- NEW
    time_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    n_teams::Int,
    n_seasons::Int,              # <--- NEW
    n_history::Int,
    n_target::Int,
    config::DynamicGoalsModel
)
    # 1. LOAD THE COMPONENTS
    # We pass n_seasons to the submodel
    inter ~ to_submodel(build_interception(config.interception_config, n_seasons))
    disp ~ to_submodel(build_dispersion(config.dispersion_config))
    ha   ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    dyn  ~ to_submodel(build_dynamics(config.dynamics_config, n_teams, n_history, n_target))

    # 2. VECTORIZED INDEXING
    idx_h = CartesianIndex.(home_team_indices, time_indices)
    idx_a = CartesianIndex.(away_team_indices, time_indices)

    att_h = view(dyn.α, idx_h)
    def_h = view(dyn.β, idx_h)
    att_a = view(dyn.α, idx_a)
    def_a = view(dyn.β, idx_a)

    home_adv = view(ha, home_team_indices)
    
    # Map the correct seasonal intercept to each match
    # If GlobalInterception: inter_seasons is [μ, μ, μ...]
    # If SeasonalInterception: inter_seasons is [μ_2021, μ_2022...]
    inter_match = view(inter, season_indices) # <--- NEW

    # 3. VECTORIZED RATES (λ)
    λ_h = exp.(inter_match .+ home_adv .+ att_h .+ def_a)
    λ_a = exp.(inter_match .+             att_a .+ def_h)

    # 4. LIKELIHOOD
    home_goals ~ arraydist(RobustNegativeBinomial.(disp.h, λ_h))
    away_goals ~ arraydist(RobustNegativeBinomial.(disp.a, λ_a))
end

function Features.required_features(model::DynamicGoalsModel)
    return [:team_ids, :goals] 
end


function build_turing_model(config::DynamicGoalsModel, feature_set::FeatureSet)
    data = feature_set.data
    
    n_teams    = Int(data[:n_teams])
    n_seasons  = Int(data[:n_seasons]) # <--- Ensure your pipeline provides this
    n_history  = Int(data[:n_history_steps])
    n_target   = Int(data[:n_target_steps])
    
    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices]) # <--- NEW
    time_idxs  = Vector{Int}(data[:time_indices])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    return build_goals_engine(
        home_ids,
        away_ids,
        season_ids,
        time_idxs,
        home_goals,
        away_goals,
        n_teams,
        n_seasons,
        n_history,
        n_target,
        config
    )
end

function extract_parameters(
    model::DynamicGoalsModel, 
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
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    team_map  = data[:team_map]

    # inter_mat is [Samples, Seasons]
    inter_mat = extract_interception(chain, model.interception_config, n_seasons)
    disp_nt   = extract_dispersion(chain, model.dispersion_config)
    ha_mat    = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn_nt  = extract_dynamics(chain, model.dynamics_config, "dyn", n_teams, n_history, n_target)

    n_samples = size(chain, 1) * size(chain, 3) # total draws across all chains
    results = Dict{Int, NamedTuple}()

#     # ==========================================
#     # 3. FIXTURE LOOP (Calculate λ for each match)
#     # ==========================================
    for row in eachrow(df)
        mid = Int(row.match_id)

        # If forecasting future matches, default to the most recent time step (n_rounds)
        # If backtesting, use the exact time_index from the DataFrame
        t_idx = hasproperty(row, :time_index) ? Int(row.time_index) : n_rounds

        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)

        # Safely extract dynamic states (Fallback to 0 if a team isn't in the mapping)
        # Note: dyn_nt.α is [Teams, Time, Samples] based on your _reconstruct_trajectory logic
        α_h = h_idx > 0 ? dyn_nt.α[h_idx, t_idx, :] : zeros(n_samples)
        β_h = h_idx > 0 ? dyn_nt.β[h_idx, t_idx, :] : zeros(n_samples)
        α_a = a_idx > 0 ? dyn_nt.α[a_idx, t_idx, :] : zeros(n_samples)
        β_a = a_idx > 0 ? dyn_nt.β[a_idx, t_idx, :] : zeros(n_samples)

        # ha_mat is [Samples, Teams]
        γ_h = h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        inter_match = inter_mat[:, s_idx] 
        # ==========================================
        # 4. FINAL LIKELIHOOD MATH
        # ==========================================
        λ_goals_h = exp.(inter_match .+ γ_h .+ α_h .+ β_a)
        λ_goals_a = exp.(inter_match .+        α_a .+ β_h)

        # Pack it exactly how model_inference() expects it!
        results[mid] = (;
            λ_h = λ_goals_h,
            λ_a = λ_goals_a,
            r_h = disp_nt.h,  # Handles Global vs Home/Away automatically
            r_a = disp_nt.a,

            # Since this is a Raw Goals model, True xG = Expected Goals
            true_xg_h = λ_goals_h, 
            true_xg_a = λ_goals_a
        )
    end

    return results
end
