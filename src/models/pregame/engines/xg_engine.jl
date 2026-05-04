# src/Models/PreGame/engines/xg_engine.jl

using Turing
using Distributions

# ==========================================
# 1. THE TURING ENGINE
# ==========================================
@model function build_xg_engine(
    # --- Base Data ---
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    time_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    # --- Expected Goals Data ---
    home_xg::Vector{Float64},
    away_xg::Vector{Float64},
    idx_xg::Vector{Int},
    idx_no_xg::Vector{Int},
    # --- Dimensions ---
    n_teams::Int,
    n_history::Int,
    n_target::Int,
    config::DynamicXGModel
)
    # ==========================================
    # 1. LOAD BASELINES & COMPONENTS
    # ==========================================
    # Sample the global baselines directly from the config distributions
    ν_xg ~ config.ν_xg

    inter ~ to_submodel(build_interception(config.interception_config))
    disp  ~ to_submodel(build_dispersion(config.dispersion_config))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    kap  ~ to_submodel(build_kappa(config.kappa_config, n_teams))
    # dyn returns the FULL matrices: dyn.α and dyn.β
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams, n_history, n_target))


    # ==========================================
    # 2. VECTORIZED INDEXING
    # ==========================================
    idx_h = CartesianIndex.(home_team_indices, time_indices)
    idx_a = CartesianIndex.(away_team_indices, time_indices)

    att_h = view(dyn.α, idx_h)
    def_h = view(dyn.β, idx_h)
    att_a = view(dyn.α, idx_a)
    def_a = view(dyn.β, idx_a)

    γ_h_flat = view(ha, home_team_indices)
    κ_h_flat = view(kap, home_team_indices)
    κ_a_flat = view(kap, away_team_indices)

    # ==========================================
    # 3. STABLE RATE GENERATION (True xG)
    # ==========================================
    # Inject the mapped Home Advantage into the Home Lambda
    log_λₕ = clamp.(inter .+ γ_h_flat .+ att_h .+ def_a, -20.0, 20.0) 
    log_λₐ = clamp.(inter .+             att_a .+ def_h, -20.0, 20.0)

    λₕ = exp.(log_λₕ) .+ 1e-6
    λₐ = exp.(log_λₐ) .+ 1e-6

    # AD-Safe Rejection: Prevent gradient explosions during warmup
    if any(isnan, λₕ) || any(isnan, λₐ) || any(isinf, λₕ) || any(isinf, λₐ)
        Turing.@addlogprob! -Inf
        return
    end

    # ==========================================
    # 4. LIKELIHOOD PIPELINE (Splitting by xG availability)
    # ==========================================
    
    # Group 1: Matches WITH xG Data
    if !isempty(idx_xg)
        λₕ_xg = λₕ[idx_xg]
        λₐ_xg = λₐ[idx_xg]
        
        # Observation 1: Expected Goals generation (Gamma)
        home_xg[idx_xg] ~ arraydist(Gamma.(ν_xg, λₕ_xg ./ ν_xg))
        away_xg[idx_xg] ~ arraydist(Gamma.(ν_xg, λₐ_xg ./ ν_xg))

        # Observation 2: Actual Goals generation (λ_goals = kappa * true_xg)
        home_goals[idx_xg] ~ arraydist(RobustNegativeBinomial.(disp.h, κ_h_flat[idx_xg] .* λₕ_xg))
        away_goals[idx_xg] ~ arraydist(RobustNegativeBinomial.(disp.a, κ_a_flat[idx_xg] .* λₐ_xg))
    end

    # Group 2: Matches WITHOUT xG Data (History / Lower Leagues)
    if !isempty(idx_no_xg)
        λₕ_no = λₕ[idx_no_xg]
        λₐ_no = λₐ[idx_no_xg]

        # Only observe Actual Goals, heavily inferring strength backward through Kappa
        home_goals[idx_no_xg] ~ arraydist(RobustNegativeBinomial.(disp.h, κ_h_flat[idx_no_xg] .* λₕ_no))
        away_goals[idx_no_xg] ~ arraydist(RobustNegativeBinomial.(disp.a, κ_a_flat[idx_no_xg] .* λₐ_no))
    end
end


# ==========================================
# 2. THE BUILDER
# ==========================================
function Features.required_features(model::DynamicXGModel)
    return [:team_ids, :goals, :xg] # Ensure xG pipeline runs!
end

function build_turing_model(config::DynamicXGModel, feature_set::FeatureSet)
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

    # --- THE FIX IS HERE ---
    # coalesce.() replaces any `missing` with `NaN`. 
    # Then we safely cast the entire clean array to Float64!
    home_xg = Vector{Float64}(coalesce.(data[:flat_home_xg], NaN))
    away_xg = Vector{Float64}(coalesce.(data[:flat_away_xg], NaN))

    # Your dynamic splitting logic works perfectly on NaN!
    idx_xg    = findall(x -> !isnan(x), home_xg)
    idx_no_xg = findall(isnan, home_xg)

    return build_xg_engine(
        home_ids, away_ids, time_idxs, 
        home_goals, away_goals, 
        home_xg, away_xg, 
        idx_xg, idx_no_xg,
        n_teams, n_history, n_target, config
    )
end


# ==========================================
# 3. THE EXTRACTOR
# ==========================================
function extract_parameters(
    model::DynamicXGModel, 
    df::AbstractDataFrame, 
    feature_set::FeatureSet,
    chain::Chains
)
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_rounds  = Int(data[:n_rounds])
    n_history = Int(data[:n_history_steps])
    n_target  = Int(data[:n_target_steps])
    team_map  = data[:team_map]

    # Unpack Baselines
    ν_xg_v = vec(Array(chain[:ν_xg]))

    # Delegate to Components
    μ_v = extract_interception(chain, model.interception_config)
    disp_nt = extract_dispersion(chain, model.dispersion_config)
    ha_mat  = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat = extract_kappa(chain, model.kappa_config, n_teams)
    dyn_nt  = extract_dynamics(chain, model.dynamics_config, "dyn", n_teams, n_history, n_target)

    n_samples = length(μ_v)
    results = Dict{Int, NamedTuple}()

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
        
        # Extract Team-Specific Conversion Rates (Kappa)
        κ_h = h_idx > 0 ? kap_mat[:, h_idx] : ones(n_samples)
        κ_a = a_idx > 0 ? kap_mat[:, a_idx] : ones(n_samples)

        # 1. Calculate True Underlying xG
        true_xg_h = exp.(μ_v .+ γ_h .+ α_h .+ β_a)
        true_xg_a = exp.(μ_v .+        α_a .+ β_h)

        # 2. Map xG to Actual Goals using Kappa
        λ_goals_h = κ_h .* true_xg_h
        λ_goals_a = κ_a .* true_xg_a

        results[mid] = (;
            λ_h = λ_goals_h,
            λ_a = λ_goals_a,
            r_h = disp_nt.h,  
            r_a = disp_nt.a,
            true_xg_h = true_xg_h, 
            true_xg_a = true_xg_a
        )
    end
    
    return results
end
