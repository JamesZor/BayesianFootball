using Turing
using Distributions

# ==========================================
# 1. THE TURING ENGINE
# ==========================================
@model function build_xg_market_engine(
    # --- Base Data ---
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    time_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    # --- xG Data ---
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
    n_history::Int,
    n_target::Int,
    config::DynamicMarketXGModel
)
    # ==========================================
    # 1. LOAD BASELINES & COMPONENTS
    # ==========================================
    ν_xg     ~ config.ν_xg
    σ_market ~ config.market_σ

    inter ~ to_submodel(build_interception(config.interception_config))
    disp  ~ to_submodel(build_dispersion(config.dispersion_config))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(build_kappa(config.kappa_config, n_teams))
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
    log_λₕ = clamp.(inter .+ γ_h_flat .+ att_h .+ def_a, -20.0, 20.0) 
    log_λₐ = clamp.(inter .+             att_a .+ def_h, -20.0, 20.0)

    λₕ = exp.(log_λₕ) .+ 1e-6
    λₐ = exp.(log_λₐ) .+ 1e-6

    if any(isnan, λₕ) || any(isnan, λₐ) || any(isinf, λₕ) || any(isinf, λₐ)
        Turing.@addlogprob! -Inf
        return
    end

    # ==========================================
    # 4. LIKELIHOOD PIPELINE (The 3 Pillars)
    # ==========================================
    
    # --- Pillar A & B: Goals and xG ---
    if !isempty(idx_xg)
        λₕ_xg = λₕ[idx_xg]
        λₐ_xg = λₐ[idx_xg]
        
        # Observation 1: Expected Goals generation (Gamma)
        home_xg[idx_xg] ~ arraydist(Gamma.(ν_xg, λₕ_xg ./ ν_xg))
        away_xg[idx_xg] ~ arraydist(Gamma.(ν_xg, λₐ_xg ./ ν_xg))

        # Observation 2: Actual Goals generation (λ_goals = kappa * true_xg)
        home_goals[idx_xg] ~ arraydist(MyDistributions.RobustNegativeBinomial.(disp.h, κ_h_flat[idx_xg] .* λₕ_xg))
        away_goals[idx_xg] ~ arraydist(MyDistributions.RobustNegativeBinomial.(disp.a, κ_a_flat[idx_xg] .* λₐ_xg))
    end

    # Fallback for matches without xG
    if !isempty(idx_no_xg)
        home_goals[idx_no_xg] ~ arraydist(MyDistributions.RobustNegativeBinomial.(disp.h, κ_h_flat[idx_no_xg] .* λₕ[idx_no_xg]))
        away_goals[idx_no_xg] ~ arraydist(MyDistributions.RobustNegativeBinomial.(disp.a, κ_a_flat[idx_no_xg] .* λₐ[idx_no_xg]))
    end

    # --- Pillar C: The Market ---
    if !isempty(idx_market)
        # The Market predicts GOALS, not xG. So we must multiply True xG by Kappa.
        # In log space: log(xG * Kappa) = log(xG) + log(Kappa)
        market_rate_h = log_λₕ[idx_market] .+ log.(κ_h_flat[idx_market])
        market_rate_a = log_λₐ[idx_market] .+ log.(κ_a_flat[idx_market])

        market_log_λ_h[idx_market] ~ arraydist(Normal.(market_rate_h, σ_market))
        market_log_λ_a[idx_market] ~ arraydist(Normal.(market_rate_a, σ_market))
    end
end

# ==========================================
# 2. THE BUILDER
# ==========================================
function Features.required_features(model::DynamicMarketXGModel)
    return [:team_ids, :goals, :xg, :market_lambda] 
end

function build_turing_model(config::DynamicMarketXGModel, feature_set::Features.FeatureSet)
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

    # 1. Clean xG Data
    home_xg = Vector{Float64}(coalesce.(data[:flat_home_xg], NaN))
    away_xg = Vector{Float64}(coalesce.(data[:flat_away_xg], NaN))
    idx_xg    = findall(x -> !isnan(x), home_xg)
    idx_no_xg = findall(isnan, home_xg)

    # 2. Clean Market Data
    market_log_h = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_home]), NaN))
    market_log_a = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_away]), NaN))
    idx_market = findall(x -> !isnan(x), market_log_h)

    return build_xg_market_engine(
        home_ids, away_ids, time_idxs, 
        home_goals, away_goals, 
        home_xg, away_xg, 
        idx_xg, idx_no_xg,
        market_log_h, market_log_a, idx_market,
        n_teams, n_history, n_target, config
    )
end

# ==========================================
# 3. THE EXTRACTOR
# ==========================================
function extract_parameters(
    model::DynamicMarketXGModel, 
    df::AbstractDataFrame, 
    feature_set::Features.FeatureSet,
    chain::Chains
)
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_rounds  = Int(data[:n_rounds])
    n_history = Int(data[:n_history_steps])
    n_target  = Int(data[:n_target_steps])
    team_map  = data[:team_map]

    # Unpack Components
    μ_v     = extract_interception(chain, model.interception_config)
    disp_nt = extract_dispersion(chain, model.dispersion_config)
    ha_mat  = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat = extract_kappa(chain, model.kappa_config, n_teams)
    dyn_nt  = extract_dynamics(chain, model.dynamics_config, "dyn", n_teams, n_history, n_target)

    # Optional: Extract Variances for diagnostics
    σ_mkt = haskey(chain, :σ_market) ? Array(chain[:σ_market]) : nothing

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
        
        κ_h = h_idx > 0 ? kap_mat[:, h_idx] : ones(n_samples)
        κ_a = a_idx > 0 ? kap_mat[:, a_idx] : ones(n_samples)

        # 1. True Underlying xG
        log_xg_h = clamp.(μ_v .+ γ_h .+ α_h .+ β_a, -20.0, 20.0)
        log_xg_a = clamp.(μ_v .+        α_a .+ β_h, -20.0, 20.0)
        
        true_xg_h = exp.(log_xg_h) .+ 1e-6
        true_xg_a = exp.(log_xg_a) .+ 1e-6

        # 2. Final Goal Expectancy (xG * Kappa)
        λ_goals_h = κ_h .* true_xg_h
        λ_goals_a = κ_a .* true_xg_a

        results[mid] = (;
            λ_h = λ_goals_h,
            λ_a = λ_goals_a,
            r_h = disp_nt.h,  
            r_a = disp_nt.a,
            true_xg_h = true_xg_h, 
            true_xg_a = true_xg_a,
            σ_market = σ_mkt
        )
    end
    
    return results
end
