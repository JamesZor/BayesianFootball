# current_development/copula_negbin/l00_frank_copula.jl

using Distributions
using Turing
using LogExpFunctions
using LinearAlgebra
using Dates
using BayesianFootball
using Random

# ==============================================================================
# 1. DISTRIBUTION: Frank Copula Negative Binomial
# ==============================================================================

struct FrankCopulaNegBin{T<:Real} <: DiscreteMultivariateDistribution
    r_h::T
    λ_h::T
    r_a::T
    λ_a::T
    κ::T
end

Base.length(::FrankCopulaNegBin) = 2

function frank_copula(u::T, v::T, κ::T) where {T<:Real}
    # Frank Copula: C(u, v) = -1/κ * log(1 + (exp(-κ*u) - 1)*(exp(-κ*v) - 1)/(exp(-κ) - 1))
    
    # AD-safe limit as κ -> 0
    if abs(κ) < 1e-5
        return u * v
    end
    
    # We use LogExpFunctions for stability
    num1 = expm1(-κ * u)
    num2 = expm1(-κ * v)
    den = expm1(-κ)
    
    # max(..., 1e-12) to ensure log argument is strictly > 0 for AD safety.
    inner = 1.0 + (num1 * num2) / den
    return -1.0 / κ * log(max(inner, 1e-12))
end

function Distributions.logpdf(d::FrankCopulaNegBin, y1::Int, y2::Int)
    if y1 < 0 || y2 < 0
        return -Inf
    end
    
    # We use RobustNegativeBinomial for stable CDF evaluations
    dist_h = RobustNegativeBinomial(d.r_h, d.λ_h)
    dist_a = RobustNegativeBinomial(d.r_a, d.λ_a)
    
    u1 = cdf(dist_h, y1)
    u0 = cdf(dist_h, y1 - 1)
    v1 = cdf(dist_a, y2)
    v0 = cdf(dist_a, y2 - 1)
    
    # Ensure probabilities are clipped to [0,1]
    u1 = clamp(u1, 0.0, 1.0)
    u0 = clamp(u0, 0.0, 1.0)
    v1 = clamp(v1, 0.0, 1.0)
    v0 = clamp(v0, 0.0, 1.0)
    
    κ = d.κ
    
    C11 = frank_copula(u1, v1, κ)
    C01 = frank_copula(u0, v1, κ)
    C10 = frank_copula(u1, v0, κ)
    C00 = frank_copula(u0, v0, κ)
    
    pmf = C11 - C01 - C10 + C00
    
    # Ensure PMF is strictly positive to avoid domain errors in log
    pmf_safe = max(pmf, 1e-12)
    return log(pmf_safe)
end

function Distributions.logpdf(d::FrankCopulaNegBin, y1::Real, y2::Real)
    return logpdf(d, Int(y1), Int(y2))
end

function Distributions._logpdf(d::FrankCopulaNegBin, x::AbstractVector{<:Real})
    return logpdf(d, x[1], x[2])
end

function Distributions.rand(rng::AbstractRNG, d::FrankCopulaNegBin)
    error("rand not yet implemented for FrankCopulaNegBin")
end


# ==============================================================================
# 2. SCORING MATRIX COMPUTATION
# ==============================================================================

function compute_score_matrix_discrete_copula(
    params; 
    max_goals::Int=12
)
    lh = params.loc_h
    la = params.loc_a
    rh = params.r_h
    ra = params.r_a
    kappa = params.κ
    
    n_samples = length(lh)
    S = zeros(Float64, max_goals, max_goals, n_samples)
    
    for k in 1:n_samples
        dist_h = RobustNegativeBinomial(rh[k], lh[k])
        dist_a = RobustNegativeBinomial(ra[k], la[k])
        
        κ_val = kappa[k]
        
        # Precompute marginal CDFs
        u = zeros(Float64, max_goals + 1)
        v = zeros(Float64, max_goals + 1)
        
        u[1] = 0.0 # CDF at -1
        v[1] = 0.0
        
        for g in 0:(max_goals-1)
            u[g+2] = cdf(dist_h, g)
            v[g+2] = cdf(dist_a, g)
        end
        
        for i in 1:max_goals
            for j in 1:max_goals
                u1 = u[i+1]
                u0 = u[i]
                v1 = v[j+1]
                v0 = v[j]
                
                C11 = frank_copula(u1, v1, κ_val)
                C01 = frank_copula(u0, v1, κ_val)
                C10 = frank_copula(u1, v0, κ_val)
                C00 = frank_copula(u0, v0, κ_val)
                
                pmf = C11 - C01 - C10 + C00
                S[i, j, k] = max(pmf, 0.0)
            end
        end
        
        sum_S = sum(S[:, :, k])
        if sum_S > 0
            S[:, :, k] ./= sum_S
        end
    end
    
    return S 
end


# ==============================================================================
# 3. TURING ENGINE
# ==============================================================================

# Time decay config for the copula engine
Base.@kwdef struct FrankCopulaTimeDecayDynamics <: BayesianFootball.Models.PreGame.AbstractDynamicsConfig
    days_half_life::Real = 180
    σ_att::ContinuousUnivariateDistribution = Gamma(2.0, 0.15)
    σ_def::ContinuousUnivariateDistribution = Gamma(2.0, 0.15)
    prior_κ::ContinuousUnivariateDistribution = Normal(0.0, 1.0)
end

### 
# ==========================================
# 2. TURING SUBMODEL
# ==========================================
@model function BayesianFootball.Models.PreGame.build_dynamics(config::FrankCopulaTimeDecayDynamics, n_teams::Int)
    # Global variance for attack and defense spread
    σ_a ~ config.σ_att
    σ_d ~ config.σ_def
    
    # Non-centered parameterization (the Z-scores)
    raw_a ~ filldist(Normal(0, 1), n_teams)
    raw_d ~ filldist(Normal(0, 1), n_teams)
    
    # Scale them
    α_scaled = raw_a .* σ_a
    β_scaled = raw_d .* σ_d
    
    # Zero-sum constraint (ensures league average is exactly 0)
    α = α_scaled .- mean(α_scaled)
    β = β_scaled .- mean(β_scaled)
    
    return (; α, β)
end

# ==========================================
# 3. EXTRACTOR
# ==========================================
function BayesianFootball.Models.PreGame.extract_dynamics(chain::Chains, ::FrankCopulaTimeDecayDynamics, prefix::String, n_teams::Int)
    n_samples = size(chain, 1) * size(chain, 3)
    
    # 1. Extract the global standard deviations
    σ_a = vec(Array(chain[Symbol("$prefix.σ_a")]))
    σ_d = vec(Array(chain[Symbol("$prefix.σ_d")]))
    
    # 2. Extract the raw Z-scores
    raw_a_matrix = zeros(n_samples, n_teams)
    raw_d_matrix = zeros(n_samples, n_teams)
    
    for i in 1:n_teams
        raw_a_matrix[:, i] = vec(Array(chain[Symbol("$prefix.raw_a[$i]")]))
        raw_d_matrix[:, i] = vec(Array(chain[Symbol("$prefix.raw_d[$i]")]))
    end
    
    # 3. Reconstruct the scaled parameters
    α_scaled = raw_a_matrix .* σ_a 
    β_scaled = raw_d_matrix .* σ_d
    
    # 4. Apply the Zero-Sum constraint
    α_matrix = α_scaled .- mean(α_scaled, dims=2)
    β_matrix = β_scaled .- mean(β_scaled, dims=2)
    
    return (; α = α_matrix, β = β_matrix)
end





# Wrapper Config for the master engine prototype
Base.@kwdef struct DynamicCopulaGoalsTimeDecayModel <: BayesianFootball.Models.PreGame.AbstractTimeDecayTeamModel
    interception_config::BayesianFootball.Models.PreGame.AbstractInterceptionConfig
    dynamics_config::BayesianFootball.Models.PreGame.AbstractDynamicsConfig
    dispersion_config::BayesianFootball.Models.PreGame.AbstractDispersionConfig
    homeadvantage_config::BayesianFootball.Models.PreGame.AbstractHomeAdvantageConfig
end

@model function build_weighted_copula_goals_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    time_indices::Vector{Int},
    month_indices::Vector{Int},
    home_goals::Vector{Int}, 
    away_goals::Vector{Int},
    match_weights::Vector{Float64},
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::DynamicCopulaGoalsTimeDecayModel 
)
    # 1. LOAD COMPONENTS
    inter ~ to_submodel(BayesianFootball.Models.PreGame.build_interception(config.interception_config, n_seasons))
    disp  ~ to_submodel(BayesianFootball.Models.PreGame.build_dispersion(config.dispersion_config, n_teams, n_months))
    ha    ~ to_submodel(BayesianFootball.Models.PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(BayesianFootball.Models.PreGame.build_dynamics(config.dynamics_config, n_teams))
    
    # Copula parameter
    κ_frank ~ config.dynamics_config.prior_κ

    # 2. VECTORIZED INDEXING 
    att_h = view(dyn.α, home_team_indices)
    def_h = view(dyn.β, home_team_indices)
    att_a = view(dyn.α, away_team_indices)
    def_a = view(dyn.β, away_team_indices)
    inter_match = view(inter, season_indices)
    home_adv = view(ha, home_team_indices)

    # --- Dispersion Construction ---
    if hasproperty(disp, :team_vol) 
        vol_h = view(disp.team_vol, home_team_indices)
        vol_a = view(disp.team_vol, away_team_indices)
        vol_m = view(disp.month_vol, month_indices)
        
        log_r_h = disp.base .+ disp.home_offset .+ vol_h .+ vol_a .+ vol_m
        log_r_a = disp.base .+ vol_h .+ vol_a .+ vol_m
        
        r_h_flat = exp.(clamp.(log_r_h, -10.0, 10.0))
        r_a_flat = exp.(clamp.(log_r_a, -10.0, 10.0))
    else 
        r_h_flat = fill(disp.h, length(home_team_indices))
        r_a_flat = fill(disp.a, length(home_team_indices))
        # Note: If disp.h is a scalar, we might need to handle it properly, 
        # but time_decay models usually have vector dispersions or broadcasting works.
    end

    # 3. VECTORIZED RATES (λ)
    λ_h = exp.(inter_match .+ home_adv .+ att_h .+ def_a)
    λ_a = exp.(inter_match .+             att_a .+ def_h)

    # 4. TIME-DECAYED COPULA LIKELIHOOD
    copula_dists = FrankCopulaNegBin.(r_h_flat, λ_h, r_a_flat, λ_a, κ_frank)
    
    log_lik_joint = logpdf.(copula_dists, home_goals, away_goals)
    
    Turing.@addlogprob! sum(log_lik_joint .* match_weights)
end

# ==============================================================================
# 4. INTERFACES FOR FEATURES & ENGINE
# ==============================================================================

function BayesianFootball.Features.required_features(model::DynamicCopulaGoalsTimeDecayModel)
    return BayesianFootball.Features.AbstractFeatureConfig[
        BayesianFootball.Features.TeamIDsFeature(), 
        BayesianFootball.Features.GoalsFeature(), 
        BayesianFootball.Features.DatesFeature(), 
        BayesianFootball.Features.MonthFeature(),
        BayesianFootball.Features.TimeIndicesFeature()
    ] 
end

function BayesianFootball.Models.PreGame.build_turing_model(model::DynamicCopulaGoalsTimeDecayModel, feature_set::BayesianFootball.Features.FeatureSet)
    data = feature_set.data
    
    n_teams    = Int(data[:n_teams])
    n_seasons  = Int(data[:n_seasons])
    n_months   = 12
    
    date_deltas = Vector{Int}(data[:dates])
    match_weights = BayesianFootball.Models.PreGame.calculate_match_weights(date_deltas, model.dynamics_config.days_half_life)
    
    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices])
    time_idxs  = Vector{Int}(data[:time_indices])
    month_indices = Vector{Int}(data[:flat_months])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    return build_weighted_copula_goals_engine(
        home_ids,
        away_ids,
        season_ids,
        time_idxs,
        month_indices,
        home_goals,
        away_goals,
        match_weights,
        n_teams,
        n_seasons,
        n_months,
        model
    )
end
