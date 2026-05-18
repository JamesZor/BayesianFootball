# src/Models/PreGame/components/dispersion.jl

# ==========================================
# 1. CONFIGURATIONS (The Interfaces)
# ==========================================
Base.@kwdef struct GlobalDispersion <: AbstractDispersionConfig
    log_r::ContinuousUnivariateDistribution = Normal(3.1, 0.4)
end

Base.@kwdef struct HomeAwayDispersion <: AbstractDispersionConfig
    log_r::ContinuousUnivariateDistribution = Normal(3.1, 0.4)      
    δ_r_home::ContinuousUnivariateDistribution = Normal(0.0, 0.5)   
end

# ADVANCED VOLATILITY DISPERSION (For LoI)
Base.@kwdef struct AdvancedVolatilityDispersion <: AbstractDispersionConfig
    log_r_base::ContinuousUnivariateDistribution = Normal(3.1, 0.4)
    δ_r_home::ContinuousUnivariateDistribution = Normal(0.0, 0.5) 
    σ_r_team::ContinuousUnivariateDistribution = truncated(Normal(0, 0.2), lower=0.0) 
    σ_r_month::ContinuousUnivariateDistribution = truncated(Normal(0, 0.2), lower=0.0)
end

# ==========================================
# 2. TURING SUBMODELS (The Math)
# ==========================================
@model function build_dispersion(config::GlobalDispersion)
    log_r ~ config.log_r
    r = exp(clamp(log_r, -10.0, 10.0))
    return (; h = r, a = r) 
end

@model function build_dispersion(config::HomeAwayDispersion)
    log_r ~ config.log_r
    δ_r_home ~ config.δ_r_home
    
    r_a = exp(clamp(log_r, -10.0, 10.0))
    r_h = exp(clamp(log_r + δ_r_home, -10.0, 10.0))
    return (; h = r_h, a = r_a) 
end

@model function build_dispersion(config::AdvancedVolatilityDispersion, n_teams::Int, n_months::Int)
    log_r_base ~ config.log_r_base
    δ_r_home ~ config.δ_r_home
    σ_r_team ~ config.σ_r_team
    σ_r_month ~ config.σ_r_month
    
    r_team_raw ~ filldist(Normal(0, 1), n_teams)
    r_month_raw ~ filldist(Normal(0, 1), n_months)
    
    r_team_vol = r_team_raw .* σ_r_team
    r_month_vol = r_month_raw .* σ_r_month
    
    return (; base = log_r_base, home_offset = δ_r_home, team_vol = r_team_vol, month_vol = r_month_vol) 
end

# Overloads to allow uniform calling from engines
@model function build_dispersion(config::GlobalDispersion, n_teams::Int, n_months::Int)
    return to_submodel(build_dispersion(config))
end

@model function build_dispersion(config::HomeAwayDispersion, n_teams::Int, n_months::Int)
    return to_submodel(build_dispersion(config))
end

# ==========================================
# 3. EXTRACTORS (The Data Pipeline)
# ==========================================
function extract_dispersion(chain, ::GlobalDispersion)
    r_val = exp.(vec(Array(chain[Symbol("disp.log_r")])))
    return (; h = r_val, a = r_val)
end

function extract_dispersion(chain, ::HomeAwayDispersion)
    log_r = vec(Array(chain[Symbol("disp.log_r")]))
    delta_r = vec(Array(chain[Symbol("disp.δ_r_home")]))
    
    r_a = exp.(log_r)
    r_h = exp.(log_r .+ delta_r)
    return (; h = r_h, a = r_a)
end

function extract_dispersion(chain, ::AdvancedVolatilityDispersion, n_teams::Int, n_months::Int)
    base = vec(Array(chain[Symbol("disp.log_r_base")]))
    home_offset = vec(Array(chain[Symbol("disp.δ_r_home")]))
    sigma_team = vec(Array(chain[Symbol("disp.σ_r_team")]))
    sigma_month = vec(Array(chain[Symbol("disp.σ_r_month")]))
    
    n_samples = length(base)
    vol_matrix_team = zeros(n_samples, n_teams)
    vol_matrix_month = zeros(n_samples, n_months)
    
    for i in 1:n_teams
        team_raw = vec(Array(chain[Symbol("disp.r_team_raw[$i]")]))
        vol_matrix_team[:, i] = team_raw .* sigma_team
    end

    for m in 1:n_months
        month_raw = vec(Array(chain[Symbol("disp.r_month_raw[$m]")]))
        vol_matrix_month[:, m] = month_raw .* sigma_month
    end
    
    return (; base = base, home_offset = home_offset, team_vol = vol_matrix_team, month_vol = vol_matrix_month)
end

# Overloads for extractors
function extract_dispersion(chain, config::GlobalDispersion, n_teams::Int, n_months::Int)
    return extract_dispersion(chain, config)
end

function extract_dispersion(chain, config::HomeAwayDispersion, n_teams::Int, n_months::Int)
    return extract_dispersion(chain, config)
end

# ==========================================
# 4. RECONSTRUCTION (Match-specific logic)
# ==========================================
"""
    reconstruct_dispersion(disp_nt::NamedTuple, h_id::Int, a_id::Int, month_idx::Int)

Calculates the final r_h and r_a vectors for a specific match given the extracted parameters.
Handles both simple (Global/HomeAway) and Advanced configurations.
"""
function reconstruct_dispersion(disp_nt::NamedTuple, h_id::Int, a_id::Int, month_idx::Int)
    # Check if we have the advanced components
    if hasproperty(disp_nt, :team_vol)
        n_samples = length(disp_nt.base)
        
        # 1. Team Volatility (Fallback to 0 if team not found)
        v_h = h_id > 0 ? disp_nt.team_vol[:, h_id] : zeros(n_samples)
        v_a = a_id > 0 ? disp_nt.team_vol[:, a_id] : zeros(n_samples)
        
        # 2. Monthly Volatility (1-12)
        v_m = (month_idx >= 1 && month_idx <= size(disp_nt.month_vol, 2)) ? 
              disp_nt.month_vol[:, month_idx] : zeros(n_samples)
        
        # 3. Sum and exponentiate
        log_r_h = disp_nt.base .+ disp_nt.home_offset .+ v_h .+ v_a .+ v_m
        log_r_a = disp_nt.base .+ v_h .+ v_a .+ v_m
        
        r_h = exp.(clamp.(log_r_h, -10.0, 10.0))
        r_a = exp.(clamp.(log_r_a, -10.0, 10.0))
        
        return (; h = r_h, a = r_a)
    else
        # Standard Global or HomeAway logic: already pre-calculated in extractor
        return (; h = disp_nt.h, a = disp_nt.a)
    end
end
