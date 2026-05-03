# src/Models/PreGame/components/dispersion.jl
# TODO: Add monthly R (config, model, extract)

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
