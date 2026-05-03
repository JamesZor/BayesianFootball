# src/Models/PreGame/types.jl

# ==========================================
# 1. ABSTRACT COMPONENT INTERFACES
# ==========================================
abstract type AbstractDispersionConfig end
abstract type AbstractHomeAdvantageConfig end
abstract type AbstractKappaConfig end
abstract type AbstractDynamicsConfig end
abstract type AbstractInterceptionConfig end

# ==========================================
# 2. MASTER ARCHITECTURE TYPES
# ==========================================
# By inheriting from your existing abstract type, the prediction engine 
# automatically knows how to handle this model during inference!
# abstract type AbstractDynamicFootballModel <: AbstractXGNegativeBinomial end

# AbstractMultiScaledNegBinModel  --- This is just the goals model
# AbstractNegBinModel  ---- a more abstract version
# AbstractXGNegativeBinomial      --- This is the xg + goal model ( kappa ) 

Base.@kwdef struct DynamicGoalsModel{
  I<:AbstractInterceptionConfig,
  T<:AbstractDynamicsConfig, 
  D<:AbstractDispersionConfig, 
  H<:AbstractHomeAdvantageConfig
  } <: AbstractNegBinModel
    interception_config::I
    dynamics_config::T
    dispersion_config::D
    homeadvantage_config::H
end

Base.@kwdef struct DynamicXGModel{
  I<:AbstractInterceptionConfig,
  T<:AbstractDynamicsConfig, 
  D<:AbstractDispersionConfig, 
  H<:AbstractHomeAdvantageConfig,
  K<:AbstractKappaConfig
} <: AbstractNegBinModel
    interception_config::I
    dynamics_config::T
    dispersion_config::D
    homeadvantage_config::H
    kappa_config::K
    ν_xg::Distribution = truncated(Normal(3.0, 0.5), lower=0.5) 
end

