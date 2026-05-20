# src/models/pregame/PreGame.jl

module PreGame

# We import the Types to extend them, but we don't need Reexport
using ...TypesInterfaces

# Macro libraries MUST be loaded at the top
using Turing, Distributions, DataFrames
using ..MyDistributions 
using ..Features
using LinearAlgebra
using Statistics
using Dates
using MCMCChains

# 2. feature_set updates & Architecture
include("./types.jl")
include("./components/dispersion.jl")
include("./components/interception.jl")
include("./components/home_advantage.jl")
include("./components/dynamics.jl")
include("./components/kappa.jl")
include("./display.jl")

include("./engines/goals_engine.jl")
include("./engines/goals_time_decay_engine.jl")
include("./engines/goals_market_time_decay_engine.jl")
include("./engines/xg_time_decay_engine.jl")
include("./engines/xg_market_time_decay_engine.jl")
include("./engines/xg_engine.jl")
include("./engines/goals_market_engine.jl")
include("./engines/xg_market_engine.jl")
include("./engines/xg_market_player_engine.jl")
export DynamicGoalsModel, DynamicGoalsTimeDecayModel, DynamicMarketGoalsTimeDecayModel, DynamicXGModel, DynamicXGTimeDecayModel, DynamicMarketGoalsModel, DynamicMarketXGModel, DynamicMarketXGTimeDecayModel, DynamicMarketXGPlayerModel
export TimeDecayDynamics, PositionalPlayerDynamics

##

export build_turing_model, extract_parameters

end # module
