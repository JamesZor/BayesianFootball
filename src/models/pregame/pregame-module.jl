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

# Team Level - Standard
include("./engines/team_level/standard/goals.jl")
include("./engines/team_level/standard/goals_market.jl")
include("./engines/team_level/standard/xg.jl")
include("./engines/team_level/standard/xg_market.jl")

# Team Level - Time Decay
include("./engines/team_level/time_decay/goals.jl")
include("./engines/team_level/time_decay/goals_market.jl")
include("./engines/team_level/time_decay/xg.jl")
include("./engines/team_level/time_decay/xg_market.jl")

# Player Level - Standard
include("./engines/player_level/standard/xg_market.jl")

# Player Level - Time Decay
include("./engines/player_level/time_decay/xg_market.jl")
include("./engines/player_level/time_decay/hierarchical_xg_market.jl")

export DynamicGoalsModel, DynamicGoalsTimeDecayModel, DynamicMarketGoalsTimeDecayModel, DynamicXGModel, DynamicXGTimeDecayModel, DynamicMarketGoalsModel, DynamicMarketXGModel, DynamicMarketXGTimeDecayModel, DynamicMarketXGPlayerModel, DynamicMarketXGPlayerTimeDecayModel, DynamicMarketXGHierarchicalPlayerTimeDecayModel
export TimeDecayDynamics, PositionalPlayerDynamics, HierarchicalPlayerDynamicsConfig
export TimeDecayDynamics, PositionalPlayerDynamics

##

export build_turing_model, extract_parameters

end # module
