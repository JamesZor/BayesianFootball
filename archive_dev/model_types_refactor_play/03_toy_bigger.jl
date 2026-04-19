# dev/toy_full_refactor.jl
using Turing, Distributions, DataFrames

# Include all required logic
include("../BayesianFootball/src/models/traits.jl")
include("../BayesianFootball/src/models/logic/dynamics.jl")
include("../BayesianFootball/src/models/logic/parameterization.jl")
include("../BayesianFootball/src/models/logic/observation.jl")
include("../BayesianFootball/src/models/football_model.jl")


# Dummy Data
n_teams = 4; n_rounds = 5; n_games = 50
home = rand(1:n_teams, n_games)
away = rand(1:n_teams, n_games)
times = rand(1:n_rounds, n_games)
g_h = rand(0:3, n_games)
g_a = rand(0:3, n_games)

println("--- Starting Step 2 Refactor Test ---")




# 1. Define a complex model simply
my_spec = FootballModel(
    GRW(σ_step_prior = Truncated(Normal(0, 0.1), 0, Inf)), # Custom Prior!
    HierarchicalHomeAdvantage(),
    NegBinObservation()
)


my_spec = FootballModel(
    Static(),
    HierarchicalHomeAdvantage(),
    PoissonObservation()
)


# 2. Run
model = football_model(n_teams, n_rounds, home, away, times, g_h, g_a, my_spec)

sample(model, NUTS(), 50)



# Test 1: Static + Hierarchical + Poisson
spec1 = FootballModel(Static(), HierarchicalHomeAdvantage(), PoissonObservation())
m1 = football_model(n_teams, n_rounds, home, away, times, g_h, g_a, spec1)
sample(m1, NUTS(), 50)

# Test 2: GRW + Hierarchical + NegBin (The one that crashed)
spec2 = FootballModel(
    GRW(σ_step_prior = Truncated(Normal(0, 0.1), 0, Inf)), 
    HierarchicalHomeAdvantage(), 
    NegBinObservation()
)
m2 = football_model(n_teams, n_rounds, home, away, times, g_h, g_a, spec2)
sample(m2, NUTS(), 50)
