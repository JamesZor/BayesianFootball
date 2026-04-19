# dev/toy_refactor.jl
using Turing, Distributions, DataFrames

# 1. Include the new files (Adjust paths if necessary)
include("../BayesianFootball/src/models/traits.jl")
include("../BayesianFootball/src/models/logic/dynamics.jl")
include("../BayesianFootball/src/models/universal_poisson.jl")

# 2. Create Dummy Data
n_teams = 4
n_rounds = 5
n_games = 20

home = rand(1:n_teams, n_games)
away = rand(1:n_teams, n_games)
times = rand(1:n_rounds, n_games)
g_h = rand(0:3, n_games)
g_a = rand(0:3, n_games)

println("--- Starting Toy Refactor Test ---")

# ==========================================
# TEST 1: STATIC MODEL
# ==========================================
println("\n1. Testing STATIC Model...")

# Define Spec: Static Dynamics
static_spec = UniversalPoisson(
    dynamics=Static(), 
    param_scheme=StandardHomeAdvantage()
)

# Compile
m_static = universal_poisson_model(
    n_teams, n_rounds, home, away, times, g_h, g_a, static_spec
)

# Sample (Short chain just to check errors)
chain_static = sample(m_static, NUTS(), 50)
println("✅ Static Model Sampled Successfully!")


# ==========================================
# TEST 2: GRW MODEL
# ==========================================
println("\n2. Testing GRW Model...")

# Define Spec: GRW Dynamics
grw_spec = UniversalPoisson(
    dynamics=GRW(), 
    param_scheme=StandardHomeAdvantage()
)

# Compile (Same function, different spec!)
m_grw = universal_poisson_model(
    n_teams, n_rounds, home, away, times, g_h, g_a, grw_spec
)

# Sample
chain_grw = sample(m_grw, NUTS(), 50)
println("✅ GRW Model Sampled Successfully!")

println("\n🎉 REFACTOR SUCCESS: One model file handled both Static and Dynamic cases.")


describe(chain_grw)
describe(chain_static)

