# dev/toy_refactor_step2.jl
using Turing, Distributions, DataFrames

# Include all required logic
include("../BayesianFootball/src/models/traits.jl")
include("../BayesianFootball/src/models/logic/dynamics.jl")
include("../BayesianFootball/src/models/logic/parameterization.jl")
include("../BayesianFootball/src/models/universal_poisson.jl")


# Dummy Data
n_teams = 4; n_rounds = 5; n_games = 50
home = rand(1:n_teams, n_games)
away = rand(1:n_teams, n_games)
times = rand(1:n_rounds, n_games)
g_h = rand(0:3, n_games)
g_a = rand(0:3, n_games)

println("--- Starting Step 2 Refactor Test ---")

# ==========================================
# MODEL 1: STATIC + STANDARD (Baseline)
# ==========================================
println("\n1. Testing Static + Standard...")
spec_1 = UniversalPoisson(
    dynamics=Static(), 
    param_scheme=StandardHomeAdvantage()
)
m1 = universal_poisson_model(n_teams, n_rounds, home, away, times, g_h, g_a, spec_1)
c1 = sample(m1, NUTS(), 20)
println("✅ Model 1 Sampled! (Look for scalar 'γ_center')")

# ==========================================
# MODEL 2: STATIC + HIERARCHICAL (The New Feature)
# ==========================================
println("\n2. Testing Static + Hierarchical HFA...")
spec_2 = UniversalPoisson(
    dynamics=Static(), 
    param_scheme=HierarchicalHomeAdvantage() # <--- THE CHANGE
)
m2 = universal_poisson_model(n_teams, n_rounds, home, away, times, g_h, g_a, spec_2)
c2 = sample(m2, NUTS(), 20)
println("✅ Model 2 Sampled! (Look for vector 'γ_raw')")

# ==========================================
# MODEL 3: GRW + HIERARCHICAL (Complex Mix)
# ==========================================
println("\n3. Testing GRW + Hierarchical HFA...")
spec_3 = UniversalPoisson(
    dynamics=GRW(), 
    param_scheme=HierarchicalHomeAdvantage()
)
m3 = universal_poisson_model(n_teams, n_rounds, home, away, times, g_h, g_a, spec_3)
c3 = sample(m3, NUTS(), 20)
println("✅ Model 3 Sampled! (Complex Matrix Dynamics + Vector HFA)")

println("\n🎉 SUCCESS: One model file is now handling 3 completely different mathematical structures.")
