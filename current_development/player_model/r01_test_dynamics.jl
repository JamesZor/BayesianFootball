# current_development/player_model/r01_test_dynamics.jl

using Revise
using BayesianFootball
using Turing
using DataFrames
using Statistics

# Include our new component logic
include("l01_player_dynamics.jl")

# 1. Define a tiny test model to verify the component
@model function tiny_test_model(config::PositionalPlayerDynamics)
    # This mirrors how the engine will call it
    dyn ~ to_submodel(BayesianFootball.Models.PreGame.build_dynamics(config, 0))
    
    # Add a dummy observation to give the sampler something to do
    # Let's say we observe a high Forward rating contribution
    1.0 ~ Normal(dyn.w_F_att, 0.1)
end

println(">>> Testing PositionalPlayerDynamics component...")

config = PositionalPlayerDynamics()
model_inst = tiny_test_model(config)

# 2. Run a quick MCMC chain
chain = sample(model_inst, NUTS(200, 0.65), 200, progress=false)

println("\n✅ Chain generated successfully.")
println(chain)

# 3. Test the Extractor
println("\n>>> Testing Extractor...")
extracted = BayesianFootball.Models.PreGame.extract_dynamics(chain, config, "dyn", 0)

expected_keys = [
    :w_G_att, :w_D_att, :w_M_att, :w_F_att,
    :w_G_def, :w_D_def, :w_M_def, :w_F_def
]

all_found = true
for k in expected_keys
    if haskey(extracted, k)
        val = extracted[k]
        println(" - $k: mean=$(round(mean(val), digits=3)), std=$(round(std(val), digits=3))")
    else
        println(" ❌ Missing key: $k")
        global all_found = false
    end
end

if all_found
    println("\n✅ Extractor verification complete.")
else
    println("\n❌ Extractor verification failed.")
end
