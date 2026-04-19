# src/samplers/initialization/map.jl

using Optimization
using OptimizationOptimJL
using Optim
using DynamicPPL 

export MapInit, get_init_params

# --- 1. Configuration ---

"""
    MapInit
Initialize chains by first finding the MAP (Maximum a Posteriori) estimate,
then jittering the starting points around it.

# Fields
- `max_iters`: Maximum optimization steps (e.g., 50 for rough, 1000 for exact).
- `jitter`: Standard deviation of noise added to MAP for each chain.
"""
Base.@kwdef struct MapInit <: AbstractInitStrategy
    max_iters::Int = 50
end

function Base.show(io::IO, init::MapInit)
  print(io, "MapInit(max_iters=$(init.max_iters))")
end

# --- 2. Logic ---

function get_init_params(model, strategy::MapInit, n_chains::Int)
    println("  [Init] Strategy: Warm Start (MAP)")
    println("         - Running optimization (limit: $(strategy.max_iters) iters)...")

    # 1. Optimization Step
    # We use LBFGS for fast convergence
    map_estimate = optimize(
        model, 
        MAP(), 
        LBFGS(),
        Optim.Options(iterations=strategy.max_iters)
    )

  # println("         - Found $(map_estimate.params)")

  return fill(Turing.InitFromParams(map_estimate.params), n_chains)
end
