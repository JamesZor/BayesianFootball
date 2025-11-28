# src/samplers/samplers-module.jl
#
module Samplers

using Turing
using Optimization      # Required for MAP
using OptimizationOptimJL # Required for MAP
using ReverseDiff, Memoization


export AbstractSamplerConfig, NUTSConfig, ADVIConfig, MAPConfig, run_sampler

# --- Abstract Type for Training Methods ---
abstract type AbstractSamplerConfig end # Renamed from AbstractTrainingMethod


# --- Concrete MCMC Config (NUTS) ---
"""
    NUTSConfig(; n_samples=1000, n_chains=4, n_warmup=500)

Specifies sampling via the No-U-Turn Sampler (NUTS).
Arguments mirror the previous NUTSMethod.
"""
struct NUTSConfig <: AbstractSamplerConfig # Renamed from NUTSMethod
    n_samples::Int
    n_chains::Int
    n_warmup::Int
end
# Default constructor
NUTSConfig(; n_samples=1000, n_chains=4, n_warmup=500) = NUTSConfig(n_samples, n_chains, n_warmup) #

# --- Concrete Variational Inference Config ---
"""
    ADVIConfig(; n_iterations=20000)

Specifies approximate inference via ADVI.
"""
struct ADVIConfig <: AbstractSamplerConfig # Renamed from ADVIMethod
    n_iterations::Int
end
ADVIConfig(; n_iterations=20000) = ADVIConfig(n_iterations) #

# --- Concrete Optimization Config (MAP) ---
"""
    MAPConfig()

Specifies finding a point estimate via Maximum a Posteriori (MAP).
"""
struct MAPConfig <: AbstractSamplerConfig end # Renamed from MAPMethod


# --- Main `run_sampler` function, dispatched on the config type ---
# Renamed from 'train' to avoid conflict with the higher-level Training module

# 1. Run with NUTS
# function run_sampler(turing_model, config::NUTSConfig)
#     println("Sampling with NUTS ($(config.n_samples) samples, $(config.n_chains) chains)...")
#     # MCMCThreads() enables parallel sampling across chains within Turing
#     chain = sample(
#         turing_model, 
#         NUTS(config.n_warmup, 0.8), 
#         MCMCThreads(), 
#         config.n_samples, 
#         config.n_chains, #
#         progress=true,
#         adtype = AutoReverseDiff(; compile=true)
#     )
#     return chain
# end
#

function run_sampler(turing_model, config::NUTSConfig)
    println("Sampling with NUTS ($(config.n_samples) samples, $(config.n_chains) chains)...")
    
    # Create the initialization strategy
    # We create a vector of strategies, one for each chain.
    # range: [-0.001, 0.001] forces the Random Walk to start at 0 (Average Team Strength).
    init_strat = Turing.InitFromUniform(-0.001, 0.001)
    
    chain = sample(
        turing_model, 
        NUTS(config.n_warmup, 0.65, max_depth=8), 
        MCMCThreads(), 
        config.n_samples, 
        config.n_chains,
        progress = true,
        adtype = AutoReverseDiff(compile=true),
        
        # FIX: Use 'initial_params' with the strategy vector
        # This bypasses the need to know length(model)
        initial_params = fill(init_strat, config.n_chains)
    )
    return chain
end

# SGLD 
export SGLDConfig

# 1. Define the Config
struct SGLDConfig <: AbstractSamplerConfig
    step_size::Float64
    n_samples::Int
end
# Default to a small step size and MANY samples (Langevin needs more samples than NUTS)
SGLDConfig(; step_size=0.001, n_samples=20000) = SGLDConfig(step_size, n_samples)

# 2. Define the Runner
function run_sampler(turing_model, config::SGLDConfig)
    println("Sampling with SGLD (Langevin Dynamics)...")
    println("  - Step Size: $(config.step_size)")
    println("  - Samples: $(config.n_samples)")
    
    # SGLD does not need specific initialization logic as much as NUTS
    # because it is robust to noise.
    chain = sample(
        turing_model, 
        SGLD(config.step_size), 
        config.n_samples,
        progress=true
    )
    return chain
end



# 2. Run with ADVI
function run_sampler(turing_model, config::ADVIConfig)
    println("Optimizing with ADVI ($(config.n_iterations) iterations)...")
    result = vi(turing_model, ADVI(10, config.n_iterations)) 
    return result # Note: Returns VI result, not Chains
end

# 3. Run with MAP
function run_sampler(turing_model, config::MAPConfig)
    println("Optimizing to find MAP estimate...")
    result = optimize(turing_model, MAP(), LBFGS()) #
    return result # Note: Returns Optimization result, not Chains
end

# Add a catch-all for safety
function run_sampler(turing_model, config::AbstractSamplerConfig)
    error("run_sampler not implemented for sampler config type: $(typeof(config))")
end


end # module Samplers

