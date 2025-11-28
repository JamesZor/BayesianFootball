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
function run_sampler(turing_model, config::NUTSConfig)
    println("Sampling with NUTS ($(config.n_samples) samples, $(config.n_chains) chains)...")
    # MCMCThreads() enables parallel sampling across chains within Turing
    chain = sample(
        turing_model, 
        NUTS(config.n_warmup, 0.8), 
        MCMCThreads(), 
        config.n_samples, 
        config.n_chains, #
        progress=true,
        adtype = AutoReverseDiff(; compile=true),

          # --- THE FIX ---
        # This tells Turing: "Don't pick random numbers between -2 and 2."
        # "Pick random numbers between -0.001 and 0.001."
        # This ensures your cumsum starts near 0, preventing the 485-million-goal explosion.
        init_params = rand(length(turing_model)) .* 0.001

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

