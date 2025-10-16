module Sampling

using Turing
using Optimization      # Required for MAP
using OptimizationOptimJL # Required for MAP

export AbstractTrainingMethod, NUTSMethod, ADVIMethod, MAPMethod, train

# --- Abstract Type for Training Methods ---
abstract type AbstractTrainingMethod end

# --- Concrete MCMC Method (NUTS) ---
"""
    NUTSMethod(; n_samples=1000, n_chains=4, n_warmup=500)

Specifies training via the No-U-Turn Sampler (NUTS), a state-of-the-art MCMC algorithm.

# Arguments
- `n_samples`: The number of posterior samples to generate per chain.
- `n_chains`: The number of independent chains to run in parallel.
- `n_warmup`: The number of warmup (adaptation) steps for each chain.
"""
struct NUTSMethod <: AbstractTrainingMethod
    n_samples::Int
    n_chains::Int
    n_warmup::Int
end
# Default constructor with keyword arguments for ease of use
NUTSMethod(; n_samples=1000, n_chains=4, n_warmup=500) = NUTSMethod(n_samples, n_chains, n_warmup)


# --- Concrete Variational Inference Method ---
"""
    ADVIMethod(; n_iterations=10000)

Specifies training via Automatic Differentiation Variational Inference (ADVI),
a fast, approximate inference method.
"""
struct ADVIMethod <: AbstractTrainingMethod
    n_iterations::Int
end
ADVIMethod(; n_iterations=20000) = ADVIMethod(n_iterations)


# --- Concrete Optimization Method (MAP) ---
"""
    MAPMethod()

Specifies finding a point estimate via Maximum a Posteriori (MAP) optimization.
This finds the single most likely set of parameters (the mode of the posterior).
"""
struct MAPMethod <: AbstractTrainingMethod end


# --- Main `train` function, dispatched on the method type ---

# 1. Train with NUTS
function train(turing_model, method::NUTSMethod)
    
    println("Sampling with NUTS ($(method.n_samples) samples, $(method.n_chains) chains)...")
    # MCMCThreads() enables parallel sampling across chains
    chain = sample(
        turing_model, 
        NUTS(method.n_warmup, 0.8), # Target acceptance rate of 0.8
        MCMCThreads(), 
        method.n_samples, 
        method.n_chains,
        progress=true,
        adtype = AutoReverseDiff(; compile=true)
    )
    return chain
end

# 2. Train with ADVI
function train(turing_model,  method::ADVIMethod)

    println("Optimizing with ADVI ($(method.n_iterations) iterations)...")
    # ADVI finds a variational distribution that approximates the posterior
    result = vi(turing_model, ADVI(10, method.n_iterations)) # 10 samples for ELBO estimate
    return result
end

# 3. Train with MAP
function train(turing_model, method::MAPMethod)
    println("Optimizing to find MAP estimate...")
    # optimize finds the mode of the posterior distribution
    result = optimize(turing_model, MAP(), LBFGS())
    return result
end

end
