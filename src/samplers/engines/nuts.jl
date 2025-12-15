# src/samplers/engines/nuts.jl

export NUTSConfig

# --- 1. Configuration ---

struct NUTSConfig <: AbstractSamplerConfig 
    n_samples::Int
    n_chains::Int
    n_warmup::Int
    accept_rate::Real 
    max_depth::Int
    initialisation::AbstractInitStrategy 
end

# Smart Constructor
function NUTSConfig(; 
    n_samples=1000, 
    n_chains=4, 
    n_warmup=500, 
    accept_rate=0.65,
    max_depth=10,
    # Helper to allow passing :map or :uniform symbol for convenience
    init_type=:uniform, 
    map_iters=50,
    jitter=0.001

)
    # Factory logic to create the correct strategy object
    strategy = if init_type == :map
        MapInit(max_iters=map_iters)
    else
        UniformInit()
    end

    return NUTSConfig(n_samples, n_chains, n_warmup, accept_rate, max_depth, strategy)
end

# --- 2. Display ---

function Base.show(io::IO, ::MIME"text/plain", c::NUTSConfig)
    printstyled(io, "NUTSConfig\n", color=:cyan, bold=true)
    println(io, "==========")
    println(io, "  Samples: $(c.n_samples)")
    println(io, "  Chains:  $(c.n_chains)")
    println(io, "  Accept Rate:  $(c.accept_rate)")
    println(io, "  Max Depth:  $(c.max_depth)")
    println(io, "  Init:    $(c.initialisation)") 
end

# --- 3. Execution ---

function run_sampler(turing_model, config::NUTSConfig)
    println("Sampling with NUTS...")
    
    # Delegate initialisation logic to the strategy
    init_params = get_init_params(turing_model, config.initialisation, config.n_chains)
    
    chain = sample(
        turing_model, 
        NUTS(config.n_warmup, config.accept_rate, max_depth=config.max_depth), 
        MCMCThreads(), 
        config.n_samples, 
        config.n_chains,
        progress = :perchain,
        adtype = AutoReverseDiff(compile=true),
        initial_params = init_params # Injection
    )
    return chain
end
