# src/samplers/initialisation/uniform.jl

export UniformInit, get_init_params

# --- 1. Configuration ---

"""
    UniformInit(a=-0.001, b=0.001)
Initialize chains from a uniform distribution U(a, b).
"""
Base.@kwdef struct UniformInit <: AbstractInitStrategy
    a::Float64 = -0.001
    b::Float64 =  0.001
end

function Base.show(io::IO, init::UniformInit)
    print(io, "UniformInit(range=[$(init.a), $(init.b)])")
end

# --- 2. Logic ---

function get_init_params(model, strategy::UniformInit, n_chains::Int)
    println("  [Init] Strategy: Uniform($(strategy.a), $(strategy.b))")
    # Turing expects a specific Init object for uniform starts
    init_dist = Turing.InitFromUniform(strategy.a, strategy.b)
    return fill(init_dist, n_chains)
end
