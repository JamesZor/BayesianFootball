"""
    compute_stake(signal::AbstractSignal, distribution::AbstractVector, odds::Number)::Float64

Calculates the percentage of bankroll to stake (0.0 to 1.0) given a posterior distribution 
and offered odds.
"""
function compute_stake(signal::AbstractSignal, distribution::AbstractVector, odds::Number)
    error("compute_stake not implemented for $(typeof(signal))")
end

"""
    signal_name(signal::AbstractSignal)::String

Returns a unique identifier for the signal strategy (e.g., "KellyCriterion").
"""
function signal_name(signal::AbstractSignal)::String
    return string(nameof(typeof(signal)))
end

"""
    signal_parameters(signal::AbstractSignal)::String

Returns a string representation of the hyperparameters (e.g., "fraction=0.5").
Useful for grouping DataFrames later.
"""
function signal_parameters(signal::AbstractSignal)::String
    fields = fieldnames(typeof(signal))
    if isempty(fields)
        return "none"
    end
    # Create a string like "fraction=0.5, cap=0.1"
    return join(["$f=$(getfield(signal, f))" for f in fields], ", ")
end

"""
    signal_description(signal::AbstractSignal)::String

Returns a short description of the mathematical logic used.
"""
function signal_description(signal::AbstractSignal)::String
    return "No description provided."
end

# --- TUI / Display ---

function Base.show(io::IO, s::AbstractSignal)
    # Compact inline print: KellyCriterion(fraction=0.5)
    print(io, "$(signal_name(s))($(signal_parameters(s)))")
end

function Base.show(io::IO, ::MIME"text/plain", s::AbstractSignal)
    # Multiline TUI print for detailed inspection
    print(io, "Signal Strategy: $(signal_name(s))\n")
    print(io, "├─ Parameters: $(signal_parameters(s))\n")
    print(io, "└─ Logic: $(signal_description(s))")
end
