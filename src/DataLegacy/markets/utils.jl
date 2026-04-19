# src/data/markets/utils.jl


"""
    _frac_to_dec(s::String)
Converts fractional odds strings ("1/2", "evens") to decimal floats.
"""
function _frac_to_dec(s::AbstractString)
    s_clean = strip(lowercase(s))

    
    if occursin("/", s_clean)
        parts = split(s_clean, "/")
        if length(parts) == 2
            num = tryparse(Float64, parts[1])
            den = tryparse(Float64, parts[2])
            if !isnothing(num) && !isnothing(den) && den != 0
                return 1.0 + (num / den)
            end
        end
    end
    
    # Fallback: try parsing as direct decimal or return NaN
    return something(tryparse(Float64, s_clean), NaN)
end

"""
    _safe_equals(val, target::Float64)
Safely compares a DataFrame value (which might be String or Int) to a Float target.
"""
function _safe_equals(val, target::Float64)
    val isa Number && return val == target
    
    # Try parsing string to float
    parsed = tryparse(Float64, string(val))
    return !isnothing(parsed) && parsed == target
end
