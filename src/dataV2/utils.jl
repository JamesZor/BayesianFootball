# src/data/utils.jl

"""
    _frac_to_dec(s::String)
Converts fractional odds strings ("1/2", "evens", "19/10") to decimal floats.
Returns NaN if parsing fails.
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
Safely compares a value (which might be a String or Int from the DataFrame) 
to a Float64 target.
"""
function _safe_equals(val, target::Float64)
    val isa Number && return val == target
    
    # Try parsing string to float
    parsed = tryparse(Float64, string(val))
    return !isnothing(parsed) && parsed == target
end


"""
    apply_schema!(df::DataFrame, schema::Dict)
Iterates through the provided schema dictionary. If the column exists in the DataFrame, 
it converts the column to the specified type (allowing for missing values).
"""
function apply_schema!(df::DataFrame, schema::Dict)
    for (col_name, target_type) in schema
        col_str = string(col_name) # DataFrame names are strings, schema keys are Symbols
        
        if hasproperty(df, col_str)
            # We use Union{Missing, target_type} because SQL pulls often contain missings
            expected_type = Union{Missing, target_type}
            
            # Only convert if it doesn't already match to save compute time
            if eltype(df[!, col_str]) != expected_type
                try
                    df[!, col_str] = convert.(expected_type, df[!, col_str])
                catch e
                    @warn "Schema conversion failed for column $col_str to $target_type."
                end
            end
        end
    end
    return df
end
