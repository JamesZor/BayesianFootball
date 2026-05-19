# current_development/player_tracking_systems/src/results_handler.jl

using DataFrames

"""
    config_to_row(config, metrics)

Uses reflection to dynamically unpack struct parameters into a dictionary.
"""
function config_to_row(config::AbstractRatingTracker, metrics::TrackerMetrics)
    row = Dict{Symbol, Any}()
    
    # 1. Tracker Type
    row[:tracker_type] = string(typeof(config))
    
    # 2. Unpack parameters via reflection and create a summary string
    params_parts = String[]
    for field in propertynames(config)
        val = getproperty(config, field)
        row[field] = val
        push!(params_parts, "$(string(field))=$(val)")
    end
    row[:parameters] = join(params_parts, ", ")
    
    # 3. Add metrics
    row[:log_loss] = metrics.log_loss
    row[:edge_coef] = metrics.glm_edge_coef
    row[:edge_pvalue] = metrics.glm_edge_pvalue
    
    return row
end

"""
    compile_results(results::Vector{Tuple{AbstractRatingTracker, TrackerMetrics}})

Compiles results into a sorted DataFrame.
"""
function compile_results(results::Vector{Tuple{T, TrackerMetrics}} where T <: AbstractRatingTracker)
    rows = [config_to_row(r[1], r[2]) for r in results]
    df = DataFrame(rows)
    
    # Sort by LogLoss (lower is better)
    sort!(df, :log_loss)
    
    return df
end
