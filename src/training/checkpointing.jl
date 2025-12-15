# src/training/checkpointing.jl

using Serialization
using Dates

export save_split_checkpoint, load_split_checkpoint, get_checkpoint_status, cleanup_checkpoints

"""
    get_checkpoint_path(dir, index)
Standardizes filename generation: `split_001.jls`, `split_002.jls`.
"""
function get_checkpoint_path(dir::String, index::Int)
    filename = "split_$(lpad(index, 3, '0')).jls"
    return joinpath(dir, filename)
end

"""
    save_split_checkpoint(dir, index, data)
Atomically saves a single split result.
"""
function save_split_checkpoint(dir::String, index::Int, data::Any)
    mkpath(dir) # Ensure directory exists
    path = get_checkpoint_path(dir, index)
    
    # Write to temp file then move to ensure atomicity (prevents partial writes on crash)
    temp_path = path * ".tmp"
    serialize(temp_path, data)
    mv(temp_path, path; force=true)
    
    # Optional: Log implies I/O, keeping it minimal here
end

"""
    load_split_checkpoint(dir, index)
Returns (true, data) if exists, or (false, nothing) if missing.
"""
function load_split_checkpoint(dir::String, index::Int)
    path = get_checkpoint_path(dir, index)
    if isfile(path)
        try
            return true, deserialize(path)
        catch e
            @warn "Failed to deserialize checkpoint $path: $e"
            return false, nothing
        end
    end
    return false, nothing
end

"""
    get_checkpoint_status(dir, total_splits)
Scans the directory to see which splits are already finished.
Returns a Vector{Any} where entries are `nothing` (missing) or the loaded result.
"""
function get_checkpoint_status(dir::Union{String, Nothing}, total_splits::Int)
    results = Vector{Any}(undef, total_splits)
    fill!(results, nothing)
    
    if isnothing(dir) || !isdir(dir)
        return results
    end
    
    println("Checking for existing checkpoints in: $dir")
    found_count = 0
    
    for i in 1:total_splits
        exists, data = load_split_checkpoint(dir, i)
        if exists
            results[i] = data
            found_count += 1
        end
    end
    
    if found_count > 0
        println("   Found $found_count / $total_splits completed splits. Resuming...")
    end
    
    return results
end

"""
    cleanup_checkpoints(dir, total_splits)
Removes the files after a successful run to save space.
"""
function cleanup_checkpoints(dir::String, total_splits::Int)
    println("Cleaning up temporary checkpoints in $dir...")
    for i in 1:total_splits
        path = get_checkpoint_path(dir, i)
        rm(path; force=true)
    end
    # Try to remove the dir if empty
    try
        rm(dir) 
    catch
        # Directory might not be empty, ignore
    end
end
