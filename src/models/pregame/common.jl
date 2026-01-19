# src/models/pregame/common.jl

using Turing
using DataFrames
using Base.Threads
using ...TypesInterfaces: FeatureSet, Vocabulary, AbstractPregameModel

# Import the abstract model families to dispatch on
using ...TypesInterfaces: AbstractPoissonModel, AbstractDixonColesModel

export prepare_model_data, extract_parameters, unpack_results


# ---  Type Definitions (The Contract) ---

"""
    PoissonRates
Standard return type for Poisson-based models (Static, Dynamic, etc.).
Returns vectors of posterior samples for Home and Away lambda.
"""
const PoissonRates = NamedTuple{(:λ_h, :λ_a), Tuple{Vector{Float64}, Vector{Float64}}}

"""
    DixonColesRates
Standard return type for Dixon-Coles models.
Includes the correlation coefficient ρ (rho).
"""
const DixonColesRates = NamedTuple{(:λ_h, :λ_a, :ρ), Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}

# --- 2. The Trait Wrapper (Function Overloading) ---

"""
    get_prediction_type(model)
Returns the concrete NamedTuple type used for the model's predictions.
"""
get_prediction_type(::AbstractPoissonModel) = PoissonRates
get_prediction_type(::AbstractDixonColesModel) = DixonColesRates

# Fallback for models not yet categorized
get_prediction_type(::AbstractPregameModel) = Dict{Symbol, Vector{Float64}}

# --- The Orchestrator (Vector Method) ---

function extract_parameters(
    model::AbstractPregameModel, 
    dfs_to_predict::AbstractDataFrame, 
    features::Tuple,
    results::Tuple
)
  return extract_parameters(model, dfs_to_predict, features[1], results[1] )

end

"""
    extract_parameters(model, dfs::Vector, vocab, results::Vector)

Iterates over multiple data splits (folds) and merges the results.
Dispatches to the single-dataframe method for each split.
"""
function extract_parameters(
    model::AbstractPregameModel, 
    dfs_to_predict::AbstractVector, 
    vocabulary::Vocabulary,
    results_vector::AbstractVector
)
    # Determine type from the first result
    ValueType = get_prediction_type(model)
    
    n_splits = length(dfs_to_predict)
    partial_results = Vector{Dict{Int64, ValueType}}(undef, n_splits)

    # Parallel Execution
    Threads.@threads for i in 1:n_splits
        if i <= length(results_vector)
            result_item = results_vector[i]
            chain = result_item isa Tuple ? result_item[1] : result_item
            
            # CALLS THE WORKER METHOD (Defined in specific model files)
            partial_results[i] = extract_parameters(model, dfs_to_predict[i], vocabulary, chain)
        end
    end

    # Efficient Merge
    total_rows = sum(nrow, dfs_to_predict)
    full_dict = Dict{Int64, ValueType}()
    sizehint!(full_dict, total_rows)

    for res in partial_results
        if isassigned(partial_results, indexin([res], partial_results)[1]) 
            merge!(full_dict, res)
        end
    end

    return full_dict
end

# --- 3. Utility: Unpack NamedTuples ---

function unpack_results(results::Vector{T}) where T <: NamedTuple
    if isempty(results); return (;); end
    keys_ = keys(results[1])
    vals = ntuple(i -> [getfield(r, keys_[i]) for r in results], length(keys_))
    return NamedTuple{keys_}(vals)
end


# --- 4. GRW Shared Utilities ---

"""
    reconstruct_states(chains, n_teams, n_rounds)

Internal helper to reconstruct 'att' and 'def' matrices (Teams x Rounds x Samples)
from the MCMC chains using the hierarchical logic (NCP).
Shared by GRWPoisson and GRWDixonColes.
"""
function reconstruct_states(chain::Chains, n_teams::Int, n_rounds::Int)
    # 1. Extract Scalars (Samples)
    σ_att_vec   = vec(chain[:σ_att])
    σ_def_vec   = vec(chain[:σ_def])
    σ_att_0_vec = vec(chain[:σ_att_0])
    σ_def_0_vec = vec(chain[:σ_def_0])

    # 2. Extract Arrays (Samples x Dimensions)
    z_att_init_raw = Array(group(chain, :z_att_init))
    z_def_init_raw = Array(group(chain, :z_def_init))
    
    n_samples = size(z_att_init_raw, 1)

    # Reshape Init: [Team, Time=1, Sample]
    Z_att_init = permutedims(reshape(z_att_init_raw, n_samples, n_teams, 1), (2, 3, 1))
    Z_def_init = permutedims(reshape(z_def_init_raw, n_samples, n_teams, 1), (2, 3, 1))

    # Reshape Steps: [Team, Time=Steps, Sample]
    z_att_steps_raw = Array(group(chain, :z_att_steps))
    z_def_steps_raw = Array(group(chain, :z_def_steps))
    
    Z_att_steps = permutedims(reshape(z_att_steps_raw, n_samples, n_teams, n_rounds-1), (2, 3, 1))
    Z_def_steps = permutedims(reshape(z_def_steps_raw, n_samples, n_teams, n_rounds-1), (2, 3, 1))

    # 3. Reshape Scalars for Broadcasting
    S_att   = reshape(σ_att_vec, 1, 1, n_samples)
    S_def   = reshape(σ_def_vec, 1, 1, n_samples)
    S_att_0 = reshape(σ_att_0_vec, 1, 1, n_samples)
    S_def_0 = reshape(σ_def_0_vec, 1, 1, n_samples)

    # 4. Reconstruction
    scaled_init_att = Z_att_init .* S_att_0
    scaled_init_def = Z_def_init .* S_def_0
    
    scaled_steps_att = Z_att_steps .* S_att
    scaled_steps_def = Z_def_steps .* S_def

    # Integrate
    raw_att = cumsum(cat(scaled_init_att, scaled_steps_att, dims=2), dims=2)
    raw_def = cumsum(cat(scaled_init_def, scaled_steps_def, dims=2), dims=2)

    # Center & Shift
    final_att = (raw_att .- mean(raw_att, dims=1)) 
    final_def = (raw_def .- mean(raw_def, dims=1))

    return final_att, final_def
end

"""
OPTIMIZED HELPER: unwraps NTuple directly into target shape
Avoids hcat and permutedims allocations.
"""
function unwrap_ntuple(tuple_of_arrays)
    # 1. Determine Dimensions
    # tuple_of_arrays is (AxisArray_1, AxisArray_2, ...)
    n_features = length(tuple_of_arrays)
    
    # Peek at the first element to get sample count (length of the array)
    n_samples = length(tuple_of_arrays[1])
    
    # 2. Pre-allocate the FINAL Matrix [Features, Samples]
    # We want Float64, assuming that's what comes out of Turing
    out = Matrix{Float64}(undef, n_features, n_samples)
    
    # 3. Fill directly (No temporary arrays)
    for (i, arr) in enumerate(tuple_of_arrays)
        out[i, :] .= vec(parent(arr))
    end
    
    return out
end

