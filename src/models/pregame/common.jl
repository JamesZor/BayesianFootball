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
