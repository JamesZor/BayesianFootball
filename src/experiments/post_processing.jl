# src/experiments/post_process.jl

using DataFrames
using ProgressMeter
using Base.Threads
using ..Data
using ..Features
using ..Models
using DataFrames
using ..TypesInterfaces: AbstractFootballModel

# We don't export here, we rely on the main module to export

# ==============================================================================
# 1. THE WRAPPER (Holy Trait Pattern)
# ==============================================================================

struct LatentStates
    df::DataFrame
    model::AbstractFootballModel # You can now store the specific model used
end


# 1. Standard Base methods
Base.getindex(ls::LatentStates, args...) = getindex(ls.df, args...)
Base.setindex!(ls::LatentStates, val, args...) = setindex!(ls.df, val, args...)
Base.size(ls::LatentStates) = size(ls.df)
Base.size(ls::LatentStates, i) = size(ls.df, i)
Base.show(io::IO, ls::LatentStates) = show(io, ls.df)

# 2. DataFrames specific methods (Use DataFrames.nrow, not Base.nrow)
DataFrames.nrow(ls::LatentStates) = nrow(ls.df)
DataFrames.ncol(ls::LatentStates) = ncol(ls.df)

# ==============================================================================
# 2. THE BRIDGE
# ==============================================================================

function extract_oos_predictions(ds::Data.DataStore, exp_results::ExperimentResults)
    # ... (Same logic as before, just no module wrapper) ...
    # 1. Reconstruct Context
    splits = Data.create_data_splits(ds, exp_results.config.splitter)
    feature_sets = Features.create_features(splits, exp_results.config.model, exp_results.config.splitter)
    results = exp_results.training_results
    n_splits = length(results)

    # 2. Extract
    split_dfs = Vector{DataFrame}(undef, n_splits)
    @showprogress for i in 1:n_splits
        split_dfs[i] = _process_split(ds, exp_results.config.model, exp_results.config.splitter, feature_sets[i], results[i])
    end

    # 3. Consolidate
    return LatentStates(vcat(split_dfs...), exp_results.config.model)
end

# ... (Include _process_split helper here too) ...
function _process_split(ds, model, splitter, feature_set, results)
    # ... (Logic from previous step) ...
    df_to_predict = Data.get_next_matches(ds, feature_set, splitter)
    if isempty(df_to_predict); return DataFrame(); end
    
    raw_preds = Models.PreGame.extract_parameters(model, df_to_predict, feature_set,  results)

    return _latent_state_dict_to_df(raw_preds)
    
end


function _latent_state_dict_to_df(raw_preds::Dict)::AbstractDataFrame
    match_ids = collect(keys(raw_preds))
    if isempty(match_ids); return DataFrame(); end
    
    first_val = raw_preds[match_ids[1]]
    cols = Dict{Symbol, Vector{Any}}(:match_id => match_ids)
    for p in keys(first_val)
        cols[p] = [raw_preds[id][p] for id in match_ids]
    end
    return DataFrame(cols)
end




