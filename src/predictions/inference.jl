# src/predictions/inference.jl


using DataFrames
using Base.Threads
using ..PredictionTypes
using ..Experiments: LatentStates

# Include the implementation methods (the Kernels)
include("methods/poisson.jl")

export model_inference

"""
    model_inference(latents::LatentStates, config::PredictionConfig)

Orchestrates the prediction pipeline:
1. Maps the `predict_match_kernel` over every row in `latents` (Threaded).
2. Flattens the resulting list of mini-DataFrames into one big 'Long' DataFrame.
"""
function model_inference(latents::LatentStates, config::PredictionConfig)
    df = latents.df
    model = latents.model
    markets = collect(config.markets)
    
    n = nrow(df)
    results = Vector{DataFrame}(undef, n)
    
    # Use eachindex for safety & extract columns once
    match_ids = df.match_id
    λ_hs = df.λ_h
    λ_as = df.λ_a
    
    @threads for i in eachindex(results)
        results[i] = predict_match_kernel(
            model, match_ids[i], λ_hs[i], λ_as[i], markets
        )
    end
    
    long_ppd = reduce(vcat, results)  # Slightly faster than vcat(results...)
    return PPD(long_ppd, model, config)
end



