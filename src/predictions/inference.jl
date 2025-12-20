# src/predictions/inference.jl

using DataFrames
using Base.Threads
using ProgressMeter
using ..Experiments: LatentStates

export model_inference

# ------------------------------------------------------------------
# 1. The Kernel (Process one match)
# ------------------------------------------------------------------
function predict_row(model, row, markets)
    # A. Dispatch to get parameters (e.g., Poissons vs DixonColes)
    params = extract_params(model, row)
    
    # B. Compute the Physics (The Score Matrix)
    S = compute_score_matrix(model, params)
    
    # C. Compute the Economics (The Markets)
    results = Dict{String, Dict{String, Vector{Float64}}}()
    
    for market in markets
        # Returns Dict("Home" => [...], "Draw" => [...])
    results[string(market)] = compute_market_probs(S, market)
    end
    
    return results
end

# ------------------------------------------------------------------
# 2. The Orchestrator (Process all matches)
# ------------------------------------------------------------------
function model_inference(latents::LatentStates; market_config=nothing)
    # Handle config default
    if isnothing(market_config)
        error("market_config must be provided")
    end

    df = latents.df
    model = latents.model
    markets = collect(market_config.markets)
    n_matches = nrow(df)
    
    println("Running Inference on $(n_matches) matches...")
    
    # 1. Compute Predictions (Threaded)
    # We collect a Vector of Dicts to avoid DataFrames threading issues
    results_vec = Vector{Dict}(undef, n_matches)
    rows = collect(eachrow(df)) # Collect to allow safe indexing in threads
    
    @threads for i in 1:n_matches
        results_vec[i] = predict_row(model, rows[i], markets)
    end
    
    # 2. Flatten into PPD Structure
    # We want a long DataFrame: match_id | market | selection | distribution
    
    v_match_ids = Real[]
    v_markets = String[]
    v_selections = String[]
    v_dists = Vector{Float64}[]
    
    for (i, res_dict) in enumerate(results_vec)
        mid = rows[i].match_id
        
        for (m_name, outcome_dict) in res_dict
            for (sel_name, dist) in outcome_dict
                push!(v_match_ids, mid)
                push!(v_markets, m_name)
                push!(v_selections, sel_name)
                push!(v_dists, dist)
            end
        end
    end
    
    ppd_df = DataFrame(
        :match_id => v_match_ids,
        :market => v_markets,
        :selection => v_selections,
        :distribution => v_dists
    )
    
    return PPD(ppd_df, model, market_config)
end
