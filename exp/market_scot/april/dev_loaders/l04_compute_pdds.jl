using Revise
using BayesianFootball
using DataFrames


function raw_preds_to_df(raw_preds)
    ids = collect(keys(raw_preds))
    
    # Capitalize Dict, Vector, and Any
    cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
    
    # assumes all entries have same keys
    first_entry = raw_preds[ids[1]]
    
    for k in keys(first_entry)
        # Convert key to Symbol if it's a string, and capitalize the comprehension logic
        cols[Symbol(k)] = [raw_preds[i][k] for i in ids]
    end
    
    # Capitalize DataFrame
    return DataFrame(cols)
end


function compute_todays_matches_pdds(data_store, experiment, todays_matches)

  feature_collection = BayesianFootball.Features.create_features(
      BayesianFootball.Data.create_data_splits(data_store, experiment.config.splitter),
      experiment.config.model, 
      experiment.config.splitter
  )

  last_split_idx = length(experiment.training_results)
  chain = experiment.training_results[last_split_idx][1]
  feature_set = feature_collection[last_split_idx][1]

  raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
      experiment.config.model, todays_matches, feature_set, chain
  )

  latents = BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds), experiment.config.model)
  ppd = BayesianFootball.Predictions.model_inference(latents)

  return ppd
end
