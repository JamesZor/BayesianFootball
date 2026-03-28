
using Revise
using BayesianFootball
using DataFrames
using BayesianFootball.Signals



ds = Data.load_extra_ds()

transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)
### load 


saved_folders = Experiments.list_experiments("exp/market_runs"; data_dir="./data")


loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
for folder in saved_folders
    try
        res = Experiments.load_experiment(folder)
        push!(loaded_results, res)
    catch e
        @warn "Could not load $folder: $e"
    end
end


m1 = loaded_results[1]

feature_collection = BayesianFootball.Features.create_features(
    BayesianFootball.Data.create_data_splits(ds, m1.config.splitter),
    m1.config.model, 
    m1.config.splitter
)


last_split_idx = length(m1.training_results)
chain1 = m1.training_results[last_split_idx][1]
feature_set = feature_collection[last_split_idx][1]



feature_set.data[:team_map]
#=
  "edinburgh-city-fc"            => 10
  "arbroath"                     => 3
  "clyde-fc"                     => 5
  "queen-of-the-south"           => 19
  "stirling-albion"              => 21
  "east-kilbride"                => 9
  "hamilton-academical"          => 14
  "forfar-athletic"              => 13
  "peterhead"                    => 18
  "stenhousemuir"                => 20
  "bonnyrigg-rose"               => 4
  "dumbarton"                    => 7
  "inverness-caledonian-thistle" => 15
  "the-spartans-fc"              => 23
  "alloa-athletic"               => 1
  "kelty-hearts-fc"              => 16
  "montrose"                     => 17
  "stranraer"                    => 22
  "cove-rangers"                 => 6
  "falkirk-fc"                   => 12
  "elgin-city"                   => 11
  "east-fife"                    => 8
  "annan-athletic"               => 2
=#



match_to_predict = DataFrame(
    match_id = [1, 2, 3],
    match_week = [999, 999, 999], 
    home_team = ["east-fife", "alloa-athletic", "hamilton-academical"], 
    away_team = ["stenhousemuir", "peterhead", "kelty-hearts-fc"]
)

using Dates
match_to_predict.match_date .= today()


raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
    m1.config.model,
    match_to_predict,
    feature_set, 
    chain1
)

function raw_preds_to_df(raw_preds::Dict)
    ids = collect(keys(raw_preds))
    cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
    # Assumes all entries have same keys
    first_entry = raw_preds[ids[1]]
    for k in keys(first_entry)
        cols[k] = [raw_preds[i][k] for i in ids]
    end
    return DataFrame(cols)
end


latents = BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds), m1.config.model)
ppd = BayesianFootball.Predictions.model_inference(latents)

function make_predictions(data_store, experiment, idx, matches_to_predict)

  function raw_preds_to_df(raw_preds::Dict)
      ids = collect(keys(raw_preds))
      cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
      # Assumes all entries have same keys
      first_entry = raw_preds[ids[1]]
      for k in keys(first_entry)
          cols[k] = [raw_preds[i][k] for i in ids]
      end
      return DataFrame(cols)
  end

  feature_collection = BayesianFootball.Features.create_features(
      BayesianFootball.Data.create_data_splits(data_store, experiment.config.splitter),
      experiment.config.model, 
      experiment.config.splitter
  )
  feature_set = feature_collection[idx][1]

  chain = experiment.training_results[idx][1]


  raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
      experiment.config.model,
      matches_to_predict,
      feature_set,
      chain
  )

  latents = BayesianFootball.Experiments.LatentStates(raw_preds_to_df(raw_preds), experiment.config.model)
  ppd = BayesianFootball.Predictions.model_inference(latents)

  return ppd

end 

pp = make_predictions(ds, m1, 1, match_to_predict) 

pp.df.prob = mean.(pp.df.distribution)
pp.df.odds = round.(1 ./ pp.df.prob, digits=2)

ppp = subset(pp.df, :match_id => ByRow(isequal(1)))

select(ppp, :market_name, :selection, :prob, :odds)

