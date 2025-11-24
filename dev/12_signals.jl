using Revise
using BayesianFootball
using DataFrames
using JLD2
using Statistics

#########

data_store = BayesianFootball.Data.load_default_datastore()
model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(data_store, model)

# filter for one season for quick training
df = filter(row -> row.season=="24/25", data_store.matches)

# we want to get the last 4 weeks - so added the game weeks
df = BayesianFootball.Data.add_match_week_column(df)

ds = BayesianFootball.Data.DataStore(
    df,
    data_store.odds,
    data_store.incidents
)


ds.matches.split_col = max.(0, ds.matches.match_week .- 14);

splitter_config = BayesianFootball.Data.ExpandingWindowCV([], ["24/25"], :split_col, :sequential) #
data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #



results = JLD2.load_object("training_results_large.jld2")

### get out of sample data - chains 
split_col_name = :split_col
all_splits = sort(unique(ds.matches[!, split_col_name]))
prediction_split_keys = all_splits[2:end] 
grouped_matches = groupby(ds.matches, split_col_name)
dfs_to_predict = [
    grouped_matches[(; split_col_name => key)] 
    for key in prediction_split_keys
]


# --- 6. Call your new function ---
all_oos_results = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    dfs_to_predict,  # Pass in the pre-split vector
    vocabulary,
    results
)



BayesianFootball.Data.DataPreprocessing.add_inital_odds_from_fractions!(ds)


####
# helpers 
#####
using Printf 

function display_odds(predict, open, close, results, symbol) 

    if !haskey(predict, symbol) || !haskey(open, symbol)
        println("symbol : $symbol not found in data")
        return 
    end

    
    chain = predict[symbol]
    o_odds = open[symbol]
    c_odds = close[symbol] 
    outcome = results[symbol]

    p_mean = mean(chain)
    p_std = std(chain)

    # 3. Market Context
    implied_prob = 1.0 / o_odds
    edge_mean = (p_mean * o_odds) - 1.0

    # --- PRINT OUTPUT ---
    
    printstyled("══════════════════════════════════════════════════════\n", color=:blue)
    printstyled(@sprintf(" MARKET ANALYSIS: :%s \n", symbol), bold=true, color=:white)
    printstyled("══════════════════════════════════════════════════════\n", color=:blue)
    
    println("Outcome:           ", outcome ? "WIN" : "LOSS")
    @printf("Open Odds :        %.3f  (Implied: %.1f%%)\n", o_odds, implied_prob*100)
    @printf("Close Odds:        %.3f  (Implied: %.1f%%)  \n", c_odds, (1 / c_odds) *100)
    @printf("Model Odds: (mean) %.3f\n", 1 / p_mean )
    println()
    
    printstyled("--- Model Posterior (MCMC) ---\n", color=:cyan)
    @printf("Mean Prob:         %.1f%%  (Edge: %.2f%%)\n", p_mean*100, edge_mean*100)
    @printf("Uncertainty (σ):   %.3f\n", p_std)
    @printf("90%% CI:            [%.1f%% - %.1f%%]\n", quantile(chain, 0.05)*100, quantile(chain, 0.95)*100)
    println()


end 

#######
#  Dev 
#######

predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )



match_id = rand(keys(all_oos_results))
r1 =  all_oos_results[match_id]
subset( ds.matches, :match_id => ByRow(isequal(match_id)))


match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...);

model_odds = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict));
model_odds

open, close, results = BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds)

display_odds(match_predict, open, close, results, :home)
display_odds(match_predict, open, close, results, :away)
display_odds(match_predict, open, close, results, :draw)

open[:over_25]
close[:over_25]
model_odds[:over_25]
results[:over_25]


BayesianFootball.Signals.bayesian_kelly(match_predict[:away], close[:away])
BayesianFootball.Signals.kelly_fraction(close[:away], mean(match_predict[:away]))
BayesianFootball.Signals.calc_analytical_shrinkage(match_predict[:away], close[:away])


