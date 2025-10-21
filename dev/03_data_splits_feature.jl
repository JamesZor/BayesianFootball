
using Revise
using BayesianFootball
using DataFrames
using Turing

# --- 1. Load the DataStore (D) ---
# This is the complete set of all data we have.
println("--- 1. Loading DataStore (D) ---")
data_store = BayesianFootball.Data.load_default_datastore();
println("✅ DataStore loaded with $(nrow(data_store.matches)) matches.\n")


splitter_static = BayesianFootball.Experiments.StaticSplit(["24/25"]); # Using a small season for speed
splitter_round  = BayesianFootball.Experiments.ExpandingWindowCV(
  ["23/24"],
  ["24/25"],
  :round,
  :sequential
)


a =BayesianFootball.Data.TimeSeriesSplits(data_store.matches, splitter_round.base_seasons, splitter_round.target_seasons, splitter_round.round_col, splitter_round.ordering)



# --- 4. Get a single data split (D_i) ---
# In a real experiment, the ExperimentRunner would loop over these.
# Here, we manually create one for demonstration.
println("--- 4. Creating a data split (D_i) ---")
train_df = filter(row -> row.season in splitter.train_seasons, data_store.matches);
println("✅ Created a data split (D_i) with $(nrow(train_df)) matches.\n")





