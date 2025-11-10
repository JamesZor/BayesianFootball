using Revise
using BayesianFootball
using DataFrames
using JLD2


# --- Phase 1: Globals (D, M, G) --- (Same as before)
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

# 1. Define your "special cases" mapping
split_map = Dict(37 => 1, 38 => 2, 39 => 3)

# 2. Use get() with a default value of 0
#    We use Ref(split_map) to tell Julia to treat the Dict as a single object
#    and not try to broadcast over its elements.
ds.matches.split_col = get.(Ref(split_map), ds.matches.match_week, 0);

# large v2 
ds.matches.split_col = max.(0, ds.matches.match_week .- 14);






splitter_config = BayesianFootball.Data.ExpandingWindowCV([], ["24/25"], :split_col, :sequential) #
data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #

# --- Phase 3: Define Training Configuration ---
# Sampler Config (Choose one)
sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=200, n_chains=2, n_warmup=100) # Use renamed struct

# Explicitly set a limit (e.g., if NUTS uses 2 chains, maybe allow 4 concurrent splits on 8 threads)
strategy_parallel_custom = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 

# training_config_limited = TrainingConfig(sampler_conf, strategy_parallel_limited)
training_config_custom  = BayesianFootball.Training.TrainingConfig(sampler_conf, strategy_parallel_custom)

# Then run:
# results = BayesianFootball.Training.train(model, training_config_custom, feature_sets)
# save and load 
#
# JLD2.save_object("training_results_large.jld2", results)

# results = JLD2.load_object("training_results.jld2")
results = JLD2.load_object("training_results_large.jld2")

### get out of sample data - chains 
# 1. Define the column you want to split on
#    (You can change this to :round, :week, etc. later)
split_col_name = :split_col

# 2. Get all unique split keys (e.g., [0, 1, 2, 3])
all_splits = sort(unique(ds.matches[!, split_col_name]))

# 3. Define the splits you want to *predict* (e.g., [1, 2, 3])
#    We skip the first key (0), as it was for training the first model
prediction_split_keys = all_splits[2:end] 

# 4. Group the data ONCE
grouped_matches = groupby(ds.matches, split_col_name)

# 5. Create the vector of DataFrames (as efficient SubDataFrame views)
#    This is the new argument for your function
dfs_to_predict = [
    grouped_matches[(; split_col_name => key)] 
    for key in prediction_split_keys
]


# --- 6. Call your new function ---
# It's now much cleaner and more flexible
all_oos_results = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    dfs_to_predict,  # Pass in the pre-split vector
    vocabulary,
    results
)



using Statistics
using Logging

# --- Copy this from your file ---
"""
Parses a fractional odds string (e.g., "19/10") into a decimal value (e.g., 2.9).
Returns 0.0 if parsing fails.
"""
function parse_fractional_to_decimal(s::AbstractString)
    parts = split(s, '/')
    if length(parts) != 2
        return 0.0
    end
    try
        n = parse(Float64, parts[1])
        d = parse(Float64, parts[2])
        if d == 0.0
            return 0.0
        end
        return (n / d) + 1.0
    catch e
        return 0.0
    end
end


# --- NEW HELPER FUNCTIONS ---

"""
Calculates the Kullback-Leibler (KL) Divergence, D_KL(Q || P).

Measures the "surprise" of using distribution P to approximate "truth" Q.
A lower score is better.

Args:
- q: The "true" distribution (e.Attribution: Q_close)
- p: The "approximate" distribution (e.g., P_open or P_model)
"""
function kl_divergence(q::NamedTuple, p::NamedTuple)
    common_keys = keys(q) ∩ keys(p)
    
    # Add a tiny epsilon to prevent log(0) -> -Inf
    epsilon = 1e-9
    
    return sum(
        q[k] * ( log(q[k] + epsilon) - log(p[k] + epsilon) )
        for k in common_keys
    )
end

"""
Removes the bookmaker's "vig" (overround) from a NamedTuple of odds.
Returns a NamedTuple of "fair" probabilities that sum to 1.0.
"""
function vig_remove(odds::NamedTuple)
    if isempty(odds)
        return (;) # Return empty NamedTuple
    end
    
    # 1. Calculate overround
    overround = sum(1.0 / v for v in values(odds))
    
    # 2. Normalize to get fair probabilities
    probs = NamedTuple(
        k => (1.0 / odds[k]) / overround 
        for k in keys(odds)
    )
    return probs
end

"""
A new function to get the OPENING odds for a match.
This is modeled on `Predictions.get_market` but reads from 
`initial_fractional_value` instead.
"""
function get_opening_market(match_id::Int, predict_config, odds_df::DataFrame)
    
    market_names = [m.market_name for m in predict_config.markets]

    # 1. Filter the odds DataFrame for this match
    match_odds_df = filter(
        row -> row.match_id == match_id && 
               row.market_name in market_names,
        odds_df
    )
    
    if isempty(match_odds_df)
        return (;) # Return empty
    end
    
    # 2. Parse the fractional odds string
    # We use our parser function here
    parsed_odds = parse_fractional_to_decimal.(match_odds_df.initial_fractional_value)
    
    # 3. Create the NamedTuple
    valid_odds = NamedTuple(
        Symbol(row.market_name) => odds
        for (row, odds) in zip(eachrow(match_odds_df), parsed_odds)
        if odds > 1.0 # Filter out failed parses or "SP"
    )
    
    return valid_odds
end

"""
Converts the `match_predict` structure (NamedTuple of vectors)
into a Vector of NamedTuples (one per posterior sample),
correctly filtering by the keys in `group`.
"""
function get_posterior_samples(match_predict::NamedTuple, group::Vector{Symbol})
    
    # 1. Filter the NamedTuple by keys that are in our group
    filtered_predict = NamedTuple(
        k => match_predict[k] 
        for k in keys(match_predict) 
        if k in group
    )
    
    # 2. Check if we found all the keys we need
    #    (e.g., did we find :home, :draw, AND :away?)
    if length(keys(filtered_predict)) != length(group)
        return [] # Return empty if a key was missing
    end
    
    # 3. Get the number of samples (all vectors should be the same length)
    #    `first(filtered_predict)` gets the first pair (e.g., :home => [0.1, ...])
    #    `.second` gets the vector itself.
    n_samples = length(first(filtered_predict).second)
    if n_samples == 0
        return []
    end
    
    # 4. "Zip" the posteriors
    #    This creates the [ (:home=>0.1, :draw=>0.3, :away=>0.6), ... ] structure
    samples = [
        NamedTuple(k => filtered_predict[k][i] for k in group)
        for i in 1:n_samples
    ]
    return samples
end


function get_posterior_samples(match_predict::NamedTuple, group::Vector{Symbol})
    
    # 1. Filter the NamedTuple by keys that are in our group
    filtered_predict = NamedTuple(
        k => match_predict[k] 
        for k in keys(match_predict) 
        if k in group
    )
    
    # 2. Check if we found all the keys we need
    if length(keys(filtered_predict)) != length(group)
        return [] # Return empty if a key was missing
    end
    
    # 3. Get the number of samples (all vectors should be the same length)
    #    `first(filtered_predict)` gets the first VALUE (the vector)
    
    # --- THIS IS THE CORRECTED LINE ---
    n_samples = length(first(filtered_predict))
    
    if n_samples == 0
        return []
    end
    
    # 4. "Zip" the posteriors
    #    This creates the [ (:home=>0.1, :draw=>0.3, :away=>0.6), ... ] structure
    samples = [
        NamedTuple(k => filtered_predict[k][i] for k in group)
        for i in 1:n_samples
    ]
    return samples
end


"""
Runs the Information Gain (vs. Close Line) analysis.

Compares the model's posterior against the opening line,
using the closing line as the "ground truth".
"""
function run_ig_analysis(
    all_oos_results::Dict, 
    model, 
    predict_config, 
    ds
)
    
    # --- For this rough draft, we'll focus on the 1x2 market ---
    MARKET_GROUP = [:home, :draw, :away]
    
    all_results = [] # To store our summary rows
    println("Starting IG analysis for $(length(all_oos_results)) matches...")
    
    # Suppress warnings about missing keys, etc.
    with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do
        
    for (match_id, r1) in all_oos_results
        try
            # 1. Get Closing Line Probs (Q_close)
            odds_close_nt = BayesianFootball.Predictions.get_market(match_id, predict_config, ds.odds)
            odds_close_filtered = NamedTuple(k => odds_close_nt[k] for k in MARKET_GROUP if haskey(odds_close_nt, k))
            
            # 2. Get Opening Line Probs (P_open)
            odds_open_nt = get_opening_market(match_id, predict_config, ds.odds)
            odds_open_filtered = NamedTuple(k => odds_open_nt[k] for k in MARKET_GROUP if haskey(odds_open_nt, k))

            # 3. Get Model Posterior (P_model_dist)
            match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...)
            p_model_samples = get_posterior_samples(match_predict, MARKET_GROUP)
            
            # --- Check: Do we have all data for this market? ---
            if (length(keys(odds_close_filtered)) != length(MARKET_GROUP) || 
                length(keys(odds_open_filtered)) != length(MARKET_GROUP) || 
                isempty(p_model_samples))
                
                # println("  - Skipping $match_id: Missing 1x2 market data.")
                continue
            end

            # --- Convert odds to "true" probabilities ---
            q_close = vig_remove(odds_close_filtered)
            p_open = vig_remove(odds_open_filtered)

            # --- 4. Calculate S_open (Benchmark Score) ---
            # D_KL(Q_close || P_open): How "bad" was the opening line?
            # This is a single number.
            s_open = kl_divergence(q_close, p_open)

            # --- 5. Calculate S_model_dist (Model Scores) ---
            # [D_KL(Q_close || P_model_1), ..., D_KL(Q_close || P_model_N)]
            # This is a vector of scores, one for each posterior sample.
            s_model_dist = [kl_divergence(q_close, p_sample) for p_sample in p_model_samples]

            # --- 6. Calculate IG Distribution ---
            # IG_dist = S_open - S_model_dist
            # (Positive numbers mean our model's score was *lower*, which is good)
            ig_dist = s_open .- s_model_dist
            
            # --- 7. Summarize and Store Results ---
            row = (
                match_id = match_id,
                s_open = s_open, # Opening line's KL score
                s_model_mean = mean(s_model_dist), # Model's average KL score
                ig_mean = mean(ig_dist), # The expected IG
                prob_model_better = mean(ig_dist .> 0.0), # Prob(IG > 0)
                ig_ci_05 = quantile(ig_dist, 0.05),
                ig_ci_95 = quantile(ig_dist, 0.95)
            )
            push!(all_results, row)
            
        catch e
            println("  - ⚠️ WARNING: Skipping match $match_id due to error: $e")
        end
    end # end for loop
    
    end # end logger
    
    println("...Analysis complete. Found $(length(all_results)) valid matches.")
    return DataFrame(all_results)
end


predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )



predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.Market1X2() )

# (This assumes all your previous code from Phase 1 & 2 has been run)
# (all_oos_results, model, predict_config, ds are all loaded)

# Run the full analysis
ig_report = run_ig_analysis(
    all_oos_results,
    model,
    predict_config,
    ds
)

# --- See the results ---

# 1. Look at the overall average
# Is the average IG positive?
println("Average IG (Mean): $(mean(ig_report.ig_mean))")

# On average, what's the probability our model was better?
println("Average Prob(Model Better): $(mean(ig_report.prob_model_better))")

# 2. Look at the top 10 best matches for the model
sort!(ig_report, :ig_mean, rev=true)
println(first(ig_report, 10))

# 3. Look at the top 10 worst matches
sort!(ig_report, :ig_mean)
println(first(ig_report, 10))


############### v2 

# (Make sure you have `using DataFrames` and `using BayesianFootball` at the top)

# 1. Make a deep copy to avoid changing your original data
ds_odds_initial = deepcopy(ds.odds)

# 2. Convert the initial fractional string to a decimal
# (This uses the `parse_fractional_to_decimal` function from our previous chat)
ds_odds_initial.initial_decimal = parse_fractional_to_decimal.(ds_odds_initial.initial_fractional_value)

# 3. Filter out rows where parsing failed (odds <= 1.0)
filter!(row -> row.initial_decimal > 1.0, ds_odds_initial)

# 4. --- THIS IS THE TRICK ---
# Overwrite the `decimal_odds` column with our new initial odds.
ds_odds_initial.decimal_odds = ds_odds_initial.initial_decimal

# 5. Create the new DataStore for "opening lines"
ds_initial = BayesianFootball.Data.DataStore(
    ds.matches,
    ds_odds_initial,
    ds.incidents
)

println("Created ds_initial with $(nrow(ds_initial.odds)) opening odds.")


# (Make sure you have `using Statistics`, `using Logging` and the helper functions
#  `kl_divergence`, `vig_remove`, and `get_posterior_samples` from our last message)

# --- CONFIG ---
# We'll stick to the 1x2 market for this simple test
MARKET_GROUP = [:home, :draw, :away]

# Pick a random match to test
match_id = rand(keys(all_oos_results))
println("\n--- 🕵️ Testing Match ID: $match_id ---")

# We'll wrap this in a try/catch to handle missing data for this one match
try
    # --- Part 1: Get Closing Line Probs (Q_close) ---
    # We use the ORIGINAL `ds` object here
    odds_close_nt = BayesianFootball.Predictions.get_market(match_id, predict_config, ds.odds)
    odds_close_filtered = NamedTuple(k => odds_close_nt[k] for k in MARKET_GROUP if haskey(odds_close_nt, k))
    
    # Check if we have all 3 outcomes
    if length(keys(odds_close_filtered)) != 3
        error("Missing closing odds for 1x2 market")
    end
    
    q_close = vig_remove(odds_close_filtered)
    println("Q_close (Closing Probs): $q_close")


    # --- Part 2: Get Opening Line Probs (P_open) ---
    # We use the NEW `ds_initial` object here
    odds_open_nt = BayesianFootball.Predictions.get_market(match_id, predict_config, ds_initial.odds)
    odds_open_filtered = NamedTuple(k => odds_open_nt[k] for k in MARKET_GROUP if haskey(odds_open_nt, k))
    
    if length(keys(odds_open_filtered)) != 3
         error("Missing opening odds for 1x2 market")
    end

    p_open = vig_remove(odds_open_filtered)
    println("P_open (Opening Probs): $p_open")


    # --- Part 3: Get Model Posterior (P_model_dist) ---
    r1 = all_oos_results[match_id]
    match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...)
    p_model_samples = get_posterior_samples(match_predict, MARKET_GROUP)

    if isempty(p_model_samples)
        error("Missing model posteriors for 1x2 market")
    end
    println("Model Samples (count): $(length(p_model_samples))")


    # --- Part 4: Calculate Scores ---
    # S_open = D_KL(Q_close || P_open)
    s_open = kl_divergence(q_close, p_open)
    println("S_open (Benchmark Score): $s_open")

    # S_model_dist = [D_KL(Q_close || P_model_1), ...]
    s_model_dist = [kl_divergence(q_close, p_sample) for p_sample in p_model_samples]
    println("S_model (Mean Score): $(mean(s_model_dist))")


    # --- Part 5: Calculate Final IG ---
    # IG_dist = S_open - S_model_dist
    ig_dist = s_open .- s_model_dist
    
    ig_mean = mean(ig_dist)
    prob_better = mean(ig_dist .> 0.0)
    
    println("----------------------------------------")
    println("✅ SUCCESS for Match $match_id")
    println("IG Mean: $ig_mean")
    println("Prob(Model Better): $prob_better")
    println("----------------------------------------")

catch e
    println("❌ FAILED for Match $match_id: $e")
end


# (This assumes ds_initial is still in your workspace from Step 1)

ig_report = run_ig_analysis(
    all_oos_results,
    model,
    predict_config,
    ds,          # <-- The original ds with CLOSING odds
    ds_initial   # <-- The new ds with OPENING odds
)

# --- AND THEN THE FINAL AGGREGATE ---

if !isempty(ig_report)
    println("\n--- 📊 FINAL REPORT (All Matches) ---")
    println("Average IG (Mean): $(mean(ig_report.ig_mean))")
    println("Average Prob(Model Better): $(mean(ig_report.prob_model_better))")
else
    println("Analysis failed to find any valid matches.")
end





##### v3 


using Statistics
using Logging

"""
(UPDATED) Calculates the Binary KL Divergence, D_KL(Q || P).

Args:
- q: The "true" probability of the event (0.0 to 1.0)
- p: The "approximate" probability of the event (0.0 to 1.0)
"""
function kl_divergence(q::Number, p::Number)
    epsilon = 1e-9 # Prevent log(0)
    
    # Add epsilon to all terms for stability
    q = clamp(q, epsilon, 1.0 - epsilon)
    p = clamp(p, epsilon, 1.0 - epsilon)
    
    # q * log(q/p) + (1-q) * log((1-q)/(1-p))
    return (q * (log(q) - log(p))) + ((1.0 - q) * (log(1.0 - q) - log(1.0 - p)))
end

"""
(NEW) Safely gets the probability of a market from a NamedTuple.
Sums keys if given a Vector.
"""
function get_prob(nt::NamedTuple, keys::Union{Symbol, Vector{Symbol}})
    # Handle single key
    if keys isa Symbol
        return haskey(nt, keys) ? nt[keys] : 0.0
    end
    
    # Handle vector of keys
    return sum(nt[k] for k in keys if haskey(nt, k); init=0.0)
end

"""
(NEW) Safely gets a probability from a vector of posterior samples.
"""
function get_prob(samples::Vector{<:NamedTuple}, keys::Union{Symbol, Vector{Symbol}})
    if isempty(samples)
        return []
    end
    
    # Handle single key
    if keys isa Symbol
        return [s[keys] for s in samples if haskey(s, keys)]
    end
    
    # Handle vector of keys
    return [sum(s[k] for k in keys if haskey(s, k); init=0.0) for s in samples]
end


"""
Runs the Binary Information Gain analysis for a specific set of markets.

Args:
- (Same as before)
- markets_to_test: A list of (market_key, complement_keys) pairs.
"""
function run_binary_ig_analysis(
    all_oos_results::Dict, 
    model, 
    predict_config, 
    ds_close::BayesianFootball.Data.DataStore,
    ds_open::BayesianFootball.Data.DataStore;
    markets_to_test::Vector
)
    
    all_results = [] # To store our summary rows
    println("Starting Binary IG analysis for $(length(all_oos_results)) matches...")
    
    with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do
        
    for (match_id, r1) in all_oos_results
        try
            # --- 1. Get All Probs (as before) ---
            odds_close_nt = BayesianFootball.Predictions.get_market(match_id, predict_config, ds_close.odds)
            q_close_all = vig_remove(odds_close_nt)

            odds_open_nt = BayesianFootball.Predictions.get_market(match_id, predict_config, ds_open.odds)
            p_open_all = vig_remove(odds_open_nt)

            match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...)
            # Note: We get *all* posterior samples, not just 1x2
            p_model_samples = get_posterior_samples(match_predict, collect(keys(match_predict)))

            if isempty(p_model_samples) || isempty(q_close_all) || isempty(p_open_all)
                continue
            end

            # --- 2. Loop over the markets we want to test ---
            for (market_key, complement_keys) in markets_to_test
                
                # --- 3. Get Binary Probs for THIS market ---
                q_close = get_prob(q_close_all, market_key)
                p_open = get_prob(p_open_all, market_key)
                p_model_dist = get_prob(p_model_samples, market_key)

                # Get complement probs
                q_close_comp = get_prob(q_close_all, complement_keys)
                p_open_comp = get_prob(p_open_all, complement_keys)
                p_model_comp_dist = get_prob(p_model_samples, complement_keys)

                # --- 4. Validation ---
                # Check if this market exists and is a valid probability
                if (q_close <= 0.0 || p_open <= 0.0 || isempty(p_model_dist) ||
                   (q_close + q_close_comp) < 0.99 || # Check if they sum to ~1
                   (p_open + p_open_comp) < 0.99)
                    # println("Skipping $match_id / $market_key: Incomplete data")
                    continue
                end

                # --- 5. Calculate Scores (using new binary kl_divergence) ---
                s_open = kl_divergence(q_close, p_open)
                s_model_dist = [kl_divergence(q_close, p) for p in p_model_dist]
                ig_dist = s_open .- s_model_dist
                
                # --- 6. Store Results ---
                row = (
                    match_id = match_id,
                    market = market_key, # <-- NEW COLUMN
                    q_close = q_close,
                    p_open = p_open,
                    p_model_mean = mean(p_model_dist),
                    s_open = s_open,
                    s_model_mean = mean(s_model_dist),
                    ig_mean = mean(ig_dist),
                    prob_model_better = mean(ig_dist .> 0.0),
                )
                push!(all_results, row)
            end # end market loop
            
        catch e
             println("  - ⚠️ WARNING: Skipping match $match_id due to error: $e")
        end
    end # end match loop
    end # end logger
    
    println("...Analysis complete. Found $(length(all_results)) valid market results.")
    return DataFrame(all_results)
end


# 1. Define the markets you want to isolate
markets_to_test = [
    # 1x2 Markets
    (:home, [:draw, :away]),
    (:draw, [:home, :away]),
    (:away, [:home, :draw]),
    
    # Over/Under 2.5
    (:over_25, [:under_25]),
    (:under_25, [:over_25]),
    
    # BTTS
    (:btts_yes, [:btts_no]),
    (:btts_no, [:btts_yes])
]

# 2. Run the new analysis function
binary_ig_report = run_binary_ig_analysis(
    all_oos_results,
    model,
    predict_config,
    ds,          # <-- The original ds with CLOSING odds
    ds_initial;  # <-- The new ds with OPENING odds
    markets_to_test=markets_to_test
)

# 3. See the results, grouped by market
if !isempty(binary_ig_report)
    # Group by market and get the average performance
    report_summary = combine(groupby(binary_ig_report, :market)) do g
        (
            n_matches = nrow(g),
            avg_ig_mean = mean(g.ig_mean),
            avg_prob_better = mean(g.prob_model_better),
            avg_p_model = mean(g.p_model_mean),
            avg_p_open = mean(g.p_open),
            avg_q_close = mean(g.q_close)
        )
    end
    
    # Sort by the most profitable (highest IG)
    sort!(report_summary, :avg_ig_mean, rev=true)
    
    println("\n--- 📊 FINAL BINARY REPORT (Grouped by Market) ---")
    println(report_summary)
else
    println("Analysis failed to find any valid matches.")
end



#### above doesnt work 


#### working or something 


using Statistics
using Logging
using DataFrames

# --- HELPER FUNCTIONS (Unchanged) ---

"""
(UPDATED) Calculates the Binary KL Divergence, D_KL(Q || P).
"""
function kl_divergence(q::Number, p::Number)
    epsilon = 1e-9 # Prevent log(0)
    q = clamp(q, epsilon, 1.0 - epsilon)
    p = clamp(p, epsilon, 1.0 - epsilon)
    return (q * (log(q) - log(p))) + ((1.0 - q) * (log(1.0 - q) - log(1.0 - p)))
end

"""
(Unchanged) Removes the bookmaker's "vig" (overround) from a NamedTuple of odds.
"""
function vig_remove(odds::NamedTuple)
    if isempty(odds)
        return (;) # Return empty NamedTuple
    end
    overround = sum(1.0 / v for v in values(odds))
    if overround == 0.0
        return (;)
    end
    probs = NamedTuple(
        k => (1.0 / odds[k]) / overround 
        for k in keys(odds)
    )
    return probs
end

"""
(v3 - Unchanged) Converts `match_predict` to a Vector of NamedTuples.
"""
function get_posterior_samples(match_predict::NamedTuple, group::Vector{Symbol})
    filtered_predict = NamedTuple(
        k => match_predict[k] 
        for k in keys(match_predict) 
        if k in group
    )
    if length(keys(filtered_predict)) != length(group)
        return []
    end
    n_samples = length(first(filtered_predict))
    if n_samples == 0
        return []
    end
    samples = [
        NamedTuple(k => filtered_predict[k][i] for k in group)
        for i in 1:n_samples
    ]
    return samples
end

"""
(Unchanged) Safely gets a probability from a vector of posterior samples.
"""
function get_prob(samples::Vector{<:NamedTuple}, keys::Union{Symbol, Vector{Symbol}})
    if isempty(samples)
        return []
    end
    if keys isa Symbol
        return [s[keys] for s in samples if haskey(s, keys)]
    end
    return [sum(s[k] for k in keys if haskey(s, k); init=0.0) for s in samples]
end


# --- (NEW) MAIN ANALYSIS FUNCTION (v3) ---

"""
Runs the Binary Information Gain analysis (v3).
Calculates vig *per market group* inside the loop.
"""
function run_binary_ig_analysis_v3(
    all_oos_results::Dict, 
    model, 
    predict_config, 
    ds_close::BayesianFootball.Data.DataStore,
    ds_open::BayesianFootball.Data.DataStore;
    markets_to_test::Vector
)
    
    all_results = [] # To store our summary rows
    println("Starting Binary IG analysis (v3) for $(length(all_oos_results)) matches...")
    
    with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do
        
    for (match_id, r1) in all_oos_results
        try
            # --- 1. Get ALL Raw Odds & All Posteriors ---
            odds_close_nt = BayesianFootball.Predictions.get_market(match_id, predict_config, ds_close.odds)
            odds_open_nt = BayesianFootball.Predictions.get_market(match_id, predict_config, ds_open.odds)
            match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...)

            if isempty(odds_close_nt) || isempty(odds_open_nt) || isempty(match_predict)
                continue
            end
            
            # --- 2. Loop over the markets we want to test ---
            for (market_key, complement_keys) in markets_to_test
                
                # Define the full group of keys for this market
                market_group_keys = [market_key, complement_keys...]
                
                # --- 3. Process CLOSING Odds (Q_close) ---
                odds_close_group = NamedTuple(
                    k => odds_close_nt[k] 
                    for k in market_group_keys if haskey(odds_close_nt, k)
                )
                if length(keys(odds_close_group)) != length(market_group_keys)
                    continue # This match didn't have all odds for this group
                end
                q_close_probs = vig_remove(odds_close_group)
                q_close = q_close_probs[market_key]

                # --- 4. Process OPENING Odds (P_open) ---
                odds_open_group = NamedTuple(
                    k => odds_open_nt[k] 
                    for k in market_group_keys if haskey(odds_open_nt, k)
                )
                if length(keys(odds_open_group)) != length(market_group_keys)
                    continue # This match didn't have all odds for this group
                end
                p_open_probs = vig_remove(odds_open_group)
                p_open = p_open_probs[market_key]

                # --- 5. Process MODEL Posteriors (P_model) ---
                p_model_dist = get_prob(get_posterior_samples(match_predict, market_group_keys), market_key)
                if isempty(p_model_dist)
                    continue # Model didn't produce posteriors for this group
                end
                
                # --- 6. Calculate Scores ---
                s_open = kl_divergence(q_close, p_open)
                s_model_dist = [kl_divergence(q_close, p) for p in p_model_dist]
                ig_dist = s_open .- s_model_dist
                
                # --- 7. Store Results ---
                row = (
                    match_id = match_id,
                    market = market_key,
                    q_close = q_close,
                    p_open = p_open,
                    p_model_mean = mean(p_model_dist),
                    s_open = s_open,
                    s_model_mean = mean(s_model_dist),
                    ig_mean = mean(ig_dist),
                    prob_model_better = mean(ig_dist .> 0.0),
                )
                push!(all_results, row)
            end # end market loop
            
        catch e
             # This will catch the KeyError or other issues
             println("  - ⚠️ WARNING: Skipping match $match_id due to error: $e")
        end
    end # end match loop
    end # end logger
    
    println("...Analysis complete. Found $(length(all_results)) valid market results.")
    return DataFrame(all_results)
end


# 1. Define the markets you want to isolate
markets_to_test = [
    # 1x2 Markets
    (:home, [:draw, :away]),
    (:draw, [:home, :away]),
    (:away, [:home, :draw]),
    
    # Over/Under 2.5
    (:over_25, [:under_25]),
    (:under_25, [:over_25]),
    
    # BTTS
    (:btts_yes, [:btts_no]),
    (:btts_no, [:btts_yes]),

    # You can add more here
    (:over_15, [:under_15]),
    (:under_15, [:over_15]),
    (:over_35, [:under_35]),
    (:under_35, [:over_35])
]

# 2. Run the new analysis function (v3)
binary_ig_report = run_binary_ig_analysis_v3(
    all_oos_results,
    model,
    predict_config,
    ds,          # <-- The original ds with CLOSING odds
    ds_initial;  # <-- The new ds with OPENING odds
    markets_to_test=markets_to_test
)

# 3. See the results, grouped by market
if !isempty(binary_ig_report)
    # Group by market and get the average performance
    report_summary = combine(groupby(binary_ig_report, :market)) do g
        (
            n_matches = nrow(g),
            avg_ig_mean = mean(g.ig_mean),
            avg_prob_better = mean(g.prob_model_better),
            avg_p_model = mean(g.p_model_mean),
            avg_p_open = mean(g.p_open),
            avg_q_close = mean(g.q_close)
        )
    end
    
    # Sort by the most profitable (highest IG)
    sort!(report_summary, :avg_ig_mean, rev=true)
    
    println("\n--- 📊 FINAL BINARY REPORT (Grouped by Market) ---")
    println(report_summary)
else
    println("Analysis failed to find any valid matches.")
end


#### lof score 
"""
Calculates the Log Score.
A lower score is better.

Args:
- p_event: The model's predicted probability of the event (0.0 to 1.0).
- y_outcome: A Bool, true if the event happened, false otherwise.
"""
function log_score(p_event::Number, y_outcome::Bool)
    epsilon = 1e-9 # Prevent log(0)
    
    # Clamp the probability to a stable range
    p = clamp(p_event, epsilon, 1.0 - epsilon)
    
    if y_outcome
        # Event happened, score is -log(p)
        return -log(p)
    else
        # Event did not happen, score is -log(1-p)
        return -log(1.0 - p)
    end
end

"""
Runs the Log Score Gain (LSG) analysis (v1).

Compares the model's posterior against the opening line,
using the ACTUAL MATCH OUTCOME as the "ground truth".
"""
function run_lsg_analysis(
    all_oos_results::Dict, 
    model, 
    predict_config, 
    ds_close::BayesianFootball.Data.DataStore, # Used to get *results*
    ds_open::BayesianFootball.Data.DataStore;  # Used to get *opening odds*
    markets_to_test::Vector
)
    
    all_results = [] # To store our summary rows
    println("Starting LSG analysis (v1) for $(length(all_oos_results)) matches...")
    
    with_logger(Logging.SimpleLogger(stderr, Logging.Error)) do
        
    for (match_id, r1) in all_oos_results
        try
            # --- 1. Get All Raw Data ---
            
            # Get match results (our "truth")
            market_results_nt = BayesianFootball.Predictions.get_market_results(match_id, predict_config, ds_close.odds)
            
            # Get opening odds (our "benchmark")
            odds_open_nt = BayesianFootball.Predictions.get_market(match_id, predict_config, ds_open.odds)
            
            # Get model posteriors (our "predictor")
            match_predict = BayesianFootball.Predictions.predict_market(model, predict_config, r1...)

            if isempty(market_results_nt) || isempty(odds_open_nt) || isempty(match_predict)
                continue
            end
            
            # --- 2. Loop over the markets we want to test ---
            for (market_key, complement_keys) in markets_to_test
                
                # Define the full group of keys for this market
                market_group_keys = [market_key, complement_keys...]
                
                # --- 3. Process Outcome (Y_outcome) ---
                if !haskey(market_results_nt, market_key)
                    continue # This match didn't have a result for this market
                end
                y_outcome = market_results_nt[market_key]::Bool

                # --- 4. Process OPENING Odds (P_open) ---
                odds_open_group = NamedTuple(
                    k => odds_open_nt[k] 
                    for k in market_group_keys if haskey(odds_open_nt, k)
                )
                if length(keys(odds_open_group)) != length(market_group_keys)
                    continue # This match didn't have all opening odds for this group
                end
                p_open_probs = vig_remove(odds_open_group)
                p_open = p_open_probs[market_key]

                # --- 5. Process MODEL Posteriors (P_model) ---
                p_model_dist = get_prob(get_posterior_samples(match_predict, market_group_keys), market_key)
                if isempty(p_model_dist)
                    continue # Model didn't produce posteriors for this group
                end
                
                # --- 6. Calculate Scores (using log_score) ---
                # A lower score is better
                s_open = log_score(p_open, y_outcome)
                s_model_dist = [log_score(p, y_outcome) for p in p_model_dist]
                
                # LSG = S_open - S_model (Positive means our score was *lower*, which is good)
                lsg_dist = s_open .- s_model_dist
                
                # --- 7. Store Results ---
                row = (
                    match_id = match_id,
                    market = market_key,
                    outcome = y_outcome,
                    p_open = p_open,
                    p_model_mean = mean(p_model_dist),
                    s_open = s_open,
                    s_model_mean = mean(s_model_dist),
                    lsg_mean = mean(lsg_dist),
                    prob_model_better = mean(lsg_dist .> 0.0),
                )
                push!(all_results, row)
            end # end market loop
            
        catch e
             println("  - ⚠️ WARNING: Skipping match $match_id due to error: $e")
        end
    end # end match loop
    end # end logger
    
    println("...Analysis complete. Found $(length(all_results)) valid market results.")
    return DataFrame(all_results)
end


# 1. Define the markets to test (can be the same as before)
markets_to_test = [
    # 1x2 Markets
    (:home, [:draw, :away]),
    (:draw, [:home, :away]),
    (:away, [:home, :draw]),
    
    # Over/Under 2.5
    (:over_25, [:under_25]),
    (:under_25, [:over_25]),
    
    # BTTS
    (:btts_yes, [:btts_no]),
    (:btts_no, [:btts_yes]),

    # Other Markets
    (:over_15, [:under_15]),
    (:under_15, [:over_15]),
    (:over_35, [:under_35]),
    (:under_35, [:over_35])
]

# 2. Run the new analysis function
lsg_report = run_lsg_analysis(
    all_oos_results,
    model,
    predict_config,
    ds,          # <-- Used for results
    ds_initial;  # <-- Used for opening odds
    markets_to_test=markets_to_test
)

# 3. See the results, grouped by market
if !isempty(lsg_report)
    # Group by market and get the average performance
    report_summary_lsg = combine(groupby(lsg_report, :market)) do g
        (
            n_matches = nrow(g),
            avg_lsg_mean = mean(g.lsg_mean), # <-- New metric
            avg_prob_better = mean(g.prob_model_better),
            avg_p_model = mean(g.p_model_mean),
            avg_p_open = mean(g.p_open),
            base_win_rate = mean(g.outcome) # <-- What was the real %?
        )
    end
    
    # Sort by the most accurate (highest LSG)
    sort!(report_summary_lsg, :avg_lsg_mean, rev=true)
    
    println("\n--- 📊 FINAL LSG REPORT (Grouped by Market) ---")
    println(report_summary_lsg)
else
    println("Analysis failed to find any valid matches.")
end


####
using DataFrames
using Statistics

# (This assumes 'binary_ig_report' is loaded in your session)

println("\n--- 📈 STARTING CONFIDENCE THRESHOLD ANALYSIS ---")

# 1. Define the confidence thresholds we want to test
#    (e.g., 50%, 60%, 70%, 80%, 90%)
thresholds_to_test = 0.5:0.1:0.9

all_threshold_reports = []

for thresh in thresholds_to_test
    
    # 2. Filter the report for *only* rows where our confidence is above the threshold
    filtered_df = filter(row -> row.prob_model_better > thresh, binary_ig_report)
    
    if isempty(filtered_df)
        println("\n--- Threshold: $(thresh * 100)% ---")
        println("No markets found with this level of confidence.")
        continue
    end
    
    # 3. Re-calculate our summary statistics on *only this filtered data*
    report_summary_filtered = combine(groupby(filtered_df, :market)) do g
        (
            n_matches_found = nrow(g), # How many bets passed our filter?
            avg_ig_mean = mean(g.ig_mean), # This is the profitability
            avg_prob_better = mean(g.prob_model_better) # Avg confidence
        )
    end
    
    # 4. Add the threshold to the report
    report_summary_filtered.threshold .= thresh
    push!(all_threshold_reports, report_summary_filtered)
end

# 5. Show the final results
if isempty(all_threshold_reports)
    println("\nNo bets were found above any threshold.")
else
    # Combine all reports into one big table
    final_threshold_report = vcat(all_threshold_reports...)
    
    # Sort by profitability (avg_ig_mean)
    sort!(final_threshold_report, :avg_ig_mean, rev=true)
    
    println("\n--- 📊 FINAL THRESHOLD REPORT (Sorted by Profitability) ---")
    println(final_threshold_report)
end


using DataFrames
using Statistics

# (This assumes 'lsg_report' is loaded in your session from the previous step)

println("\n--- 📈 STARTING LSG CONFIDENCE THRESHOLD ANALYSIS ---")

# 1. Define the confidence thresholds we want to test
thresholds_to_test_lsg = 0.5:0.1:0.9

all_lsg_threshold_reports = []

for thresh in thresholds_to_test_lsg
    
    # 2. Filter the report for *only* rows where our confidence is above the threshold
    filtered_df_lsg = filter(row -> row.prob_model_better > thresh, lsg_report)
    
    if isempty(filtered_df_lsg)
        println("\n--- LSG Threshold: $(thresh * 100)% ---")
        println("No markets found with this level of confidence.")
        continue
    end
    
    # 3. Re-calculate our summary statistics on *only this filtered data*
    report_summary_filtered_lsg = combine(groupby(filtered_df_lsg, :market)) do g
        (
            n_matches_found = nrow(g), # How many bets passed our filter?
            avg_lsg_mean = mean(g.lsg_mean), # This is the ACCURACY score
            avg_prob_better = mean(g.prob_model_better) # Avg confidence
        )
    end
    
    # 4. Add the threshold to the report
    report_summary_filtered_lsg.threshold .= thresh
    push!(all_lsg_threshold_reports, report_summary_filtered_lsg)
end

# 5. Show the final results
if isempty(all_lsg_threshold_reports)
    println("\nNo bets were found above any threshold.")
else
    # Combine all reports into one big table
    final_lsg_threshold_report = vcat(all_lsg_threshold_reports...)
    
    # Sort by accuracy (avg_lsg_mean)
    sort!(final_lsg_threshold_report, :avg_lsg_mean, rev=true)
    
    println("\n--- 📊 FINAL LSG THRESHOLD REPORT (Sorted by Accuracy) ---")
    println(final_lsg_threshold_report)
end



##### testing ig diff 

function calculate_pnl(stake::Number, odds::Number, winner::Bool)
    if stake <= 0.0
        return 0.0 # No bet was placed
    end
    
    if winner
        return stake * (odds - 1.0) # Profit
    else
        return -stake # Loss
    end
end



using DataFrames
using Statistics

# (This assumes 'binary_ig_report', 'ds', 'parse_fractional_to_decimal', 
#  and 'calculate_pnl' are all loaded in your session)

# --- 1. Define Our Trading Rule ---
const TRADE_TRIGGER = 0.121  # The minimum 'abs_disagreement' we found
const MARKET_TO_TEST = :over_25
const MARKET_GROUP_STR = "Match goals" # From your ds.odds
const CHOICE_GROUP_FLOAT = 2.5        # From your ds.odds

# --- 2. Get Our 31 "Winner" Signals ---
# These are the 31 rows we identified
trade_signals = filter(binary_ig_report) do row
    row.market == MARKET_TO_TEST &&
    abs(row.perceived_edge) > TRADE_TRIGGER
end

println("Found $(nrow(trade_signals)) trade signals to test...")

# --- 3. Filter ds.odds for just the markets we care about ---
# This makes the lookup much faster
odds_to_check = filter(ds.odds) do row
    !ismissing(row.winning) &&
    row.market_group == MARKET_GROUP_STR &&
    row.choice_group == CHOICE_GROUP_FLOAT
end

# Group by match_id for fast lookups
grouped_odds = groupby(odds_to_check, :match_id)

# --- 4. Loop Through Our Signals and Calculate PnL ---
pnl_results = []

for signal in eachrow(trade_signals)
    match_id = signal.match_id
    
    # Find the odds for this specific match
    if !haskey(grouped_odds, (match_id=match_id,))
        println("  - ⚠️ WARNING: No odds found for match $(match_id). Skipping.")
        continue
    end
    
    match_odds_df = grouped_odds[(match_id=match_id,)]
    
    # Determine which way to bet (Over or Under)
    bet_direction = (signal.p_model_mean > signal.p_open) ? "Over" : "Under"
    
    # Find the row for that specific bet
    bet_row = filter(row -> row.choice_name == bet_direction, match_odds_df)
    
    if isempty(bet_row)
        println("  - ⚠️ WARNING: No '$bet_direction' odds found for match $(match_id). Skipping.")
        continue
    end
    
    bet = first(bet_row)
    
    # --- Get the data for our PnL calculation ---
    stake = 1.0 # Use a 1-unit flat stake for simplicity
    
    # We MUST use the initial odds
    odds = parse_fractional_to_decimal(bet.initial_fractional_value)
    
    # The actual outcome!
    winner = bet.winning::Bool
    
    # Calculate the PnL for this one bet
    pnl = calculate_pnl(stake, odds, winner)
    
    push!(pnl_results, (
        match_id = match_id,
        bet_on = bet_direction,
        odds_paid = odds,
        winner = winner,
        pnl = pnl
    ))
end

# --- 5. Report The Final Results ---
results_df = DataFrame(pnl_results)

if isempty(results_df)
    println("No valid bets were found to calculate PnL.")
else
    total_pnl = sum(results_df.pnl)
    n_wins = sum(results_df.winner)
    n_bets = nrow(results_df)
    win_rate = n_wins / n_bets
    avg_odds = mean(results_df.odds_paid)
    
    println("\n--- 📊 SIMPLE PnL BACKTEST (1-Unit Stakes) ---")
    println("Market:             :$(MARKET_TO_TEST)")
    println("Trigger:            abs(perceived_edge) > $(TRADE_TRIGGER)")
    println("-------------------------------------------------")
    println("Total Bets Placed:  $(n_bets)")
    println("Total Wins:         $(n_wins)")
    println("Total Losses:       $(n_bets - n_wins)")
    println("Win Rate:           $(round(win_rate * 100, digits=2))%")
    println("Average Odds Paid:  $(round(avg_odds, digits=2))")
    println("\nTotal PnL:          $(round(total_pnl, digits=3)) units")
    println("ROI:                $(round((total_pnl / n_bets) * 100, digits=2))%")
    
    println("\n--- Full Bet List ---")
    # println(results_df)
end


## overfitting so running a split test 
