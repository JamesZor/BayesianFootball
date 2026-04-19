
using Revise
using BayesianFootball
using DataFrames
using JLD2
using Statistics



save_dir = "dev_exp/simple_dixon/"




data_store = BayesianFootball.Data.load_default_datastore()
model = BayesianFootball.Models.PreGame.StaticDixonColes()
vocabulary = BayesianFootball.Features.create_vocabulary(data_store, model)



# sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=600, n_chains=2, n_warmup=100) # Use renamed struct

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=200, n_chains=2, n_warmup=100) # Use renamed struct
strategy_parallel_custom = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=4) 
training_config_custom  = BayesianFootball.Training.TrainingConfig(sampler_conf, strategy_parallel_custom)

# seasons_to_train = ["20/21","21/22","22/23","23/24","24/25"]
seasons_to_train = ["24/25"]


for season_str in seasons_to_train

    println("Processing season $season_str.")

    # create the data set 

    # filter for one season for quick training
    df = filter(row -> row.season==season_str, data_store.matches)
    # we want to get the last 4 weeks - so added the game weeks
    df = BayesianFootball.Data.add_match_week_column(df)
    df.split_col = max.(0, df.match_week .- 14);

    ds = BayesianFootball.Data.DataStore(
        df,
        data_store.odds,
        data_store.incidents
    )

    ## Set the sets

    splitter_config = BayesianFootball.Data.ExpandingWindowCV([], [season_str], :split_col, :sequential) #
    data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
    feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #

    ## run  
    results = BayesianFootball.Training.train(model, training_config_custom, feature_sets)

    ## save 
    save_season_name_str = save_dir * "s_" * replace(season_str, "/" => "_") * ".jld2"
    
    JLD2.save_object(save_season_name_str, results)

    
println("Finished season $season_str.")



end 



season_to_load = seasons_to_train[1]

season_to_load_str = save_dir * "s_" * replace(season_to_load, "/" => "_") * ".jld2"

results_dixon = JLD2.load_object(season_to_load_str)

results_poisson = JLD2.load_object("training_results_large.jld2")

df = filter(row -> row.season==season_to_load, data_store.matches)
# we want to get the last 4 weeks - so added the game weeks
df = BayesianFootball.Data.add_match_week_column(df)
df.split_col = max.(0, df.match_week .- 14);

ds = BayesianFootball.Data.DataStore(
    df,
    data_store.odds,
    data_store.incidents
)

# here we want to use the open line odds
BayesianFootball.Data.DataPreprocessing.add_inital_odds_from_fractions!(data_store)
ds = data_store

split_col_name = :split_col
all_splits = sort(unique(ds.matches[!, split_col_name]))
prediction_split_keys = all_splits[2:end] 
grouped_matches = groupby(ds.matches, split_col_name)

dfs_to_predict = [
    grouped_matches[(; split_col_name => key)] 
    for key in prediction_split_keys
]

oos_dixon = BayesianFootball.Models.PreGame.extract_parameters(
    model,
    dfs_to_predict,  # Pass in the pre-split vector
    vocabulary,
    results_dixon
)


model_pos = BayesianFootball.Models.PreGame.StaticPoisson()
oos_poisson = BayesianFootball.Models.PreGame.extract_parameters(
    model_pos,
    dfs_to_predict,  # Pass in the pre-split vector
    vocabulary,
    results_poisson
)


predict_config = BayesianFootball.Predictions.PredictionConfig( BayesianFootball.Markets.get_standard_markets() )




match_id = rand(keys(oos_dixon))
r_dixon =  oos_dixon[match_id]
r_poisson =  oos_poisson[match_id]

subset( ds.matches, :match_id => ByRow(isequal(match_id)))


match_predict_dixon = BayesianFootball.Predictions.predict_market(model, predict_config, r_dixon...);
model_odds_poisson = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict_poisson));
model_odds_poisson

match_predict_poisson = BayesianFootball.Predictions.predict_market(model_pos, predict_config, r_poisson...);

model_odds_dixon = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict_dixon));
model_odds_dixon


model_odds_poisson = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict_poisson));
model_odds_poisson


open, close, results = BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds)


using StatsPlots
sym = :over_25
density( match_predict_dixon[sym], label="dixon")
density!( match_predict_poisson[sym], label="poisson")


sym = :draw
BayesianFootball.Signals.bayesian_kelly(match_predict_dixon, open)
BayesianFootball.Signals.bayesian_kelly(match_predict_poisson, open)




######


using Statistics
using Printf

"""
    summarize_chain(chain, market_odds)

Helper to calculate probability, fair odds, and edge from a posterior chain.
"""
function summarize_chain(chain, market_odds)
    # Calculate Model Probability (Post. Mean)
    model_prob = mean(chain)
    
    # Calculate Implied Probability from Market
    implied_prob = 1.0 / market_odds
    
    # Calculate Fair Odds (1 / Model Prob)
    model_odds = model_prob > 0 ? 1.0 / model_prob : Inf
    
    # Calculate Edge (Expected Value)
    # EV = (Probability * Decimal Odds) - 1
    ev = (model_prob * market_odds) - 1.0
    
    return model_prob, model_odds, ev, implied_prob
end

"""
    print_model_comparison(symbol, predict_dixon, predict_poisson, open_odds, close_odds, result, kelly_dixon, kelly_poisson)

Displays a detailed comparison block for a single market symbol (e.g., :home, :over_25).
"""
function print_model_comparison(symbol::Symbol, 
                                pred_dixon, pred_poisson, 
                                open_odds, close_odds, result,
                                kelly_dixon, kelly_poisson)

    # 1. Check if symbol exists in all datasets
    if !haskey(open_odds, symbol) || !haskey(pred_dixon, symbol)
        return # Skip if data missing
    end

    # 2. Get Market Context
    o_odds = open_odds[symbol]
    c_odds = close_odds[symbol]
    outcome_str = result[symbol] ? "WIN" : "LOSS"
    outcome_color = result[symbol] ? :green : :red

    # 3. Calculate Stats for Both Models
    prob_d, odds_d, edge_d, imp_p = summarize_chain(pred_dixon[symbol], o_odds)
    prob_p, odds_p, edge_p, _     = summarize_chain(pred_poisson[symbol], o_odds)

    # 4. Get Kelly Stakes (Handle cases where kelly returns 0.0 or missing)
    k_d = get(kelly_dixon, symbol, 0.0)
    k_p = get(kelly_poisson, symbol, 0.0)

    # --- RENDER OUTPUT ---
    printstyled("──────────────────────────────────────────────────────────────\n", color=:light_black)
    printstyled(@sprintf(" MARKET: :%-10s ", symbol), bold=true, color=:white)
    printstyled("RESULT: ", color=:white)
    printstyled("$outcome_str\n", bold=true, color=outcome_color)
    
    # Market Info
    @printf(" Market Open: %6.3f (Imp: %4.1f%%) | Close: %6.3f\n", o_odds, imp_p*100, c_odds)
    println()

    # Comparison Header
    printstyled(@sprintf(" %-12s | %-10s | %-10s | %-8s | %-8s\n", "Model", "Prob", "Fair Odds", "Edge", "Kelly"), color=:light_blue)
    println(" " * "-"^60)

    # Row: Dixon-Coles
    # Color code the edge: Green if > 0, Red if < 0
    d_color = edge_d > 0 ? :green : :light_black
    printstyled(@sprintf(" %-12s | %5.1f%%     | %6.3f     | ", "Dixon-Coles", prob_d*100, odds_d), color=:white)
    printstyled(@sprintf("%+5.1f%%", edge_d*100), color=d_color)
    printstyled(@sprintf("   | %5.2f%%\n", k_d*100), color=(k_d > 0 ? :yellow : :light_black))

    # Row: Poisson
    p_color = edge_p > 0 ? :green : :light_black
    printstyled(@sprintf(" %-12s | %5.1f%%     | %6.3f     | ", "Poisson", prob_p*100, odds_p), color=:white)
    printstyled(@sprintf("%+5.1f%%", edge_p*100), color=p_color)
    printstyled(@sprintf("   | %5.2f%%\n", k_p*100), color=(k_p > 0 ? :yellow : :light_black))
    println()
end

"""
    compare_all_markets(match_id, pred_dixon, pred_poisson, open, close, results, kelly_dixon, kelly_poisson)

Main function to loop through standard markets and display the dashboard.
"""
function compare_all_markets(match_id, 
                             pred_dixon, pred_poisson, 
                             open, close, results, 
                             kelly_dixon, kelly_poisson;
                             markets=[:home, :draw, :away, :over_25, :under_25, :btts_yes])
    
    printstyled("\n══════════════════════════════════════════════════════════════\n", color=:magenta)
    printstyled(@sprintf(" BAYESIAN MODEL COMPARISON | MATCH ID: %d \n", match_id), bold=true, color=:white)
    printstyled("══════════════════════════════════════════════════════════════\n", color=:magenta)

    for sym in markets
        print_model_comparison(sym, pred_dixon, pred_poisson, open, close, results, kelly_dixon, kelly_poisson)
    end
    printstyled("══════════════════════════════════════════════════════════════\n", color=:magenta)
end


##
#
match_id = rand(keys(oos_dixon))
r_dixon =  oos_dixon[match_id]
r_poisson =  oos_poisson[match_id]

subset( ds.matches, :match_id => ByRow(isequal(match_id)))


match_predict_dixon = BayesianFootball.Predictions.predict_market(model, predict_config, r_dixon...);
match_predict_poisson = BayesianFootball.Predictions.predict_market(model_pos, predict_config, r_poisson...);

model_odds_dixon = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict_dixon));
model_odds_dixon


model_odds_poisson = Dict(key => median(1 ./ value) for (key, value) in pairs(match_predict_poisson));
model_odds_poisson


open, close, results = BayesianFootball.Predictions.get_market_data(match_id, predict_config, ds.odds)


# 1. Calculate Kelly for both (you already did this in your history)
kelly_dixon_res   = BayesianFootball.Signals.bayesian_kelly(match_predict_dixon, open)
kelly_poisson_res = BayesianFootball.Signals.bayesian_kelly(match_predict_poisson, open)

# 2. Run the Comparison Dashboard
# You can customize the `markets` list to see just what you care about
compare_all_markets(
    match_id, 
    match_predict_dixon, 
    match_predict_poisson, 
    open, 
    close, 
    results, 
    kelly_dixon_res, 
    kelly_poisson_res;
    markets=[:home, :draw, :away, :over_25, :under_25, :btts_yes, :btts_no]
)


subset( ds.matches, :match_id => ByRow(isequal(match_id)))

###
using StatsPlots, Measures

"""
    plot_model_internals(r_dixon, r_poisson, match_id; title_text="")

Visualizes the posterior distributions of the Expected Goals (Lambda) 
to understand why the models disagree with the market.
"""
function plot_model_internals(r_dixon, r_poisson, match_id; title_text="")
    
    # 1. Extract Lambdas
    # r_dixon/poisson are NamedTuples: (λ_h=..., λ_a=..., ρ=...)
    d_lh, d_la = r_dixon.λ_h, r_dixon.λ_a
    p_lh, p_la = r_poisson.λ_h, r_poisson.λ_a
    
    # 2. Setup Layout
    l = @layout [a b; c]
    
    # --- PLOT A: Home Expected Goals (λ_h) ---
    p1 = density(d_lh, label="Dixon (Home)", color=:blue, fill=true, alpha=0.2, linewidth=2)
    density!(p1, p_lh, label="Poisson (Home)", color=:cyan, linestyle=:dash, linewidth=2)
    title!(p1, "Home Attack Strength (λ_h)")
    xlabel!(p1, "Expected Goals")
    ylabel!(p1, "Density")
    
    # --- PLOT B: Away Expected Goals (λ_a) ---
    p2 = density(d_la, label="Dixon (Away)", color=:red, fill=true, alpha=0.2, linewidth=2)
    density!(p2, p_la, label="Poisson (Away)", color=:orange, linestyle=:dash, linewidth=2)
    title!(p2, "Away Attack Strength (λ_a)")
    xlabel!(p2, "Expected Goals")
    
    # --- PLOT C: Expected Goal Difference (λ_h - λ_a) ---
    # This acts as a proxy for the Handicap / Match Winner confidence
    d_diff = d_lh .- d_la
    p_diff = p_lh .- p_la
    
    p3 = density(d_diff, label="Dixon Margin", color=:purple, fill=true, alpha=0.2, linewidth=2)
    density!(p3, p_diff, label="Poisson Margin", color=:magenta, linestyle=:dash, linewidth=2)
    
    # Add a vertical line at 0 (Draw line)
    vline!(p3, [0.0], color=:black, label="Draw Line", linewidth=1.5)
    
    title!(p3, "Expected Goal Difference (Home - Away)")
    xlabel!(p3, "< Away Wins   |   Home Wins >")
    
    # Final Plot Construction
    plot(p1, p2, p3, layout=l, size=(1000, 700), margin=5mm,
         plot_title="Model Internals Match ID: $match_id $title_text")
end


plot_model_internals(r_dixon, r_poisson, match_id, title_text="| Arbroath vs Kelty Hearts")


using DataFrames, PrettyTables, Statistics

function compare_probabilities(model_dixon, model_poisson, match_id, predicted_df_dixon, predicted_df_poisson)
    
    # 1. Get the specific match data
    r_dixon = predicted_df_dixon[match_id]
    rho_val = mean(r_dixon.ρ) 
    
    println("------------------------------------------------")
    println("DIXON-COLES DIAGNOSTIC | Match ID: $match_id")
    println("------------------------------------------------")
    println("Correlation Parameter (ρ): $(round(rho_val, digits=4))")
    
    if abs(rho_val) < 0.05
        printstyled("ℹ Low Correlation: Dixon-Coles is acting almost exactly like Poisson.\n", color=:yellow)
    else
        printstyled("✓ Significant Correlation: DC is adjusting low scores.\n", color=:green)
    end
    println()
    
    # 2. Compare the Expected Goals (The "Engine")
    lambda_h_d = mean(r_dixon.λ_h)
    lambda_a_d = mean(r_dixon.λ_a)
    
    lambda_h_p = mean(predicted_df_poisson[match_id].λ_h)
    lambda_a_p = mean(predicted_df_poisson[match_id].λ_a)

    data = [
        "Home Exp. Goals (λ_h)" lambda_h_p lambda_h_d (lambda_h_d - lambda_h_p);
        "Away Exp. Goals (λ_a)" lambda_a_p lambda_a_d (lambda_a_d - lambda_a_p)
    ]
    
    # FIX: Pass headers as the second argument (Positional), not as a keyword
    headers = ["Metric", "Poisson", "Dixon-Coles", "Diff"]
    pretty_table(data, headers)
    
    println("\nConclusion:")
    if abs(lambda_h_d - lambda_h_p) < 0.1
        println("The models have nearly identical opinions on team strength.")
        println("The failure was not the distribution choice (Poisson vs DC),")
        println("but the rating of the teams themselves.")
    end
end
using DataFrames, PrettyTables, Statistics

function compare_probabilities(model_dixon, model_poisson, match_id, predicted_df_dixon, predicted_df_poisson)
    
    # 1. Get the specific match data
    r_dixon = predicted_df_dixon[match_id]
    rho_val = mean(r_dixon.ρ) 
    
    println("------------------------------------------------")
    println("DIXON-COLES DIAGNOSTIC | Match ID: $match_id")
    println("------------------------------------------------")
    println("Correlation Parameter (ρ): $(round(rho_val, digits=4))")
    
    if abs(rho_val) < 0.05
        printstyled("ℹ Low Correlation: Dixon-Coles is acting almost exactly like Poisson.\n", color=:yellow)
    else
        printstyled("✓ Significant Correlation: DC is adjusting low scores.\n", color=:green)
    end
    println()
    
    # 2. Compare the Expected Goals (The "Engine")
    lambda_h_d = mean(r_dixon.λ_h)
    lambda_a_d = mean(r_dixon.λ_a)
    
    lambda_h_p = mean(predicted_df_poisson[match_id].λ_h)
    lambda_a_p = mean(predicted_df_poisson[match_id].λ_a)

    # FIX: Use a DataFrame. This avoids the Matrix/Header syntax ambiguity.
    df_compare = DataFrame(
        Metric = ["Home Exp. Goals (λ_h)", "Away Exp. Goals (λ_a)"],
        Poisson = [lambda_h_p, lambda_a_p],
        DixonColes = [lambda_h_d, lambda_a_d],
        Diff = [(lambda_h_d - lambda_h_p), (lambda_a_d - lambda_a_p)]
    )
    
    # PrettyTables handles DataFrames automatically
    pretty_table(df_compare)
    
    println("\nConclusion:")
    if abs(lambda_h_d - lambda_h_p) < 0.1
        println("The models have nearly identical opinions on team strength.")
        println("The failure was not the distribution choice (Poisson vs DC),")
        println("but the rating of the teams themselves.")
    end
end


# Usage:
compare_probabilities(model, model_pos, match_id, oos_dixon, oos_poisson)



using DataFrames, Dates, Printf

"""
    check_recent_form(match_id, matches_df)

Prints the last 6 games for Home and Away teams leading up to the target match
to visually check for form drifts that the Static model missed.
"""
function check_recent_form(match_id, matches_df)
    # 1. Get Target Match Details
    target = subset(matches_df, :match_id => ByRow(==(match_id)))
    if nrow(target) == 0
        println("Match ID not found.")
        return
    end
    
    t_date = target.match_date[1]
    h_team = target.home_team[1]
    a_team = target.away_team[1]
    
    println("---------------------------------------------------------")
    printstyled(" FORM GUIDE: $h_team vs $a_team \n", bold=true, color=:white)
    println(" Target Match Date: $t_date")
    println("---------------------------------------------------------")
    
    # 2. Define Helper to get last N games
    function get_last_n(team, date, N=6)
        # Find games where team played either home or away, BEFORE target date
        mask = ((matches_df.home_team .== team) .| (matches_df.away_team .== team)) .& 
               (matches_df.match_date .< date)
        
        hist = sort(matches_df[mask, :], :match_date, rev=true)
        return first(hist, N)
    end

    # 3. Print Home Team Form
    h_form = get_last_n(h_team, t_date)
    printstyled("\n HOME: $h_team (Last 6)\n", color=:cyan)
    println(" Date       | Opponent        | Result | Scored | Conceded")
    println("------------|-----------------|--------|--------|----------")
    
    for row in eachrow(h_form)
        is_home = row.home_team == h_team
        opp = is_home ? row.away_team : row.home_team
        scr = is_home ? row.home_score : row.away_score
        conc = is_home ? row.away_score : row.home_score
        res = scr > conc ? "W" : (scr == conc ? "D" : "L")
        
        # Color code result
        c = res == "W" ? :green : (res == "D" ? :yellow : :red)
        
        @printf(" %s | %-15s | ", row.match_date, first(opp, 15))
        printstyled("$res     ", color=c)
        @printf("| %d      | %d\n", scr, conc)
    end

    # 4. Print Away Team Form
    a_form = get_last_n(a_team, t_date)
    printstyled("\n AWAY: $a_team (Last 6)\n", color=:magenta)
    println(" Date       | Opponent        | Result | Scored | Conceded")
    println("------------|-----------------|--------|--------|----------")
    
    for row in eachrow(a_form)
        is_home = row.home_team == a_team
        opp = is_home ? row.away_team : row.home_team
        scr = is_home ? row.home_score : row.away_score
        conc = is_home ? row.away_score : row.home_score
        res = scr > conc ? "W" : (scr == conc ? "D" : "L")
        
        c = res == "W" ? :green : (res == "D" ? :yellow : :red)
        
        @printf(" %s | %-15s | ", row.match_date, first(opp, 15))
        printstyled("$res     ", color=c)
        @printf("| %d      | %d\n", scr, conc)
    end
    println()
end


check_recent_form(match_id, ds.matches)
