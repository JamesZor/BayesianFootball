# ----------------------------------------------
# 1. The set up 
# ----------------------------------------------
using Revise
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)


# -----
data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)

# --- setup 1 
cv_config = BayesianFootball.Data.CVConfig(
    # tournament_ids = [56,57],
    tournament_ids = [55],
    target_seasons = ["22/23"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
  warmup_period = 34,
  # warmup_period = 15,
    # stop_early = false
    stop_early = true
)

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)
model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 
feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 
sampler_conf = Samplers.NUTSConfig(
                100,
                2,
                100,
                0.65,
                10,
  Samplers.UniformInit(-0.05, 0.05)
)
training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

# --- set up 2 

cv_config = BayesianFootball.Data.CVConfig(
    # tournament_ids = [56,57],
    tournament_ids = [55],
    target_seasons = ["22/23"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
    warmup_period = 34,
    stop_early = true
)

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)
model = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 
feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 
sampler_conf = Samplers.NUTSConfig(
                200,
                2,
                100,
                0.65,
                10,
  Samplers.UniformInit(-0.05, 0.05)
)
training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

## --- GRW Dixon coles
grw_dixoncoles_model = Models.PreGame.GRWDixonColes()

exp_conf_grw_dc = Experiments.ExperimentConfig(
                    name = "grw poisson",
                    model = grw_dixoncoles_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

grw_dc_results = Experiments.run_experiment(ds, exp_conf_grw_dc)

using Turing

describe(grw_dc_results.training_results[1][1]) 


fset = feature_sets[end][1]
chain = grw_dc_results.training_results[end][1]
df_trends = Models.PreGame.extract_trends(grw_dixoncoles_model, fset, chain)
Models.PreGame.extract_trends(grw_poisson_model, fset, chain)


using BayesianFootball.Signals

baker = BayesianKelly()
my_signals = [baker]

ledger = BayesianFootball.BackTesting.run_backtest(ds, [grw_dc_results], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)
ledger = BayesianFootball.BackTesting.run_backtest(ds, [grw_dc_results, grw_poisson_results], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)
ledger = BayesianFootball.BackTesting.run_backtest(ds, [grw_poisson_results, shp_results], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)


a = BayesianFootball.BackTesting.generate_tearsheet(ledger)



## --- GRW poisson 

grw_poisson_model = Models.PreGame.GRWPoisson()

exp_conf_grw = Experiments.ExperimentConfig(
                    name = "grw poisson",
                    model = grw_poisson_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

grw_poisson_results = Experiments.run_experiment(ds, exp_conf_grw)


using Turing

describe(grw_poisson_results.training_results[1][1]) 



fset = feature_sets[end][1]
chain = grw_poisson_results.training_results[end][1]
df_trends = Models.PreGame.extract_trends(grw_poisson_model, fset, chain)
Models.PreGame.extract_trends(grw_poisson_model, fset, chain)



using Plots, StatsPlots

# 1. Plot Attack Strengths
@df df_trends plot(:round, :att, group=:team, 
    title="Evolution of Attack Strength",
    xlabel="Round", ylabel="Attack (Log Scale)",
    legend=:outertopright, lw=2
)


# 2. Plot Defense Strengths
@df df_trends plot(:round, :def, group=:team, 
    title="Evolution of Defense Strength",
    xlabel="Round", ylabel="Defense (Log Scale)",
    legend=:outertopright, lw=2
)

# Plot trajectory of one team (e.g., Raith Rovers)
unique(df_trends.team)

team_data = filter(row -> row.team == "dundee-fc", df_trends)

team_data_1= filter(row -> row.team == "ayr-united", df_trends)
team_data_2= filter(row -> row.team == "partick-thistle", df_trends)

# 1. Start the plot with the first team (Dundee FC)
plot(team_data.att, team_data.def, 
    label="Dundee FC",
    title="Tactical Evolution: Attack vs Defense",
    xlabel="Attack Strength", ylabel="Defense Strength",
    marker=:circle, arrow=true, lw=2, 
    legend=:outertopright,  # Position the legend outside
    yflip=true              # OPTIONAL: Flip Y axis if lower defense is "better"
)

# 2. Add the second team (Ayr United) to the SAME plot
plot!(team_data_1.att, team_data_1.def, 
    label="Ayr United",
    marker=:circle, arrow=true, lw=2
)

plot!(team_data_2.att, team_data_2.def, 
    label="partick-thistle",
    marker=:circle, arrow=true, lw=2
)



using BayesianFootball.Signals

baker = BayesianKelly()
my_signals = [baker]

ledger = BayesianFootball.BackTesting.run_backtest(ds, [grw_poisson_results], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)
ledger = BayesianFootball.BackTesting.run_backtest(ds, [grw_poisson_results, shp_results], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)


a = BayesianFootball.BackTesting.generate_tearsheet(ledger)



## -- mix copula model 

mix_copula_model = Models.PreGame.StaticMixtureCopula()


exp_conf = Experiments.ExperimentConfig(
                    name = "mixture copula",
                    model = mix_copula_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

mix_copula_results = Experiments.run_experiment(ds, exp_conf)



using BayesianFootball.Signals

baker = BayesianKelly()
my_signals = [baker]

ledger = BayesianFootball.BackTesting.run_backtest(ds, [mix_copula_results, DC], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)


a = BayesianFootball.BackTesting.generate_tearsheet(ledger)

c=unique(a.selection)

for cc in c
  show(subset(a, :selection => ByRow(isequal(cc))))
end

latents_mc = Experiments.extract_oos_predictions(ds, mix_copula_results)
latents = Experiments.extract_oos_predictions(ds, mvpln_results)
latents_bp = Experiments.extract_oos_predictions(ds, BP_results)
latents_shp = Experiments.extract_oos_predictions(ds, shp_results)
latents_dc = Experiments.extract_oos_predictions(ds, DC)


a = Predictions.model_inference(latents)
b = Predictions.model_inference(latents_bp)
c = Predictions.model_inference(latents_shp)
d = Predictions.model_inference(latents_mc)
e = Predictions.model_inference(latents_dc)



fs = feature_sets[1]
daaa = Data.get_next_matches(ds, fs, cv_config)
mid = daaa[4,:match_id]
market_data = Data.prepare_market_data(ds)

subset(market_data.df, :match_id => ByRow(isequal(mid)))
subset(ds.matches, :match_id => ByRow(isequal(mid)))

using StatsPlots

sym = :draw
a1 = subset( a.df, :selection => ByRow(isequal(sym)), :match_id => ByRow(isequal(mid)))[1, :]
b1 = subset( b.df, :selection => ByRow(isequal(sym)), :match_id => ByRow(isequal(mid)))[1, :]
c1 = subset( c.df, :selection => ByRow(isequal(sym)), :match_id => ByRow(isequal(mid)))[1, :]
d1 = subset( d.df, :selection => ByRow(isequal(sym)), :match_id => ByRow(isequal(mid)))[1, :]
mean(1 ./ a1.distribution)
mean(1 ./ b1.distribution)
mean(1 ./ c1.distribution)
mean(1 ./ d1.distribution)
mean(a1.distribution)
mean(b1.distribution)
mean(c1.distribution)
mean(d1.distribution)
subset(market_data.df, :match_id => ByRow(isequal(mid)), :selection => ByRow(isequal(sym)))

density(a1.distribution, title="$sym",label="mvpln")
density!(b1.distribution, label="bivariatePoisson")
density!(c1.distribution, label="Heirarical")
density!(d1.distribution, label="mix copula - clayton + frank")

#- 

compare_models_to_market(mid, :under_35, market_data, a, b, c, d, e)

using DataFrames, StatsPlots, Statistics, PrettyTables

function compare_models_to_market(
    mid::Int, 
    sym::Symbol, 
    market_data, 
    # Pass your model objects here. I've defaulted names to match your snippet
    models...; 
    # Map your model objects to the specific labels you want in the legend/table
    labels = ["mvpln", "bivariatePoisson", "Hierarchical", "mix copula - clayton + frank", "dixoncoles"]
)

    # 1. Setup containers
    # We will store results in a DataFrame for the "nice table"
    results_df = DataFrame(
        Source = String[], 
        Mean_Prob = Float64[], 
        Mean_Odds = Float64[],
        Type = String[]
    )
    
    # Initialize the plot
    p = density(
        title="Posterior Distributions: $sym (Match $mid)", 
        xlabel="Probability", 
        ylabel="Density", 
        legend=:outertopright,  # Places legend outside the plot area
        size=(800, 500)         # Increases width to accommodate the external legend
    )

    # 2. Loop through the models provided
    for (i, model_obj) in enumerate(models)
        # Handle cases where user might pass more models than labels
        lbl = get(labels, i, "Model $i")
        
        # Access the dataframe (assuming structure is model.df)
        # If your models are just DataFrames, remove the `.df` access
        df = hasproperty(model_obj, :df) ? model_obj.df : model_obj

        # Filter Logic
        row_subset = subset(df, 
            :selection => ByRow(isequal(sym)), 
            :match_id => ByRow(isequal(mid))
        )

        if isempty(row_subset)
            @warn "No data found for $lbl in match $mid"
            continue
        end

        # Extract the distribution array
        # row_subset[1, :] gets the first row
        dist = row_subset[1, :distribution]

        # Calculate Statistics based on your logic
        # Odds = mean(1 ./ distribution)
        # Prob = mean(distribution)
        mean_odds = mean(1 ./ dist)
        mean_prob = mean(dist)

        # Update Table
        push!(results_df, (lbl, mean_prob, mean_odds, "Model"))

        # Update Plot
        # We use i==1 to determine if we start the plot or append to it, 
        # but StatsPlots handles `density!` well if the plot object `p` is passed.
        density!(p, dist, label=lbl, linewidth=2, alpha=0.7)
    end

    # 3. Process Market Data
    mkt_df = hasproperty(market_data, :df) ? market_data.df : market_data
    
    mkt_row = subset(mkt_df, 
        :match_id => ByRow(isequal(mid)), 
        :selection => ByRow(isequal(sym))
    )

    if !isempty(mkt_row)
        # Extract Market Open
        odds_open = mkt_row[1, :odds_open]
        prob_open = mkt_row[1, :prob_implied_open]
        push!(results_df, ("Market Open", prob_open, odds_open, "Market"))
        
        # Extract Market Close
        odds_close = mkt_row[1, :odds_close]
        prob_close = mkt_row[1, :prob_implied_close]
        push!(results_df, ("Market Close", prob_close, odds_close, "Market"))

        # Optional: Add vertical lines to the plot for Market Implied Probability
        vline!(p, [prob_open], label="Mkt Open", linestyle=:dash, color=:black)
        vline!(p, [prob_close], label="Mkt Close", linestyle=:dot, color=:gray)
    end

    # 4. Display Results
    
    # Print the table nicely
    println("\n--- Comparison for Selection: $sym ---")
    pretty_table(results_df)

    # Return the plot object so it displays
    return p
end



using Turing

r = mix_copula_results.training_results[1][1]
describe(r) 

"""
julia> describe(r) 
Chains MCMC chain (300×396×2 Array{Float64, 3}):

Iterations        = 201:1:500
Number of chains  = 2
Samples per chain = 300
Wall duration     = 519.01 seconds
Compute duration  = 884.4 seconds
parameters        = μ, γ, σ_att, σ_def, σ_h, σ_a, w[1], w[2], w[3], ρ_raw, θ_clay_log, θ_frank, αₛ[1], αₛ[2], αₛ[3], αₛ[4], αₛ[5], αₛ[6], αₛ[7], αₛ[8], αₛ[9], αₛ[10], βₛ[1], βₛ[2], βₛ[3], βₛ[4], βₛ[5], βₛ[6], βₛ[7], βₛ[8], βₛ[9], βₛ[10], ϵ_h_raw[1], ϵ_h_raw[2], ϵ_h_raw[3], ϵ_h_raw[4], ϵ_h_raw[5], ϵ_h_raw[6], ϵ_h_raw[7], ϵ_h_raw[8], ϵ_h_raw[9], ϵ_h_raw[10], ϵ_h_raw[11], ϵ_h_raw[12], ϵ_h_raw[13], ϵ_h_raw[14], ϵ_h_raw[15], ϵ_h_raw[16], ϵ_h_raw[17], ϵ_h_raw[18], ϵ_h_raw[19], ϵ_h_raw[20], ϵ_h_raw[21], ϵ_h_raw[22], ϵ_h_raw[23], ϵ_h_raw[24], ϵ_h_raw[25], ϵ_h_raw[26], ϵ_h_raw[27], ϵ_h_raw[28], ϵ_h_raw[29], ϵ_h_raw[30], ϵ_h_raw[31], ϵ_h_raw[32], ϵ_h_raw[33], ϵ_h_raw[34], ϵ_h_raw[35], ϵ_h_raw[36], ϵ_h_raw[37], ϵ_h_raw[38], ϵ_h_raw[39], ϵ_h_raw[40], ϵ_h_raw[41], ϵ_h_raw[42], ϵ_h_raw[43], ϵ_h_raw[44], ϵ_h_raw[45], ϵ_h_raw[46], ϵ_h_raw[47], ϵ_h_raw[48], ϵ_h_raw[49], ϵ_h_raw[50], ϵ_h_raw[51], ϵ_h_raw[52], ϵ_h_raw[53], ϵ_h_raw[54], ϵ_h_raw[55], ϵ_h_raw[56], ϵ_h_raw[57], ϵ_h_raw[58], ϵ_h_raw[59], ϵ_h_raw[60], ϵ_h_raw[61], ϵ_h_raw[62], ϵ_h_raw[63], ϵ_h_raw[64], ϵ_h_raw[65], ϵ_h_raw[66], ϵ_h_raw[67], ϵ_h_raw[68], ϵ_h_raw[69], ϵ_h_raw[70], ϵ_h_raw[71], ϵ_h_raw[72], ϵ_h_raw[73], ϵ_h_raw[74], ϵ_h_raw[75], ϵ_h_raw[76], ϵ_h_raw[77], ϵ_h_raw[78], ϵ_h_raw[79], ϵ_h_raw[80], ϵ_h_raw[81], ϵ_h_raw[82], ϵ_h_raw[83], ϵ_h_raw[84], ϵ_h_raw[85], ϵ_h_raw[86], ϵ_h_raw[87], ϵ_h_raw[88], ϵ_h_raw[89], ϵ_h_raw[90], ϵ_h_raw[91], ϵ_h_raw[92], ϵ_h_raw[93], ϵ_h_raw[94], ϵ_h_raw[95], ϵ_h_raw[96], ϵ_h_raw[97], ϵ_h_raw[98], ϵ_h_raw[99], ϵ_h_raw[100], ϵ_h_raw[101], ϵ_h_raw[102], ϵ_h_raw[103], ϵ_h_raw[104], ϵ_h_raw[105], ϵ_h_raw[106], ϵ_h_raw[107], ϵ_h_raw[108], ϵ_h_raw[109], ϵ_h_raw[110], ϵ_h_raw[111], ϵ_h_raw[112], ϵ_h_raw[113], ϵ_h_raw[114], ϵ_h_raw[115], ϵ_h_raw[116], ϵ_h_raw[117], ϵ_h_raw[118], ϵ_h_raw[119], ϵ_h_raw[120], ϵ_h_raw[121], ϵ_h_raw[122], ϵ_h_raw[123], ϵ_h_raw[124], ϵ_h_raw[125], ϵ_h_raw[126], ϵ_h_raw[127], ϵ_h_raw[128], ϵ_h_raw[129], ϵ_h_raw[130], ϵ_h_raw[131], ϵ_h_raw[132], ϵ_h_raw[133], ϵ_h_raw[134], ϵ_h_raw[135], ϵ_h_raw[136], ϵ_h_raw[137], ϵ_h_raw[138], ϵ_h_raw[139], ϵ_h_raw[140], ϵ_h_raw[141], ϵ_h_raw[142], ϵ_h_raw[143], ϵ_h_raw[144], ϵ_h_raw[145], ϵ_h_raw[146], ϵ_h_raw[147], ϵ_h_raw[148], ϵ_h_raw[149], ϵ_h_raw[150], ϵ_h_raw[151], ϵ_h_raw[152], ϵ_h_raw[153], ϵ_h_raw[154], ϵ_h_raw[155], ϵ_h_raw[156], ϵ_h_raw[157], ϵ_h_raw[158], ϵ_h_raw[159], ϵ_h_raw[160], ϵ_h_raw[161], ϵ_h_raw[162], ϵ_h_raw[163], ϵ_h_raw[164], ϵ_h_raw[165], ϵ_h_raw[166], ϵ_h_raw[167], ϵ_h_raw[168], ϵ_h_raw[169], ϵ_h_raw[170], ϵ_h_raw[171], ϵ_h_raw[172], ϵ_h_raw[173], ϵ_h_raw[174], ϵ_h_raw[175], ϵ_a_raw[1], ϵ_a_raw[2], ϵ_a_raw[3], ϵ_a_raw[4], ϵ_a_raw[5], ϵ_a_raw[6], ϵ_a_raw[7], ϵ_a_raw[8], ϵ_a_raw[9], ϵ_a_raw[10], ϵ_a_raw[11], ϵ_a_raw[12], ϵ_a_raw[13], ϵ_a_raw[14], ϵ_a_raw[15], ϵ_a_raw[16], ϵ_a_raw[17], ϵ_a_raw[18], ϵ_a_raw[19], ϵ_a_raw[20], ϵ_a_raw[21], ϵ_a_raw[22], ϵ_a_raw[23], ϵ_a_raw[24], ϵ_a_raw[25], ϵ_a_raw[26], ϵ_a_raw[27], ϵ_a_raw[28], ϵ_a_raw[29], ϵ_a_raw[30], ϵ_a_raw[31], ϵ_a_raw[32], ϵ_a_raw[33], ϵ_a_raw[34], ϵ_a_raw[35], ϵ_a_raw[36], ϵ_a_raw[37], ϵ_a_raw[38], ϵ_a_raw[39], ϵ_a_raw[40], ϵ_a_raw[41], ϵ_a_raw[42], ϵ_a_raw[43], ϵ_a_raw[44], ϵ_a_raw[45], ϵ_a_raw[46], ϵ_a_raw[47], ϵ_a_raw[48], ϵ_a_raw[49], ϵ_a_raw[50], ϵ_a_raw[51], ϵ_a_raw[52], ϵ_a_raw[53], ϵ_a_raw[54], ϵ_a_raw[55], ϵ_a_raw[56], ϵ_a_raw[57], ϵ_a_raw[58], ϵ_a_raw[59], ϵ_a_raw[60], ϵ_a_raw[61], ϵ_a_raw[62], ϵ_a_raw[63], ϵ_a_raw[64], ϵ_a_raw[65], ϵ_a_raw[66], ϵ_a_raw[67], ϵ_a_raw[68], ϵ_a_raw[69], ϵ_a_raw[70], ϵ_a_raw[71], ϵ_a_raw[72], ϵ_a_raw[73], ϵ_a_raw[74], ϵ_a_raw[75], ϵ_a_raw[76], ϵ_a_raw[77], ϵ_a_raw[78], ϵ_a_raw[79], ϵ_a_raw[80], ϵ_a_raw[81], ϵ_a_raw[82], ϵ_a_raw[83], ϵ_a_raw[84], ϵ_a_raw[85], ϵ_a_raw[86], ϵ_a_raw[87], ϵ_a_raw[88], ϵ_a_raw[89], ϵ_a_raw[90], ϵ_a_raw[91], ϵ_a_raw[92], ϵ_a_raw[93], ϵ_a_raw[94], ϵ_a_raw[95], ϵ_a_raw[96], ϵ_a_raw[97], ϵ_a_raw[98], ϵ_a_raw[99], ϵ_a_raw[100], ϵ_a_raw[101], ϵ_a_raw[102], ϵ_a_raw[103], ϵ_a_raw[104], ϵ_a_raw[105], ϵ_a_raw[106], ϵ_a_raw[107], ϵ_a_raw[108], ϵ_a_raw[109], ϵ_a_raw[110], ϵ_a_raw[111], ϵ_a_raw[112], ϵ_a_raw[113], ϵ_a_raw[114], ϵ_a_raw[115], ϵ_a_raw[116], ϵ_a_raw[117], ϵ_a_raw[118], ϵ_a_raw[119], ϵ_a_raw[120], ϵ_a_raw[121], ϵ_a_raw[122], ϵ_a_raw[123], ϵ_a_raw[124], ϵ_a_raw[125], ϵ_a_raw[126], ϵ_a_raw[127], ϵ_a_raw[128], ϵ_a_raw[129], ϵ_a_raw[130], ϵ_a_raw[131], ϵ_a_raw[132], ϵ_a_raw[133], ϵ_a_raw[134], ϵ_a_raw[135], ϵ_a_raw[136], ϵ_a_raw[137], ϵ_a_raw[138], ϵ_a_raw[139], ϵ_a_raw[140], ϵ_a_raw[141], ϵ_a_raw[142], ϵ_a_raw[143], ϵ_a_raw[144], ϵ_a_raw[145], ϵ_a_raw[146], ϵ_a_raw[147], ϵ_a_raw[148], ϵ_a_raw[149], ϵ_a_raw[150], ϵ_a_raw[151], ϵ_a_raw[152], ϵ_a_raw[153], ϵ_a_raw[154], ϵ_a_raw[155], ϵ_a_raw[156], ϵ_a_raw[157], ϵ_a_raw[158], ϵ_a_raw[159], ϵ_a_raw[160], ϵ_a_raw[161], ϵ_a_raw[162], ϵ_a_raw[163], ϵ_a_raw[164], ϵ_a_raw[165], ϵ_a_raw[166], ϵ_a_raw[167], ϵ_a_raw[168], ϵ_a_raw[169], ϵ_a_raw[170], ϵ_a_raw[171], ϵ_a_raw[172], ϵ_a_raw[173], ϵ_a_raw[174], ϵ_a_raw[175]
internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, lp, logprior, loglikelihood

Summary Statistics

   parameters      mean       std      mcse    ess_bulk   ess_tail      rhat   ess_per_sec 
       Symbol   Float64   Float64   Float64     Float64    Float64   Float64       Float64 

            μ    0.1261    0.0750    0.0036    424.5671   439.3956    1.0091        0.4801
            γ    0.2489    0.0920    0.0036    671.1483   484.0819    1.0041        0.7589
        σ_att    0.2547    0.1044    0.0069    208.5552   311.9514    1.0129        0.2358
        σ_def    0.1675    0.0844    0.0060    188.3569   202.3586    1.0053        0.2130
          σ_h    0.2095    0.1189    0.0095    154.0646   216.9092    1.0196        0.1742
          σ_a    0.2508    0.1329    0.0147     82.9380   255.5413    1.0380        0.0938
         w[1]    0.3475    0.2409    0.0322     64.1951   137.3689    1.0202        0.0726
         w[2]    0.3214    0.2207    0.0239     89.3817   165.1454    1.0151        0.1011
         w[3]    0.3311    0.2277    0.0477     23.0887   416.0095    1.0676        0.0261
        ρ_raw   -0.0363    1.0029    0.3382      9.2237    22.9975    1.1496        0.0104
   θ_clay_log    0.4977    0.4887    0.0284    298.6871   358.7761    1.0222        0.3377
      θ_frank   -0.0947    1.9913    0.1132    302.9141   388.7093    1.0121        0.3425
        αₛ[1]   -1.1956    0.6656    0.0333    397.0236   201.9984    1.0055        0.4489
        αₛ[2]    0.5905    0.6181    0.0215    766.7647   438.4549    1.0069        0.8670
        αₛ[3]   -0.5685    0.6382    0.0235    742.0724   416.6616    1.0037        0.8391
        αₛ[4]    0.7456    0.5982    0.0237    618.0846   420.1744    0.9981        0.6989
        αₛ[5]    0.1866    0.6218    0.0228    755.3919   471.3031    1.0013        0.8541
        αₛ[6]   -1.0681    0.7427    0.0263    812.3838   414.2036    1.0006        0.9186
        αₛ[7]    0.1693    0.6272    0.0226    778.4521   512.7486    1.0047        0.8802
        αₛ[8]    0.8443    0.6543    0.0279    551.8980   442.3282    1.0002        0.6240
        αₛ[9]    0.6851    0.6165    0.0283    492.3156   319.9525    1.0021        0.5567
       αₛ[10]   -0.1986    0.6147    0.0224    751.2776   512.5033    1.0033        0.8495
        βₛ[1]   -0.1628    0.7159    0.0238    874.5946   402.8195    1.0049        0.9889
        βₛ[2]   -0.3536    0.7828    0.0312    645.7571   480.8662    1.0040        0.7302
        βₛ[3]    1.3320    0.7753    0.0434    335.6051   364.8838    1.0006        0.3795
        βₛ[4]   -0.6184    0.7285    0.0270    749.5289   464.0568    1.0001        0.8475
        βₛ[5]   -0.4101    0.7466    0.0354    456.7828   315.7874    1.0042        0.5165
        βₛ[6]    0.8338    0.7225    0.0282    655.3348   421.1801    1.0081        0.7410
        βₛ[7]   -0.1582    0.7523    0.0277    730.8700   290.2757    1.0008        0.8264
        βₛ[8]   -0.2723    0.7660    0.0237   1024.6102   458.8754    1.0028        1.1585
        βₛ[9]   -0.0050    0.7800    0.0270    814.7932   464.6109    0.9999        0.9213
       βₛ[10]   -0.0299    0.7308    0.0305    575.1932   330.4897    1.0014        0.6504
   ϵ_h_raw[1]    0.0878    0.9773    0.0362    735.5372   539.2328    1.0012        0.8317
   ϵ_h_raw[2]    0.0046    0.9367    0.0413    511.2556   418.1347    0.9987        0.5781
   ϵ_h_raw[3]   -0.1592    0.9689    0.0450    462.7154   331.7737    1.0003        0.5232
   ϵ_h_raw[4]    0.0808    0.9948    0.0452    493.0681   458.2842    1.0050        0.5575
   ϵ_h_raw[5]   -0.3075    1.0102    0.0326    943.0134   493.5380    0.9995        1.0663
   ϵ_h_raw[6]    0.0529    0.9767    0.0335    846.6050   463.8996    1.0039        0.9573
   ϵ_h_raw[7]   -0.2180    0.9674    0.0451    478.7649   362.0066    1.0035        0.5413
   ϵ_h_raw[8]   -0.3071    0.9195    0.0436    459.4566   309.6527    1.0097        0.5195
   ϵ_h_raw[9]   -0.2919    0.8638    0.0334    672.7690   546.8736    1.0041        0.7607
  ϵ_h_raw[10]   -0.2384    0.9677    0.0330    869.1112   480.3437    1.0056        0.9827
  ϵ_h_raw[11]    0.5197    0.9956    0.0402    637.5115   388.0973    1.0041        0.7208
  ϵ_h_raw[12]    0.1498    0.9530    0.0335    804.9696   514.0621    0.9981        0.9102
            ⋮         ⋮         ⋮         ⋮           ⋮          ⋮         ⋮             ⋮

                                                                            338 rows omitted

Quantiles

   parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
       Symbol   Float64   Float64   Float64   Float64   Float64 

            μ   -0.0181    0.0760    0.1268    0.1802    0.2633
            γ    0.0869    0.1830    0.2483    0.3106    0.4288
        σ_att    0.0938    0.1803    0.2419    0.3139    0.4931
        σ_def    0.0335    0.1111    0.1617    0.2077    0.3640
          σ_h    0.0076    0.1157    0.2083    0.3005    0.4359
          σ_a    0.0331    0.1409    0.2458    0.3481    0.5090
         w[1]    0.0090    0.1445    0.2976    0.5248    0.8447
         w[2]    0.0186    0.1355    0.2864    0.4745    0.8052
         w[3]    0.0138    0.1432    0.2959    0.4963    0.8177
        ρ_raw   -1.9467   -0.7418   -0.0559    0.6335    1.8379
   θ_clay_log   -0.4237    0.1820    0.4904    0.8184    1.4803
      θ_frank   -3.8662   -1.3560   -0.0805    1.1801    3.9269
        αₛ[1]   -2.5703   -1.5921   -1.1953   -0.7033    0.0195
        αₛ[2]   -0.6271    0.1998    0.6031    0.9643    1.8008
        αₛ[3]   -1.8069   -1.0226   -0.5413   -0.1614    0.6130
        αₛ[4]   -0.3333    0.3418    0.7139    1.0972    2.0160
        αₛ[5]   -1.0472   -0.2170    0.1977    0.5842    1.4188
        αₛ[6]   -2.6492   -1.5374   -1.0636   -0.5638    0.2645
        αₛ[7]   -1.0743   -0.2313    0.1636    0.6174    1.4786
        αₛ[8]   -0.3583    0.4240    0.7930    1.2616    2.0508
        αₛ[9]   -0.4948    0.2968    0.6556    1.0985    1.8132
       αₛ[10]   -1.4548   -0.5923   -0.1719    0.2263    0.9645
        βₛ[1]   -1.6746   -0.5499   -0.1768    0.2721    1.2359
        βₛ[2]   -1.9401   -0.8326   -0.3627    0.1399    1.2275
        βₛ[3]   -0.3746    0.8679    1.3548    1.8479    2.8257
        βₛ[4]   -1.9203   -1.1266   -0.6237   -0.1508    0.8727
        βₛ[5]   -1.8466   -0.9113   -0.3712    0.0642    1.0850
        βₛ[6]   -0.5173    0.3510    0.8137    1.3010    2.3404
        βₛ[7]   -1.7171   -0.6012   -0.1568    0.2695    1.3759
        βₛ[8]   -1.7567   -0.7483   -0.2798    0.2521    1.2768
        βₛ[9]   -1.6025   -0.5003   -0.0541    0.4928    1.5815
       βₛ[10]   -1.5337   -0.4771   -0.0214    0.4483    1.3854
   ϵ_h_raw[1]   -1.8222   -0.5933    0.0835    0.7783    1.9448
   ϵ_h_raw[2]   -1.8462   -0.6239   -0.0085    0.6216    1.8718
   ϵ_h_raw[3]   -2.0733   -0.8903   -0.1020    0.4673    1.7531
   ϵ_h_raw[4]   -1.8502   -0.5621    0.0314    0.7100    2.0951
   ϵ_h_raw[5]   -2.3853   -0.9066   -0.2776    0.3602    1.5631
   ϵ_h_raw[6]   -1.9271   -0.5212    0.0529    0.7057    1.9647
   ϵ_h_raw[7]   -2.1643   -0.7722   -0.2255    0.4066    1.6939
   ϵ_h_raw[8]   -2.1715   -0.8806   -0.2891    0.2920    1.4794
   ϵ_h_raw[9]   -1.8796   -0.8640   -0.2915    0.2983    1.3176
  ϵ_h_raw[10]   -2.2368   -0.8291   -0.2692    0.4346    1.7517
  ϵ_h_raw[11]   -1.5298   -0.1069    0.5616    1.1343    2.4096
  ϵ_h_raw[12]   -1.6630   -0.5011    0.1730    0.7587    1.9981
            ⋮         ⋮         ⋮         ⋮         ⋮         ⋮
                                                 338 rows omitted
"""

###


DC_model = Models.PreGame.DixonColesNCP(
            )


exp_conf_DC = Experiments.ExperimentConfig(
                    name = "DC",
                    model = DC_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

DC = Experiments.run_experiment(ds, exp_conf_DC)


##
mvpln_model = Models.PreGame.StaticMVPLN()

exp_conf_mvpln = Experiments.ExperimentConfig(
                    name = "mvpln",
                    model = mvpln_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

mvpln_results = Experiments.run_experiment(ds, exp_conf_mvpln)

using Turing

r = mvpln_results.training_results[1][1]

describe(r)

### 
#
shp_model = Models.PreGame.StaticHierarchicalPoisson()

experiment_conf_shp = Experiments.ExperimentConfig(
                    name = "static poisson",
                    model = shp_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

shp_results = Experiments.run_experiment(ds, experiment_conf_shp)

describe(shp_results.training_results[1][1])


model_BP = Models.PreGame.BivariatePoissonNCP()

experiment_conf_BP = Experiments.ExperimentConfig(
                    name = "BivariatePoissonNCP v3 ",
                    model = model_BP,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

BP_results = Experiments.run_experiment(ds, experiment_conf_BP)

describe( BP_results.training_results[1][1])



######

using BayesianFootball.Signals

baker = BayesianKelly()
my_signals = [baker]

ledger = BayesianFootball.BackTesting.run_backtest(ds, [mvpln_results], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)
ledger = BayesianFootball.BackTesting.run_backtest(ds, [mvpln_results, shp_results, BP_results], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)

ledger = BayesianFootball.BackTesting.run_backtest(ds, [DC, exp_results_2], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)
ledger = BayesianFootball.BackTesting.run_backtest(ds, [DC], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)

a = BayesianFootball.BackTesting.generate_tearsheet(ledger)

c=unique(a.selection)

for cc in c
  show(subset(a, :selection => ByRow(isequal(cc))))
end


subset(ds.matches, :match_week => ByRow(isequal(35)), :tournament_id => ByRow(isequal(55)))


latents = Experiments.extract_oos_predictions(ds, mvpln_results)
latents_bp = Experiments.extract_oos_predictions(ds, BP_results)
latents_shp = Experiments.extract_oos_predictions(ds, shp_results)


a = Predictions.model_inference(latents)
b = Predictions.model_inference(latents_bp)
c = Predictions.model_inference(latents_shp)



daaa = Data.get_next_matches(ds, fs, cv_config)
mid = daaa[3,:match_id]
market_data = Data.prepare_market_data(ds)

subset(market_data.df, :match_id => ByRow(isequal(mid)))
subset(ds.matches, :match_id => ByRow(isequal(mid)))

using StatsPlots

sym = :draw
a1 = subset( a.df, :selection => ByRow(isequal(sym)), :match_id => ByRow(isequal(mid)))[1, :]
b1 = subset( b.df, :selection => ByRow(isequal(sym)), :match_id => ByRow(isequal(mid)))[1, :]
c1 = subset( c.df, :selection => ByRow(isequal(sym)), :match_id => ByRow(isequal(mid)))[1, :]
mean(1 ./ a1.distribution)
mean(1 ./ b1.distribution)
mean(1 ./ c1.distribution)
mean(a1.distribution)
mean(b1.distribution)
mean(c1.distribution)
subset(market_data.df, :match_id => ByRow(isequal(mid)), :selection => ByRow(isequal(sym)))

density(a1.distribution, title="$sym",label="mvpln")
density!(b1.distribution, label="bi")
density!(c1.distribution, label="shp")




symbols = [:model_name, :model_parameters, :signal_name, :signal_params]

a = BayesianFootball.BackTesting.generate_tearsheet(ledger; groupby_cols=symbols)

a2 = DC.training_results[1][1]
using Turing
describe(a2)

model_2 = Models.PreGame.StaticHierarchicalPoisson()

experiment_conf_2 = Experiments.ExperimentConfig(
                    name = "test_static_hierarchicalPoisson",
                    model = model_2,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

exp_results_2 = Experiments.run_experiment(ds, experiment_conf_2)



model_3 = Models.PreGame.BivariatePoissonNCP(
            )

experiment_conf_3 = Experiments.ExperimentConfig(
                    name = "BivariatePoissonNCP v3 ",
                    model = model_3,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

exp_results_3 = Experiments.run_experiment(ds, experiment_conf_3)


a = exp_results_2.training_results[1][1]
describe(a)

