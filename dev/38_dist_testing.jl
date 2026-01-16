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
    tournament_ids = [56],
    target_seasons = ["22/23"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
  warmup_period = 36,
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
                100,
                2,
                10,
                0.65,
                10,
  Samplers.MapInit(50)
)


sampler_conf1 = Samplers.NUTSConfig(
                100,
                2,
                10,
                0.65,
                10,
  Samplers.UniformInit(-0.05, 0.05)
)



training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

training_config1 = Training.TrainingConfig(sampler_conf1, train_cfg, nothing, false)

## --- GRW Negative binomial 
grw_negbin_model = Models.PreGame.GRWNegativeBinomial()


exp_conf_grw_nb = Experiments.ExperimentConfig(
                    name = "grw nb ",
                    model = grw_negbin_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

grw_nb_results = Experiments.run_experiment(ds, exp_conf_grw_nb)


using Turing

describe(grw_nb_results.training_results[1][1]) 

df_trends = Models.PreGame.extract_trends(grw_negbin_model, feature_sets[end][1], grw_nb_results.training_results[end][1])


using Plots, StatsPlots

# 1. Plot Attack Strengths
@df df_trends plot(:round, :att, group=:team, 
    title="Evolution of Attack Strength",
    xlabel="Round", ylabel="Attack (Log Scale)",
    legend=:outertopright, lw=2,
    palette = :tab10
)


# 2. Plot Defense Strengths
@df df_trends plot(:round, :def, group=:team, 
    title="Evolution of Defense Strength",
    xlabel="Round", ylabel="Defense (Log Scale)",
    legend=:outertopright, lw=2,
    palette = :tab10
)


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


exp_conf_grw1 = Experiments.ExperimentConfig(
                    name = "grw poisson",
                    model = grw_poisson_model,
                    splitter = cv_config,
                    training_config = training_config1,
                    save_dir ="./data/junk"
)

grw_poisson_results1 = Experiments.run_experiment(ds, exp_conf_grw1)


using Turing

describe(grw_poisson_results.training_results[1][1]) 
describe(grw_poisson_results1.training_results[1][1]) 

a = grw_poisson_results.training_results[1][1]
a1 = grw_poisson_results1.training_results[1][1]
describe(a[:is_accept])
describe(a1[:is_accept])

describe(a[:step_size])
describe(a1[:step_size])

describe(a[:tree_depth])
describe(a1[:tree_depth])


describe(a[:hamiltonian_energy])
describe(a1[:hamiltonian_energy])


describe(a[:max_hamiltonian_energy_error])
describe(a1[:max_hamiltonian_energy_error])




fset = feature_sets[end][1]
chain = grw_poisson_results.training_results[end][1]
df_trends = Models.PreGame.extract_trends(grw_poisson_model, fset, chain)
Models.PreGame.extract_trends(grw_poisson_model, fset, chain)

using Turing

describe(chain)


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



flat_strat = FlatStake(0.05)
# 2. Conservative Kelly: Quarter Kelly (0.25)
kelly_strat = KellyCriterion(0.25)

# 3. Bayesian/Shrinkage Kelly: Uses the Baker-McHale analytical approximation
shrink_strat = AnalyticalShrinkageKelly()

baker = BayesianKelly()

my_signals = [flat_strat, kelly_strat, shrink_strat, baker]

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

latents = Experiments.extract_oos_predictions(ds, grw_poisson_results)


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

sym = :away
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

compare_models_to_market(mid, :away, market_data, a)

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



## static negative bino 
neg_bi_model = Models.PreGame.StaticDoubleNegBin()


exp_conf_dnb = Experiments.ExperimentConfig(
                    name = "neg_bi_model",
                    model = neg_bi_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

dnb = Experiments.run_experiment(ds, exp_conf_dnb)


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

describe(dnb.training_results[2][1])
describe(shp_results.training_results[2][1])

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

ledger = BayesianFootball.BackTesting.run_backtest(ds, [dnb], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)



ledger = BayesianFootball.BackTesting.run_backtest(ds, [dnb, shp_results, BP_results,DC], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)


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

