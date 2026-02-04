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



using Turing

data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)


# ----------------------------------------------
# 2. Experiment Configs Set up 
# ----------------------------------------------

# --- setup 1 
cv_config = BayesianFootball.Data.CVConfig(
    # tournament_ids = [56,57],
    tournament_ids = [56],
    target_seasons = ["22/23"],
    history_seasons = 0,
    dynamics_col = :match_week,
    # warmup_period = 36,
    warmup_period = 36,
    stop_early = true
)

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)
model = BayesianFootball.Models.PreGame.StaticPoisson() # place holder
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 
feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2) 
sampler_conf = Samplers.NUTSConfig(
                400,
                4,
                100,
                0.65,
                10,
  Samplers.UniformInit(-0.05, 0.05),
                :perchain,
)
training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)


# ----------------------------------------------
# 3. GRW models
# ----------------------------------------------

# A: ---  GRW Poisson

grw_poisson_model = Models.PreGame.GRWPoisson()

conf_poisson = Experiments.ExperimentConfig(
                    name = "grw poisson",
                    model = grw_poisson_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

results_poisson = Experiments.run_experiment(ds, conf_poisson)

describe(results_poisson.training_results[1][1]) 
df_trends_poisson = Models.PreGame.extract_trends(grw_poisson_model, feature_sets[end][1], results_poisson.training_results[end][1])

# B: ---  GRW Dixon Coles DC 

grw_dixoncoles_model = Models.PreGame.GRWDixonColes()

conf_dixoncoles = Experiments.ExperimentConfig(
                    name = "grw dixon coles",
                    model = grw_dixoncoles_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

results_dixoncoles = Experiments.run_experiment(ds, conf_dixoncoles)


describe(results_dixoncoles.training_results[1][1]) 
df_trends_dixoncoles = Models.PreGame.extract_trends(grw_dixoncoles_model, feature_sets[end][1], results_dixoncoles.training_results[end][1])


# C: ---  GRW Negative binomial 


grw_negbin_model = Models.PreGame.GRWNegativeBinomial()

conf_negbin = Experiments.ExperimentConfig(
                    name = "grw negative binomial",
                    model = grw_negbin_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

results_negbin = Experiments.run_experiment(ds, conf_negbin)


describe(results_negbin.training_results[1][1]) 
df_trends_negbin = Models.PreGame.extract_trends(grw_negbin_model, feature_sets[end][1], results_negbin.training_results[end][1])


# D: ---  GRW Bivariate Poisson  BP

grw_bipoisson_model = Models.PreGame.GRWBivariatePoisson()

conf_bipoisson = Experiments.ExperimentConfig(
                    name = "grw bivariate poisson",
                    model = grw_bipoisson_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

results_bipoisson = Experiments.run_experiment(ds, conf_bipoisson)


describe(results_bipoisson.training_results[1][1]) 
df_trends_bipoisson = Models.PreGame.extract_trends(grw_bipoisson_model, feature_sets[end][1], results_bipoisson.training_results[end][1])


# E : ---  GRW Negative binomial  phi - later  added 


grw_negbin_model = Models.PreGame.GRWNegativeBinomialPhi()

conf_negbin = Experiments.ExperimentConfig(
                    name = "grw negative binomial phi",
                    model = grw_negbin_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

results_negbin = Experiments.run_experiment(ds, conf_negbin)


describe(results_negbin.training_results[1][1]) 
df_trends_negbin = Models.PreGame.extract_trends(grw_negbin_model, feature_sets[end][1], results_negbin.training_results[end][1])


# F : --- GRW Negative binomial mu - 

grw_negbin_model = Models.PreGame.GRWNegativeBinomialMu()

conf_negbin = Experiments.ExperimentConfig(
                    name = "grw negative binomial mu",
                    model = grw_negbin_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

results_negbin = Experiments.run_experiment(ds, conf_negbin)

describe(results_negbin.training_results[1][1]) 

df_trends_negbin = Models.PreGame.extract_mu_trends(grw_negbin_model, feature_sets[end][1], results_negbin.training_results[end][1])

using Plots

# Ensure data is sorted by round (usually is, but good practice)
sort!(df_trends_negbin, :round)

# Extract vectors for plotting
x = df_trends_negbin.round
y = df_trends_negbin.mu_mean
lower = df_trends_negbin.mu_lower
upper = df_trends_negbin.mu_upper

# Create the plot
plot(
    x, y,
    ribbon = (y .- lower, upper .- y), # Plots.jl expects (distance_to_lower, distance_to_upper)
    fillalpha = 0.3,                   # Transparency of the ribbon
    color = :blue,
    lw = 2,                            # Line width of the mean
    label = "League Baseline (μ) Trend",
    title = "Time-Varying Global Goal Rate (μ)",
    xlabel = "Match Week",
    ylabel = "Log-Goal Rate (μ)",
    legend = :topright,
    grid = true,
    minorgrid = true
)

# Add a horizontal line for the static prior mean (0.2) to see the drift
hline!([0.2], label="Static Prior (0.2)", linestyle=:dash, color=:red)

# #

df = BayesianFootball.Models.PreGame.extract_trends(grw_negbin_model, feature_sets[end][1], results_negbin.training_results[end][1])

"""
    plot_all_teams_strength(df; teams_to_plot=nothing)

Plots trajectories using the 'Tab10' color palette.
"""
function plot_all_teams_strength(df::DataFrame; sym::Symbol = :total_att, teams_to_plot::Union{Vector{String}, Nothing}=nothing)
    
    # 1. Filter Data
    data_to_plot = if isnothing(teams_to_plot)
        df
    else
        filter(row -> row.team in teams_to_plot, df)
    end
    
    # Sort for correct line connecting
    sort!(data_to_plot, [:team, :round])

    # 2. Extract League Baseline (from the first available team)
    first_team = data_to_plot.team[1]
    baseline_df = filter(row -> row.team == first_team, data_to_plot)

    # 3. Create Plot with Tab10 Palette
    p = plot(
        size = (900, 600),
        title = "Total Attack Strength (League Adjusted)",
        xlabel = "Match Week",
        ylabel = "Total Log-Rate (μ + att)",
        legend = :outerright,
        margin = 5Plots.mm,
        palette = :tab10  # <--- SETS THE COLOUR SCHEME
    )

    # A. Plot League Baseline (Black Dash for contrast)
    plot!(p, baseline_df.round, baseline_df.mu_global, 
          label="League Avg (μ)", color=:black, lw=3, linestyle=:dash, alpha=0.5)

    # B. Plot Teams (Cycling through Tab10)
    plot!(p, 
          data_to_plot.round, 
          data_to_plot[!, sym], 
          group = data_to_plot.team, 
          lw = 2.5,
          alpha = 0.9
    )

    return p
end

# Usage:
# target_teams = ["airdrieonians", "queen-of-the-south", "falkirk"]
# plot_all_teams_strength(df, teams_to_plot=target_teams)

plot_all_teams_strength(df)
plot_all_teams_strength(df; sym=:def)

####
#
####

using BayesianFootball.Signals

baker = BayesianKelly()
my_signals = [baker]

ledger = BayesianFootball.BackTesting.run_backtest(ds, [results_poisson, results_negbin, results_bipoisson, results_dixoncoles], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)

ledger = BayesianFootball.BackTesting.run_backtest(ds, [results_negbin], my_signals; market_config = Data.Markets.DEFAULT_MARKET_CONFIG)

a = BayesianFootball.BackTesting.generate_tearsheet(ledger)



c=unique(a.selection)

for cc in c
  show(subset(a, :selection => ByRow(isequal(cc))))
end




#=
Needing to improve the calibration of the models, its been suggest that 
we use more of an empirical estimation for some of the choice of priors, 
namely μ and γ

=# 

##

using Statistics, Distributions

"""
    make_league_priors(df_train)

Calculates the 'Physics' of the league from raw data to set intelligent priors.
Returns NamedTuple with Normal distributions for μ and γ.
"""
function make_league_priors(df_train)
    # 1. Calculate Average Goals (The "Energy" of the league)
    # Total goals divided by total matches
    avg_goals_per_match = (sum(df_train.home_score) + sum(df_train.away_score)) / nrow(df_train)
    
    # 2. Calculate Home Advantage (Ratio)
    avg_h = mean(df_train.home_score)
    avg_a = mean(df_train.away_score)
    # Avoid division by zero in weird edge cases
    raw_home_adv = avg_a > 0 ? avg_h / avg_a : 1.3 

    println("\n--- ⚡ Data-Driven Priors Calculated ⚡ ---")
    println("  Avg Goals/Match: $(round(avg_goals_per_match, digits=3))")
    println("  Implied μ (team): $(round(log(avg_goals_per_match/2), digits=3))")
    println("  Home Adv Ratio:  $(round(raw_home_adv, digits=3))")

    # 3. Create Priors
    # We use log() because your model uses Log-Links
    # avg_goals_per_match = exp(μ_h) + exp(μ_a) ≈ 2 * exp(μ)
    # -> μ ≈ log(avg / 2)
    
    target_mu = log(avg_goals_per_match / 2.0)
    target_gamma = log(raw_home_adv)

    # We return Normal distributions centered on the truth, 
    # but with enough variance (0.2) to let the sampler adjust slightly.
    return (;
        prior_μ = Normal(target_mu, 0.05),
        prior_γ = Normal(target_gamma, 0.05)
    )
end



emperical_priors = make_league_priors(
        subset( ds.matches, :tournament_id => ByRow(x -> x ∈[56,57])) 
)


grw_negbin_model = Models.PreGame.GRWNegativeBinomial(
            μ = emperical_priors.prior_μ,  # <--- INJECTED HERE
            γ = emperical_priors.prior_γ,  # <--- INJECTED HERE
            σ_k = Gamma(2, 0.05), 
            σ_0 = Gamma(2, 0.05)
        )

conf_negbin = Experiments.ExperimentConfig(
                    name = "grw negative binomial",
                    model = grw_negbin_model,
                    splitter = cv_config,
                    training_config = training_config,
                    save_dir ="./data/junk"
)

results_negbin = Experiments.run_experiment(ds, conf_negbin)




describe(results_negbin.training_results[1][1]) 




symbols =[:μ, :γ, :σ_att, :σ_def] 

grw_negbin_model 
describe(results_negbin.training_results[1][1][symbols]) 


#= 
(prior_μ = Normal{Float64}(μ=0.3191977392306569, σ=0.25), prior_γ = Normal{Float64}(μ=0.12681652075405134, σ=0.25))
julia> describe(results_negbin.training_results[1][1][symbols]) 
Chains MCMC chain (500×4×4 Array{Float64, 3}):

Iterations        = 101:1:600
Number of chains  = 4
Samples per chain = 500
Wall duration     = 2563.68 seconds
Compute duration  = 9968.76 seconds
parameters        = μ, γ, σ_att, σ_def
internals         = 

Summary Statistics

  parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec 
      Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64 

           μ    0.1667    0.0686    0.0016   1886.2430   1390.2821    0.9999        0.1892
           γ    0.2072    0.0865    0.0020   1795.9024   1388.4144    1.0033        0.1802
       σ_att    0.0617    0.0252    0.0009    810.4998   1161.4447    1.0061        0.0813
       σ_def    0.0292    0.0175    0.0004   1496.8695   1474.1310    0.9999        0.1502


Quantiles

  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           μ    0.0335    0.1208    0.1671    0.2107    0.3011
           γ    0.0315    0.1502    0.2060    0.2626    0.3839
       σ_att    0.0157    0.0439    0.0604    0.0772    0.1140
       σ_def    0.0045    0.0161    0.0261    0.0391    0.0682


julia> grw_negbin_model 
   GRWNegativeBinomial
   -----------------
     μ ~ Normal{Float64}(μ=0.3191977392306569, σ=0.05)
     γ ~ Normal{Float64}(μ=0.12681652075405134, σ=0.05)
     log_r_prior ~ Normal{Float64}(μ=1.5, σ=1.0)
     σ_k ~ Gamma{Float64}(α=2.0, θ=0.05)
     σ_0 ~ Gamma{Float64}(α=2.0, θ=0.4)
     z_init ~ Normal{Float64}(μ=0.0, σ=1.0)
     z_steps ~ Normal{Float64}(μ=0.0, σ=1.0)


julia> describe(results_negbin.training_results[1][1][symbols]) 
Chains MCMC chain (200×4×4 Array{Float64, 3}):

Iterations        = 101:1:300
Number of chains  = 4
Samples per chain = 200
Wall duration     = 1320.62 seconds
Compute duration  = 5218.17 seconds
parameters        = μ, γ, σ_att, σ_def
internals         = 

Summary Statistics

  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   ess_per_sec 
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64       Float64 

           μ    0.2694    0.0373    0.0014   745.2661   657.4916    1.0063        0.1428
           γ    0.1218    0.0426    0.0016   745.1975   598.4214    1.0007        0.1428
       σ_att    0.0512    0.0245    0.0011   472.8216   696.0120    1.0045        0.0906
       σ_def    0.0297    0.0162    0.0006   811.2845   725.9000    1.0014        0.1555


Quantiles

  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           μ    0.2003    0.2446    0.2688    0.2946    0.3413
           γ    0.0412    0.0949    0.1223    0.1489    0.2045
       σ_att    0.0092    0.0338    0.0487    0.0668    0.1029
       σ_def    0.0053    0.0177    0.0274    0.0393    0.0629





julia> grw_negbin_model
   GRWNegativeBinomial
   -----------------
     μ ~ Normal{Float64}(μ=0.3191977392306569, σ=0.05)
     γ ~ Normal{Float64}(μ=0.12681652075405134, σ=0.05)
     log_r_prior ~ Normal{Float64}(μ=1.5, σ=1.0)
     σ_k ~ Gamma{Float64}(α=2.0, θ=0.05)
     σ_0 ~ Gamma{Float64}(α=2.0, θ=0.05)
     z_init ~ Normal{Float64}(μ=0.0, σ=1.0)
     z_steps ~ Normal{Float64}(μ=0.0, σ=1.0)


julia> describe(results_negbin.training_results[1][1][symbols])
Chains MCMC chain (200×4×4 Array{Float64, 3}):

Iterations        = 101:1:300
Number of chains  = 4
Samples per chain = 200
Wall duration     = 1260.87 seconds
Compute duration  = 4988.9 seconds
parameters        = μ, γ, σ_att, σ_def
internals         = 

Summary Statistics

  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   ess_per_sec 
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64       Float64 

           μ    0.2784    0.0373    0.0013   803.2344   657.3302    1.0000        0.1610
           γ    0.1252    0.0451    0.0015   909.4781   721.8158    1.0004        0.1823
       σ_att    0.0675    0.0221    0.0011   381.4785   579.9310    1.0071        0.0765
       σ_def    0.0348    0.0181    0.0008   485.8106   698.8495    1.0010        0.0974


Quantiles

  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           μ    0.2083    0.2520    0.2767    0.3031    0.3510
           γ    0.0384    0.0942    0.1244    0.1543    0.2161
       σ_att    0.0273    0.0524    0.0667    0.0829    0.1142
       σ_def    0.0064    0.0205    0.0329    0.0473    0.0757

=#


df_trends_negbin = Models.PreGame.extract_trends(grw_negbin_model, feature_sets[end][1], results_negbin.training_results[end][1])

