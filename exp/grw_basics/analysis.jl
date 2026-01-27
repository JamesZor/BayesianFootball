# exp/grw_basics/analysis.jl

using Revise
using BayesianFootball
using DataFrames
using BayesianFootball.Signals

# Load DataStore again (Data is lightweight, models are heavy)
data_store = BayesianFootball.Data.load_default_datastore()
ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)

# 1. Load Experiments from Disk
# =============================
exp_dir = "./data/exp/grw_basics"
println("Scanning for results in: $exp_dir")

# This helper lists the folders it finds
saved_folders = Experiments.list_experiments("exp/grw_basics"; data_dir="./data")
saved_folders = Experiments.list_experiments("exp/grw_basics_pl_ch"; data_dir="./data")

# Load them all into a list
loaded_results = Vector{BayesianFootball.Experiments.ExperimentResults}([])
for folder in saved_folders
    try
        res = Experiments.load_experiment(folder)
        push!(loaded_results, res)
    catch e
        @warn "Could not load $folder: $e"
    end
end

if isempty(loaded_results)
    error("No results loaded! Did you run runner.jl?")
end

# 2. Run Backtest
# ===============
println("\nRunning Backtest on $(length(loaded_results)) models...")

baker = BayesianKelly()
my_signals = [baker]

# Run backtest on ALL loaded models at once
ledger = BayesianFootball.BackTesting.run_backtest(
    ds, 
    loaded_results, 
    my_signals; 
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)

# 3. Analyze
# ==========
tearsheet = BayesianFootball.BackTesting.generate_tearsheet(ledger)

println("\n=== TEARSHEET SUMMARY ===")
println(tearsheet)

# Breakdown by Model (Selection)
println("\n=== BREAKDOWN BY MODEL ===")
model_names = unique(tearsheet.selection)

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub)
end



# 4. turing 
using Turing

symbols =[:μ, :γ, :σ_att, :σ_def] 

for m in loaded_results 
  println("\n Model: $(m.config.name) \n")
  println( 
describe(m.training_results[1][1][symbols])
)
end 

using StatsPlots
plot(loaded_results[1].training_results[end][1][:μ])
      

describe(loaded_results[1].training_results[end][1][symbols])
describe(loaded_results[2].training_results[end][1][symbols])
describe(loaded_results[3].training_results[end][1][symbols])
describe(loaded_results[4].training_results[end][1][symbols])


#= 
 Model: grw_neg_bin                                                                                                  
                                                                                                                     
Chains MCMC chain (250×4×2 Array{Float64, 3}):                                                                       
                                                                                                                     
Iterations        = 51:1:300                                                                                         
Number of chains  = 2                                                                                                
Samples per chain = 250                                                                                              
Wall duration     = 249.56 seconds                                                                                   
Compute duration  = 497.66 seconds                                                                                   
parameters        = μ, γ, σ_att, σ_def                                                                               
internals         =                                                                                                  
                                                                                                                     
Summary Statistics                                                                                                   
                                                                                                                     
  parameters      mean       std      mcse   ess_bulk   ess_tail      rhat   ess_per_sec                             
      Symbol   Float64   Float64   Float64    Float64    Float64   Float64       Float64                             
                                                                                                                     
           μ    0.2693    0.0869    0.0042   436.3733   311.9187    1.0174        0.8769                             
           γ    0.1701    0.1108    0.0050   496.8303   317.8725    1.0023        0.9983                             
       σ_att    0.0461    0.0266    0.0012   486.8139   360.3088    1.0023        0.9782                             
       σ_def    0.0604    0.0355    0.0017   411.4484   352.9079    0.9980        0.8268                             
                                                                                                                     
                                                                                                                     
Quantiles                                                                                                            
                                                                                                                     
  parameters      2.5%     25.0%     50.0%     75.0%     97.5%                                                       
      Symbol   Float64   Float64   Float64   Float64   Float64                                                       
                                                                                                                     
           μ    0.1074    0.2116    0.2675    0.3225    0.4525                                                       
           γ   -0.0474    0.1028    0.1709    0.2416    0.4049                                                       
       σ_att    0.0086    0.0260    0.0430    0.0601    0.1070                                                       
       σ_def    0.0081    0.0339    0.0541    0.0810    0.1442   



=# 


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
        prior_μ = Normal(target_mu, 0.25),
        prior_γ = Normal(target_gamma, 0.25)
    )
end



using DataFrames
dd = subset( ds.matches, :tournament_id => ByRow(x -> x ∈[56,57])) 
ddd = subset( ds.matches, :tournament_id => ByRow(x -> x ∈[54,55])) 

make_league_priors(dd)
make_league_priors(ddd)

names(ds.matches)

