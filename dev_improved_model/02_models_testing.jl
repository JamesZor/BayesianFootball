using Revise
using BayesianFootball
using DataFrames

using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)


ds = Data.load_extra_ds()
transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)


cv_config = BayesianFootball.Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons = ["24/25"],
    history_seasons = 3,
    dynamics_col = :match_month,
    warmup_period = 9,
    stop_early = true
)



# -------------------------------------
#  MS kappa
# -------------------------------------

model = BayesianFootball.Models.PreGame.MSNegativeBinomialKappa()

splits = BayesianFootball.Data.create_data_splits(ds, cv_config)

feature_sets = BayesianFootball.Features.create_features(
    splits, model, cv_config
)

splits = BayesianFootball.Data.create_data_splits(ds, cv_config) 
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=2)  

sampler_conf = Samplers.NUTSConfig(                                                                                                                                                                                                                                     
                       200,                                                                                                                                                                                                                                                    
                       16,                                                                                                                                                                                                                                                     
                       100,                                                                                                                                                                                                                                                    
                       0.65,                                                                                                                                                                                                                                                   
                       10,                                                                                                                                                                                                                                                     
         Samplers.UniformInit(-1, 1),                                                                                                                                                                                                                                      
                       :perchain,                                                                                                                                                                                                                                              
       )

training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false) 

conf_basic = Experiments.ExperimentConfig(                                                                                                                                                                                                                              
                           name = "multi_basic_test",                                                                                                                                                                                                                          
                           model = model,                                                                                                                                                                                                                                      
                           splitter = cv_config,                                                                                                                                                                                                                               
                           training_config = training_config,                                                                                                                                                                                                                  
                           save_dir ="./data/junk"                                                                                                                                                                                                                             
       )  

results_basic = Experiments.run_experiment(ds, conf_basic)    



c = results_basic.training_results[1][1]  

typeof(c)


describe(c)

#=
                                                                                                                                                                                                                   
Summary Statistics                                                                                                                                                                                                 
                                                                                                                                                                                                                   
    parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec                                                                                                                       
        Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64                                                                                                                       
                                                                                                                                                                                                                   
             μ    0.1307    0.0505    0.0009   3380.4520   2714.2905    0.9994        0.0180                                                                                                                       
             γ    0.1360    0.0320    0.0006   3237.9329   2561.4171    1.0025        0.0173                                                                                                                       
         log_r    3.3476    0.3157    0.0057   3201.4945   2117.9919    1.0082        0.0171                                                                                                                       
         δₘ[1]   -0.0257    0.0538    0.0010   3126.6653   2759.6589    1.0027        0.0167                                                                                                                       
         δₘ[2]   -0.0499    0.0516    0.0009   3472.3087   2874.4275    1.0018        0.0185                                                                                                                       
         δₘ[3]   -0.0425    0.0504    0.0009   3069.3742   2688.0708    1.0016        0.0164                                                                                                                       
         δₘ[4]   -0.0278    0.0497    0.0009   3231.4724   3103.6598    1.0000        0.0172                                                                                                                       
         δₘ[5]    0.0498    0.0814    0.0015   3077.4184   2140.3797    1.0028        0.0164                                                                                                                       
         δₘ[6]   -0.0006    0.0995    0.0017   3430.7052   2097.9790    1.0063        0.0183                                                                                                                       
         δₘ[7]   -0.0874    0.0843    0.0015   3059.9509   2261.4831    1.0071        0.0163                                                                                                                       
         δₘ[8]   -0.0193    0.0533    0.0009   3246.2276   2930.9046    1.0022        0.0173                                                                                                                       
         δₘ[9]    0.0401    0.0547    0.0010   3247.5775   2536.5448    1.0023        0.0173                                                                                                                       
        δₘ[10]    0.0404    0.0556    0.0010   3036.7051   2431.2697    1.0037        0.0162                                                                                                                       
        δₘ[11]   -0.0058    0.0550    0.0009   3389.1915   2982.0973    0.9993        0.0181                                                                                                                       
        δₘ[12]    0.1217    0.0519    0.0009   3436.3575   2937.9814    1.0040        0.0183                                                                                                                       
            δₙ    0.0093    0.0493    0.0008   3548.2617   2176.7070    1.0029        0.0189                                                                                                                       
            δₚ    0.1039    0.0422    0.0007   3332.2456   3174.9045    1.0025        0.0178                                                                                                                       
          α.σ₀    0.2097    0.0477    0.0013   1366.0218   2277.6677    1.0107        0.0073                                                                                                                       
          α.σₛ    0.0920    0.0304    0.0008   1376.7357   1163.9546    1.0107        0.0073                                                                                                                       
          α.σₖ    0.0230    0.0176    0.0004   2045.4544   1673.3945    1.0023        0.0109                                                                                                                       
   α.z_init[1]    1.6626    0.5307    0.0094   3221.0170   2653.8971    1.0030        0.0172



=#


using Plots
using StatsPlots
using MCMCChains

# 1. Set backend and create directory
plotlyjs()
mkpath("figures")

# Define the chain variable explicitly (change 'c' to whatever your chain is named)
my_chain = c 

println("Generating plots... (This may take a moment)")

# ==============================================================================
# PLOT 1: Global Baselines and Single-Flag Modifiers
# ==============================================================================
base_syms = [Symbol("μ"), Symbol("γ"), Symbol("log_r"), Symbol("δₙ"), Symbol("δₚ")]
base_chain = my_chain[base_syms]

p_base = plot(
    base_chain, 
    size=(1600, 1000), 
    # title="Global Hyperparameters (Baselines & Flags)",
    margin=5Plots.mm
);
savefig(p_base, "figures/mcmc_global_hyperparams.html")
println("Saved: figures/mcmc_global_hyperparams.html")


# ==============================================================================
# PLOT 2: Monthly Modifiers (Using the `group` function)
# ==============================================================================
# `group` automatically grabs all δₘ[1] through δₘ[12]
month_chain = group(my_chain, :δₘ)

p_month = plot(
    month_chain, 
    size=(1200, 1800), # Made taller to fit 12 parameters neatly
    title="Monthly Calendar Modifiers (δₘ)",
    margin=5Plots.mm
);
savefig(p_month, "figures/mcmc_monthly_effects.html")
println("Saved: figures/mcmc_monthly_effects.html")


# ==============================================================================
# PLOT 3: The Multi-Scale Volatilities (Sigmas)
# ==============================================================================
sigma_syms = [
    Symbol("α.σ₀"), Symbol("α.σₛ"), Symbol("α.σₖ"),
    Symbol("β.σ₀"), Symbol("β.σₛ"), Symbol("β.σₖ")
]
sigma_chain = my_chain[sigma_syms]

p_sigmas = plot(
    sigma_chain, 
    size=(1600, 1600), 
    title="Multi-Scale Variances (Macro vs Micro)",
    margin=5Plots.mm
);
savefig(p_sigmas, "figures/mcmc_sigmas.html")
println("Saved: figures/mcmc_sigmas.html")

println("All plots successfully generated!")

