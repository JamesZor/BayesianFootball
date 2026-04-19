"""
In this dev file we want to test a frame work to edit the parameters of the models, namley the priors 
and or if we can pass the distributions through without effecting the reversediff tape size etc. 

Here we can use the julia base.kwags on the struct for the models. 

First we whill look at the static model. 

 --- old verison:

# pre-game model - StaticPoisson 
    struct StaticPoisson <: AbstractStaticPoissonModel end

@model function static_poisson_model_train(n_teams, home_ids, away_ids, 
                                    home_goals, away_goals, ::Type{T} = Float64) where {T}
        # --- Priors ---
        log_α_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
        log_β_raw ~ MvNormal(zeros(n_teams), 0.5 * I)
        home_adv ~ Normal(log(1.3), 0.2)

 --- updates 

# CHANGE 1: Update the struct to hold configuration

Base.@kwdef struct StaticPoisson <: AbstractStaticPoissonModel 
  sigma::Float64 = 0.5 
end 




# CHANGE 2: update the struct to hold distributions

using Distributions

# Use a type parameter `D` to keep it type-stable, or just `Distribution` if you don't mind a tiny bit of dynamic dispatch.
Base.@kwdef struct GRWPoisson{D<:Distribution} <: AbstractDynamicPoissonModel 

Base.@kwdef struct StaticPoisson{D<:Distribution} <: AbstractStaticPoissonModel
    # We just store a standard Normal. 
    # filldist will handle the 'n_teams' part later.
    prior::D = Normal(0, 0.5) 
end

in the model:
      log_α_raw ~ filldist(model.prior, n_teams)


or  "Factory Function"

Base.@kwdef struct StaticPoisson{F<:Function} <: AbstractStaticPoissonModel
    # A function that takes `n` and returns a Distribution
    prior_factory::F = (n) -> MvNormal(zeros(n), 0.5 * I)
end

in the model as :
      prior_dist = model.prior_factory(n_teams)

"""


using BayesianFootball
using DataFrames, Statistics, LinearAlgebra
using ThreadPinning
using BenchmarkTools
using Turing, DynamicPPL, ReverseDiff, LogDensityProblems


# ==============================================================================
#  Tape Compilation & Benchmarking
# ==============================================================================

function benchmark_tape(ldf, theta, name)
    println("\n=== Benchmarking: $name ===")
    
    # A. Define the function wrapper for ReverseDiff
    # We want the gradient of the log density
    f_tape = x -> LogDensityProblems.logdensity(ldf, x)

    # B. Record the Tape (One-time cost)
    println("Recording tape...")
    tape = ReverseDiff.GradientTape(f_tape, theta)
    
    # C. Compile the Tape (One-time cost - optimizes instructions)
    println("Compiling tape...")
    compiled_tape = ReverseDiff.compile(tape)

    # D. Benchmark the Gradient Calculation
    # This is what happens thousands of times during NUTS sampling
    result_buffer = similar(theta)
    
    # We use $ to interpolate variables to avoid global variable scope issues in benchmark
    b = @benchmark ReverseDiff.gradient!($result_buffer, $compiled_tape, $theta)
    
    display(b)
    return compiled_tape
end



pinthreads(:cores)
BLAS.set_num_threads(1) 



#######
# 1. Setup & Data Loading
# ------------------------------------------------------------------------------

ds, true_params = BayesianFootball.SyntheticData.generate_synthetic_data_with_params(
    n_teams=100, 
    n_seasons=1,
    legs_per_season=4,
    in_season_volatility=0.05 # Made it volatile so you can see movement in plots!
)


#
model_v1 = BayesianFootball.Models.PreGame.StaticPoisson()

# v2 
model_baseline = BayesianFootball.Models.PreGame.StaticPoisson()
model_tight    = BayesianFootball.Models.PreGame.StaticPoisson(σ =0.05)
model_loose    = BayesianFootball.Models.PreGame.StaticPoisson(σ =1.0)

# v3 
model_norm = BayesianFootball.Models.PreGame.StaticPoisson()

# Option A: Standard T-Dist (df=3, scale=1.0)
# This will be much "wider" than your Normal(0, 0.5)
model_tdist_std = BayesianFootball.Models.PreGame.StaticPoisson(
    prior = TDist(3)
)

# Option B: Scaled T-Dist (df=3, scale=0.5) -> RECOMMENDED
# This matches your Normal(0, 0.5) scale but allows for outliers (heavy tails)
model_tdist_scaled = BayesianFootball.Models.PreGame.StaticPoisson(
    prior = LocationScale(0.0, 0.5, TDist(3))
)


#---------

vocabulary = BayesianFootball.Features.create_vocabulary(ds, model_v1)

splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["2020/21"], 
    round_col = :round
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

fs = BayesianFootball.Features.create_features(data_splits, vocabulary, model_v1, splitter_config)

data = BayesianFootball.Models.PreGame.TuringHelpers.prepare_data(model_v1, fs[1][1])


# --- rev diff bits 
# v1 
turing_m1 = BayesianFootball.Models.PreGame.build_turing_model(model_v1, fs[1][1]) 
ldf_1 = Turing.LogDensityFunction(turing_m1)
theta_1 = Float64.(randn(LogDensityProblems.dimension(ldf_1)))


# v2 ---
turing_tight = BayesianFootball.Models.PreGame.build_turing_model(model_tight, fs[1][1]) 
turing_loose = BayesianFootball.Models.PreGame.build_turing_model(model_loose, fs[1][1]) 

ldf_tight = Turing.LogDensityFunction(turing_tight)
ldf_loose = Turing.LogDensityFunction(turing_loose)

theta_tight = Float64.(randn(LogDensityProblems.dimension(ldf_tight)))
theta_loose = Float64.(randn(LogDensityProblems.dimension(ldf_loose)))

# v3 --- 

turing_norm = BayesianFootball.Models.PreGame.build_turing_model(model_norm, fs[1][1]) 
ldf_norm = Turing.LogDensityFunction(turing_norm)
theta_norm = Float64.(randn(LogDensityProblems.dimension(ldf_norm)))

turing_tdist_std = BayesianFootball.Models.PreGame.build_turing_model(model_tdist_std, fs[1][1]) 
turing_tdist_scaled = BayesianFootball.Models.PreGame.build_turing_model(model_tdist_scaled, fs[1][1]) 

ldf_tdist_std = Turing.LogDensityFunction(turing_tdist_std)
ldf_tdist_scaled = Turing.LogDensityFunction(turing_tdist_scaled)

theta_tdist_std = Float64.(randn(LogDensityProblems.dimension(ldf_tdist_std)))
theta_tdist_scaled = Float64.(randn(LogDensityProblems.dimension(ldf_tdist_scaled)))




# Run Benchmarks
bench_v1 = benchmark_tape(ldf_1, theta_1, "Version 1: MvNormal (Identity)")

bench_tight = benchmark_tape(ldf_tight, theta_tight, "Version 2: tight ")
bench_loose = benchmark_tape(ldf_loose, theta_loose, "Version 2: loose ")


# v3 

bench_norm = benchmark_tape(ldf_norm, theta_norm, "Version 1: Norm")
bench_tdist_std = benchmark_tape(ldf_tdist_std, theta_tdist_std, "Version 2: tdist_std ")
bench_tdist_scaled = benchmark_tape(ldf_tdist_scaled, theta_tdist_scaled, "Version 2: tdist_scaled ")






"""
# ---- v1 ----- 

normal verison before changes:  

=== Benchmarking: Version 1: MvNormal (Identity) ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 363 samples with 1 evaluation per sample.
 Range (min … max):  13.305 ms …  18.238 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     13.625 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   13.752 ms ± 609.522 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▄▄██▅███▆▅▃                                                   
  ████████████▇▆▅▆▁▅▁▅▅▆▇▁▆▁▁▁▅▁▁▆▅▁▅▁▅▅▅▁▁▁▅▁▁▁▁▁▅▁▁▁▁▁▅▁▆▁▁▅ ▇
  13.3 ms       Histogram: log(frequency) by time      16.5 ms <

 Memory estimate: 464.27 KiB, allocs estimate: 9.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

# ---- v2 ----

=== Benchmarking: Version 2: tight  ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 461 samples with 1 evaluation per sample.
 Range (min … max):  10.502 ms …  17.397 ms  ┊ GC (min … max): 0.00% … 38.67%
 Time  (median):     10.683 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   10.835 ms ± 542.005 μs  ┊ GC (mean ± σ):  0.24% ±  2.21%

  ▂▄▇██▆▅▄▂ ▂ ▁                                                 
  ███████████▇██▆▆▄▄█▆▆█▆▆▆▄▆██▆▁▆▆▁▁▁▁▁▆▁▁▁▄▄▆▄▁▁▁▁▁▁▁▁▁▁▁▁▁▄ ▇
  10.5 ms       Histogram: log(frequency) by time      12.8 ms <

 Memory estimate: 464.27 KiB, allocs estimate: 9.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

julia> bench_loose = benchmark_tape(ldf_loose, theta_loose, "Version 2: loose ")

=== Benchmarking: Version 2: loose  ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 355 samples with 1 evaluation per sample.
 Range (min … max):  13.763 ms …  17.973 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     13.970 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   14.092 ms ± 409.652 μs  ┊ GC (mean ± σ):  0.02% ± 0.39%

  ▁▄▆▆▇█▇▄▂▃▁               ▁                                   
  ████████████▆▄▇▇▁▆▄▆▁▆▄▇▄███▄▄▁▆▁▄▆▁▁▁▄▁▄▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄ ▇
  13.8 ms       Histogram: log(frequency) by time        16 ms <

 Memory estimate: 464.27 KiB, allocs estimate: 9.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)


# ---- v3 ----

julia> bench_norm = benchmark_tape(ldf_norm, theta_norm, "Version 1: Norm")

=== Benchmarking: Version 1: Norm ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 503 samples with 1 evaluation per sample.
 Range (min … max):  9.527 ms …  16.993 ms  ┊ GC (min … max): 0.00% … 42.36%
 Time  (median):     9.745 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   9.927 ms ± 631.103 μs  ┊ GC (mean ± σ):  0.33% ±  2.41%

   ▃▅▇█▇▆▄▂▂▂                                                  
  ▆███████████▄▅▆▅▇█▅▇▄▆▄▅▅▄▇▆▇▅▄▅▁▄▆▄▄▄▄▁▁▁▄▄▁▄▄▁▄▁▁▁▄▁▁▄▄▁▇ ▇
  9.53 ms      Histogram: log(frequency) by time        12 ms <

 Memory estimate: 464.27 KiB, allocs estimate: 9.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

julia> bench_tdist_std = benchmark_tape(ldf_tdist_std, theta_tdist_std, "Version 2: tdist_std ")

=== Benchmarking: Version 2: tdist_std  ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 392 samples with 1 evaluation per sample.
 Range (min … max):  12.422 ms …  17.455 ms  ┊ GC (min … max): 0.00% … 5.22%
 Time  (median):     12.599 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   12.752 ms ± 494.299 μs  ┊ GC (mean ± σ):  0.11% ± 1.18%

  ▁▆▆█▆▃▂▂▁▁                                                    
  ██████████▇▅▄▅▇▇▁▄▄▄▇▇▆▄▄▅▆▄▅▄▅▅▄▁▁▁▄▁▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▄ ▆
  12.4 ms       Histogram: log(frequency) by time      15.2 ms <

 Memory estimate: 464.27 KiB, allocs estimate: 9.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

julia> bench_tdist_scaled = benchmark_tape(ldf_tdist_scaled, theta_tdist_scaled, "Version 2: tdist_scaled ")

=== Benchmarking: Version 2: tdist_scaled  ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 388 samples with 1 evaluation per sample.
 Range (min … max):  12.567 ms …  19.857 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     12.772 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   12.875 ms ± 486.202 μs  ┊ GC (mean ± σ):  0.03% ± 0.47%

    ▂▆▆█▇▄▃▂▁                                                   
  ▄▇█████████▇▇▆▁▅▆▅▄▁▁▄▄▄▅▁▁▅▄▅▅▁▄▁▁▄▁▁▁▁▄▁▁▄▁▁▁▁▄▄▁▁▁▁▁▁▄▁▁▄ ▆
  12.6 ms       Histogram: log(frequency) by time      14.7 ms <

 Memory estimate: 464.27 KiB, allocs estimate: 9.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)


"""


# --------------------------------------------------------------


ds, true_params = BayesianFootball.SyntheticData.generate_synthetic_data_with_params(
    n_teams=10, 
    n_seasons=1,
    legs_per_season=4,
    in_season_volatility=0.05 # Made it volatile so you can see movement in plots!
)


# v3 
model_norm = BayesianFootball.Models.PreGame.StaticPoisson()

# Option A: Standard T-Dist (df=3, scale=1.0)
# This will be much "wider" than your Normal(0, 0.5)
model_tdist_std = BayesianFootball.Models.PreGame.StaticPoisson(
    prior = TDist(3)
)

# Option B: Scaled T-Dist (df=3, scale=0.5) -> RECOMMENDED
# This matches your Normal(0, 0.5) scale but allows for outliers (heavy tails)
model_tdist_scaled = BayesianFootball.Models.PreGame.StaticPoisson(
    prior = LocationScale(0.0, 0.5, TDist(3))
)


#---------

vocabulary = BayesianFootball.Features.create_vocabulary(ds, model_norm) 

splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["2020/21"], 
    round_col = :round
)


data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model_norm, splitter_config)

train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=3) 

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=500, n_chains=2, n_warmup=500) # Use renamed struct

training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)


results_norm = BayesianFootball.Training.train(model_norm, training_config, feature_sets)

results_tdist_std = BayesianFootball.Training.train(model_tdist_std, training_config, feature_sets)

results_tdist_scaled = BayesianFootball.Training.train(model_tdist_scaled, training_config, feature_sets)


alpha_matrix_norm = Array(group(results_norm[1][1], :log_α))
beta_matrix_norm = Array(group(results_norm[1][1], :log_β))


alpha_matrix_tdist_std = Array(group(results_tdist_std[1][1], :log_α))
beta_matrix_tdist_std = Array(group(results_tdist_std[1][1], :log_β))

alpha_matrix_tdist_scaled = Array(group(results_tdist_scaled[1][1], :log_α))
beta_matrix_tdist_scaled = Array(group(results_tdist_scaled[1][1], :log_β))



p_overview_norm = BayesianFootball.SyntheticData.plot_parameter_comparison(alpha_matrix_norm, beta_matrix_norm, true_params)

p_overview_tdist_std = BayesianFootball.SyntheticData.plot_parameter_comparison(alpha_matrix_tdist_std, beta_matrix_tdist_std, true_params)


p_overview_tdist_scaled = BayesianFootball.SyntheticData.plot_parameter_comparison(alpha_matrix_tdist_scaled, beta_matrix_tdist_scaled, true_params)

p_aa = BayesianFootball.SyntheticData.plot_static_fit_over_time(5, alpha_matrix_norm, beta_matrix_norm, true_params)

p_aa = BayesianFootball.SyntheticData.plot_static_fit_over_time(5, alpha_matrix_tdist_std, beta_matrix_tdist_std, true_params)


p_aa = BayesianFootball.SyntheticData.plot_static_fit_over_time(5, alpha_matrix_tdist_scaled, beta_matrix_tdist_scaled, true_params)


using StatsPlots
density(alpha_matrix_norm[:,1], label="normal")
density!(alpha_matrix_tdist_std[:,1], label="tdist std")
density!(alpha_matrix_tdist_scaled[:,1], label="tdist scaled ")


num = 4
density(alpha_matrix_norm[:,num], label="normal")
density!(alpha_matrix_tdist_std[:,num], label="tdist std")
density!(alpha_matrix_tdist_scaled[:,num], label="tdist scaled ")

density(rand(model_norm.prior, 10000), label="normal")
density!(rand(model_tdist_std.prior, 10000), label=" tdist norm ")
density!(rand(model_tdist_scaled.prior, 10000), label="tdist scaled")




# -------------------------------------------------------------------------------------------
#
"""
  Testing the grw model 
    
  #T1: reversediff tape 

  #T2: plots
"""

# ---- T1 reverse diff Tape ----

using BayesianFootball
using DataFrames, Statistics, LinearAlgebra
using ThreadPinning
using BenchmarkTools
using Turing, DynamicPPL, ReverseDiff, LogDensityProblems

pinthreads(:cores)
BLAS.set_num_threads(1) 

using StatsPlots 
using Distributions, StatsPlots, Random

# 1. Your Current Baseline (Half-Normal)
# σ ~ |N(0, 0.05)|
hn = Truncated(Normal(0, 0.05), 0, Inf)

# 2. The Heavy Tail (Half-Cauchy)
# σ ~ |Cauchy(0, 0.05)|
# Cauchy has undefined mean/variance and extremely heavy tails
hc = Truncated(Cauchy(0, 0.05), 0, Inf)

# 3. The "Owen" Recommendation (Gamma on Variance)
# Paper: Variance(σ²) ~ Gamma(α=2, β=250) (Rate=250 => Scale=1/250)
# We sample Variance, then sqrt() to see the distribution of Sigma
g_var_dist = Gamma(2, 1/250) 

# Generate samples for plotting
n = 100_000
samples_hn = rand(hn, n)
samples_hc = rand(hc, n)
samples_g  = sqrt.(rand(g_var_dist, n)) # Transform σ² -> σ

# Plotting
density(samples_hn, label="Half-Normal (σ=0.05)", linewidth=2, 
        title="Prior Distributions for Random Walk Sigma", 
        xlabel="Sigma (Step Size)", ylabel="Density", xlims=(0, 0.25))

density!(samples_hc, label="Half-Cauchy (γ=0.05)", linewidth=2, linestyle=:dash)

density!(samples_g, label="Owen's Gamma (Transformed)", linewidth=2, color=:green)

#  -----
# ==============================================================================
# 0. Setup & Data Generation (Dynamic Context)
# ==============================================================================

# We generate data with volatility to justify the dynamic model
ds_dynamic, true_params_dynamic = BayesianFootball.SyntheticData.generate_synthetic_data_with_params(
    n_teams=100, 
    n_seasons=1,
    legs_per_season=4, # 4 games per pair = temporal history exists
    in_season_volatility=0.05 
)

# ==============================================================================
# 1. Define the 3 Model Variants
# ==============================================================================

# 1. Standard: Half-Normal (Baseline)
# Tightly focused at zero, but allows drift.
m_std = BayesianFootball.Models.PreGame.GRWPoisson(
    σ_att = Truncated(Normal(0, 0.05), 0, Inf),
    σ_def = Truncated(Normal(0, 0.05), 0, Inf)
)

# 2. Owen's Approximation: Gamma
# Pushes density AWAY from zero.
# Gamma(2, 0.04) has shape similar to the paper's finding
m_owen = BayesianFootball.Models.PreGame.GRWPoisson(
    σ_att = Gamma(2, 0.04), 
    σ_def = Gamma(2, 0.04)
)

# 3. Heavy Tail: Half-Cauchy
# Theoretically dangerous for NUTS (funnels), let's see if it's slower.
m_cauchy = BayesianFootball.Models.PreGame.GRWPoisson(
    σ_att = Truncated(Cauchy(0, 0.05), 0, Inf),
    σ_def = Truncated(Cauchy(0, 0.05), 0, Inf)
)

# ==============================================================================
# 2. Feature Pipeline
# ==============================================================================

# We only need to create features once using any valid GRW model.
# The priors do not change the structure of the data (Time indices, IDs, Goals).

vocab_dyn = BayesianFootball.Features.create_vocabulary(ds_dynamic, m_std)

split_conf_dyn = BayesianFootball.Data.StaticSplit(
    train_seasons = ["2020/21"], 
    round_col = :round
)

splits_dyn = BayesianFootball.Data.create_data_splits(ds_dynamic, split_conf_dyn)

# Important: Use m_std to generate features so the system knows to build 
# the 'time_indices' required for the dynamic model.
fs_dyn = BayesianFootball.Features.create_features(splits_dyn, vocab_dyn, m_std, split_conf_dyn)

# We grab the first training set
train_data_dyn = fs_dyn[1][1] 

# ==============================================================================
# 3. Build Turing Models & LogDensity Functions
# ==============================================================================

println("\nBuilding Turing models...")

# A. Standard Model
turing_std = BayesianFootball.Models.PreGame.build_turing_model(m_std, train_data_dyn)
ldf_std = Turing.LogDensityFunction(turing_std)
theta_std = Float64.(randn(LogDensityProblems.dimension(ldf_std)));

# B. Owen (Gamma) Model
turing_owen = BayesianFootball.Models.PreGame.build_turing_model(m_owen, train_data_dyn)
ldf_owen = Turing.LogDensityFunction(turing_owen)
theta_owen = Float64.(randn(LogDensityProblems.dimension(ldf_owen)));

# C. Cauchy Model
turing_cauchy = BayesianFootball.Models.PreGame.build_turing_model(m_cauchy, train_data_dyn)
ldf_cauchy = Turing.LogDensityFunction(turing_cauchy)
theta_cauchy = Float64.(randn(LogDensityProblems.dimension(ldf_cauchy)));

# ==============================================================================
# 4. Run Benchmarks
# ==============================================================================

# 1. Standard Half-Normal
# Expectation: Very fast, similar to your Static benchmarks
bench_grw_std = benchmark_tape(ldf_std, theta_std, "GRW: Half-Normal (Baseline)")

# 2. Owen's Gamma
# Expectation: Should be as fast as Normal (both are simple scalars in the graph)
bench_grw_owen = benchmark_tape(ldf_owen, theta_owen, "GRW: Owen's Gamma")

# 3. Half-Cauchy
# Expectation: Might be slightly slower per gradient due to heavier math operations (log(1+x^2)),
# BUT the real cost usually comes during sampling (divergences), not just one gradient eval.
bench_grw_cauchy = benchmark_tape(ldf_cauchy, theta_cauchy, "GRW: Half-Cauchy")

"""
standard half normal 

julia> bench_grw_std = benchmark_tape(ldf_std, theta_std, "GRW: Half-Normal (Baseline)")                                              
                                                                                                                                      
=== Benchmarking: GRW: Half-Normal (Baseline) ===                                                                                     
Recording tape...                                                                                                                     
Compiling tape...                                                                                                                     
BenchmarkTools.Trial: 69 samples with 1 evaluation per sample.                                                                        
 Range (min … max):  72.172 ms …  76.836 ms  ┊ GC (min … max): 0.00% … 0.00%                                                          
 Time  (median):     73.269 ms               ┊ GC (median):    0.00%                                                                  
 Time  (mean ± σ):   73.294 ms ± 578.262 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%                                                          
                                                                                                                                      
                        ▃▁ ▁ ▃▁▃ ▁   █ ▁▁▃                                                                                            
  ▄▁▁▁▁▁▁▁▄▁▄▁▄▄▇▄▇▄▇▄▄▄██▁█▄███▇█▄▄▄█▄███▄▁▇▁▁▁▁▁▁▄▁▁▁▁▁▁▁▁▁▄ ▁                                                                      
  72.2 ms         Histogram: frequency by time         74.4 ms <                                                                      
                                                                                                                                      
 Memory estimate: 1.21 MiB, allocs estimate: 68.                                                                                      
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)                                                                                
                                                                                                                                      
julia> bench_grw_std = benchmark_tape(ldf_std, theta_std, "GRW: Half-Normal (Baseline)")                                              
                                                                                                                                      
=== Benchmarking: GRW: Half-Normal (Baseline) ===                                                                                     
Recording tape...                                                                                                                     
Compiling tape...                                                                                                                     
BenchmarkTools.Trial: 62 samples with 1 evaluation per sample.                                                                        
 Range (min … max):  80.837 ms …  82.860 ms  ┊ GC (min … max): 0.00% … 0.00%                                                          
 Time  (median):     81.637 ms               ┊ GC (median):    0.00%                                                                  
 Time  (mean ± σ):   81.669 ms ± 389.948 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%                                                          
                                                                                                                                      
  ▁         ▁▁      ▁▄▁  ▄▄▁█ ▁ ▁  ██  ▄ ▄▁ ▁                                                                                         
  █▁▁▁▁▁▆▁▆▁██▁▁▆▆▁▁███▁▆████▆█▆█▆▆██▆▁█▁██▁█▁▁▁▁▆▁▁▁▁▁▁▆▁▁▆▁▆ ▁                                                                      
  80.8 ms         Histogram: frequency by time         82.6 ms <                                                                      
                                                                                                                                      
 Memory estimate: 1.21 MiB, allocs estimate: 68.                                                                                      
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)                                                                                
                                                                                                                                      
julia> bench_grw_std = benchmark_tape(ldf_std, theta_std, "GRW: Half-Normal (Baseline)")                                              
                                                                                                                                      
=== Benchmarking: GRW: Half-Normal (Baseline) ===                                                                                     
Recording tape...                                                                                                                     
Compiling tape...                                                                                                                     
BenchmarkTools.Trial: 53 samples with 1 evaluation per sample.                                                                        
 Range (min … max):  94.092 ms …  96.620 ms  ┊ GC (min … max): 0.00% … 0.00%                                                          
 Time  (median):     94.926 ms               ┊ GC (median):    0.00%                                                                  
 Time  (mean ± σ):   94.905 ms ± 444.331 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%                                                          
                                                                                                                                      
      █    ▃    ▃  ▃▃    ▃   ▃▃ ▃▃█▃▃ █ ▃    ▃▃                                                                                       
  ▇▁▁▁█▁▁▁▁█▁▁▇▇█▇▁██▁▇▁▁█▇▁▁██▇█████▁█▁█▇▁▇▇██▇▁▁▇▁▁▁▁▁▇▁▁▁▇▇ ▁                                                                      
  94.1 ms         Histogram: frequency by time         95.7 ms <                                                                      
                                                                                                                                      
 Memory estimate: 1.21 MiB, allocs estimate: 68.                                                                                      
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)   



# ---- owen gamma 
julia> bench_grw_owen = benchmark_tape(ldf_owen, theta_owen, "GRW: Owen's Gamma")

=== Benchmarking: GRW: Owen's Gamma ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 70 samples with 1 evaluation per sample.
 Range (min … max):  70.987 ms …  73.523 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     71.811 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   71.846 ms ± 423.891 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

        ▂              ▂▅ █  ▅▅       ▂  ▂                      
  ▅▁▁▅▁▁█▁▁▅▅▁▁██▅█▁▁█▅█████▅██▅▅▅▅█▅██▁▁█▅██▁▅▁▅▅▁▅█▁▁▁▁▁▁▁▁▅ ▁
  71 ms           Histogram: frequency by time         72.8 ms <

 Memory estimate: 1.21 MiB, allocs estimate: 68.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

julia> bench_grw_owen = benchmark_tape(ldf_owen, theta_owen, "GRW: Owen's Gamma")

=== Benchmarking: GRW: Owen's Gamma ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 47 samples with 1 evaluation per sample.
 Range (min … max):  106.115 ms … 108.668 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     107.185 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   107.210 ms ± 582.966 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

     ▃      ▃     ▃ █  ▃  █     █ ▃▃  ▃▃       ▃                 
  ▇▇▁█▁▇▁▁▇▁█▁▇▇▁▁█▁█▇▁█▇▇█▇▁▇▇▇█▇██▁▁██▇▁▁▇▇▁▇█▁▁▁▁▁▁▁▁▁▇▁▁▁▁▇ ▁
  106 ms           Histogram: frequency by time          109 ms <

 Memory estimate: 1.21 MiB, allocs estimate: 68.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

julia> bench_grw_owen = benchmark_tape(ldf_owen, theta_owen, "GRW: Owen's Gamma")

=== Benchmarking: GRW: Owen's Gamma ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 48 samples with 1 evaluation per sample.
 Range (min … max):  104.515 ms … 107.255 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     105.893 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   105.862 ms ± 760.756 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

           ▄█ ▁         ▁   ▁            ▁   ▄ ▁ █               
  ▆▁▁▁▆▆▁▆▁██▁█▁▆▁▆▁▆▆▁▁█▆▁▁█▁▆▆▁▆▁▆▆▁▆▁▆█▁▁▆█▆█▆█▁▁▆▁▆▁▁▆▁▆▁▁▆ ▁
  105 ms           Histogram: frequency by time          107 ms <

 Memory estimate: 1.21 MiB, allocs estimate: 68.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)


# --- half cauchy 

julia> bench_grw_cauchy = benchmark_tape(ldf_cauchy, theta_cauchy, "GRW: Half-Cauchy")

=== Benchmarking: GRW: Half-Cauchy ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 44 samples with 1 evaluation per sample.
 Range (min … max):  112.794 ms … 114.993 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     113.633 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   113.673 ms ± 466.018 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▁ ▁               ▁ ▁▄ ▁    ▁█▁▁     ▁    ▄                    
  █▁█▁▁▁▁▆▁▁▆▆▆▆▆▁▆▆█▆██▆█▁▆▆▁████▁▆▁▁▁█▁▁▁▆█▁▁▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▆ ▁
  113 ms           Histogram: frequency by time          115 ms <

 Memory estimate: 1.21 MiB, allocs estimate: 68.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

julia> bench_grw_cauchy = benchmark_tape(ldf_cauchy, theta_cauchy, "GRW: Half-Cauchy")

=== Benchmarking: GRW: Half-Cauchy ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 39 samples with 1 evaluation per sample.
 Range (min … max):  129.455 ms … 132.223 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     130.080 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   130.156 ms ± 441.438 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

          ▁█  ██▁▁▄  ▁▁▁                                         
  ▆▁▆▆▁▁▆▁██▆▆█████▆▆███▆▁▁▆▁▁▁▁▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▆ ▁
  129 ms           Histogram: frequency by time          132 ms <

 Memory estimate: 1.21 MiB, allocs estimate: 68.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

julia> bench_grw_cauchy = benchmark_tape(ldf_cauchy, theta_cauchy, "GRW: Half-Cauchy")

=== Benchmarking: GRW: Half-Cauchy ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 41 samples with 1 evaluation per sample.
 Range (min … max):  123.816 ms … 126.786 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     124.448 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   124.493 ms ± 468.766 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

      ▂  ▂   █ ▅    ▂                                            
  ▅▁▅▁█▁███▅▅███▅█████▁▁▁▅▅▁▁▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅ ▁
  124 ms           Histogram: frequency by time          127 ms <

 Memory estimate: 1.21 MiB, allocs estimate: 68.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

""" 

# ---- T2 plots ----

using BayesianFootball
using DataFrames, Statistics, LinearAlgebra
using ThreadPinning
using BenchmarkTools
using Turing, DynamicPPL, ReverseDiff, LogDensityProblems


ds, true_params = BayesianFootball.SyntheticData.generate_synthetic_data_with_params(
    n_teams=10, 
    n_seasons=1,
    legs_per_season=4,
    in_season_volatility=0.05 # Made it volatile so you can see movement in plots!
)


model = BayesianFootball.Models.PreGame.GRWPoisson()

model_low = BayesianFootball.Models.PreGame.GRWPoisson(
                σ_att = Truncated(Normal(0, 0.001), 0, Inf),
                σ_def = Truncated(Normal(0, 0.001), 0, Inf),
            )


model_high = BayesianFootball.Models.PreGame.GRWPoisson(
                σ_att = Truncated(Normal(0, 0.3), 0, Inf),
                σ_def = Truncated(Normal(0, 0.3), 0, Inf),
            )


vocabulary = BayesianFootball.Features.create_vocabulary(ds, model)



splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["2020/21"], 
    round_col = :round
)


data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config)

train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=3) 

sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=500, n_chains=2, n_warmup=500) # Use renamed struct

training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)


results = BayesianFootball.Training.train(model, training_config, feature_sets)

results_low = BayesianFootball.Training.train(model_low, training_config, feature_sets)

results_high = BayesianFootball.Training.train(model_high, training_config, feature_sets)



# reconstruct_vectorized 
att_tensor = BayesianFootball.Models.PreGame.Implementations.reconstruct_vectorized(results[1][1], 10, :z_att_steps, :z_att_init, :σ_att)
def_tensor = BayesianFootball.Models.PreGame.Implementations.reconstruct_vectorized(results[1][1], 10, :z_def_steps, :z_def_init, :σ_def)

att_tensor_low = BayesianFootball.Models.PreGame.Implementations.reconstruct_vectorized(results_low[1][1], 10, :z_att_steps, :z_att_init, :σ_att)
def_tensor_low = BayesianFootball.Models.PreGame.Implementations.reconstruct_vectorized(results_low[1][1], 10, :z_def_steps, :z_def_init, :σ_def)


att_tensor_high = BayesianFootball.Models.PreGame.Implementations.reconstruct_vectorized(results_high[1][1], 10, :z_att_steps, :z_att_init, :σ_att)
def_tensor_high = BayesianFootball.Models.PreGame.Implementations.reconstruct_vectorized(results_high[1][1], 10, :z_def_steps, :z_def_init, :σ_def)



num = 8

p1 = BayesianFootball.SyntheticData.plot_dynamic_trajectory(
    num, 
    true_params, 
    att_tensor, 
    def_tensor; 
)


p1 = BayesianFootball.SyntheticData.plot_dynamic_trajectory(
    num, 
    true_params, 
    att_tensor_low, 
    def_tensor_low; 
)


p1 = BayesianFootball.SyntheticData.plot_dynamic_trajectory(
    num, 
    true_params, 
    att_tensor_high, 
    def_tensor_high; 
)


describe(results[1][1])
"""
parameters      mean       std      mcse    ess_bulk   ess_tail      rhat   ess_per_sec       
              Symbol   Float64   Float64   Float64     Float64    Float64   Float64       Float64       
                                                                                                        
               σ_att    0.0429    0.0248    0.0015    280.7046   532.0800    1.0116        0.3811       
               σ_def    0.0320    0.0212    0.0011    343.0629   529.0743    1.0028        0.4658       
            home_adv    0.2613    0.0638    0.0023    808.1696   251.9757    1.0076        1.0973       
       z_att_init[1]    0.3122    0.5099    0.0230    494.3274   638.1024    0.9991        0.6712       
       z_att_init[2]   -0.2626    0.4948    0.0216    524.7319   527.1603    1.0038        0.7125       
       z_att_init[3]    0.0135    0.5100    0.0266    383.3531   413.3780    1.0092        0.5205       
       z_att_init[4]   -0.7372    0.5114    0.0234    481.8123   589.2150    1.0163        0.6542       
       z_att_init[5]    0.6013    0.4743    0.0212    503.1237   550.1780    1.0033        0.6831
       z_att_init[6]    0.5439    0.4926    0.0235    434.5747   442.4736    1.0075        0.5901
       z_att_init[7]   -0.2436    0.5204    0.0257    416.7152   625.8879    1.0071        0.5658
       z_att_init[8]   -0.1043    0.5031    0.0239    453.2373   501.4651    1.0081        0.6154
       z_att_init[9]    0.0269    0.5047    0.0262    394.7592   350.8359    1.0198        0.5360
      z_att_init[10]   -0.3539    0.5063    0.0235    465.1526   397.2294    1.0050        0.6316
       z_def_init[1]    0.2956    0.4551    0.0236    385.5422   439.4565    1.0005        0.5235
       z_def_init[2]    0.2608    0.4553    0.0241    352.7169   627.2548    1.0018        0.4789
       z_def_init[3]   -0.2355    0.4649    0.0204    505.1705   654.3335    1.0066        0.6859
       z_def_init[4]    0.7934    0.4454    0.0283    247.8487   749.4816    1.0017        0.3365
       z_def_init[5]   -0.3458    0.4588    0.0204    523.5084   610.1878    1.0005        0.7108
       z_def_init[6]   -0.2571    0.4610    0.0208    477.9045   543.6425    1.0040        0.6489
       z_def_init[7]   -0.0231    0.4632    0.0208    493.6841   654.9202    1.0016        0.6703
       z_def_init[8]   -0.2652    0.4780    0.0203    554.0433   640.4033    1.0083        0.7523
       z_def_init[9]    0.3942    0.4332    0.0234    348.3381   554.7300    1.0023        0.4730
      z_def_init[10]   -0.4025    0.5134    0.0304    290.1978   719.9207    1.0071        0.3940
   z_att_steps[1, 1]   -0.0287    1.0042    0.0301   1104.7726   607.9891    1.0025        1.5000
   z_att_steps[2, 1]   -0.1172    1.0226    0.0264   1470.2307   647.0476    1.0047        1.9963
   z_att_steps[3, 1]    0.0243    0.9799    0.0248   1554.9879   578.9956    1.0084        2.1113
   z_att_steps[4, 1]   -0.0761    1.0129    0.0296   1168.9051   656.1957    1.0024        1.5871
   z_att_steps[5, 1]    0.0584    0.9619    0.0234   1707.6028   815.4633    1.0004        2.3186
   z_att_steps[6, 1]    0.0927    0.9668    0.0262   1386.9234   626.9076    1.0042        1.8831
   z_att_steps[7, 1]   -0.0489    0.9533    0.0271   1254.5169   818.1474    0.9995        1.7034
   z_att_steps[8, 1]    0.0053    0.9916    0.0302   1080.7913   595.8776    0.9992        1.4675
   z_att_steps[9, 1]    0.0811    1.0185    0.0326    967.7271   670.3873    1.0116        1.3140
  z_att_steps[10, 1]    0.0473    0.9680    0.0260   1391.7649   732.5656    0.9995        1.8897


"""

describe(results_high[1][1])
"""

Summary Statistics                                                                    19:34:15 [55/6145]
                                                                                                        
          parameters      mean       std      mcse    ess_bulk   ess_tail      rhat   ess_per_sec       
              Symbol   Float64   Float64   Float64     Float64    Float64   Float64       Float64       
                                                                                                        
               σ_att    0.0686    0.0345    0.0018    369.1677   571.1288    1.0032        0.1902       
               σ_def    0.0428    0.0274    0.0013    419.6312   647.3567    1.0061        0.2162       
            home_adv    0.2544    0.0586    0.0017   1186.1366   779.5640    0.9986        0.6111       
       z_att_init[1]    0.4316    0.5466    0.0195    800.6629   683.3507    0.9996        0.4125       
       z_att_init[2]   -0.2249    0.5493    0.0181    924.9092   812.2325    0.9997        0.4765       
       z_att_init[3]   -0.0200    0.5250    0.0168    979.0920   772.3788    1.0050        0.5044       
       z_att_init[4]   -0.7760    0.5640    0.0176   1037.8110   619.4064    0.9993        0.5347       
       z_att_init[5]    0.6334    0.5101    0.0163    976.4446   688.2093    1.0023        0.5030       
       z_att_init[6]    0.3866    0.5316    0.0187    843.1018   774.6489    1.0002        0.4343       
       z_att_init[7]   -0.1563    0.5784    0.0191    968.9884   562.4638    0.9997        0.4992       
       z_att_init[8]   -0.0336    0.5349    0.0170    999.3163   682.9774    1.0000        0.5148       
       z_att_init[9]   -0.0127    0.5037    0.0175    838.9983   564.6565    1.0046        0.4322       
      z_att_init[10]   -0.3144    0.5274    0.0178    887.9256   684.2640    0.9985        0.4574       
       z_def_init[1]    0.2506    0.4857    0.0160    916.1737   834.5763    1.0010        0.4720       
       z_def_init[2]    0.2232    0.4891    0.0165    875.1059   699.1905    1.0017        0.4508       
       z_def_init[3]   -0.2413    0.4953    0.0178    768.0854   808.0083    1.0020        0.3957       
       z_def_init[4]    0.8005    0.4926    0.0205    574.1128   793.6096    1.0046        0.2958       
       z_def_init[5]   -0.3752    0.5385    0.0195    837.8555   621.1162    0.9983        0.4316       
       z_def_init[6]   -0.2644    0.5248    0.0174    912.6690   843.9370    1.0012        0.4702       
       z_def_init[7]   -0.0035    0.4934    0.0178    763.1811   700.5473    1.0009        0.3932       
       z_def_init[8]   -0.2980    0.5194    0.0168    957.2726   694.6341    1.0002        0.4932       
       z_def_init[9]    0.3710    0.4598    0.0167    779.3078   756.4823    1.0055        0.4015       
      z_def_init[10]   -0.3489    0.5138    0.0182    797.7443   759.9384    1.0004        0.4110       
   z_att_steps[1, 1]   -0.0478    0.9963    0.0315   1015.3912   614.0572    1.0024        0.5231       
   z_att_steps[2, 1]   -0.0504    0.9589    0.0268   1283.8755   871.5987    0.9990        0.6614       
   z_att_steps[3, 1]   -0.0071    0.9608    0.0295   1065.1346   586.8406    1.0027        0.5487       
   z_att_steps[4, 1]   -0.0827    0.9715    0.0299   1056.3421   852.9083    1.0036        0.5442       
   z_att_steps[5, 1]    0.1841    0.9762    0.0282   1219.7997   483.6878    1.0002        0.6284       
   z_att_steps[6, 1]    0.1461    1.0286    0.0299   1180.8922   841.4352    0.9997        0.6084       
   z_att_steps[7, 1]   -0.0948    1.0040    0.0332    934.6628   770.7974    1.0000        0.4815       
   z_att_steps[8, 1]   -0.0319    1.0026    0.0282   1249.6588   658.2991    0.9985        0.6438       
   z_att_steps[9, 1]    0.0231    1.0438    0.0317   1079.5360   647.4273    1.0000        0.5561       
  z_att_steps[10, 1]   -0.0430    0.9536    0.0308    949.4470   653.8199    1.0009        0.4891

"""


describe(results_low[1][1])
"""
Summary Statistics                                                                                      
                                                                                                        
          parameters      mean       std      mcse    ess_bulk   ess_tail      rhat   ess_per_sec       
              Symbol   Float64   Float64   Float64     Float64    Float64   Float64       Float64       
                                                                                                        
               σ_att    0.0008    0.0006    0.0000    858.5122   556.0693    1.0017        1.1717       
               σ_def    0.0008    0.0006    0.0000    875.3266   518.5063    1.0062        1.1947       
            home_adv    0.2760    0.0643    0.0014   2156.0955   538.1747    1.0015        2.9427       
       z_att_init[1]    0.0854    0.4054    0.0186    475.7673   751.6250    1.0083        0.6493       
       z_att_init[2]   -0.3531    0.4479    0.0187    569.6619   689.3679    1.0026        0.7775       
       z_att_init[3]    0.1388    0.4376    0.0192    520.2994   612.6818    1.0067        0.7101       
       z_att_init[4]   -0.6653    0.4444    0.0182    603.2352   771.5397    1.0019        0.8233       
       z_att_init[5]    0.5700    0.4214    0.0187    506.9246   826.7118    1.0059        0.6919       
       z_att_init[6]    0.8240    0.4087    0.0185    486.5504   641.1469    1.0099        0.6641       
       z_att_init[7]   -0.2988    0.4574    0.0193    554.6592   639.1617    1.0025        0.7570       
       z_att_init[8]   -0.1842    0.4346    0.0197    486.4707   740.0137    1.0091        0.6639       
       z_att_init[9]    0.1198    0.4329    0.0195    484.6771   850.2513    1.0059        0.6615       
      z_att_init[10]   -0.3899    0.4375    0.0188    536.2955   556.0504    1.0034        0.7319       
       z_def_init[1]    0.3578    0.3887    0.0157    615.0774   601.9425    1.0020        0.8395       
       z_def_init[2]    0.3194    0.3917    0.0177    491.2995   785.7523    1.0002        0.6705       
       z_def_init[3]   -0.3477    0.4162    0.0165    637.3615   673.7452    1.0024        0.8699       
       z_def_init[4]    0.5929    0.3904    0.0166    554.4704   605.3038    0.9994        0.7568       
       z_def_init[5]   -0.3407    0.4284    0.0164    675.0130   429.2310    1.0003        0.9213       
       z_def_init[6]   -0.3151    0.4102    0.0159    640.0349   753.6123    1.0025        0.8735       
       z_def_init[7]   -0.0705    0.4187    0.0179    552.0492   663.3891    1.0006        0.7534       
       z_def_init[8]   -0.1856    0.4242    0.0175    591.8529   841.7226    1.0036        0.8078       
       z_def_init[9]    0.3734    0.4053    0.0163    614.4267   731.8591    1.0005        0.8386       
      z_def_init[10]   -0.5047    0.4342    0.0174    615.6336   654.9990    1.0000        0.8402       
   z_att_steps[1, 1]   -0.0221    0.9732    0.0245   1578.9589   589.2431    1.0046        2.1550       
   z_att_steps[2, 1]   -0.0157    0.9412    0.0217   1881.7770   520.3001    1.0038        2.5683       
   z_att_steps[3, 1]    0.0148    1.0013    0.0230   1912.6192   699.6282    1.0001        2.6104       
   z_att_steps[4, 1]   -0.0315    1.0698    0.0244   1905.8651   532.4855    0.9996        2.6012       
   z_att_steps[5, 1]   -0.0217    0.9975    0.0237   1801.1870   474.6770    1.0138        2.4583       
   z_att_steps[6, 1]    0.0094    0.9918    0.0198   2379.3889   649.7970    1.0029        3.2474       
   z_att_steps[7, 1]    0.0018    0.9990    0.0226   1968.3938   791.2914    1.0056        2.6865       
   z_att_steps[8, 1]    0.0104    1.0470    0.0271   1593.4480   571.6077    1.0021        2.1748       
   z_att_steps[9, 1]    0.0170    1.0489    0.0205   2746.2618   505.2853    1.0041        3.7482       
  z_att_steps[10, 1]   -0.0208    1.0021    0.0263   1412.2740   675.7457    1.0025        1.9275  

"""


using Plots, StatsPlots
plot(results[1][1], :σ_att)
plot(results_high[1][1], :σ_att)
plot(results_low[1][1], :σ_att)

describe(results[1][1][:tree_depth])
describe(results_high[1][1][:tree_depth])
describe(results_low[1][1][:tree_depth])




##################################################

using BayesianFootball
using DataFrames, Statistics, LinearAlgebra
using Turing, DynamicPPL

# 1. Generate Dynamic Data (with volatility to justify the random walk)
ds, true_params = BayesianFootball.SyntheticData.generate_synthetic_data_with_params(
    n_teams=10, 
    n_seasons=1,
    legs_per_season=4,
    in_season_volatility=0.05 
)

# --- Define the 3 Variants ---

# A. Standard Half-Normal (Baseline)
# Fast, stable, peaks at zero.
model_std = BayesianFootball.Models.PreGame.GRWPoisson(
    σ_att = Truncated(Normal(0, 0.05), 0, Inf),
    σ_def = Truncated(Normal(0, 0.05), 0, Inf)
)

# B. Owen's Gamma (Paper Recommendation)
# [cite_start]Pushes density AWAY from zero to force adaptation[cite: 55].
# [cite_start]Gamma(2, 0.04) creates a peak around 0.04, matching the "settled" variance found in the paper[cite: 217].
model_owen = BayesianFootball.Models.PreGame.GRWPoisson(
    σ_att = Gamma(2, 0.04),
    σ_def = Gamma(2, 0.04)
)

# C. Half-Cauchy (Heavy Tail)
# theoretically allows massive jumps, likely slower sampling.
model_cauchy = BayesianFootball.Models.PreGame.GRWPoisson(
    σ_att = Truncated(Cauchy(0, 0.05), 0, Inf),
    σ_def = Truncated(Cauchy(0, 0.05), 0, Inf)
)




# Create vocabulary and splits using the standard model (structure is identical)
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model_std)

splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["2020/21"], 
    round_col = :round
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model_std, splitter_config)

# Sampler Configuration
# Note: You might need to increase warmup for Cauchy if it has divergences
train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=3) 
sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=500, n_chains=6, n_warmup=500)

training_config = BayesianFootball.Training.TrainingConfig(sampler_conf, train_cfg)



println("Training Standard Model...")
results_std = BayesianFootball.Training.train(model_std, training_config, feature_sets)


println("Training Owen Gamma Model...")
results_owen = BayesianFootball.Training.train(model_owen, training_config, feature_sets)

println("Training Cauchy Model...")
results_cauchy = BayesianFootball.Training.train(model_cauchy, training_config, feature_sets)


"""
julia> results_std = BayesianFootball.Training.train(model_std, training_config, feature_sets)                                        
Dispatching to train(model, config, ::Vector{Tuple{FeatureSet, String}})...                                                           
🚀 Starting Independent training for 1 splits sequentially...                                                                         
                                                                                                                                      
--- Training Split 1: static_seasons_2020/21 ---                                                                                      
Dispatching to train(model, config, ::FeatureSet)...                                                                                  
Sampling with NUTS (500 samples, 6 chains)...                                                                                         
                                                                                                                                      
┌ Info: Found initial step size                                                                               |  ETA: N/A             
└   ϵ = 0.05                                                                                                                          
┌ Info: Found initial step size                                                                                                       
└   ϵ = 0.05                                                                                                                          
┌ Info: Found initial step size                                                                                                       
└   ϵ = 0.05                                                                                                                          
┌ Info: Found initial step size                                                                                                       
└   ϵ = 0.05                                                                                                                          
┌ Info: Found initial step size                                                                                                       
└   ϵ = 0.05                                                                                                                          
┌ Info: Found initial step size                                                                                                       
└   ϵ = 0.05                                                                                                                          
Chain 2/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:14:06         
Chain 4/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:15:04         
Chain 1/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:15:11         
Chain 3/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:16:56         
Chain 5/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:17:30         
Chain 6/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:45:27         
Sampling (6 threads) 100%|████████████████████████████████████████████████████████████████████████████████████| Time: 0:45:27         
✅ Finished Split 1.                                                                                                                  
                                                                                                                                      
Sequential training complete!                                                                                                         
1-element Vector{Tuple{Any, String}}:                                                                                                 
 (MCMC chain (500×737×6 Array{Float64, 3}), "static_seasons_2020/21")  


julia> results_owen = BayesianFootball.Training.train(model_owen, training_config, feature_sets)                                      
Dispatching to train(model, config, ::Vector{Tuple{FeatureSet, String}})...                                                           
🚀 Starting Independent training for 1 splits sequentially...                                                                         
                                                                                                                                      
--- Training Split 1: static_seasons_2020/21 ---                                                                                      
Dispatching to train(model, config, ::FeatureSet)...                                                                                  
Sampling with NUTS (500 samples, 6 chains)...                                                                                         
┌ Info: Found initial step size                                                                                                       
└   ϵ = 0.05
┌ Info: Found initial step size
└   ϵ = 0.05
┌ Info: Found initial step size
└   ϵ = 0.05
┌ Info: Found initial step size
└   ϵ = 0.05
┌ Info: Found initial step size
└   ϵ = 0.05
┌ Info: Found initial step size
└   ϵ = 0.05
Chain 1/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:15:06
Chain 2/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:15:24
Chain 5/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:15:25
Chain 3/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:15:30
Chain 4/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:16:56
Chain 6/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:18:39
Sampling (6 threads) 100%|████████████████████████████████████████████████████████████████████████████████████| Time: 0:18:39
✅ Finished Split 1.

Sequential training complete!
1-element Vector{Tuple{Any, String}}:
 (MCMC chain (500×737×6 Array{Float64, 3}), "static_seasons_2020/21")


julia> results_cauchy = BayesianFootball.Training.train(model_cauchy, training_config, feature_sets)
Dispatching to train(model, config, ::Vector{Tuple{FeatureSet, String}})...
🚀 Starting Independent training for 1 splits sequentially...

--- Training Split 1: static_seasons_2020/21 ---
Dispatching to train(model, config, ::FeatureSet)...
Sampling with NUTS (500 samples, 6 chains)...
┌ Info: Found initial step size
└   ϵ = 0.05
┌ Info: Found initial step size
└   ϵ = 0.05
┌ Info: Found initial step size
└   ϵ = 0.05
┌ Info: Found initial step size
└   ϵ = 0.05
┌ Info: Found initial step size
└   ϵ = 0.05
┌ Info: Found initial step size
└   ϵ = 0.05
Chain 1/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:15:27
Chain 6/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:19:23
Chain 4/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:19:46
Chain 3/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:19:51
Chain 5/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:20:12
Chain 2/6 100%|███████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:20:46
Sampling (6 threads) 100%|████████████████████████████████████████████████████████████████████████████████████| Time: 0:20:46
✅ Finished Split 1.

Sequential training complete!
1-element Vector{Tuple{Any, String}}:
 (MCMC chain (500×737×6 Array{Float64, 3}), "static_seasons_2020/21")

"""


### 
# Helper to extract and format tensors for a specific team (e.g., Team 1)
function get_trajectories(result, model_type)
    chain = result[1][1]
    
    # We use your 'reconstruct_vectorized' helper
    att = BayesianFootball.Models.PreGame.Implementations.reconstruct_vectorized(
        chain, 10, :z_att_steps, :z_att_init, :σ_att
    )
    def = BayesianFootball.Models.PreGame.Implementations.reconstruct_vectorized(
        chain, 10, :z_def_steps, :z_def_init, :σ_def
    )
    return att, def
end

# Extract
att_std, def_std = get_trajectories(results_std, "Standard")
att_owen, def_owen = get_trajectories(results_owen, "Owen")
att_cauchy, def_cauchy = get_trajectories(results_cauchy, "Cauchy")

# Plot Team 1 (or any volatile team)
team_id = 2 

# Plot 1: Standard
p1 = BayesianFootball.SyntheticData.plot_dynamic_trajectory(
    team_id, true_params, att_std, def_std; 
)

# Plot 2: Owen (Gamma)
p2 = BayesianFootball.SyntheticData.plot_dynamic_trajectory(
    team_id, true_params, att_owen, def_owen; 
)

# Plot 3: Cauchy
p3 = BayesianFootball.SyntheticData.plot_dynamic_trajectory(
    team_id, true_params, att_cauchy, def_cauchy; 
)

display(plot(p1, p2, p3, layout=(3,1), size=(800, 900)))

println("=== STANDARD HALF-NORMAL ===")
describe(results_std[1][1][[:σ_att, :σ_def]])

println("\n=== OWEN'S GAMMA ===")
describe(results_owen[1][1][[:σ_att, :σ_def]])

println("\n=== HALF-CAUCHY ===")
describe(results_cauchy[1][1][[:σ_att, :σ_def]])


"""


julia> describe(results_std[1][1][[:σ_att, :σ_def]])
Chains MCMC chain (500×2×6 Array{Float64, 3}):

Iterations        = 501:1:1000
Number of chains  = 6
Samples per chain = 500
Wall duration     = 2727.08 seconds
Compute duration  = 7436.89 seconds
parameters        = σ_att, σ_def
internals         = 

Summary Statistics

  parameters      mean       std      mcse   ess_bulk    ess_tail      rhat   ess_per_sec 
      Symbol   Float64   Float64   Float64    Float64     Float64   Float64       Float64 

       σ_att    0.0470    0.0260    0.0010   654.1427   1224.5080    1.0109        0.0880
       σ_def    0.0332    0.0219    0.0009   606.2230   1072.6179    1.0169        0.0815


Quantiles

  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

       σ_att    0.0031    0.0278    0.0457    0.0644    0.0997
       σ_def    0.0012    0.0150    0.0310    0.0488    0.0805


julia> describe(results_owen[1][1][[:σ_att, :σ_def]])
Chains MCMC chain (500×2×6 Array{Float64, 3}):

Iterations        = 501:1:1000
Number of chains  = 6
Samples per chain = 500
Wall duration     = 1118.95 seconds
Compute duration  = 5812.87 seconds
parameters        = σ_att, σ_def
internals         = 

Summary Statistics

  parameters      mean       std      mcse    ess_bulk    ess_tail      rhat   ess_per_sec 
      Symbol   Float64   Float64   Float64     Float64     Float64   Float64       Float64 

       σ_att    0.0617    0.0290    0.0009    884.9263   1237.9545    1.0071        0.1522
       σ_def    0.0426    0.0226    0.0006   1274.8027   1736.3477    0.9998        0.2193


Quantiles

  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

       σ_att    0.0133    0.0404    0.0589    0.0795    0.1259
       σ_def    0.0073    0.0258    0.0401    0.0568    0.0926


julia> describe(results_cauchy[1][1][[:σ_att, :σ_def]])
Chains MCMC chain (500×2×6 Array{Float64, 3}):

Iterations        = 501:1:1000
Number of chains  = 6
Samples per chain = 500
Wall duration     = 1245.99 seconds
Compute duration  = 6914.96 seconds
parameters        = σ_att, σ_def
internals         = 

Summary Statistics

  parameters      mean       std      mcse   ess_bulk    ess_tail      rhat   ess_per_sec 
      Symbol   Float64   Float64   Float64    Float64     Float64   Float64       Float64 

       σ_att    0.0499    0.0317    0.0014   555.9152    492.0809    1.0058        0.0804
       σ_def    0.0317    0.0232    0.0008   854.4486   1275.4118    1.0148        0.1236


Quantiles

  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

       σ_att    0.0028    0.0253    0.0464    0.0698    0.1200
       σ_def    0.0012    0.0136    0.0275    0.0457    0.0879


"""
