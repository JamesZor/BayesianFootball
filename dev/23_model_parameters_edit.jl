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


ds, true_params = BayesianFootball.SyntheticData.generate_synthetic_data_with_params(
    n_teams=10, 
    n_seasons=1,
    legs_per_season=4,
    in_season_volatility=0.05 # Made it volatile so you can see movement in plots!
)


model = BayesianFootball.Models.PreGame.GRWPoisson()

model_low = BayesianFootball.Models.PreGameGRWPoisson(
                σ_att = Truncated(Normal(0, 0.001), 0, Inf),
                σ_def = Truncated(Normal(0, 0.001), 0, Inf),
            )


model_high = BayesianFootball.Models.PreGameGRWPoisson(
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

