""" 
In this workspace we want to explore and define some methods 
for the computaional efficenty of models, namely the reverse diff tape of the models 
"""


""" 
Explore workspace 1: Here want to test on the statict poisson to see if the MvNormal Normal makes a 
differecnes
"""

# load the data and packages #

using BayesianFootball
using DataFrames, Statistics, LinearAlgebra
using ThreadPinning
using BenchmarkTools
using Turing, DynamicPPL, ReverseDiff, LogDensityProblems

# 1. Setup & Data Loading
# ------------------------------------------------------------------------------
pinthreads(:cores)
BLAS.set_num_threads(1) 

ds = BayesianFootball.load_scottish_data("24/25", split_week=0)
model_struct = BayesianFootball.Models.PreGame.StaticPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model_struct)

splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["24/25"], 
    round_col = :split_col 
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
fs = BayesianFootball.Features.create_features(data_splits, vocabulary, model_struct, splitter_config)

# ==============================================================================
# 3. Instantiation
# ==============================================================================

# Helper to prepare data (assuming this exists in your package)
data = BayesianFootball.Models.PreGame.TuringHelpers.prepare_data(model_struct, fs[1][1])

# Build Instances
# turing_m1 = static_poisson_v1(data.n_teams, data.flat_home_ids, data.flat_away_ids, data.flat_home_goals, data.flat_away_goals)
# turing_m2 = static_poisson_v2(data.n_teams, data.flat_home_ids, data.flat_away_ids, data.flat_home_goals, data.flat_away_goals)
#

turing_m1 = BayesianFootball.Models.PreGame.build_turing_model(model_struct, fs[1][1]) 
turing_m2 = BayesianFootball.Models.PreGame.build_turing_model(model_struct, fs[1][1], Val(:v2)) 


# Create LogDensityFunctions (wraps model + data into a function f(θ))
ldf_1 = Turing.LogDensityFunction(turing_m1)
ldf_2 = Turing.LogDensityFunction(turing_m2)

# Generate a valid parameter set θ to test with
theta_1 = Float64.(randn(LogDensityProblems.dimension(ldf_1)))
theta_2 = Float64.(randn(LogDensityProblems.dimension(ldf_2)))

# ==============================================================================
# 4. Tape Compilation & Benchmarking
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

# Run Benchmarks
bench_v1 = benchmark_tape(ldf_1, theta_1, "Version 1: MvNormal (Identity)")
bench_v2 = benchmark_tape(ldf_2, theta_2, "Version 2: filldist (Optimized)")

println("\nDone.")

""" 
# run results: 

# model v1 

run 1:
julia> bench_v1 = benchmark_tape(ldf_1, theta_1, "Version 1: MvNormal (Identity)")

=== Benchmarking: Version 1: MvNormal (Identity) ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  235.992 μs …  5.972 ms  ┊ GC (min … max): 0.00% … 95.41%
 Time  (median):     244.789 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   244.830 μs ± 57.539 μs  ┊ GC (mean ± σ):  0.23% ±  0.95%

              ▁▆█▅                  ▆▆▅▁                        
  ▂▁▁▂▂▂▂▂▂▂▃▅█████▅▃▃▂▃▄▄▅▅▆▇▇▆▆▄▅█████▆▅▅▅▆▅▅▄▅▅▆▆▄▄▃▃▃▃▃▃▂▂ ▄
  236 μs          Histogram: frequency by time          252 μs <

 Memory estimate: 19.52 KiB, allocs estimate: 9.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

- run 2: 
=== Benchmarking: Version 1: MvNormal (Identity) ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  233.618 μs …  3.717 ms  ┊ GC (min … max): 0.00% … 92.92%
 Time  (median):     236.423 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   237.932 μs ± 34.990 μs  ┊ GC (mean ± σ):  0.15% ±  0.93%

       ▆▇█▇▅▃▂                                                  
  ▁▁▃▅████████▇▆▄▄▃▄▄▅▅▇▇▇▇▆▇▆▄▄▃▃▃▄▄▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁ ▃
  234 μs          Histogram: frequency by time          246 μs <

 Memory estimate: 19.52 KiB, allocs estimate: 9.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

run 3: 
=== Benchmarking: Version 1: MvNormal (Identity) ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  245.500 μs …  4.021 ms  ┊ GC (min … max): 0.00% … 93.19%
 Time  (median):     247.494 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   249.631 μs ± 38.054 μs  ┊ GC (mean ± σ):  0.15% ±  0.93%

      ▂▇█▇▄                                                     
  ▁▂▃▆█████▇▄▂▂▁▁▁▁▁▂▃▄▅▇▇▆▅▄▃▃▂▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁ ▂
  246 μs          Histogram: frequency by time          258 μs <

 Memory estimate: 19.52 KiB, allocs estimate: 9.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)


# --- model v2 
julia> bench_v2 = benchmark_tape(ldf_2, theta_2, "Version 2: filldist (Optimized)")

=== Benchmarking: Version 2: filldist (Optimized) ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  249.387 μs …  3.787 ms  ┊ GC (min … max): 0.00% … 92.61%
 Time  (median):     251.591 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   253.313 μs ± 35.857 μs  ┊ GC (mean ± σ):  0.14% ±  0.93%

        ▃▆█▇▄▁                                                  
  ▁▁▁▂▃▆██████▅▄▃▂▁▁▁▂▂▂▃▄▅▆▆▆▅▄▃▂▂▂▁▂▂▂▂▂▂▂▁▁▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  249 μs          Histogram: frequency by time          261 μs <

 Memory estimate: 19.52 KiB, allocs estimate: 9.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

run 2:
=== Benchmarking: Version 2: filldist (Optimized) ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  237.124 μs …  3.769 ms  ┊ GC (min … max): 0.00% … 92.54%
 Time  (median):     238.938 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   240.866 μs ± 35.518 μs  ┊ GC (mean ± σ):  0.14% ±  0.93%

       ▆█▇▄                                                     
  ▁▁▂▄█████▇▄▃▂▂▁▁▁▁▂▂▂▃▄▆▇▅▄▃▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
  237 μs          Histogram: frequency by time          249 μs <

 Memory estimate: 19.52 KiB, allocs estimate: 9.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

run 3:
=== Benchmarking: Version 2: filldist (Optimized) ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 10000 samples with 1 evaluation per sample.
 Range (min … max):  243.586 μs …  3.689 ms  ┊ GC (min … max): 0.00% … 92.60%
 Time  (median):     245.641 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   247.601 μs ± 34.845 μs  ┊ GC (mean ± σ):  0.14% ±  0.93%

      ▁▆█▇▃                                                     
  ▁▁▂▅██████▅▃▃▂▂▂▂▁▂▂▂▄▆▇▇▅▄▃▃▂▂▂▃▃▃▂▂▂▂▂▂▂▂▂▂▂▁▁▁▂▂▂▁▁▁▁▁▁▁▁ ▂
  244 μs          Histogram: frequency by time          256 μs <

 Memory estimate: 19.52 KiB, allocs estimate: 9.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

"""



# ==============================================================================
# WORKSPACE: GRW Gradient Benchmarking
# Goal: Compare CartesianIndex (V1) vs Linear Indexing (V2) for Dynamic Models
# ==============================================================================

using BayesianFootball
using DataFrames, Statistics, LinearAlgebra
using ThreadPinning
using BenchmarkTools
using Turing, DynamicPPL, ReverseDiff, LogDensityProblems

pinthreads(:cores)
BLAS.set_num_threads(1) 

# 1. Load Data
ds = BayesianFootball.load_scottish_data("24/25", split_week=0)
model_struct = BayesianFootball.Models.PreGame.GRWPoisson()
vocabulary = BayesianFootball.Features.create_vocabulary(ds, model_struct)

# Create "Fake" full season data to stress test the Random Walk
# We want to test the loop over time
splitter_config = BayesianFootball.Data.StaticSplit(train_seasons = ["24/25"], round_col = :split_col)
data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)
fs = BayesianFootball.Features.create_features(data_splits, vocabulary, model_struct, splitter_config)

# 2. Model Definitions
# ==============================================================================

# VERSION 1: Your implementation (CartesianIndex)
@model function grw_v1(n_teams, n_rounds, flat_home_ids, flat_away_ids, 
                       flat_home_goals, flat_away_goals, time_indices)
    
    # --- Priors ---
    σ_att ~ Truncated(Normal(0, 0.05), 0, Inf)
    σ_def ~ Truncated(Normal(0, 0.05), 0, Inf)
    home_adv ~ Normal(log(1.3), 0.2)

    # --- Random Walk ---
    z_att_init ~ filldist(Normal(0, 1), n_teams)
    z_def_init ~ filldist(Normal(0, 1), n_teams)
    z_att_steps ~ filldist(Normal(0, 1), n_teams, n_rounds - 1)
    z_def_steps ~ filldist(Normal(0, 1), n_teams, n_rounds - 1)

    # Reconstruction (allocates hcat)
    att_raw = cumsum(hcat(z_att_init .* 0.5, z_att_steps .* σ_att), dims=2)
    def_raw = cumsum(hcat(z_def_init .* 0.5, z_def_steps .* σ_def), dims=2)

    att = att_raw .- mean(att_raw, dims=1)
    def = def_raw .- mean(def_raw, dims=1)

    # --- Likelihood (Cartesian View) ---
    # This creates a vector of objects, then views into the matrix
    att_h = view(att, CartesianIndex.(flat_home_ids, time_indices))
    def_a = view(def, CartesianIndex.(flat_away_ids, time_indices))
    att_a = view(att, CartesianIndex.(flat_away_ids, time_indices))
    def_h = view(def, CartesianIndex.(flat_home_ids, time_indices))

    log_λs = home_adv .+ att_h .+ def_a
    log_μs = att_a .+ def_h

    flat_home_goals ~ arraydist(LogPoisson.(log_λs))
    flat_away_goals ~ arraydist(LogPoisson.(log_μs))
end

# VERSION 2: Linear Indexing (Tape Optimized)
# ReverseDiff often prefers simple integer arrays over Arrays of CartesianIndexes
@model function grw_v2(n_teams, n_rounds, flat_home_ids, flat_away_ids, 
                       flat_home_goals, flat_away_goals, time_indices)
    
    σ_att ~ Truncated(Normal(0, 0.05), 0, Inf)
    σ_def ~ Truncated(Normal(0, 0.05), 0, Inf)
    home_adv ~ Normal(log(1.3), 0.2)

    z_att_init ~ filldist(Normal(0, 1), n_teams)
    z_def_init ~ filldist(Normal(0, 1), n_teams)
    z_att_steps ~ filldist(Normal(0, 1), n_teams, n_rounds - 1)
    z_def_steps ~ filldist(Normal(0, 1), n_teams, n_rounds - 1)

    att_raw = cumsum(hcat(z_att_init .* 0.5, z_att_steps .* σ_att), dims=2)
    def_raw = cumsum(hcat(z_def_init .* 0.5, z_def_steps .* σ_def), dims=2)

    att = att_raw .- mean(att_raw, dims=1)
    def = def_raw .- mean(def_raw, dims=1)

    # --- Likelihood (Linear Indexing Optimization) ---
    # Convert 2D lookup (row, col) into 1D lookup (idx)
    # Formula: idx = row + (col - 1) * n_rows
    # We broadcast this calculation. It uses basic math operations which AD loves.
    
    idx_h = flat_home_ids .+ (time_indices .- 1) .* n_teams
    idx_a = flat_away_ids .+ (time_indices .- 1) .* n_teams

    # Direct 1D access (Reshapes implicitly if needed, or treated as flat memory)
    att_h = att[idx_h]
    def_a = def[idx_a]
    att_a = att[idx_a]
    def_h = def[idx_h]

    log_λs = home_adv .+ att_h .+ def_a
    log_μs = att_a .+ def_h

    flat_home_goals ~ arraydist(LogPoisson.(log_λs))
    flat_away_goals ~ arraydist(LogPoisson.(log_μs))
end

# 3. Instantiate & Prepare
# ==============================================================================
data_raw = fs[1][1].data # Access the dictionary directly

# Flatten helpers (as per your script)
f_home = vcat(data_raw[:round_home_ids]...);
f_away = vcat(data_raw[:round_away_ids]...);
f_hg = vcat(collect.(data_raw[:round_home_goals])...);
f_ag = vcat(collect.(data_raw[:round_away_goals])...);

t_indices = Int[]
for (t, r) in enumerate(data_raw[:round_home_ids])
    append!(t_indices, fill(t, length(r)))
end

n_t = data_raw[:n_teams]
n_r = data_raw[:n_rounds]

# Build Models
m1 = grw_v1(n_t, n_r, f_home, f_away, f_hg, f_ag, t_indices)
m2 = grw_v2(n_t, n_r, f_home, f_away, f_hg, f_ag, t_indices)

ldf_1 = Turing.LogDensityFunction(m1)
ldf_2 = Turing.LogDensityFunction(m2)

theta_1 = Float64.(randn(LogDensityProblems.dimension(ldf_1)));
theta_2 = Float64.(randn(LogDensityProblems.dimension(ldf_2)));

# 4. Benchmark Function
# ==============================================================================
function benchmark_tape(ldf, theta, name)
    println("\n=== Benchmarking: $name ===")
    
    tape = ReverseDiff.GradientTape(x -> LogDensityProblems.logdensity(ldf, x), theta)
    compiled_tape = ReverseDiff.compile(tape)
    
    result_buffer = similar(theta)
    b = @benchmark ReverseDiff.gradient!($result_buffer, $compiled_tape, $theta)
    display(b)
end

benchmark_tape(ldf_1, theta_1, "V1: CartesianIndex")
benchmark_tape(ldf_2, theta_2, "V2: Linear Indexing")



""" 
# run grw model results: 

# model v1 ++++++++++++++++++++++++++++++++++

run 1: ------
julia> benchmark_tape(ldf_1, theta_1, "V1: CartesianIndex")                                                                           
                                                                                                                                      
=== Benchmarking: V1: CartesianIndex ===                                                                                              
BenchmarkTools.Trial: 1296 samples with 1 evaluation per sample.                                                                      
 Range (min … max):  3.628 ms …   7.788 ms  ┊ GC (min … max): 0.00% … 46.60%                                                          
 Time  (median):     3.843 ms               ┊ GC (median):    0.00%                                                                   
 Time  (mean ± σ):   3.854 ms ± 159.220 μs  ┊ GC (mean ± σ):  0.12% ±  1.60%                                                          
                                                                                                                                      
                                    ▃▅▇█▅▁                                                                                            
  ▂▁▁▁▁▁▁▁▁▂▁▁▁▁▁▁▂▁▁▁▁▂▁▂▁▂▂▂▁▁▃▄▅▇██████▆▆▅▄▃▃▂▃▂▂▂▂▁▂▂▂▁▁▂ ▃                                                                       
  3.63 ms         Histogram: frequency by time        3.97 ms <                                                                       
                                                                                                                                      
 Memory estimate: 120.25 KiB, allocs estimate: 68.     


run 2: ------

julia> benchmark_tape(ldf_1, theta_1, "V1: CartesianIndex")

=== Benchmarking: V1: CartesianIndex ===
BenchmarkTools.Trial: 1062 samples with 1 evaluation per sample.
 Range (min … max):  4.607 ms …   8.651 ms  ┊ GC (min … max): 0.00% … 41.80%
 Time  (median):     4.689 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   4.705 ms ± 177.857 μs  ┊ GC (mean ± σ):  0.07% ±  1.28%

            ▃▅▅▆▅▆█▃▅▂                                                                                                                
  ▂▂▃▃▂▃▄▄▆▇████████████▇▇▄▅▅▄▃▃▃▃▃▃▃▂▁▂▂▃▂▁▂▂▂▁▁▁▁▂▁▁▂▁▁▁▁▁▂ ▄                                                                       
  4.61 ms         Histogram: frequency by time        4.91 ms <                                                                       
                                                                                                                                      
 Memory estimate: 120.25 KiB, allocs estimate: 68.   

run 3: ------

julia> benchmark_tape(ldf_1, theta_1, "V1: CartesianIndex")

=== Benchmarking: V1: CartesianIndex ===
BenchmarkTools.Trial: 936 samples with 1 evaluation per sample.
 Range (min … max):  5.131 ms …   9.510 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     5.312 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   5.335 ms ± 161.569 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                 ▃▄██▇▂                                        
  ▂▁▁▁▂▁▂▂▂▃▃▁▃▃▆██████▇▆▅▄▄▄▃▄▄▃▃▂▃▃▂▂▂▂▂▁▃▂▂▂▁▂▂▁▁▂▁▁▂▁▁▂▂▂ ▃
  5.13 ms         Histogram: frequency by time        5.69 ms <

 Memory estimate: 120.25 KiB, allocs estimate: 68.



# model v2 ++++++++++++++++++++++++++++++++++

run 1: ------

julia> benchmark_tape(ldf_2, theta_2, "V2: Linear Indexing")                                                                          
                                                                                                                                      
=== Benchmarking: V2: Linear Indexing ===                                                                                             
BenchmarkTools.Trial: 1043 samples with 1 evaluation per sample.                                                                      
 Range (min … max):  4.396 ms …  10.043 ms  ┊ GC (min … max): 0.00% … 35.49%                                                          
 Time  (median):     4.777 ms               ┊ GC (median):    0.00%                                                                   
 Time  (mean ± σ):   4.791 ms ± 209.526 μs  ┊ GC (mean ± σ):  0.07% ±  1.10%                                                          
                                                                                                                                      
                                      ▂▆▇█▄                                                                                           
  ▂▁▁▁▁▁▁▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▂▁▁▁▁▁▁▁▂▂▂▃▅██████▆▄▃▃▃▂▂▂▁▂▂▂▁▂▁▁▂ ▃                                                                       
  4.4 ms          Histogram: frequency by time        4.97 ms <                                                                       
                                                                                                                                      
 Memory estimate: 120.25 KiB, allocs estimate: 68.    

run 2: ------


julia> benchmark_tape(ldf_2, theta_2, "V2: Linear Indexing")

=== Benchmarking: V2: Linear Indexing ===
BenchmarkTools.Trial: 837 samples with 1 evaluation per sample.
 Range (min … max):  5.840 ms …   8.976 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     5.959 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   5.973 ms ± 149.008 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

              ▄▆▇▃▇██▆▃                                         
  ▂▁▃▃▂▂▃▃▃▄▇████████████▆▄▄▃▃▃▃▂▃▃▂▂▂▃▃▂▁▂▂▂▂▁▂▁▁▂▁▂▁▁▁▂▁▁▁▂ ▃
  5.84 ms         Histogram: frequency by time        6.25 ms <

 Memory estimate: 120.25 KiB, allocs estimate: 68.

run 3: ------

julia> benchmark_tape(ldf_2, theta_2, "V2: Linear Indexing")

=== Benchmarking: V2: Linear Indexing ===
BenchmarkTools.Trial: 817 samples with 1 evaluation per sample.
 Range (min … max):  5.949 ms …   9.101 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     6.110 ms               ┊ GC (median):    0.00%
 Time  (mean ± σ):   6.120 ms ± 129.452 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                            ▁▁ ▁▂▄▅█▄▆▅▂▁▄▃▃                   
  ▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▃▂▃▃▆▆██████████████████▆▇▆▆▅▄▃▃▂▂▃▂▃▂▂ ▄
  5.95 ms         Histogram: frequency by time        6.22 ms <

 Memory estimate: 120.25 KiB, allocs estimate: 68.

"""
