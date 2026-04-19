# dev/tape_analysis_funnel.jl

using Revise
using BayesianFootball
using DataFrames, Statistics, LinearAlgebra
using ThreadPinning
using BenchmarkTools
using Turing, DynamicPPL, ReverseDiff, LogDensityProblems
using InteractiveUtils # for @code_warntype
using JET 

# ==============================================================================
# 1. SETUP & DATA LOADING
# ==============================================================================
pinthreads(:cores)
BLAS.set_num_threads(1) 

println("Loading Data...")

# Using load_extra_ds() as Funnel model needs shots/SOT data, not just goals
ds = BayesianFootball.Data.load_extra_ds()

# Standard Preprocessing from your dev_funnel_model example
transform!(ds.matches, :match_week => ByRow(w -> cld(w, 4)) => :match_month)
# Filter for a specific season/tournament to keep the tape manageable for dev
df = DataFrames.subset(ds.matches, :tournament_id => ByRow(in(56)), :season => ByRow(isequal("24/25")))

# Config - keeping it short (warmup_period=2) for fast tape compilation during dev
cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [56],
    target_seasons = ["24/25"],
    history_seasons = 0,
    dynamics_col = :match_month, 
    warmup_period = 2,
    stop_early = true
)

println("Creating Splits & Features...")
# Create splits
splits = BayesianFootball.Data.create_data_splits(ds, cv_config)

# Instantiate the model struct
model_struct = BayesianFootball.Models.PreGame.SequentialFunnelModel()

# Create features (this populates :flat_home_shots, :flat_home_sot etc.)
feature_sets = BayesianFootball.Features.create_features(splits, model_struct, cv_config)

# Grab the first training set for analysis
# feature_sets is usually a Vector of Tuples (train, test), we want the first train set
feature_set_train = feature_sets[end][1] 

# ==============================================================================
# 2. MODEL INSTANTIATION
# ==============================================================================
println("Building Turing Model...")

# Build the Turing model using the specific builder for SequentialFunnelModel
turing_model = BayesianFootball.Models.PreGame.build_turing_model(model_struct, feature_set_train)

# Create LogDensityFunction (wraps model + data into f(θ))
ldf = Turing.LogDensityFunction(turing_model)

# Generate a valid parameter set θ (flat vector of Float64)
theta = Float64.(randn(LogDensityProblems.dimension(ldf)))

println("Model Dimension: $(length(theta))")

# ==============================================================================
# 3. TAPE BENCHMARKING
# ==============================================================================

function benchmark_tape(ldf, theta, name)
    println("\n=== Benchmarking: $name ===")
    
    # A. Define the function wrapper for ReverseDiff
    f_tape = x -> LogDensityProblems.logdensity(ldf, x)

    # B. Record the Tape (One-time cost)
    println("Recording tape...")
    tape = ReverseDiff.GradientTape(f_tape, theta)
    
    # C. Compile the Tape (One-time cost - optimizes instructions)
    println("Compiling tape...")
    compiled_tape = ReverseDiff.compile(tape)

    # D. Benchmark the Gradient Calculation
    result_buffer = similar(theta)
    
    # Use $ to interpolate variables to avoid global scope issues
    b = @benchmark ReverseDiff.gradient!($result_buffer, $compiled_tape, $theta)
    
    display(b)
    return compiled_tape
end

# Run the benchmark
# This tests the speed of one gradient step (what happens inside NUTS)
bench_funnel = benchmark_tape(ldf, theta, "Sequential Funnel Model (Compiled Tape)")
#=


=== Benchmarking: Sequential Funnel Model (Compiled Tape) ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 7088 samples with 1 evaluation per sample.
 Range (min … max):  689.449 μs …  1.980 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     700.630 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   702.204 μs ± 24.509 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

                     ▃▆██▅                                      
  ▁▁▁▁▁▁▁▁▂▂▃▅▇██▆▅▄▆█████▇▆▄▃▃▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▂▂▂▂▁ ▂
  689 μs          Histogram: frequency by time          721 μs <

 Memory estimate: 14.62 KiB, allocs estimate: 192.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

julia> bench_funnel = benchmark_tape(ldf, theta, "Sequential Funnel Model (Compiled Tape)")

=== Benchmarking: Sequential Funnel Model (Compiled Tape) ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 7653 samples with 1 evaluation per sample.
 Range (min … max):  640.358 μs …  1.530 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     648.934 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   651.434 μs ± 18.022 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

          ▃▆██▅▂ ▁▂▃▁                                           
  ▂▄▆▆█▇▇████████████▇▆▅▄▄▃▃▃▂▂▃▃▂▃▃▂▂▂▂▂▂▂▂▃▃▂▂▂▃▃▂▂▂▂▁▂▂▁▂▂▂ ▄
  640 μs          Histogram: frequency by time          682 μs <

 Memory estimate: 14.62 KiB, allocs estimate: 192.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)

julia> bench_funnel = benchmark_tape(ldf, theta, "Sequential Funnel Model (Compiled Tape)")

=== Benchmarking: Sequential Funnel Model (Compiled Tape) ===
Recording tape...
Compiling tape...
BenchmarkTools.Trial: 6599 samples with 1 evaluation per sample.
 Range (min … max):  740.674 μs …  2.639 ms  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     752.637 μs              ┊ GC (median):    0.00%
 Time  (mean ± σ):   755.766 μs ± 38.217 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

     ▁▁▄▆▅▄▇█▇▅▅▅▅▄▄▃▁▁▁▂▂▁▁▂▂                                 ▂
  ▃▅▇██████████████████████████▇▆▇▆▆▅▆▃▅▅▄▄▅▁▅▁▃▃▁▄▄▁▃▁▄▃▅▃▁▃▅ █
  741 μs        Histogram: log(frequency) by time       806 μs <

 Memory estimate: 14.62 KiB, allocs estimate: 192.
typename(ReverseDiff.CompiledTape)(#benchmark_tape##0)



=#

# ==============================================================================
# 4. STATIC ANALYSIS & TYPE STABILITY
# ==============================================================================

println("\n==============================================================================")
println("=== REPORT 1: @code_warntype (Type Stability) ===")
println("==============================================================================")
println("Checking 'logdensity' call. Look for RED text or 'Any' (performance killers)...\n")

# Inspect the log density call itself. 
# If this shows type instabilities, the tape recorder might generate inefficient code.
@code_warntype LogDensityProblems.logdensity(ldf, theta)

println("\n==============================================================================")
println("=== REPORT 2: JET Analysis (Dynamic Dispatch) ===")
println("==============================================================================")
println("Checking for runtime dispatch errors or performance pitfalls...\n")

# JET will detect if Julia can't infer types at compile time, leading to dynamic dispatch.
@report_opt LogDensityProblems.logdensity(ldf, theta)

println("\nDone.")
