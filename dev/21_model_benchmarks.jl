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
using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1) 


data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.load_scottish_data("24/25", split_week=0)


model = BayesianFootball.Models.PreGame.StaticPoisson()

vocabulary = BayesianFootball.Features.create_vocabulary(ds, model)

splitter_config = BayesianFootball.Data.StaticSplit(
    train_seasons = ["24/25"], 
    round_col = :split_col 
)

data_splits = BayesianFootball.Data.create_data_splits(ds, splitter_config)

# feature_sets = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #


fs = BayesianFootball.Features.create_features(data_splits, vocabulary, model, splitter_config) #


# --- Benchmarking tools 

using BenchmarkTools
using DynamicPPL
using Turing
using ReverseDiff
using LogDensityProblems

# 1. Instantiate the model (You already did this successfully)
turing_model_v1 = BayesianFootball.Models.PreGame.build_turing_model(model, fs[1][1]) 
turing_model_v2 = BayesianFootball.Models.PreGame.build_turing_model(model, fs[1][1], :v2) 

# 2. Extract the current parameter values
vi_v1 = DynamicPPL.VarInfo(turing_model_v1)
vi_v2 = DynamicPPL.VarInfo(turing_model_v1)


initial_theta = vi[DynamicPPL.SampleFromPrior()]

ldf_v1 = Turing.LogDensityFunction(turing_model_v1)
ldf_v2 = Turing.LogDensityFunction(turing_model_v2)


dim_v1 = LogDensityProblems.dimension(ldf_v1)
dim_v2 = LogDensityProblems.dimension(ldf_v2)

initial_theta_v1 = randn(dim_v1);
initial_theta_v2 = randn(dim_v2);

# 3. Define the "Log Density" function
log_density_function_v1 = Turing.LogDensityFunction(turing_model_v1)
log_density_function_v2 = Turing.LogDensityFunction(turing_model_v2)

# 4. Benchmark
println("--- Benchmarking Gradient Calculation ---")
println("Parameter Count: $(length(initial_theta))")

# Test A: ReverseDiff (Compiled)
println("\nTesting ReverseDiff (Compiled)...")
try
    # Compile the tape
    tape = ReverseDiff.GradientTape(q -> Turing.LogDensityProblems.logdensity(log_density_function, q), initial_theta)
    compiled_tape = ReverseDiff.compile(tape)
    
    # Benchmark
    # We allocate the result buffer once to be fair
    result_buffer = similar(initial_theta)
    b_rev = @benchmark ReverseDiff.gradient!($result_buffer, $compiled_tape, $initial_theta)
    display(b_rev)
catch e
    println("ReverseDiff Failed: ", e)
end

