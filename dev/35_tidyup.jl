using Revise 
using BayesianFootball

using DataFrames
using Statistics
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)

# BLAS.set_num_threads(1) 


data_store = BayesianFootball.Data.load_default_datastore()

ds = BayesianFootball.Data.DataStore( 
    Data.add_match_week_column(data_store.matches),
    data_store.odds,
    data_store.incidents
)

cv_config = BayesianFootball.Data.CVConfig(
    tournament_ids = [55],
    target_seasons = ["22/23"],
    history_seasons = 0, # Will auto-include "23/24" if available
    dynamics_col = :match_week,
  warmup_period = 33,
    stop_early = false
)

splits = Data.create_data_splits(ds, cv_config)

model = BayesianFootball.Models.PreGame.StaticPoisson()


vocabulary = BayesianFootball.Features.create_vocabulary(ds, model) 


feature_sets = BayesianFootball.Features.create_features(
    splits, vocabulary, model, cv_config
)





# sampler_conf = BayesianFootball.Samplers.NUTSConfig(n_samples=100, n_chains=2, n_warmup=100) # Use renamed struct

# init_conf = Samplers.MapInit(50,0.001) 
#
# sampler_conf = Samplers.NUTSConfig(
#                 100,
#                 2,
#                 100,
#                 0.65,
#                 10,
#                 init_conf 
# )
#
#
# training_config = Training.TrainingConfig(sampler_conf, train_cfg)
#
#
#
# results = Training.train(model, training_config, feature_sets)
#

train_cfg = BayesianFootball.Training.Independent(parallel=true, max_concurrent_splits=1) 

sampler_conf = Samplers.NUTSConfig(
                100,
                2,
                100,
                0.65,
                10,
  Samplers.UniformInit(-0.05, 0.05)
)


training_config = Training.TrainingConfig(sampler_conf, train_cfg)

results = Training.train(model, training_config, feature_sets)



"""
# ----------------------------
# training 
"""



"""
Test 1: Enable Checkpointing & Verify Files

First, we modify your configuration to write to a local tmp folder. Run this script:

"""

# 1. Setup Config with Checkpointing ENABLED
# We set cleanup=false so we can inspect the files manually after the run
training_config = BayesianFootball.Training.TrainingConfig(
    sampler = sampler_conf,
    strategy = train_cfg,
    checkpoint_dir = "./data/tmp_checkpoints",   # <--- NEW: Local folder
    cleanup_checkpoints = false             # <--- NEW: Keep files for inspection
)

println("Test 1: Running Training with Checkpoints...")
results_run_1 = BayesianFootball.Training.train(model, training_config, feature_sets)

# 2. Verification
println("\nVerifying Checkpoints...")
if isdir("./data/tmp_checkpoints")
    files = readdir("./data/tmp_checkpoints")
    println("   Found $(length(files)) files in checkpoint dir: $files")
    
    # Check if we have the expected number of .jls files (should match splits length)
    expected = length(feature_sets)
    actual = count(f -> endswith(f, ".jls"), files)
    
    if expected == actual
        println("   ✅ SUCCESS: Created $actual checkpoint files for $expected splits.")
    else
        println("   ❌ FAIL: Expected $expected checkpoints, found $actual.")
    end
else
    println("   ❌ FAIL: Checkpoint directory was not created.")
end


"""
Test 2: The "Resume" Capability

Now we simulate a crash/restart. We will re-run the exact same command.

    Expected Behavior: The system should see the files in ./tmp_checkpoints, print a message like "Found X checkpoints. Resuming...", and finish instantly without running the sampler again.

"""

println("\n🧪 Test 2: Resuming (Should skip all work)...")

# We use the SAME config (pointing to the existing ./tmp_checkpoints)
@time results_run_2 = BayesianFootball.Training.train(model, training_config, feature_sets)

# Verification
# If the time is near 0.0 seconds, it worked.
# You should see logs: "✅ All splits already completed via checkpoints."


"""
Test 3: Resilience (Simulate Partial Failure)

Now we delete one file to simulate a crash that happened partway through.

    Expected Behavior: The system should re-train only the missing split.

"""


println("\n Test 3: Partial Resume (Simulating crash)...")

# 1. Delete the checkpoint for Split #1 (or any index)
split_to_delete = 2
file_to_delete = joinpath("./data/tmp_checkpoints", "split_$(lpad(split_to_delete, 3, '0')).jls")
rm(file_to_delete; force=true)

println("   Deleted checkpoint: $file_to_delete")

# 2. Run Training Again
println("   Re-running training...")
results_run_3 = BayesianFootball.Training.train(model, training_config, feature_sets)

# Verification
# Look at the logs. You should see:
# "Starting Independent training for 1 splits..." (instead of total splits)
# And it should verify that only Split 1 was processed.


println("\n Test 4: Cleanup...")

# 1. Update config to enable cleanup
cleanup_config = BayesianFootball.Training.TrainingConfig(
    sampler = sampler_conf,
    strategy = train_cfg,
    checkpoint_dir = "./data/tmp_checkpoints",
    cleanup_checkpoints = true  # <--- NEW: Delete after success
)

# 2. Run (will likely resume Split 1 if Test 3 finished, or just finish instantly)
BayesianFootball.Training.train(model, cleanup_config, feature_sets)

# 3. Verify directory is gone/empty
if !isdir("./tmp_checkpoints") || isempty(readdir("./data/tmp_checkpoints"))
    println("   ✅ SUCCESS: Checkpoint directory cleaned up.")
else
    println("   ❌ FAIL: Checkpoint directory still exists/not empty.")
end
