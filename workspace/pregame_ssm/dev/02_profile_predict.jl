

using Profile
using BenchmarkTools

# 1. Isolate the prediction logic
function profile_prediction()
    return BayesianFootball.predict(
        ssm_neg_m.config.model_def,
        chain,
        features,
        mapping
    )
end

# 2. Run a warmup
println("Running a warmup prediction...")
profile_prediction() 

# 3. Clear any previous profiling data
Profile.clear_malloc_data() # Use this function to clear allocation data

# 4. Profile the code (Corrected syntax)
println("\nProfiling memory allocations...")
@profile profile_prediction()

# 5. Print the allocation profile
println("\n--- Allocation Profile (Top 10 lines) ---")
# We'll use the Profile module's standard print function and tell it to focus on memory
# Note: The output format might vary slightly with Julia versions, but this is the modern way.
Profile.print(format = :flat, sortedby = :allocs, noisefloor=2.0)

println("\n--- BenchmarkTools Summary ---")
@btime profile_prediction();


