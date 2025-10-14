# scripts/run_dev.jl

# Activate the project environment
using Pkg
Pkg.activate(".")

# Load Revise.jl for automatic code reloading
using Revise

# Load your package
using BayesianFootball

# --- Example Usage of the Data Module ---
# Define the path to your data (adjust if necessary)
# You might need to go up one level from the scripts directory
data_path = BayesianFootball.Data.DataPaths.scotland

# Create DataFiles and DataStore objects
try
    data_files = BayesianFootball.Data.DataFiles(data_path)
    println("Successfully created DataFiles object.")
    
    data_store = BayesianFootball.Data.DataStore(data_files)
    println("Successfully created DataStore object.")
    
    # Display the first few rows of the matches DataFrame
    println("\n--- Sample of Matches Data ---")
    println(first(data_store.matches, 5))
    
catch e
    println("An error occurred: ", e)
    println("\nPlease ensure the data path is correct and the CSV files exist.")
end



# --- Example Usage of the Feature Module ---

f = BayesianFootball.Features.create_features(data_store)


# --- Example Usage of the pregame model Module ---

model = BayesianFootball.Models.PreGame.PregameModel(
    BayesianFootball.Models.PreGame.PoissonGoal(),
    BayesianFootball.Models.PreGame.AR1(),
    true
)

model_1 = BayesianFootball.Models.PreGame.PregameModel(
    BayesianFootball.Models.PreGame.PoissonGoal(),
    BayesianFootball.Models.PreGame.Static(),
    true
)


# --- 2. Define a Model ---
static_model = BayesianFootball.Models.PreGame.PregameModel(
  BayesianFootball.Models.PreGame.PoissonGoal(),
  BayesianFootball.Models.PreGame.Static(),
  true
)


feature_set = f
# --- 3. Build the Turing Model ---
# This calls our API to create the actual @model block
turing_model = BayesianFootball.Models.PreGame.build_turing_model(static_model, feature_set)
println("✅ Turing model built successfully.")

# --- 4. Sample from the Model ---
using Turing
# We use the NUTS sampler, a standard choice for this kind of model.
# We'll run it for 1000 iterations: 200 for warmup and 800 for sampling.
chain = sample(turing_model, NUTS(0.65), 10)
println("✅ Sampling complete!")

# --- 5. Inspect the Results ---
println("\n--- MCMC Chain Summary ---")
# Printing the 'chain' object gives a nice summary of the posterior distributions
# for all the parameters in our model.
println(chain)
