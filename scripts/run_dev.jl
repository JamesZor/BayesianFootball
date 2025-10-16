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
data_store = BayesianFootball.Data.load_default_datastore()
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


# This now works out-of-the-box
using Turing
data_store = BayesianFootball.Data.load_default_datastore()
feature_set = BayesianFootball.Features.create_features(data_store)

# --- 1. Define Models ---
static_poisson_model = BayesianFootball.Models.PreGame.PregameModel(
  BayesianFootball.Models.PreGame.PoissonGoal(),
  BayesianFootball.Models.PreGame.Static(),
  true
)

# --- 2. Build and Sample ---
turing_model_1 = BayesianFootball.Models.PreGame.build_turing_model(static_poisson_model, feature_set)
chain_1 = sample(turing_model_1, NUTS(), 10)


static_nb_model = BayesianFootball.Models.PreGame.PregameModel(
  BayesianFootball.Models.PreGame.NegativeBinomialGoal(),
  BayesianFootball.Models.PreGame.Static(),
  true
)

# --- 2. Build and Sample ---
turing_model_2 = BayesianFootball.Models.PreGame.build_turing_model(static_nb_model, feature_set)
chain_1 = sample(turing_model_2, NUTS(), 10)

dynamic_poisson_model = BayesianFootball.Models.PreGame.PregameModel(
  BayesianFootball.Models.PreGame.PoissonGoal(),
  BayesianFootball.Models.PreGame.Static(),
  true
)
dynamic_nb_model = PregameModel(NegativeBinomial(), AR1(), true)
turing_model_2 = build_turing_model(dynamic_nb_model, feature_set)
chain_2 = sample(turing_model_2, NUTS(), 100)





#######
using Revise

# Load your package
using BayesianFootball
using Turing

# --- Prepare Data (this part is unchanged) ---
println("--- Loading data and features ---")
data_store = BayesianFootball.Data.load_default_datastore()
feature_set = BayesianFootball.Features.create_features(data_store)

# --- NEW, SIMPLER WORKFLOW ---
println("\n--- Running the Static Poisson Model ---")

# 1. Instantiate the concrete model struct
static_poisson = BayesianFootball.Models.PreGame.StaticPoisson()

# 2. Build the Turing model by calling the dispatched method
turing_model_1 = BayesianFootball.Models.PreGame.build_turing_model(static_poisson, feature_set)
println("✅ Static Poisson model built.")

# 3. Sample
chain_1 = sample(turing_model_1, NUTS(), 10)
println(chain_1)


println("\n--- Running the Static Simplex Poisson Model ---")

# 1. Instantiate the new model struct
static_simplex_poisson = BayesianFootball.Models.PreGame.StaticSimplexPoisson()

# 2. Build the model
turing_model_4 = BayesianFootball.Models.PreGame.build_turing_model(static_simplex_poisson, feature_set)
println("✅ Static Simplex Poisson model built.")

# 3. Sample
chain_4 = sample(turing_model_4, NUTS(), 10)
println(chain_4)


println("\n--- Running the Hierarchical Simplex Poisson Model ---")

# 1. Instantiate the new model struct
simplex_poisson = BayesianFootball.Models.PreGame.HierarchicalSimplexPoisson()

# 2. Build the model
turing_model_3 = BayesianFootball.Models.PreGame.build_turing_model(simplex_poisson, feature_set)
println("✅ Hierarchical Simplex Poisson model built.")

# 3. Sample
chain_3 = sample(turing_model_3, NUTS(), 10)
println(chain_3)
