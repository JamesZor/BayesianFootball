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
        BayesianFootball.Models.PreGame.Poisson(),
        BayesianFootball.Models.PreGame.AR1(),
        true
    )
