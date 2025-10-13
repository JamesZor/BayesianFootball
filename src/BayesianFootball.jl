module BayesianFootball

# Using DataFrames, CSV, and Dates from the Data module
using DataFrames, CSV, Dates

# Include the source code for the Data submodule
include("data/data-module.jl")
include("features/features-module.jl")

# Make the modules and their functions available to the user
export Data, Features

end
