module BayesianFootball

# Using DataFrames, CSV, and Dates from the Data module
using DataFrames, CSV, Dates

# Include the source code for the Data submodule
include("data/data-module.jl")

# Make the Data module and its functions available to the user
export Data

end
