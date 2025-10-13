module BayesianFootball

# Using DataFrames, CSV, and Dates from the Data module
# using DataFrames, CSV, Dates

# abstract type AbstractFootballModel end
# Include the source code for the Data submodule
include("data/data-module.jl")
include("features/features-module.jl")

include("models/models-module.jl")
# Make the modules and their functions available to the user

export Data, Features, Models, AbstractFootballModel

end
