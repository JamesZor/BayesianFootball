# Import the necessary modules that will be used across all tests
using Test
using BayesianFootball
using DataFrames, Dates, InlineStrings # Add any other packages your tests need here

# Start the main test suite for the entire package
@testset "BayesianFootball.jl Tests" begin

    println("Running Data Module tests...")
    include("data_tests.jl")

    # As you create new modules and their tests, you'll add them here.
    # For example:
    # println("Running Features Module tests...")
    # include("features_tests.jl")

end

