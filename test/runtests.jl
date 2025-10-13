# Import the necessary modules that will be used across all tests
using Test
using BayesianFootball
using DataFrames, Dates, InlineStrings # Add any other packages your tests need here

# Start the main test suite for the entire package
@testset "BayesianFootball.jl Tests" begin

    println("running data module tests...")
    include("data_tests.jl")

    println("Running Features Module tests...")
    include("features_tests.jl")

end

