# Use @testset to group tests for the Data module
@testset "Data Module" begin
    
    # Define the path to our mock data directory
    # Note: @__DIR__ here refers to the 'test' directory, where runtests.jl is located
    mock_data_path = joinpath(@__DIR__, "mock_data")

    # Test 1: Can we create a DataFiles object correctly?
    @testset "File Path Handling" begin
        @test begin
            data_files = BayesianFootball.Data.DataFiles(mock_data_path)
            # Check if the object was created and if the paths are correct
            data_files isa BayesianFootball.Data.DataFiles &&
            basename(data_files.match) == "football_data_mixed_matches.csv"
        end
    end

    # Test 2: Can we create a DataStore from the mock files?
    @testset "DataStore Creation" begin
        @test begin
            data_files = BayesianFootball.Data.DataFiles(mock_data_path)
            data_store = BayesianFootball.Data.DataStore(data_files)
            # Check if we got a DataStore object with three DataFrames
            data_store isa BayesianFootball.Data.DataStore &&
            data_store.matches isa DataFrame &&
            data_store.odds isa DataFrame &&
            data_store.incidents isa DataFrame
        end
    end

    # Test 3: Are the column types correct after loading?
    @testset "Column Type Validation" begin
        data_files = BayesianFootball.Data.DataFiles(mock_data_path)
        data_store = BayesianFootball.Data.DataStore(data_files)
        matches_df = data_store.matches

        # Test 3a: Check score columns are subtypes of Union{Missing, Int}
        # This passes for both Int and Union{Missing, Int} column types.
        @test eltype(matches_df.home_score) <: Union{Missing, Int}
        @test eltype(matches_df.away_score) <: Union{Missing, Int}

        # Test 3b: Check match_date is a Date
        @test eltype(matches_df.match_date) == Date
        
        # Test 3c: Check team name is the correct InlineString type
        @test eltype(matches_df.home_team) == InlineStrings.String31
    end

end
