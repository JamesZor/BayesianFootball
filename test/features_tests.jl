@testset "Features Module" begin

    # Setup: Create a DataStore using the mock data
    mock_data_path = joinpath(@__DIR__, "mock_data")
    data_files = BayesianFootball.Data.DataFiles(mock_data_path)
    data_store = BayesianFootball.Data.DataStore(data_files)

    # Create features from the data_store
    feature_set = BayesianFootball.Features.create_features(data_store)

    # Test 1: Check the type of the returned object
    @testset "FeatureSet Creation" begin
        @test feature_set isa BayesianFootball.Features.FeatureSet
    end

    # Test 2: Validate the team mappings
    @testset "Team Mappings" begin
        # In our mock data, we have 4 unique teams
        @test feature_set.n_teams == 4
        @test length(feature_set.team_to_id) == 4
        @test length(feature_set.id_to_team) == 4

        # Check if a specific team is mapped correctly
        @test haskey(feature_set.team_to_id, "motherwell")
    end

    # Test 3: Check for new columns in the DataFrame
    @testset "DataFrame Transformation" begin
        df = feature_set.data
        @test "home_team_id" in names(df)
        @test "away_team_id" in names(df)

        # Check that the IDs are integers
        @test eltype(df.home_team_id) == Int
    end

end
