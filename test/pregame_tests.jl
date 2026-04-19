@testset "Models Module" begin

    # Define a simple pre-game model for testing
    model = BayesianFootball.Models.PreGame.PregameModel(
        BayesianFootball.Models.PreGame.Poisson(),
        BayesianFootball.Models.PreGame.AR1(),
        true
    )

    @testset "Model Definition" begin
        @test model isa BayesianFootball.Models.PreGame.AbstractPregameModel
        @test model.home_advantage == true
    end

    @testset "Required Features API" begin
        features = BayesianFootball.Models.PreGame.required_features(model)
        
        # Check that it gathers features from all components
        @test :home_score in features
        @test :away_score in features
        @test :global_round in features
        @test :is_home in features # From home_advantage = true
    end

    @testset "Turing Model Instantiation" begin
        # Setup mock features to pass to the model builder
        mock_data_path = joinpath(@__DIR__, "mock_data")
        data_store = BayesianFootball.Data.DataStore(
            BayesianFootball.Data.DataFiles(mock_data_path)
        )
        feature_set = BayesianFootball.Features.create_features(data_store)

        # Test if the model can be instantiated without errors
        turing_model = BayesianFootball.Models.PreGame.build_turing_model(model, feature_set)
        
        @test turing_model isa Turing.Model
    end

end
