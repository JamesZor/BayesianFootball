using Test
using BayesianFootball
using DataFrames
using Dates
using MCMCChains

include(joinpath(@__DIR__, "..", "experiments", "scotland",
                 "01_time_decay_cross_tier_and_priors", "l01_cross_tier_loader.jl"))

@testset "cross-tier Stage-1 deterministic contracts" begin
    @testset "candidate manifest and paired lower boundaries" begin
        candidates = cross_tier_smoke_candidates()
        @test length(candidates) == 5
        @test [candidate.name for candidate in candidates] == [
            "m00_control",
            "m01_all_spfl_league_offsets",
            "m02_all_spfl_hierarchical_tiers",
            "m03_prior_structural_offset",
            "m04_prior_market_derived",
        ]
        @test all(candidate.fit_config.execution isa BayesianFootball.AutoExecution
                  for candidate in candidates)
        @test all(candidate.splitter.dynamics_col === :match_biweek for candidate in candidates)
        @test all(candidate.splitter.warmup_period == 0 for candidate in candidates)
    end

    @testset "cold counts include fitted target rows" begin
        data = Dict{Symbol, Any}(
            :history_match_ids => Set([1]),
            :target_match_ids => Set([2]),
            :ordered_match_ids => [1, 2],
            :flat_home_ids => [1, 1],
            :flat_away_ids => [2, 2],
        )
        @test ct_cold_mask(data, Dict("home" => 1, "away" => 2)) == Bool[true, true]

        data[:target_match_ids] = Set([2, 3, 4, 5])
        data[:ordered_match_ids] = [1, 2, 3, 4, 5]
        data[:flat_home_ids] = fill(1, 5)
        data[:flat_away_ids] = fill(2, 5)
        @test ct_cold_mask(data, Dict("home" => 1, "away" => 2)) == Bool[false, false]
    end

    @testset "A2 vectorised tape and unknown roster refusal" begin
        fs = FeatureSet(Dict{Symbol, Any}(
            :flat_home_ids => Int[1, 2], :flat_away_ids => Int[2, 1],
            :flat_home_goals => Int[1, 0], :flat_away_goals => Int[0, 1],
            :dates => Float64[1, 0], :n_teams => 2,
            :first_observed_tier_by_team => Int[1, 4],
        ))
        model = CTM02HierarchicalTiers()
        @test BayesianFootball.Models.PreGame.build_turing_model(model, fs) !== nothing
        @test_throws ErrorException BayesianFootball.Models.PreGame.build_turing_model(
            model, FeatureSet(Dict{Symbol, Any}(
                :flat_home_ids => Int[1], :flat_away_ids => Int[2],
                :flat_home_goals => Int[0], :flat_away_goals => Int[0],
                :dates => Float64[0], :n_teams => 2,
                :first_observed_tier_by_team => Int[1, 0],
            )))
    end

    @testset "A2 ordered locations keep draws-by-teams shape" begin
        parameters = Symbol[:μ, :γ, :σ_a, :σ_d, Symbol("raw_a[1]"),
                            Symbol("raw_a[2]"), Symbol("raw_d[1]"),
                            Symbol("raw_d[2]"), :d_1, :d_2, :d_3]
        draws = zeros(2, length(parameters), 2)
        draws[:, 1, :] .= 0.2
        draws[:, 2, :] .= 0.1
        draws[:, 3:4, :] .= 1.0
        draws[:, 9, :] .= 0.1
        draws[:, 10, :] .= 0.2
        draws[:, 11, :] .= 0.3
        chain = Chains(draws, parameters)
        fs = FeatureSet(Dict{Symbol, Any}(
            :n_teams => 2, :team_map => Dict("h" => 1, "a" => 2),
            :first_observed_tier_by_team => [1, 4],
        ))
        oos = DataFrame(match_id = [42], home_team = ["h"], away_team = ["a"])
        rates = BayesianFootball.Models.PreGame.extract_parameters(
            CTM02HierarchicalTiers(), oos, fs, chain)[42]
        @test length(rates.λ_h) == 4
        @test rates.λ_h ≈ fill(exp(0.9), 4)
        @test rates.λ_a ≈ fill(exp(-0.4), 4)
    end

    @testset "two-chain rate extraction flattens in MCMCChains order" begin
        # draws=2, parameters=8, chains=2. Values are deliberately distinct by chain.
        names = Symbol[:μ, :γ, :σ_a, :σ_d, Symbol("raw_a[1]"), Symbol("raw_a[2]"),
                       Symbol("raw_d[1]"), Symbol("raw_d[2]")]
        values = Array{Float64}(undef, 2, length(names), 2)
        values[:, 1, :] .= 0.2
        values[:, 2, :] .= 0.1
        values[:, 3, :] .= 1.0
        values[:, 4, :] .= 1.0
        values[:, 5, 1] .= 0.5
        values[:, 6, 1] .= -0.5
        values[:, 7, 1] .= 0.25
        values[:, 8, 1] .= -0.25
        values[:, 5, 2] .= 1.0
        values[:, 6, 2] .= -1.0
        values[:, 7, 2] .= 0.5
        values[:, 8, 2] .= -0.5
        chain = Chains(values, names)
        fs = FeatureSet(Dict{Symbol, Any}(
            :n_teams => 2, :team_map => Dict("h" => 1, "a" => 2),
        ))
        oos = DataFrame(match_id = [42], home_team = ["h"], away_team = ["a"],
                        match_date = [Date(2025, 1, 1)])
        rates = BayesianFootball.Models.PreGame.extract_parameters(CTM00Control(), oos, fs, chain)
        @test length(rates[42].λ_h) == 4
        @test rates[42].λ_h[1] ≈ exp(0.2 + 0.1 + 0.5 - 0.25)
        @test rates[42].λ_h[3] ≈ exp(0.2 + 0.1 + 1.0 - 0.5)
    end
end
