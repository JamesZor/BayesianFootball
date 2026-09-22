# Deterministic contract tests; no database, real data snapshot, or sampling.
using Test, DataFrames, Dates, Distributions
include(joinpath(@__DIR__, "l11_decompression_loader.jl"))
const D = DecompressionPXG

@testset "Proxy-xG rolling filtration" begin
    matches = DataFrame(
        match_id = [1, 2, 3, 4, 5, 6],
        match_date = Date.( ["2025-01-01", "2025-01-01", "2025-01-08",
                             "2025-01-08", "2025-01-15", "2025-01-15"]),
        match_hour = fill(15, 6),
        home_team = ["A", "C", "A", "B", "A", "B"],
        away_team = ["B", "D", "C", "D", "D", "C"],
    )
    observations = Dict(
        1 => (h = 2.0, a = 0.5, source = :commentary),
        2 => (h = 1.0, a = 1.0, source = :commentary),
        3 => (h = 1.5, a = 0.7, source = :commentary),
        4 => (h = 0.8, a = 1.2, source = :commentary),
        5 => (h = 50.0, a = 0.01, source = :commentary),
        6 => (h = 0.01, a = 50.0, source = :commentary),
    )
    feature = D.ProxyXGFormCovariate().feature
    original = D.GPH_FEATURES._pxg_rolling_lookup(observations, matches, feature)
    changed = copy(observations)
    changed[5] = (h = 5000.0, a = 5000.0, source = :commentary)
    changed[6] = (h = 5000.0, a = 5000.0, source = :commentary)
    perturbed = D.GPH_FEATURES._pxg_rolling_lookup(changed, matches, feature)
    @test original[1] == original[2] == (
        att_h = 0.0, att_a = 0.0, def_h = 0.0, def_a = 0.0,
        supremacy = 0.0, level = 0.0, available = 0.0)
    @test all(original[id] == perturbed[id] for id in 1:6)
    @test original[5].available == 1.0
    @test original[6].available == 1.0
end

@testset "Coefficient parameterisation and prior" begin
    covariate = D.ProxyXGFormCovariate()
    fake = (data = Dict{Symbol,Any}(
        :flat_pxg_supremacy => [2.0, -1.0, 0.0],
        :pxg_supremacy_by_match_id => Dict(1 => 2.0, 2 => -1.0),
    ),)
    design = D.GPH_PG.covariate_column(covariate, fake)
    @test design == [1.0, -0.5, 0.0]
    @test mean(D.GPH_PG.covariate_prior(covariate)) == 0.60
    @test std(D.GPH_PG.covariate_prior(covariate)) == 0.20
    q = 0.6 .* design
    home, away = D.GPH_PG.covariate_sides(D.GPH_PG.covariate_role(covariate), q)
    @test home - away == 0.6 .* [2.0, -1.0, 0.0]
end

@testset "Three-arm architecture" begin
    arms = Dict(D.models())
    @test arms["m01_poisson_time_decay"].observation isa D.PoissonObservation
    @test arms["m02_joint_gamma_poisson"].observation isa D.JointGammaPoissonObservation
    candidate = arms["m03_negbin_pxg_covariate"]
    @test candidate.observation isa D.NegativeBinomialObservation
    @test only(candidate.covariates) isa D.ProxyXGFormCovariate
    @test candidate.guard isa D.ArrayClampGuard
    @test D.B.observation_family(candidate.observation) == :negbin
end
