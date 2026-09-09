# Real-data deterministic tests only. No database access or MCMC.
using BayesianFootball, Test, DataFrames, LinearAlgebra, ThreadPinning
import TOML
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l08_workflow.jl"))
include(joinpath(@__DIR__, "l08_decomposed_models.jl"))
include(joinpath(@__DIR__, "l08_model_checks.jl"))

registry, sha = GoalDecompositionIncidentData.load_registry(joinpath(@__DIR__, "results"))
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)
splitter = l08_splitter()
boundaries = Data.create_id_boundaries(ds, splitter)
entries = l08_models(registry, sha; registry_hash = sha)
reports = Dict{String,Any}()
prepared = Dict{Symbol,Any}()

@testset "08 real Fold 1 deterministic contract (NOT MCMC)" begin
    @test length(boundaries) == 40
    for entry in entries
        @testset "$(entry.name)" begin
            original = Features.create_features(boundaries[1:1], ds, entry.model, splitter)
            oos = [Data.get_next_matches(ds, original[1], splitter)]
            declared = l08_declare_prediction_teams(original, oos)
            fs = first(declared[1])
            @test fs.data[:goal_decomposition_prior_only_teams] == ["arbroath", "inverness-caledonian-thistle"]
            @test fs.data[:goal_decomposition_training_ids] == first(original[1]).data[:goal_decomposition_training_ids]
            @test length(fs.data[:flat_home_ids]) == 720
            @test fs.data[:n_teams] == 25
            @test nrow(oos[1]) == 20
            # The declaration reads identities ONLY: corrupt every supplied outcome.
            changed = deepcopy(oos)
            changed[1].home_score .= 97
            changed[1].away_score .= 83
            redeclared = l08_declare_prediction_teams(original, changed)
            @test isequal(first(redeclared[1]).data, fs.data)
            report = l08_deterministic_checks(entry.model, fs, oos[1])
            @test report["parameters"] == l08_expected_params(entry.name, 25, fs.data[:n_referees])
            @test report["gradient_allocations"] == 0
            @test report["instructions"] == report["duplicate_instructions"]
            @test report["extraction"]["grid_allocations"] == 0
            # Logit evaluation remains finite when logistic itself rounds to 0/1.
            for logit in (-100.0, 100.0)
                @test all(isfinite, _gd_logbinomial([0.0, 1.0], [0.0, 2.0], [0.0, log(2.0)], [logit]))
            end
            reports[string(entry.name)] = report
            prepared[entry.name] = (; model = entry.model, fs)
            println(entry.name, " ", report)
        end
    end
    # Identical-spine control: remove labels from m01 and its complete density is m00.
    control, baseline = prepared[:m00_recombined_control], prepared[:m01_decomposed_baseline]
    data = deepcopy(baseline.fs.data)
    data[:incident_complete] .= 0.0
    pc = l08_logdensity_problem(control.model, control.fs)
    pb = l08_logdensity_problem(baseline.model, typeof(baseline.fs)(data))
    @test pc.θ == pb.θ
    @test pc.f(pc.θ) == pb.f(pc.θ)
end

source_files = ["l08_decomposed_models.jl", "l08_model_checks.jl", "test08_model_contract.jl"]
manifest = Dict{String,Any}("incident_sha" => sha, "julia_version" => string(VERSION),
    "models" => reports, "mcmc_executed" => false,
    "source_hashes" => Dict(name => l08_file_hash(joinpath(@__DIR__, name)) for name in source_files))
path = joinpath(@__DIR__, "results", "deterministic_checks_julia_$(VERSION).toml")
open(path, "w") do io
    TOML.print(io, manifest; sorted = true)
end
println("Deterministic evidence: ", path)
