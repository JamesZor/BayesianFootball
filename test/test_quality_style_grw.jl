using Test
using BayesianFootball
using DataFrames
using Distributions
using DynamicPPL
using ForwardDiff
using LinearAlgebra
using LogDensityProblems
using MCMCChains
using Random
using ReverseDiff
using Statistics

const QSPG = BayesianFootball.Models.PreGame
const QSAPI = QSPG.Builder

qs_model(name = :qs_test; config = QualityStyleGRW()) = CountModelBuilder(name) |>
    add(GlobalInterception()) |> add(config) |>
    add(GlobalHomeAdvantage()) |> add(PoissonObservation()) |> build

"A small fold with the same time-index/state-count contract as the real fold."
function qs_features(; n_history = 3, n_target = 2)
    n_rounds = n_history + n_target
    n = 8
    return BayesianFootball.FeatureSet(Dict{Symbol,Any}(
        :flat_home_ids => [mod1(i, 4) for i in 1:n],
        :flat_away_ids => [mod1(i + 1, 4) for i in 1:n],
        :season_indices => ones(Int, n),
        :time_indices => [mod1(i, n_rounds) for i in 1:n],
        :flat_months => [mod1(i, 12) for i in 1:n],
        :flat_home_goals => [mod(i, 3) for i in 1:n],
        :flat_away_goals => [mod(i + 1, 3) for i in 1:n],
        :dates => collect(0.0:(n - 1)),
        :n_teams => 4, :n_seasons => 1, :n_rounds => n_rounds,
        :n_history_steps => n_history, :n_target_steps => n_target,
        :team_map => Dict("t$i" => i for i in 1:4),
    ))
end

"Construct a posterior with distinct τ, r and z values, including a flat no-target fold."
function qs_chain(n_teams, n_history, n_target; draws = 5)
    columns = String["dyn.τ₀", "dyn.τₛ", "dyn.r₀", "dyn.rₛ"]
    if n_target > 0
        append!(columns, ["dyn.τₖ", "dyn.rₖ"])
    end
    for side in ("q", "s")
        append!(columns, ["dyn.$side.z_init[$i]" for i in 1:n_teams])
        append!(columns, ["dyn.$side.z_season[$i,$t]"
                          for t in 1:(n_history - 1) for i in 1:n_teams])
        append!(columns, ["dyn.$side.z_target[$i,$t]"
                          for t in 1:n_target for i in 1:n_teams])
    end
    values = zeros(Float64, draws, length(columns), 1)
    for (j, name) in enumerate(columns), k in 1:draws
        values[k, j, 1] = occursin("τ", name) ? 0.08 + 0.01k :
                          occursin(".r", name) ? 0.3 + 0.03k :
                          (j % 7 - 3) * 0.1 + 0.02k * (j % 3)
    end
    return Chains(values, Symbol.(columns))
end

@testset "QualityStyleGRW" begin
    @testset "interface, priors and state geometry" begin
        config = QualityStyleGRW()
        @test QualityStyleGRW === QSPG.QualityStyleGRW
        @test QualityStyleGRW <: QSPG.AbstractDynamicsConfig
        @test mean(config.τ₀) ≈ 0.16
        @test mean(config.τₛ) ≈ 0.085
        @test mean(config.τₖ) ≈ 0.027
        @test median(config.r₀) ≈ 0.3
        @test median(config.rₛ) ≈ 0.4
        @test median(config.rₖ) ≈ 0.3
        weak = QualityStyleGRW(r₀ = LogNormal(0, 1), rₛ = LogNormal(0, 1),
                               rₖ = LogNormal(0, 1))
        @test all(median(getproperty(weak, site)) == 1 for site in (:r₀, :rₛ, :rₖ))
        @test occursin("α = q + s", QSPG.math_formula(config))

        model = qs_model()
        @test model isa PoissonCountModel
        @test all(r.pass for r in validate(CountModelBuilder(:qs_validate) |>
            add(GlobalInterception()) |> add(config) |>
            add(GlobalHomeAdvantage()) |> add(PoissonObservation())))
        @test QSAPI.dynamics_match_weights(config, [0.0, 90.0]) == ones(2)
        sites = cb_varinfo_sites(model)
        for leaf in ("τ₀", "τₛ", "τₖ", "r₀", "rₛ", "rₖ")
            @test Symbol("dyn.$leaf") in sites
        end
        for side in ("q", "s"), leaf in ("z_init", "z_season", "z_target")
            @test Symbol("dyn.$side.$leaf") in sites
        end
        @test_throws ErrorException cb_chain_columns(model, 4)

        fs = qs_features()
        design = QSAPI.dynamics_design(config, fs, 8)
        reference = QSAPI.dynamics_design(MultiScaleGRW(), fs, 8)
        @test design isa GRWDynamicsDesign
        @test design.n_rounds == reference.n_rounds == 5
        @test design.n_history == reference.n_history == 3
        @test design.n_target == reference.n_target == 2
        @test design.home_state_indices == reference.home_state_indices
        @test design.away_state_indices == reference.away_state_indices
        @test design.initial_accumulator == reference.initial_accumulator
        @test design.season_accumulator == reference.season_accumulator
        @test design.target_accumulator == reference.target_accumulator
        no_target = QSAPI.dynamics_design(config, qs_features(n_target = 0), 8)
        @test no_target.target_marker === Val(false)
        @test size(no_target.target_accumulator) == (0, 3)
        bad = deepcopy(fs)
        bad.data[:n_rounds] = 6
        @test_throws ErrorException QSAPI.dynamics_design(config, bad, 8)
    end

    @testset "prior geometry and rotation" begin
        rng = MersenneTwister(20261009)
        for r in (1.0, 0.3)
            τ = 0.16
            zq, zs = randn(rng, 100_000), randn(rng, 100_000)
            q = τ / sqrt(1 + r^2) .* zq
            s = r * τ / sqrt(1 + r^2) .* zs
            α, β = q .+ s, s .- q
            @test cor(α, β) ≈ (r^2 - 1) / (r^2 + 1) atol = 0.015
            @test var(α) ≈ τ^2 rtol = 0.015
            @test var(β) ≈ τ^2 rtol = 0.015
        end
    end

    @testset "chain reconstruction, centring and OOS micro step" begin
        config = QualityStyleGRW()
        chain = qs_chain(4, 3, 2)
        counts = QSPG.qs_step_counts(chain, "dyn")
        @test counts == (; n_history = 3, n_target = 2)
        states = QSAPI._cb_extract_dynamics(chain, config, "dyn", 4, qs_features())
        @test size(states.α) == size(states.β) == size(states.q) == size(states.s) == (4, 5, 5)
        @test states.α == states.q .+ states.s
        @test states.β == states.s .- states.q
        for path in (states.q, states.s, states.α, states.β)
            @test maximum(abs.(sum(path, dims = 1))) < 1e-12
        end
        # Independent recomputation at the first target step for the first draw.
        z(chain_name, team, step) = Array(chain[QSPG._grw_chain_symbol(chain, chain_name, team, step)])[1, 1, 1]
        τ = Array(chain[:"dyn.τₖ"])[1, 1, 1]
        r = Array(chain[:"dyn.rₖ"])[1, 1, 1]
        qstep = [z("dyn.q.z_target", i, 1) for i in 1:4] .* (τ / sqrt(1 + r^2))
        sstep = [z("dyn.s.z_target", i, 1) for i in 1:4] .* (τ * r / sqrt(1 + r^2))
        @test states.q[:, 4, 1] .- states.q[:, 3, 1] ≈ qstep .- mean(qstep)
        @test states.s[:, 4, 1] .- states.s[:, 3, 1] ≈ sstep .- mean(sstep)
        @test states.α[:, 4, 1] == states.q[:, 4, 1] .+ states.s[:, 4, 1]
        @test states.β[:, 4, 1] == states.s[:, 4, 1] .- states.q[:, 4, 1]

        @test maximum(abs.(sum(states.oos_q, dims = 2))) < 1e-12
        @test maximum(abs.(sum(states.oos_s, dims = 2))) < 1e-12
        # Reproduce the actual held-out draw from the chain's micro-scale τₖ/rₖ;
        # carrying the last state forward, or using a macro scale, cannot pass.
        rng = MersenneTwister(0)
        zq, zs = randn(rng, 5, 4), randn(rng, 5, 4)
        τk = vec(Array(chain[:"dyn.τₖ"]))
        rk = vec(Array(chain[:"dyn.rₖ"]))
        σq = τk ./ sqrt.(1 .+ rk.^2)
        @test states.oos_q ≈ (zq .- mean(zq, dims = 2)) .* σq
        @test states.oos_s ≈ (zs .- mean(zs, dims = 2)) .* (rk .* σq)
        oos = QSAPI._cb_oos_dynamics(config, states, Dict(), 7, 2, 3, 5)
        @test oos.att_h ≈ vec(states.α[2, end, :]) .+ states.oos_q[:, 2] .+ states.oos_s[:, 2]
        @test oos.def_a ≈ vec(states.β[3, end, :]) .- states.oos_q[:, 3] .+ states.oos_s[:, 3]
        @test QSAPI._cb_oos_dynamics(config, states, Dict(), 7, 0, 3, 5).att_h == zeros(5)
        @test QSAPI._cb_extract_dynamics(chain, config, "dyn", 4, qs_features()).oos_q == states.oos_q

        empty = qs_chain(4, 3, 0)
        @test QSPG.qs_step_counts(empty, "dyn") == (; n_history = 3, n_target = 0)
        held = QSAPI._cb_extract_dynamics(empty, config, "dyn", 4, qs_features(n_target = 0))
        @test size(held.α) == (4, 3, 5)
        @test all(isfinite, held.oos_q)
        @test all(isfinite, held.oos_s)
    end

    @testset "full-model AD on a pinned real Scottish Lower fold" begin
        model = qs_model(:qs_real_fold)
        raw = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10^6)
        scope = Data.DataScope(name = "lower", train_tournaments = [56, 57],
            cups = :none, target_tournaments = [56, 57], monitor_tournaments = Int[],
            clock_tournaments = [56, 57], target_seasons = ["24/25", "25/26"],
            history_seasons = 2, dynamics_col = :match_biweek)
        ds = Data.apply_scope(raw, scope)
        splitter = Data.ScopedWalkForwardCV(scope)
        boundaries = Data.create_id_boundaries(ds, splitter)
        @test length(boundaries) == 40
        fs = first(Features.create_features(boundaries[1:1], ds, model, splitter))
        fs = fs isa Tuple ? first(fs) : fs
        tm = QSPG.build_turing_model(model, fs)
        Random.seed!(20261009)
        vi = DynamicPPL.VarInfo(tm)
        tm(vi)
        θ = copy(vi[:])
        density = DynamicPPL.LogDensityFunction(tm)
        f = x -> LogDensityProblems.logdensity(density, x)
        @test isfinite(f(θ))
        tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, θ))
        relerr(a, b) = norm(a - b) / max(norm(a), norm(b), 1.0)
        probe_rng = MersenneTwister(20261010)
        for δ in (0.0, 0.001, -0.002)
            point = θ .+ δ .* randn(probe_rng, length(θ))
            compiled = similar(point)
            ReverseDiff.gradient!(compiled, tape, point)
            forward = ForwardDiff.gradient(f, point)
            @test all(isfinite, compiled)
            @test relerr(compiled, forward) <= 1e-6
            @test relerr(compiled, ReverseDiff.gradient(f, point)) <= 1e-8
            for j in eachindex(point)
                h = 1e-5
                plus, minus = copy(point), copy(point)
                plus[j] += h
                minus[j] -= h
                finite = (f(plus) - f(minus)) / (2h)
                @test abs(compiled[j] - finite) / max(abs(compiled[j]), abs(finite), 1.0) <= 1e-6
            end
        end
    end
end
println("QUALITY_STYLE_GRW_TESTS_DONE")
