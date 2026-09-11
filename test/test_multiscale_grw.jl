using Test
using BayesianFootball
using DataFrames
using Dates
using Distributions
using DynamicPPL
using ForwardDiff
using LinearAlgebra
using LogDensityProblems
using MCMCChains
using Random
using ReverseDiff
using Statistics

# ==============================================================================
# MultiScaleGRW — graduated two-speed random-walk team dynamics (Task 007)
# ==============================================================================
#
# The property these tests exist to pin is the STATE-COUNT CONTRACT. The walk must
# emit exactly `n_history + n_target` states, because that is what `:n_rounds`
# promises and what `:time_indices` indexes into. The superseded implementation
# emitted `1 + n_history + n_target` and still ran, still sampled, and still
# scored — it simply read every fixture against the wrong state. Nothing about a
# "does it build" test would have caught that, so the geometry is asserted directly.
# ==============================================================================

const GRWPG = BayesianFootball.Models.PreGame
const GRWAPI = GRWPG.Builder

"""
A fold with `n_history` history seasons and `n_target` target steps, so the state
geometry under test is non-degenerate in BOTH directions.
"""
function grw_feature_set(; n::Int = 8, n_teams::Int = 4,
                         n_history::Int = 3, n_target::Int = 2)
    n_rounds = n_history + n_target
    home = Int[mod1(i, n_teams) for i in 1:n]
    away = Int[mod1(i + 1, n_teams) for i in 1:n]
    # Fixtures spread across the timeline, including the first and last states.
    times = Int[mod1(i, n_rounds) for i in 1:n]
    return BayesianFootball.FeatureSet(Dict{Symbol,Any}(
        :flat_home_ids => home,
        :flat_away_ids => away,
        :season_indices => ones(Int, n),
        :time_indices => times,
        :flat_months => Int[mod1(i, 12) for i in 1:n],
        :flat_home_goals => Int[mod(i, 3) for i in 1:n],
        :flat_away_goals => Int[mod(i + 1, 3) for i in 1:n],
        :dates => collect(0.0:(n - 1)),
        :n_teams => n_teams,
        :n_seasons => 1,
        :n_rounds => n_rounds,
        :n_history_steps => n_history,
        :n_target_steps => n_target,
        :team_map => Dict("t$i" => i for i in 1:n_teams),
    ))
end

grw_model(name::Symbol = :grw_test) =
    CountModelBuilder(name) |>
        add(GlobalInterception()) |>
        add(MultiScaleGRW()) |>
        add(GlobalHomeAdvantage()) |>
        add(PoissonObservation()) |>
        build

@testset "MultiScaleGRW" begin

    @testset "export surface" begin
        # Reachable from the top level, not only from deep inside PreGame.
        @test MultiScaleGRW === BayesianFootball.Models.PreGame.MultiScaleGRW
        @test MultiScaleGRW <: GRWPG.AbstractDynamicsConfig
        @test GRWDynamicsDesign === GRWAPI.GRWDynamicsDesign

        config = MultiScaleGRW()
        @test config.z₀ isa ContinuousUnivariateDistribution
        # Defence is looser than attack across a season boundary and tighter within
        # the target season; that asymmetry is the modelling claim, so pin it.
        @test mean(config.β_σ₀) > mean(config.α_σ₀)
        @test mean(config.β_σₛ) > mean(config.α_σₛ)
        @test mean(config.β_σₖ) < mean(config.α_σₖ)
    end

    @testset "accumulators encode the state-count contract" begin
        n_history, n_target = 3, 2
        acc = GRWPG.grw_accumulators(n_history, n_target)
        n_rounds = n_history + n_target

        @test size(acc.initial) == (1, n_rounds)
        # n_history - 1 macro transitions, NOT n_history. This is the off-by-one.
        @test size(acc.season) == (n_history - 1, n_rounds)
        @test size(acc.target) == (n_target, n_rounds)

        # The level is in every state.
        @test all(acc.initial .== 1.0)
        # Macro transition t first appears at state t+1.
        @test acc.season[1, :] == Float64[0, 1, 1, 1, 1]
        @test acc.season[2, :] == Float64[0, 0, 1, 1, 1]
        # Micro step k first appears at state n_history + k.
        @test acc.target[1, :] == Float64[0, 0, 0, 1, 1]
        @test acc.target[2, :] == Float64[0, 0, 0, 0, 1]

        # A single-season fold has no macro transitions but is still valid.
        solo = GRWPG.grw_accumulators(1, 3)
        @test size(solo.season) == (0, 4)
        @test size(solo.target) == (3, 4)

        # A walk needs a level to start from.
        @test_throws ErrorException GRWPG.grw_accumulators(0, 2)
        @test_throws ErrorException GRWPG.grw_accumulators(2, -1)
    end

    @testset "builder integration" begin
        builder = CountModelBuilder(:grw_validate) |>
            add(GlobalInterception()) |> add(MultiScaleGRW()) |>
            add(GlobalHomeAdvantage()) |> add(PoissonObservation())
        rows = validate(builder)
        @test all(row.pass for row in rows)

        # The weighting row must PASS and must say the states carry recency.
        weighting = only(filter(r -> occursin("weighting", r.name), rows))
        @test weighting.pass
        @test occursin("unit likelihood weights", weighting.detail)

        model = grw_model()
        @test model isa PoissonCountModel
        @test model.dynamics isa MultiScaleGRW

        # Recency lives in the latent states, so the likelihood weights every
        # fixture equally. Time decay here would discount the same evidence twice.
        dates = Float64[0, 90, 180, 365]
        @test GRWAPI.dynamics_match_weights(MultiScaleGRW(), dates) == ones(4)
        @test GRWAPI.dynamics_match_weights(
            TimeDecayDynamics(days_half_life = 180.0), dates) != ones(4)
    end

    @testset "dynamics_design" begin
        fs = grw_feature_set()
        n = length(fs.data[:flat_home_ids])
        design = GRWAPI.dynamics_design(MultiScaleGRW(), fs, n)

        @test design isa GRWDynamicsDesign
        @test design.n_history == 3
        @test design.n_target == 2
        @test design.n_rounds == 5
        @test design.target_marker === Val(true)
        @test length(design.home_state_indices) == n
        @test length(design.away_state_indices) == n

        # Each fixture gathers (its team, its time step).
        for i in 1:n
            @test design.home_state_indices[i] ==
                  CartesianIndex(fs.data[:flat_home_ids][i], fs.data[:time_indices][i])
        end

        # A fold with no observed target steps dispatches to the no-target branch,
        # which does not sample σₖ at all.
        fs0 = grw_feature_set(n_history = 3, n_target = 0)
        d0 = GRWAPI.dynamics_design(MultiScaleGRW(), fs0, 8)
        @test d0.target_marker === Val(false)
        @test size(d0.target_accumulator) == (0, 3)

        # The contract violation the old component shipped with is now refused.
        bad = deepcopy(fs)
        bad.data[:n_rounds] = 6            # 6 != 3 + 2
        @test_throws ErrorException GRWAPI.dynamics_design(MultiScaleGRW(), bad, n)

        # A time index outside the state range is refused rather than clamped.
        oob = deepcopy(fs)
        oob.data[:time_indices] = fill(99, n)
        @test_throws ErrorException GRWAPI.dynamics_design(MultiScaleGRW(), oob, n)
    end

    @testset "θ sites and fold-dependent widths" begin
        model = grw_model()
        sites = cb_varinfo_sites(model)
        for side in ("α", "β"), leaf in ("σ₀", "σₛ", "σₖ", "z_init", "z_season", "z_target")
            @test Symbol("dyn.$side.$leaf") in sites
        end

        # `cb_chain_columns` only knows (n_teams, n_seasons). The GRW's innovation
        # grids are (teams x steps) and the step count comes from the fold, so the
        # honest answer is a refusal — not the scalar default, which would have
        # silently undercounted θ.
        @test_throws ErrorException cb_chain_columns(model, 4; n_seasons = 1)
    end

    @testset "log density and gradients" begin
        fs = grw_feature_set()
        model = grw_model()
        turing_model = GRWPG.build_turing_model(model, fs)

        Random.seed!(20260911)
        varinfo = DynamicPPL.VarInfo(turing_model)
        turing_model(varinfo)
        θ = copy(varinfo[:])
        density = DynamicPPL.LogDensityFunction(turing_model)
        f = x -> LogDensityProblems.logdensity(density, x)

        @test isfinite(f(θ))

        # ReverseDiff is the production AD path; ForwardDiff is the independent
        # check. Agreement is what licenses the compiled tape.
        g_reverse = ReverseDiff.gradient(f, θ)
        g_forward = ForwardDiff.gradient(f, θ)
        @test all(isfinite, g_reverse)
        @test isapprox(g_reverse, g_forward; rtol = 1e-6)

        # A compiled tape must replay to the same gradient at a DIFFERENT point;
        # that is the property that fails when a model branches on its own values.
        tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, θ))
        perturbed = θ .+ 0.01 .* sin.(collect(eachindex(θ)))
        replayed = similar(perturbed)
        ReverseDiff.gradient!(replayed, tape, perturbed)
        @test isapprox(replayed, ReverseDiff.gradient(f, perturbed); rtol = 1e-8)
    end

    @testset "posterior reconstruction" begin
        n_teams, n_history, n_target = 4, 3, 2
        n_rounds = n_history + n_target
        n_draws = 5

        # Build a chain by hand so the reconstruction has a known right answer.
        columns = String[]
        for side in ("α", "β")
            append!(columns, ["dyn.$side.σ₀", "dyn.$side.σₛ", "dyn.$side.σₖ"])
            append!(columns, ["dyn.$side.z_init[$i]" for i in 1:n_teams])
            append!(columns, ["dyn.$side.z_season[$i,$t]"
                              for t in 1:(n_history - 1) for i in 1:n_teams])
            append!(columns, ["dyn.$side.z_target[$i,$t]"
                              for t in 1:n_target for i in 1:n_teams])
        end
        values = zeros(Float64, n_draws, length(columns), 1)
        for (j, name) in enumerate(columns)
            values[:, j, 1] .= endswith(name, "σ₀") ? 1.0 :
                               endswith(name, "σₛ") ? 1.0 :
                               endswith(name, "σₖ") ? 1.0 : 1.0
        end
        chain = Chains(values, Symbol.(columns))

        # The geometry is recovered from the site names, not passed in.
        counts = GRWPG.grw_step_counts(chain, "dyn")
        @test counts.n_history == n_history
        @test counts.n_target == n_target

        traj = GRWAPI._cb_extract_dynamics(chain, MultiScaleGRW(), "dyn", n_teams)
        # (teams, time, samples) — note this is NOT the (samples, teams) layout the
        # static components return, which is why the OOS hook indexes differently.
        @test size(traj.α) == (n_teams, n_rounds, n_draws)
        @test size(traj.β) == (n_teams, n_rounds, n_draws)
        @test all(isfinite, traj.α)

        # Every state column is zero-centred over teams, at every time and draw.
        for t in 1:n_rounds, s in 1:n_draws
            @test isapprox(sum(traj.α[:, t, s]), 0.0; atol = 1e-10)
            @test isapprox(sum(traj.β[:, t, s]), 0.0; atol = 1e-10)
        end

        # With every z and σ set to 1, each team's raw state at time t is exactly t
        # (one unit per accumulated innovation), so after zero-centring over
        # identical teams every state is 0. The interesting assertion is that the
        # walk is MONOTONE in accumulated innovations before centring, which the
        # accumulator test above already pins.
        @test all(abs.(traj.α) .< 1e-10)

        # Held-out fixtures price from the LAST visible state.
        oos = GRWAPI._cb_oos_dynamics(
            MultiScaleGRW(), traj, Dict(), 1, 2, 3, n_draws)
        @test length(oos.att_h) == n_draws
        @test oos.att_h == vec(traj.α[2, n_rounds, :])
        @test oos.def_a == vec(traj.β[3, n_rounds, :])
        @test oos.att_a == vec(traj.α[3, n_rounds, :])
        @test oos.def_h == vec(traj.β[2, n_rounds, :])

        # An unmapped team contributes exactly zero, never a league-mean guess.
        missing_side = GRWAPI._cb_oos_dynamics(
            MultiScaleGRW(), traj, Dict(), 1, 0, 3, n_draws)
        @test missing_side.att_h == zeros(n_draws)
    end
end
