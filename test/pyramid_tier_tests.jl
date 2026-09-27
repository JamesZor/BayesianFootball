using Test
using BayesianFootball
using MCMCChains
using Statistics
using DataFrames
using Dates
using Distributions
using DynamicPPL
using LogDensityProblems
using ReverseDiff
using ForwardDiff
using LinearAlgebra
using Turing
import Serialization

const PyramidGRW = BayesianFootball.Models.PreGame
const PyramidBuilder = PyramidGRW.Builder

function _pyramid_empty_target_chain(n_teams::Int, n_draws::Int)
    columns = String[]
    for side in ("α", "β")
        append!(columns, ["dyn.$side.σ₀", "dyn.$side.σₛ"])
        append!(columns, ["dyn.$side.z_init[$team]" for team in 1:n_teams])
        append!(columns, ["dyn.$side.z_season[$team,1]" for team in 1:n_teams])
    end
    values = zeros(n_draws, length(columns), 1)
    for (column, name) in enumerate(columns)
        values[:, column, 1] .= occursin("σₛ", name) ? 0.4 : 0.0
    end
    return Chains(values, Symbol.(columns))
end

@testset "Pyramid-tier MultiScaleGRW" begin
    @testset "macro accumulator geometry" begin
        for (n_history, n_target) in ((2, 0), (2, 3), (3, 2))
            old = PyramidGRW.grw_accumulators(n_history, n_target)
            flagged = PyramidGRW.grw_accumulators(
                n_history, n_target; target_season_step = true)
            @test old.initial == flagged.initial
            @test old.target == flagged.target
            if n_target == 0
                @test flagged.season == old.season
            else
                @test size(flagged.season, 1) == size(old.season, 1) + 1
                @test flagged.season[1:size(old.season, 1), :] == old.season
                @test flagged.season[end, :] ==
                      Float64[state >= n_history + 1 for state in 1:(n_history + n_target)]
            end
        end

        @test PyramidGRW.grw_accumulators(2, 3).season == Float64[0 1 1 1 1;]
        @test PyramidGRW.grw_accumulators(2, 3; target_season_step = true).season ==
              Float64[0 1 1 1 1; 0 0 1 1 1]
        @test PyramidGRW.grw_accumulators(3, 2).season ==
              Float64[0 1 1 1 1; 0 0 1 1 1]
    end

    @testset "empty-target OOS boundary innovation" begin
        n_teams, n_draws = 3, 4_000
        chain = _pyramid_empty_target_chain(n_teams, n_draws)
        config = MultiScaleGRW(target_season_step = true)
        feature_set = FeatureSet(Dict{Symbol,Any}(:grw_oos_seed => 8128))
        draw_a = PyramidBuilder._cb_extract_dynamics(
            chain, config, "dyn", n_teams, feature_set)
        draw_b = PyramidBuilder._cb_extract_dynamics(
            chain, config, "dyn", n_teams, feature_set)
        @test draw_a.oos_α == draw_b.oos_α
        @test draw_a.oos_β == draw_b.oos_β
        @test size(draw_a.oos_α) == (n_draws, n_teams)
        @test all(abs.(vec(sum(draw_a.oos_α, dims = 2))) .< 1e-12)
        @test all(abs.(vec(sum(draw_a.oos_β, dims = 2))) .< 1e-12)
        # A serialized chain (as in a reloaded fit) must regenerate identical OOS draws.
        saved = IOBuffer()
        Serialization.serialize(saved, chain)
        seekstart(saved)
        reloaded = Serialization.deserialize(saved)
        draw_reloaded = PyramidBuilder._cb_extract_dynamics(
            reloaded, config, "dyn", n_teams, feature_set)
        @test draw_reloaded.oos_α == draw_a.oos_α
        @test draw_reloaded.oos_β == draw_a.oos_β
        @test abs(mean(draw_a.oos_α)) < 0.015
        @test isapprox(std(vec(draw_a.oos_α)), 0.4 * sqrt(2 / 3); rtol = 0.05)

        # The OOS hook applies that one innovation to each side's carried state.
        effects = PyramidBuilder._cb_oos_dynamics(
            config, draw_a, Dict(), 1, 1, 2, n_draws)
        @test effects.att_h == draw_a.oos_α[:, 1]
        @test effects.def_a == draw_a.oos_β[:, 2]
    end

    @testset "default-off regression surface" begin
        default = MultiScaleGRW()
        explicit_off = MultiScaleGRW(target_season_step = false)
        @test default == explicit_off
        @test string(default) == string(explicit_off)
        @test PyramidGRW.grw_accumulators(3, 2) ==
              PyramidGRW.grw_accumulators(3, 2; target_season_step = false)
    end
end

function _pyramid_rows()
    rows = NamedTuple[]
    add(id, tournament, season, date, home, away; biweek = 1) = push!(rows, (;
        match_id = id, tournament_id = tournament, season = String(season),
        match_date = Date(date), match_hour = 15, match_week = biweek,
        match_biweek = biweek, match_month = month(Date(date)),
        home_team = String(home), away_team = String(away),
        home_score = 1, away_score = 0, neutral_venue = false))
    add(1, 55, "21/22", "2022-04-01", "changer", "champ")
    add(2, 54, "21/22", "2022-04-02", "celtic", "top")
    add(3, 56, "21/22", "2022-04-03", "l1", "l1b")
    add(4, 57, "21/22", "2022-04-04", "promoted", "l2")
    add(5, 56, "22/23", "2023-04-01", "changer", "l1")
    add(6, 55, "22/23", "2023-04-02", "champ", "champ2")
    add(7, 54, "22/23", "2023-04-03", "celtic", "top")
    add(8, 57, "22/23", "2023-04-04", "promoted", "l2")
    add(9, 56, "23/24", "2023-08-05", "changer", "promoted")
    add(10, 57, "23/24", "2023-08-05", "entered", "l2")
    add(11, 55, "23/24", "2023-08-06", "champ", "champ2")
    add(12, 54, "23/24", "2023-08-06", "celtic", "top")
    add(13, 73, "2023", "2023-08-12", "changer", "celtic"; biweek = 2)
    return DataFrame(rows)
end

function _pyramid_store(rows = _pyramid_rows())
    empty = DataFrame()
    return BayesianFootball.Data.DataStore(BayesianFootball.Data.ScottishPyramid(),
        rows, empty, empty, empty, empty, empty, empty, empty)
end

function _pyramid_scope(; lower = false)
    return BayesianFootball.Data.DataScope(
        name = lower ? "lower" : "spfl_cups",
        train_tournaments = lower ? [56, 57] : [54, 55, 56, 57],
        cups = lower ? :none : :senior_spfl_ties,
        target_tournaments = [56, 57], monitor_tournaments = lower ? Int[] : [54, 55],
        clock_tournaments = [56, 57], target_seasons = ["23/24"],
        history_seasons = 2, dynamics_col = :match_biweek)
end

function _pyramid_model(; tiers = PyramidTiers(carry = EstimatedCarry(), jump = EstimatedJump()))
    return CountModelBuilder(:pyramid_synthetic) |> add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |> add(GlobalHomeAdvantage()) |>
        add(tiers) |> add(JointGammaPoissonObservation()) |> build
end

@testset "Pyramid tier features and reductions" begin
    raw = _pyramid_store()
    lower = BayesianFootball.Data.apply_scope(raw, _pyramid_scope(lower = true))
    @test DataFrames.metadata(lower.matches, "full_club_season_tiers", nothing) !== nothing
    ordered = Int[3, 4, 5, 8]
    team_map = Dict(name => i for (i, name) in enumerate(sort(unique(vcat(
        lower.matches.home_team, lower.matches.away_team)))))
    F_full = Dict{Symbol,Any}(:target_season => "23/24")
    BayesianFootball.Features.add_feature!(F_full,
        PyramidTierFeature(anchor = :full_pyramid, jump = EstimatedJump()),
        ordered, team_map, lower)
    F_lower = Dict{Symbol,Any}(:target_season => "23/24")
    BayesianFootball.Features.add_feature!(F_lower,
        PyramidTierFeature(anchor = :first_in_window, jump = EstimatedJump()),
        ordered, team_map, lower)
    changer_row = findfirst(==(5), ordered)
    @test F_full[:pyramid_anchor_home][changer_row] == -1.0
    @test F_lower[:pyramid_anchor_home][changer_row] == -2.0

    events = F_full[:pyramid_events]
    changer_event = findfirst(e -> e.club == "changer" && e.season == "22/23", events)
    promoted_event = findfirst(e -> e.club == "promoted" && e.season == "23/24", events)
    @test changer_event !== nothing
    @test promoted_event !== nothing
    target = F_full[:pyramid_oos_bridge][9]
    @test target.home.incidence[changer_event] == 1.0
    @test target.away.incidence[promoted_event] == 1.0
    @test all(e -> e.club != "entered", events)

    fixed = Dict{Symbol,Any}(:target_season => "23/24")
    BayesianFootball.Features.add_feature!(fixed,
        PyramidTierFeature(anchor = :full_pyramid, jump = FixedTransitionPriors()),
        ordered, team_map, lower)
    fixed_events = fixed[:pyramid_events]
    @test Set((e.from, e.to) for e in fixed_events) == Set([(55, 56), (57, 56)])
    rel = only(filter(e -> (e.from, e.to) == (55, 56), fixed_events))
    @test (rel.α_mean, rel.α_sd, rel.β_mean, rel.β_sd) == (0.12, 0.20, -0.16, 0.26)

    pooled = BayesianFootball.Data.apply_scope(raw, _pyramid_scope())
    all_ids = Int.(pooled.matches.match_id)
    pooled_map = Dict(name => i for (i, name) in enumerate(sort(unique(vcat(
        pooled.matches.home_team, pooled.matches.away_team)))))
    F_pooled = Dict{Symbol,Any}(:target_season => "23/24")
    BayesianFootball.Features.add_feature!(F_pooled,
        PyramidTierFeature(anchor = :first_in_window, jump = NoJump()),
        all_ids, pooled_map, pooled)
    cup = F_pooled[:pyramid_oos_bridge][13]
    @test cup.home.anchor + cup.home.delta == -2.0
    @test cup.away.anchor + cup.away.delta == 0.0
    @test cup.away.old_firm == 1.0

    # Production plumbing supplies the declared target season before pyramid
    # extraction. A scheduled tier from target+1 must not widen the event design.
    future_rows = _pyramid_rows()
    push!(future_rows, (match_id = 14, tournament_id = 55, season = "24/25",
        match_date = Date("2024-08-03"), match_hour = 15, match_week = 1,
        match_biweek = 1, match_month = 8, home_team = "changer",
        away_team = "champ", home_score = 0, away_score = 0,
        neutral_venue = false))
    future_scoped = BayesianFootball.Data.apply_scope(
        _pyramid_store(future_rows), _pyramid_scope())
    boundary = BayesianFootball.Data.SplitBoundary(1, 1, collect(1:8), collect(9:13))
    meta = BayesianFootball.Data.GroupedSplitMetaData(
        [54, 55, 56, 57], "22/23", "23/24", 2, 1, 0)
    splitter = BayesianFootball.Data.ScopedWalkForwardCV(_pyramid_scope())
    future_fs = first(BayesianFootball.Features.create_features(
        [(boundary, meta)], future_scoped, _pyramid_model(), splitter))[1]
    base_fs = first(BayesianFootball.Features.create_features(
        [(boundary, meta)], pooled, _pyramid_model(), splitter))[1]
    @test future_fs.data[:target_season] == "23/24"
    for key in filter(key -> startswith(String(key), "pyramid_"), keys(base_fs.data))
        @test future_fs.data[key] == base_fs.data[key]
    end

    source = (; home = (; anchor = -1.0, delta = -1.0, old_firm = 0.0,
                         incidence = Float64[]),
                away = (; anchor = -1.0, delta = 0.0, old_firm = 0.0,
                         incidence = Float64[]), events = NamedTuple[])
    bridge = Dict(99 => source)
    row = (; match_id = 99)
    base_draw = (; step = [1.0], carry = [0.0], old_firm = [0.0],
                  jump = nothing, attack_share = 0.48)
    carried_draw = merge(base_draw, (; carry = [1.0]))
    no_carry = BayesianFootball.predictor_oos(
        PyramidTiers(carry = NoCarry(), jump = NoJump()), base_draw, bridge, row)
    fixed_zero = BayesianFootball.predictor_oos(
        PyramidTiers(carry = FixedCarry(0), jump = NoJump()), base_draw, bridge, row)
    fixed_one = BayesianFootball.predictor_oos(
        PyramidTiers(carry = FixedCarry(1), jump = NoJump()), carried_draw, bridge, row)
    @test no_carry == fixed_zero
    @test fixed_one.h - no_carry.h == [-0.48]
    @test fixed_one.a - no_carry.a == [0.52]

    old_source = merge(source, (; home = merge(source.home, (; old_firm = 1.0))))
    old_draw = merge(base_draw, (; old_firm = [1.15]))
    old = BayesianFootball.predictor_oos(PyramidTiers(jump = NoJump()), old_draw,
        Dict(99 => old_source), row)
    plain = BayesianFootball.predictor_oos(PyramidTiers(jump = NoJump()), old_draw,
        bridge, row)
    @test old.h - plain.h == [0.48 * 1.15]
    @test old.a - plain.a == [-0.52 * 1.15]
end

@testset "Pyramid and league-delta defaults and recipe identity" begin
    tiers = PyramidTiers()
    @test tiers.tier_term
    @test tiers.anchor == :first_in_window
    @test tiers.attack_share == 0.48
    @test tiers.carry isa NoCarry
    @test tiers.jump isa NoJump
    @test tiers.step_prior.untruncated == Normal(0.47, 0.16)
    @test tiers.old_firm_prior.untruncated == Normal(1.15, 0.10)
    @test EstimatedCarry().prior == Beta(1.5, 4.0)
    @test EstimatedJump().scale_prior == Gamma(2.0, 0.08)
    @test CompetitionKappa().intercept_scale == 0.10
    @test CompetitionKappa().kappa_scale == 0.25
    @test DEFAULT_TRANSITION_PRIORS == (
        (from = 55, to = 56, α_mean = 0.12, α_sd = 0.20,
         β_mean = -0.16, β_sd = 0.26),
        (from = 56, to = 57, α_mean = 0.03, α_sd = 0.20,
         β_mean = -0.04, β_sd = 0.24),
        (from = 57, to = 56, α_mean = 0.00, α_sd = 0.21,
         β_mean = 0.01, β_sd = 0.31),
    )

    base = CountModelBuilder(:base) |> add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |> add(JointGammaPoissonObservation()) |> build
    same = CountModelBuilder(:same) |> add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |>
        add(GlobalHomeAdvantage()) |> add(JointGammaPoissonObservation()) |> build
    @test string(base) == string(same)
    c0 = Harness.Candidate(name = "a", model = base, scope = _pyramid_scope(), role = :control)
    c1 = Harness.Candidate(name = "b", model = same, scope = _pyramid_scope(), role = :candidate)
    @test Harness.recipe_hash(c0) == Harness.recipe_hash(c1)

    w1_lower = BayesianFootball.Data.DataScope(
        name = "lower", train_tournaments = [56, 57], cups = :none,
        target_tournaments = [56, 57], monitor_tournaments = Int[],
        clock_tournaments = [56, 57], target_seasons = ["24/25", "25/26"],
        history_seasons = 2, dynamics_col = :match_biweek)
    w1_joint = JointGammaPoissonObservation(
        feature = BayesianFootball.Features.MatchProxyXGFeature(k = 25.0, fallback = :none),
        shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
        log_kappa_prior = Normal(0.0, 0.2))
    w1_model = CountModelBuilder(:td_lower_joint) |> add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = 180.0)) |> add(GlobalHomeAdvantage()) |>
        add(w1_joint) |> build
    w1_candidate = Harness.Candidate(name = "td_lower_joint", model = w1_model,
        scope = w1_lower, role = :control)
    @test Harness.recipe_hash(w1_candidate) ==
          "722ed88a7efe8c82eb013f789f4d41936fbcea597fa8b9d8cef3f620d062e9fc"
    w1_grw = CountModelBuilder(:grw_lower_joint) |> add(GlobalInterception()) |>
        add(MultiScaleGRW()) |> add(GlobalHomeAdvantage()) |> add(w1_joint) |> build
    @test Harness.recipe_hash(Harness.Candidate(name = "grw_lower_joint",
        model = w1_grw, scope = w1_lower)) ==
          "d373b3393d5079129fa7bed773b28e23cf2c5b67a2904b9ea189f9d37a55f8a6"
    @test string(_pyramid_model(tiers = PyramidTiers(carry = NoCarry()))) !=
          string(_pyramid_model(tiers = PyramidTiers(carry = EstimatedCarry())))
end

function _league_delta_model(dynamics)
    observation = JointGammaPoissonObservation(
        feature = BayesianFootball.Features.MatchProxyXGFeature(k = 25.0, fallback = :none),
        shape_prior = truncated(Normal(4.0, 1.5), 0.5, Inf),
        log_kappa_prior = Normal(0.0, 0.2), kappa = CompetitionKappa())
    return CountModelBuilder(:league_delta_synthetic) |> add(GlobalInterception()) |>
        add(dynamics) |> add(GlobalHomeAdvantage()) |>
        add(PyramidTiers(carry = EstimatedCarry(), jump = EstimatedJump())) |>
        add(observation) |> build
end

@testset "Pyramid and league deltas are compiled-AD safe" begin
    pooled = BayesianFootball.Data.apply_scope(_pyramid_store(), _pyramid_scope())
    boundary = BayesianFootball.Data.SplitBoundary(1, 1, collect(1:8), collect(9:13))
    no_jump_model = _pyramid_model(tiers = PyramidTiers(carry = NoCarry(), jump = NoJump()))
    no_jump_fs = BayesianFootball.Features.create_features(
        boundary, pooled, no_jump_model, :match_biweek)
    no_jump_turing = PyramidGRW.build_turing_model(no_jump_model, no_jump_fs)
    @test length(DynamicPPL.VarInfo(no_jump_turing)[:]) > 0
    for dynamics in (TimeDecayDynamics(days_half_life = 180.0),
                     MultiScaleGRW(target_season_step = true))
        model = _league_delta_model(dynamics)
        fs = BayesianFootball.Features.create_features(
            boundary, pooled, model, :match_biweek)
        design = PyramidBuilder.cb_design(model, fs)
        @test design.observation_data.n_tiers == 4
        @test design.observation_data.n_competitions == 5
        audit = BayesianFootball.Harness._gradient_audit(model, fs; replays = 3)
        @test audit.reversediff_forward_error <= 1.0e-6
        @test audit.worst_perturbed_error <= 1.0e-8
        @test isfinite(audit.log_density)
    end
end

@testset "Pyramid and league-delta extraction parity" begin
    pooled = BayesianFootball.Data.apply_scope(_pyramid_store(), _pyramid_scope())
    boundary = BayesianFootball.Data.SplitBoundary(1, 1, collect(1:8), collect(9:13))
    meta = BayesianFootball.Data.GroupedSplitMetaData(
        [54, 55, 56, 57], "22/23", "23/24", 2, 1, 0)
    splitter = BayesianFootball.Data.ScopedWalkForwardCV(_pyramid_scope())
    model = _league_delta_model(TimeDecayDynamics(days_half_life = 180.0))
    fs = first(BayesianFootball.Features.create_features(
        [(boundary, meta)], pooled, model, splitter))[1]
    turing_model = PyramidGRW.build_turing_model(model, fs)
    chain = Turing.sample(turing_model, Turing.Prior(), 2; progress = false)

    tiers = only(model.covariates)
    draws = BayesianFootball.predictor_extract(chain, tiers, "pyramid_tiers")
    match_id = first(boundary.history_match_ids)
    source = fs.data[:pyramid_oos_bridge][match_id]
    actual = BayesianFootball.predictor_oos(
        tiers, draws, fs.data[:pyramid_oos_bridge], (; match_id))
    design = first(PyramidBuilder.cb_design(model, fs).predictor_designs)
    row = findfirst(==(match_id), fs.data[:ordered_match_ids])
    home_base = draws.step .* (design.anchor_home[row] .+
        draws.carry .* design.delta_home[row]) .+
        draws.old_firm .* design.old_firm_home[row]
    away_base = draws.step .* (design.anchor_away[row] .+
        draws.carry .* design.delta_away[row]) .+
        draws.old_firm .* design.old_firm_away[row]
    jump_h = draws.jump.α * vec(design.events_home[row, :])
    jump_a = draws.jump.α * vec(design.events_away[row, :])
    concede_h = draws.jump.β * vec(design.events_home[row, :])
    concede_a = draws.jump.β * vec(design.events_away[row, :])
    expected_h = 0.48 .* home_base .- 0.52 .* away_base .+ jump_h .+ concede_a
    expected_a = 0.48 .* away_base .- 0.52 .* home_base .+ jump_a .+ concede_h
    @test actual.h ≈ expected_h atol = 1.0e-12
    @test actual.a ≈ expected_a atol = 1.0e-12

    observation = PyramidBuilder._cb_extract_observation(
        model.observation, chain, fs.data[:n_teams], fs)
    league = observation.bridge[match_id]
    rates = PyramidBuilder._cb_rates(model.observation, ones(2), ones(2),
        observation, 1, 1, match_id)
    expected_home = exp.(observation.δ_intercept[:, league.tier_home_idx])
    expected_away = exp.(observation.δ_intercept[:, league.tier_away_idx])
    expected_finishing = observation.κ .*
        exp.(observation.δ_kappa[:, league.competition_idx])
    @test rates.μ_h ≈ expected_home atol = 1.0e-12
    @test rates.μ_a ≈ expected_away atol = 1.0e-12
    @test rates.κ_competition ≈ expected_finishing atol = 1.0e-12
    @test rates.λ_h ≈ expected_home .* expected_finishing atol = 1.0e-12
    @test rates.λ_a ≈ expected_away .* expected_finishing atol = 1.0e-12
end
