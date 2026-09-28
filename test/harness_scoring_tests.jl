using Test
using BayesianFootball
using BayesianFootball: Data, CountLatents
using DataFrames
using Dates
using Distributions
using Random
using UUIDs
using Statistics

const H = BayesianFootball.Harness

function harness_store(matches)
    empty = DataFrame()
    return Data.DataStore(Data.ScottishAll(), matches, empty, empty, empty, empty,
                          empty, empty, empty)
end

function add_fixture!(rows, id, date, season, tier, home, away)
    push!(rows, (; match_id = id, match_date = date, season = season,
                  tournament_id = tier, home_team = home, away_team = away,
                  home_score = 1, away_score = 0))
end

function toy_transition_store()
    rows = NamedTuple[]
    id = 1
    # Previous-season memberships establish all four transition directions.
    for (tier, club, opponent) in ((55, "R", "R0"), (57, "P", "P0"),
                                    (56, "D", "D0"), (54, "A", "A0"),
                                    (56, "I56", "I56b"), (57, "I57", "I57b"),
                                    (55, "I55", "I55b"))
        add_fixture!(rows, id, Date(2023, 2, id), "22/23", tier, club, opponent)
        id += 1
    end
    # Intermediate season: R relegates into L1, P promotes into L1, D drops to L2,
    # A changes 54 -> 55 and therefore belongs only to :any.
    current = ((56, "R", "I56"), (56, "P", "I56"),
               (57, "D", "I57"), (55, "A", "I55"),
               (57, "E", "I57")) # E was absent while 22/23 was covered: entered SPFL.
    current_ids = Dict{String,Vector{Int}}()
    for (tier, club, opponent) in current
        current_ids[club] = Int[]
        for n in 1:12
            push!(current_ids[club], id)
            add_fixture!(rows, id, Date(2024, 1, 1) + Day(id), "23/24",
                         tier, club, opponent)
            id += 1
        end
    end
    # Third season keeps every club in place, proving the previous-season comparison
    # does not label all fixtures of a historically mobile club.
    for (tier, club, opponent) in current
        add_fixture!(rows, id, Date(2025, 1, 1) + Day(id), "24/25",
                     tier, club, opponent)
        id += 1
    end
    return harness_store(DataFrame(rows)), current_ids
end

@testset "Harness transition cohorts" begin
    ds, current = toy_transition_store()
    tiers = H.club_season_tiers(ds)
    panel = collect(Iterators.flatten(values(current)))

    relegated = H.transition_cohort(ds, panel, tiers;
        first_n = 10, direction = :relegated_into_L1)
    promoted = H.transition_cohort(ds, panel, tiers;
        first_n = 10, direction = :promoted_into_L1)
    l1_l2 = H.transition_cohort(ds, panel, tiers;
        first_n = 10, direction = :l1_l2)
    entered = H.transition_cohort(ds, panel, tiers;
        first_n = 10, direction = :entered_spfl)
    any_move = H.transition_cohort(ds, panel, tiers;
        first_n = 10, direction = :any)

    @test relegated.match_ids == Set(current["R"][1:10])
    @test promoted.match_ids == Set(current["P"][1:10])
    @test l1_l2.match_ids == union(Set(current["P"][1:10]), Set(current["D"][1:10]))
    @test entered.match_ids == Set(current["E"][1:10])
    @test any_move.match_ids == union((Set(current[c][1:10]) for c in ("R", "P", "D", "A"))...)
    @test !(current["R"][11] in relegated.match_ids)
    @test H.transition_cohort(ds, panel, tiers;
        first_n = 20, direction = :relegated_into_L1).match_ids == Set(current["R"])
    @test all(relegated.clubs[m] == ["R"] for m in relegated.match_ids)
    @test_throws ArgumentError H.transition_cohort(ds, panel, tiers;
        first_n = 9, direction = :any)
    @test H._previous_season("00/01") == "99/00"
end

@testset "Harness fixture-clustered bootstrap" begin
    rng = MersenneTwister(9)
    a = DataFrame(match_id = Int[], selection = Symbol[], family = String[],
                  ll_model = Float64[], ll_market = Float64[])
    b = similar(a, 0)
    fixture_effects = repeat([-0.9, 1.1], 20) # known row-weighted mean = 0.1
    selections = (:home, :draw, :away, :over_25)
    for (mid, effect) in enumerate(fixture_effects), selection in selections
        push!(a, (mid, selection, selection === :over_25 ? "OU2.5" : "1X2",
                  1.5 + effect, 1.0))
        push!(b, (mid, selection, selection === :over_25 ? "OU2.5" : "1X2",
                  1.5, 1.0))
    end
    clustered = H._paired_bootstrap(a, b; B = 5_000, seed = 2)
    @test clustered.delta ≈ 0.1 atol = 1e-12
    @test clustered.lo <= 0.1 <= clustered.hi

    d = a.ll_model .- b.ll_model
    row_stats = [mean(rand(rng, d, length(d))) for _ in 1:5_000]
    row_width = quantile(row_stats, 0.975) - quantile(row_stats, 0.025)
    @test clustered.hi - clustered.lo > row_width
end

function poisson_1x2(lambda_home, lambda_away; max_goals = 24)
    home = draw = away = 0.0
    for h in 0:max_goals, a in 0:max_goals
        p = pdf(Poisson(lambda_home), h) * pdf(Poisson(lambda_away), a)
        h > a ? (home += p) : h == a ? (draw += p) : (away += p)
    end
    total = home + draw + away
    return (home / total, draw / total, away / total)
end

@testset "Harness compression slope convention" begin
    model_sup = [-0.30, -0.10, 0.10, 0.30, 0.50]
    market_sup = 2.0 .* model_sup
    ids = collect(101:105)
    draws = 30
    lambda_home = reduce(vcat, [fill(exp(s / 2), 1, draws) for s in model_sup])
    lambda_away = reduce(vcat, [fill(exp(-s / 2), 1, draws) for s in model_sup])
    latents = CountLatents(ids, lambda_home, lambda_away)

    odds = DataFrame(match_id = Int[], market_name = String[], market_line = Float64[],
                     selection = Symbol[], prob_fair_close = Float64[])
    for (id, supremacy) in zip(ids, market_sup)
        p = poisson_1x2(exp(supremacy / 2), exp(-supremacy / 2))
        for (selection, probability) in zip((:home, :draw, :away), p)
            push!(odds, (id, "1x2", 0.0, selection, probability))
        end
    end
    slopes = H.compression_slopes(latents, odds, ids)
    @test slopes.n_fixtures == length(ids)
    @test slopes.compression_slope > 1.0
    @test slopes.compression_slope ≈ 2.0 atol = 0.15
end

@testset "Harness monitor subsets from synthetic latents" begin
    matches = DataFrame(
        match_id = [1, 2, 3, 4], tournament_id = [56, 57, 54, 55],
        season = fill("24/25", 4), match_date = Date.(fill("2024-08-03", 4)),
        home_team = ["h$i" for i in 1:4], away_team = ["a$i" for i in 1:4])
    ds = harness_store(matches)
    latents = CountLatents([1, 2, 3, 4], fill(1.2, 4, 3), fill(1.0, 4, 3))
    fit = (; latents)
    target = H._season_panel(ds, fit, ["24/25"])
    scored = H._season_panel(ds, fit, ["24/25"];
                             tournaments = (54, 55, 56, 57))
    subsets, _ = H._subsets(ds, target, scored,
        DataFrame(match_id = Int[], market_name = String[],
                  prob_fair_close = Float64[]), H.club_season_tiers(ds))
    @test target == [1, 2]
    @test scored == [1, 2, 3, 4]
    @test subsets["monitor_t54"] == Set([3])
    @test subsets["monitor_t55"] == Set([4])
    lower = Data.DataStore(Data.ScottishLower(), matches[matches.tournament_id .>= 56, :],
                           DataFrame(), DataFrame(), DataFrame(), DataFrame(),
                           DataFrame(), DataFrame(), DataFrame())
    lower_target = H._season_panel(lower, fit, ["24/25"])
    @test lower_target == target
    lower_subsets, _ = H._subsets(lower, lower_target, lower_target,
        DataFrame(match_id = Int[], market_name = String[],
                  prob_fair_close = Float64[]), H.club_season_tiers(lower))
    @test all(lower_subsets[label] == subsets[label] for label in ("target", "t56", "t57"))
    @test all(!startswith(name, "monitor_") || name in ("monitor_t54", "monitor_t55")
              for name in keys(subsets))
end

@testset "Harness scorecard version defaults" begin
    @test H.SCORECARD_VERSION == "v1.2"
    @test H._harness_score_version(nothing) == H.SCORECARD_VERSION
    @test H._harness_score_version(missing) == H.SCORECARD_VERSION
end

@testset "Club-season transition bias bootstrap and paired absolute bias" begin
    matches = DataFrame(match_id = collect(1:80), season = fill("24/25", 80),
        home_team = repeat(["A", "B", "C", "D"], inner = 20),
        away_team = fill("Opponent", 80))
    ds = harness_store(matches)
    clubs = Dict(i => [matches.home_team[i]] for i in 1:80)
    obs = DataFrame(match_id = collect(1:80), selection = fill(:home, 80),
        family = fill("1X2", 80), p_market = fill(0.4, 80),
        p_model = repeat([0.25, 0.35, 0.45, 0.55], inner = 20))
    bias = H.transition_bias_pp(obs, ds, clubs; B = 2_000, seed = 11)
    @test bias.value == 100 * mean([obs.p_model[mid] - obs.p_market[mid]
                                      for (mid, _) in clubs])  # v1.1 iteration order
    @test bias.n_obs == 80
    @test bias == H.transition_bias_pp(obs, ds, clubs; B = 2_000, seed = 11)
    rng = MersenneTwister(11)
    fixture_ci = quantile([100 * mean(rand(rng, obs.p_model .- obs.p_market, 80))
                           for _ in 1:2_000], [0.025, 0.975])
    @test bias.hi - bias.lo > fixture_ci[2] - fixture_ci[1]
    identical = H._delta_abs_bias_vs_control(obs, obs, ds, clubs; B = 2_000)
    @test identical.value == identical.lo == identical.hi == 0.0
    control = copy(obs)
    control.p_model .= 0.45
    improved = H._delta_abs_bias_vs_control(obs, control, ds, clubs; B = 2_000)
    @test improved.value < 0
end

@testset "Panel isolation in leaderboard" begin
    ref = H.RunRef("same", "synthetic", UUID("11111111-1111-1111-1111-111111111111"), :control)
    one = H._score_row(ref, "target", "all", "logloss", 0.3, NaN, NaN, 1, 1;
                       panel = "56+57|24/25,25/26|n=710")
    two = merge(one, (; panel = "56+57|23/24,24/25,25/26|n=1070", value = 0.8))
    board = H.leaderboard(DataFrame([one, two]))
    @test nrow(board) == 2
    @test Set(board.panel) == Set([one.panel, two.panel])
    @test Dict(r.panel => r.target_logloss_all for r in eachrow(board)) ==
          Dict(one.panel => 0.3, two.panel => 0.8)
    script = Module(:LeaderboardScriptTest)
    Base.include(script, joinpath(@__DIR__, "..", "scripts", "leaderboard.jl"))
    md = script.leaderboard_markdown(board, DataFrame())
    @test occursin("## Posterior grid — panel `$(one.panel)`", md)
    @test occursin("## Posterior grid — panel `$(two.panel)`", md)
    @test count("| `same` |", md) == 12  # headline + five cohort tables per panel
end

@testset "Leaderboard retains every within-panel control pairing" begin
    panel_a = "56+57|24/25,25/26|n=710"
    panel_b = "56+57|23/24,24/25,25/26|n=1070"
    ctl_a = H.RunRef("base_a", "synthetic", uuid4(), :control)
    ctl_b = H.RunRef("base_b", "synthetic", uuid4(), :control)
    arm = H.RunRef("arm", "synthetic", uuid4(), :candidate)
    scores = [H._score_row(ref, "target", "all", "logloss", 0.6, NaN, NaN, 1, 1;
                           panel = panel)
              for panel in (panel_a, panel_b), ref in (ctl_a, ctl_b, arm)]
    rows = NamedTuple[vec(scores)...]
    append!(rows, [H._score_row(arm, "target", "all", "delta_logloss_vs_control",
                                delta, delta - 0.01, delta + 0.01, 1, 1;
                                control_run_id = ctl.run_id, panel = panel)
                   for panel in (panel_a, panel_b), (ctl, base_delta) in ((ctl_a, -0.01), (ctl_b, 0.02))
                   for delta in (panel == panel_a ? base_delta : ctl.run_id == ctl_a.run_id ? 0.07 : -0.04,)])
    board = H.leaderboard(DataFrame(rows))
    paired = filter(:model => ==("arm"), board)
    @test nrow(paired) == 4
    @test all(g -> Set(g.control_name) == Set(["base_a", "base_b"]),
              groupby(paired, :panel))
    @test Dict((r.panel, r.control_name) => r.delta_vs_control for r in eachrow(paired)) ==
          Dict((panel_a, "base_a") => -0.01, (panel_a, "base_b") => 0.02,
               (panel_b, "base_a") => 0.07, (panel_b, "base_b") => -0.04)
end

@testset "Leaderboard keeps metrics for a UUID scored under two labels" begin
    # The runbook §2 shape: re-scoring under an alias moves the NULL-control metric rows to the
    # alias, while the native label keeps only its delta rows (write_scores! replace-by-key).
    panel = "56+57|24/25,25/26|n=710"
    ctl_id, arm_id = uuid4(), uuid4()
    ctl_alias = H.RunRef("s12_m02_td_joint", "synthetic", ctl_id, :candidate)
    ctl_native = H.RunRef("td_lower_joint", "synthetic", ctl_id, :control)
    arm_alias = H.RunRef("g1_grw_all_spfl", "synthetic", arm_id, :candidate)
    arm_native = H.RunRef("grw_spfl_poisson", "synthetic", arm_id, :candidate)
    m12 = H.RunRef("m12_td", "synthetic", uuid4(), :control)
    cohort = "transition_$(first(H.HARNESS_DIRECTIONS))_first10"
    metrics(ref, ll) = [
        H._score_row(ref, "target", "all", "logloss", ll, NaN, NaN, 1, 1; panel),
        H._score_row(ref, "target", "1X2", "logloss", ll + 0.3, NaN, NaN, 1, 1; panel),
        H._score_row(ref, "target", "all", "ece", 0.02, NaN, NaN, 1, 1; panel),
        H._score_row(ref, "target", "1X2", "compression_slope", 1.1, NaN, NaN, 1, 1; panel),
        H._score_row(ref, cohort, "all", "logloss", ll + 0.05, NaN, NaN, 1, 12; panel)]
    delta(ref, control, value) = H._score_row(ref, "target", "all", "delta_logloss_vs_control",
        value, value - 0.01, value + 0.01, 1, 1; control_run_id = control.run_id, panel)
    rows = NamedTuple[
        metrics(ctl_alias, 0.64)..., delta(ctl_alias, m12, 0.0004),  # alias rows come first
        delta(ctl_native, ctl_native, 0.0),
        metrics(arm_alias, 0.66)..., delta(arm_alias, m12, 0.02), delta(arm_native, ctl_native, 0.01),
        metrics(m12, 0.65)..., delta(m12, m12, 0.0)]
    for frame in (DataFrame(rows), DataFrame(reverse(rows)))
        board = H.leaderboard(frame)
        @test nrow(board) == 5
        @test all(isfinite, board.target_logloss_all)
        @test all(isfinite, board.target_logloss_1x2)
        @test all(isfinite, board.target_ece_all)
        @test all(isfinite, board.compression_slope)
        @test all(isfinite, board[!, Symbol(cohort * "_logloss")])
        pairs = Dict((r.model, r.control_name) => (r.target_logloss_all, r.delta_vs_control)
                     for r in eachrow(board))
        @test pairs == Dict(
            ("td_lower_joint", "td_lower_joint") => (0.64, 0.0),
            ("s12_m02_td_joint", "m12_td") => (0.64, 0.0004),
            ("grw_spfl_poisson", "td_lower_joint") => (0.66, 0.01),
            ("g1_grw_all_spfl", "m12_td") => (0.66, 0.02),
            ("m12_td", "m12_td") => (0.65, 0.0))
    end
end

@testset "Harness per-subset score row counts" begin
    frame = DataFrame(match_id = repeat([1, 2], inner = 7),
        selection = repeat([:home, :draw, :away, :over_25, :under_25,
                            :btts_yes, :btts_no], 2),
        family = repeat(["1X2", "1X2", "1X2", "OU2.5", "OU2.5", "BTTS", "BTTS"], 2),
        p_model = fill(0.4, 14), p_market = fill(0.4, 14),
        y = repeat([1.0, 0, 0, 1, 0, 1, 0], 2))
    frame.ll_model = BayesianFootball.Evaluation.calc_logloss.(frame.p_model, frame.y)
    frame.ll_market = copy(frame.ll_model)
    @test H._observation_scores(frame, "all").n_obs == 14
    @test H._observation_scores(frame, "1X2").n_obs == 6
    @test H._observation_scores(frame, "OU2.5").n_obs == 4
    @test H._observation_scores(frame, "BTTS").n_obs == 4
    @test H._observation_scores(frame[in.(frame.match_id, Ref(Set([1]))), :], "all").n_fixtures == 1
end
