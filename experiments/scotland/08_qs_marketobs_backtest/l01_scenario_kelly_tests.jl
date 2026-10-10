"Pure mathematical/interface checks. No fits, database reads or financial backtest."
function verify_scenario_allocator(out; native::Bool = false, regression::Bool = false)
    mkpath(out)
    allocator = scenario_allocator(native)
    exec = PF.ExecutionConfig(min_selection_stake = 0.0)
    observations = NamedTuple[]
    golden = NamedTuple[]
    mode = native ? "native" : "prototype"
    golden_path = joinpath(out, "phase2_legacy_golden.csv")
    reference = native ? CSV.read(golden_path, DF.DataFrame; types = Dict(:bits => UInt64)) : nothing
    function note(test, value, expected, tolerance)
        push!(observations, (; test, value = Float64(value), expected = Float64(expected),
                            tolerance = Float64(tolerance), passed = abs(value - expected) <= tolerance))
    end
    function legacy_golden(label, p, R, cfg)
        a = PF.allocate(PF.KellyLogUtility(), p, R, cfg).a
        for (j, value) in enumerate(a)
            bits = reinterpret(UInt64, value)
            push!(golden, (; test = label, leg = j, stake = value, bits))
            if native
                row = filter(r -> r.test == label && r.leg == j, reference)
                Test.@test DF.nrow(row) == 1
                Test.@test bits == row.bits[1]
            end
        end
        return a
    end

    Test.@testset "Scenario Kelly $(mode)" begin
        Test.@testset "Closed-form single bet and commission" begin
            sel = PF.Selection("1X2_home", "1X2", 0.0, :home, 2.0, 2.0, 0.6, 0.5)
            R = PF.payoff_matrix([sel], 2, 1, exec.commission)
            p = [0.4, 0.6]
            b = (2.0 - 1.0) * (1.0 - 0.02)
            expected = (0.6 * (1 + b) - 1) / b
            result = PF.allocate(allocator, scenario_weights(native, p), R, exec)
            Test.@test R[:, 1] == [-1.0, b]
            Test.@test result.converged
            Test.@test isapprox(result.a[1], expected; atol = 2e-5, rtol = 0)
            Test.@test result.a[1] >= 0
            note("single_bet_closed_form_net_commission", result.a[1], expected, 2e-5)
            legacy_golden("single_bet", p, R, exec)
        end

        Test.@testset "Independent bets versus independent brute-force oracle" begin
            p = [0.36, 0.24, 0.24, 0.16]
            R = [1.0 1.0; 1.0 -1.0; -1.0 1.0; -1.0 -1.0]
            result = PF.allocate(allocator, scenario_weights(native, p), R, exec)
            # Re-derive the oracle directly: never call Portfolio.growth/allocate for it.
            objective(a) = sum(p[s] * log(1.0 + sum(R[s, j] * a[j] for j in 1:2)) for s in 1:4)
            best, best_a = -Inf, [0.0, 0.0]
            for x in 0.0:0.0025:0.5, y in 0.0:0.0025:0.5
                x + y <= exec.budget || continue
                u = objective([x, y])
                if u > best
                    best, best_a = u, [x, y]
                end
            end
            actual = objective(result.a)
            Test.@test result.converged
            Test.@test actual >= best - 1e-7
            Test.@test maximum(abs.(result.a .- best_a)) <= 0.0025
            Test.@test all(isapprox.(result.a, 5 / 26; atol = 2e-5, rtol = 0))
            Test.@test all(result.a .>= 0) && sum(result.a) <= exec.budget
            note("independent_grid_objective", actual, best, 1e-5)
            note("independent_analytic_stake", result.a[1], 5 / 26, 2e-5)
            legacy_golden("independent_bets", p, R, exec)
        end

        Test.@testset "Shared posterior draws and full slate" begin
            slate, grids, cfg = toy_slate()
            groups = ["toy_posterior_fold1_chain_layout1", "toy_posterior_fold1_chain_layout1"]
            original = [b.a_kelly[:] for b in slate.books]
            joint = scenario_books(allocator, slate, grids, groups, cfg; native = native)
            problem = joint.problem
            Test.@test length(problem.draw_indices) == 1000
            Test.@test size(problem.R) == (1000, 2)
            Test.@test problem.R[:, 1] == problem.R[:, 2]
            Test.@test problem.score_rows[:, 1] == problem.score_rows[:, 2]
            Test.@test count(==(1.0), problem.R[:, 1]) == 600
            Test.@test abs(sum(joint.solution.a) - 0.2) <= 2e-5
            Test.@test abs(sum(sum(b.a_kelly) for b in slate.books) - 0.4) <= 4e-5
            Test.@test all(joint.solution.a .>= 0)
            Test.@test [b.a_kelly for b in slate.books] == original
            Test.@test all(joint.slate.books[i].R === slate.books[i].R for i in 1:2)
            Test.@test all(joint.slate.books[i].k_shrink == slate.books[i].k_shrink for i in 1:2)
            note("shared_posterior_joint_total_stake", sum(joint.solution.a), 0.2, 2e-5)
            Test.@test_throws ArgumentError scenario_problem(allocator, slate, grids, ["fold1", "fold2"]; native = native)
            Test.@test_throws ArgumentError scenario_problem(allocator, slate, grids, ["", ""]; native = native)
            Test.@test_throws DimensionMismatch scenario_problem(allocator, slate,
                [grids[1], grids[2][:, :, 1:9]], groups; native = native)
            bad = grids[1][:, :, :]
            bad[1, 1, 1] = -0.1
            Test.@test_throws ArgumentError scenario_problem(allocator, slate, [bad, grids[2]], groups; native = native)

            # Check the original production pipeline, not a transcription of its steps.
            policy = PF.PolicySpec(trust = PF.FlatTrust(0.5), risk = PF.NoRisk(), cap = PF.FixedCap(0.9))
            ctx = PF.SlateContext(1, slate.window, 1.0)
            staked = PF.stake_slate(policy, joint.slate, ctx)
            Test.@test all(isapprox(staked.stakes[i][1], joint.solution.a[i] * 0.5 * 0.30;
                                  atol = 1e-14, rtol = 0) for i in 1:2)
            Test.@test !staked.capped
            capped = PF.stake_slate(PF.PolicySpec(trust = PF.FlatTrust(1.0), risk = PF.NoRisk(),
                                                 cap = PF.FixedCap(0.01)), joint.slate, ctx)
            Test.@test capped.capped && isapprox(capped.exposure, 0.01; atol = 1e-14)
            risked = PF.stake_slate(PF.PolicySpec(trust = PF.FlatTrust(1.0),
                                                 risk = PF.SlateDrawdown(23.0), cap = PF.FixedCap(0.25)),
                                    joint.slate, ctx)
            Test.@test 0 <= risked.k_risk <= 1 && 0 <= risked.exposure <= 0.25
            Test.@test [b.a_kelly for b in slate.books] == original
        end

        Test.@testset "Deterministic seed, fixture order and scoreline randomness" begin
            slate, grids, cfg = toy_slate(shared = false)
            groups = ["toy_fold", "toy_fold"]
            first = scenario_problem(allocator, slate, grids, groups; native = native)
            again = scenario_problem(allocator, slate, grids, groups; native = native)
            Test.@test first.draw_indices == again.draw_indices
            Test.@test first.score_rows == again.score_rows && first.R == again.R
            reversed = scenario_problem(allocator, PF.Slate(slate.window, reverse(slate.books)),
                                        reverse(grids), groups; native = native)
            Test.@test first.R == reversed.R[:, [2, 1]]
            changed = scenario_problem(scenario_allocator(native; seed = 20261011),
                                       slate, grids, groups; native = native)
            Test.@test first.score_rows != changed.score_rows
            a = PF.allocate(allocator, first.weights, first.R, cfg).a
            b = PF.allocate(allocator, again.weights, again.R, cfg).a
            Test.@test a == b
            note("seed_replay_max_stake_difference", maximum(abs.(a .- b)), 0.0, 0.0)
        end

        Test.@testset "Constraints and no silent mean-vector fallback" begin
            p = scenario_weights(native, [1.0])
            R = ones(1, 2)
            cfg = PF.ExecutionConfig(budget = 0.07, max_selection_stake = 0.04, min_selection_stake = 0.0)
            result = PF.allocate(allocator, p, R, cfg)
            Test.@test result.converged
            Test.@test all(>=(0.0), result.a) && all(<=(0.04), result.a) && sum(result.a) <= 0.07
            empty = PF.allocate(allocator, p, zeros(1, 0), cfg)
            Test.@test isempty(empty.a) && empty.converged
            Test.@test_throws ArgumentError PF.allocate(allocator, [1.0], R, cfg)
            Test.@test_throws ArgumentError scenario_allocator(native; n_scenarios = 1001)
            Test.@test_throws ArgumentError scenario_allocator(native; n_scenarios = 0)
            Test.@test_throws ArgumentError PF.allocate(allocator, scenario_weights(native, [-0.1, 1.1]), zeros(2, 1), cfg)
            Test.@test_throws ArgumentError PF.allocate(allocator, p, fill(NaN, 1, 1), cfg)
            Test.@test_throws ArgumentError PF.allocate(allocator, p, fill(-1.1, 1, 1), cfg)
            Test.@test_throws ArgumentError PF.allocate(allocator, p, R, PF.ExecutionConfig(budget = 1.01))
            if native
                small = PF.ExecutionConfig(budget = 0.01, max_selection_stake = 0.005, min_selection_stake = 0.0)
                wide = PF.allocate(allocator, p, ones(1, 20), small)
                Test.@test wide.converged && sum(wide.a) <= 0.01 && all(wide.a .>= 0)
            end
        end
    end
    CSV.write(joinpath(out, "phase2_" * mode * "_tests.csv"), DF.DataFrame(observations))
    if !native
        CSV.write(golden_path, DF.DataFrame(golden))
    end
    if regression
        Test.@testset "Existing portfolio regression" begin
            Base.invokelatest(Base.include, @__MODULE__, joinpath(pkgdir(BF), "test", "portfolio_tests.jl"))
        end
    end
    println(native ? "PHASE2_PASS — native allocator, experimental adapter and checks; stop before Phase3" :
                     "PROTOTYPE_PASS — validate before src graduation; no backtest ran")
    flush(stdout)
    return nothing
end
