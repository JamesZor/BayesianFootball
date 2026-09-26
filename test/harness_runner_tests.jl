using Test
using BayesianFootball
using DataFrames
using Dates
using UUIDs

@testset "Harness Runner and Gate Verification" begin

    @testset "1. recipe_hash stability and sensitivity" begin
        scope1 = Data.DataScope(name = "scope1", train_tournaments = [56, 57])
        scope2 = Data.DataScope(name = "scope2", train_tournaments = [54, 55, 56, 57])

        model1 = CountModelBuilder(:m1) |>
            add(GlobalInterception()) |>
            add(TimeDecayDynamics(days_half_life = 180.0)) |>
            add(GlobalHomeAdvantage()) |>
            add(PoissonObservation()) |>
            build

        model2 = CountModelBuilder(:m2) |>
            add(GlobalInterception()) |>
            add(MultiScaleGRW()) |>
            add(GlobalHomeAdvantage()) |>
            add(PoissonObservation()) |>
            build

        sampler1 = Samplers.QueuedNUTSConfig(n_samples = 1000, n_warmup = 500, n_chains = 4)
        sampler2 = Samplers.QueuedNUTSConfig(n_samples = 500, n_warmup = 250, n_chains = 2)

        c1 = Harness.Candidate(name = "c1", model = model1, scope = scope1, sampler = sampler1)
        c2 = Harness.Candidate(name = "c2", model = model1, scope = scope1, sampler = sampler2)
        c3 = Harness.Candidate(name = "c3", model = model2, scope = scope1, sampler = sampler1)
        c4 = Harness.Candidate(name = "c4", model = model1, scope = scope2, sampler = sampler1)

        # Same model and scope but different sampler -> identical recipe_hash
        @test Harness.recipe_hash(c1) == Harness.recipe_hash(c2)
        # Different model -> different recipe_hash
        @test Harness.recipe_hash(c1) != Harness.recipe_hash(c3)
        # Different scope -> different recipe_hash
        @test Harness.recipe_hash(c1) != Harness.recipe_hash(c4)
    end

    @testset "2. Severity routing (hard check fails vs review diagnostic)" begin
        store = Harness.InMemoryCheckStore()
        base = (;
            run_id = uuid4(),
            recipe_hash = "fake_recipe_hash",
            experiment = "synth_exp",
            candidate = "synth_cand",
            stage = "smoke",
            git_sha = "abcd123"
        )
        records = NamedTuple[]

        # Hard failure MUST throw and record fail with severity hard
        @test_throws Harness.HarnessCheckError Harness._run_hard_check!(records, base, "hard_check") do
            error("simulated critical invariant breach")
        end
        @test length(records) == 1
        @test records[1].check == "hard_check"
        @test records[1].severity == "hard"
        @test records[1].status == "fail"
        @test occursin("simulated critical invariant breach", records[1].detail)

        # Review diagnostic MUST NOT throw and records fail with severity review
        diag_val = Harness._run_diagnostic!(records, base, "convergence_audit", "review") do
            error("max R-hat 1.05 exceeds 1.02 gate")
        end
        @test diag_val === nothing
        @test length(records) == 2
        @test records[2].check == "convergence_audit"
        @test records[2].severity == "review"
        @test records[2].status == "fail"
        @test occursin("max R-hat 1.05", records[2].detail)

        # Passing hard check
        pass_val = Harness._run_hard_check!(records, base, "passing_hard") do
            (; n_matches = 40)
        end
        @test pass_val.n_matches == 40
        @test length(records) == 3
        @test records[3].check == "passing_hard"
        @test records[3].severity == "hard"
        @test records[3].status == "pass"

        Harness.write_checks!(store, records)
        stored_df = Harness.read_checks(store; candidate = "synth_cand")
        @test nrow(stored_df) == 3
        @test sum(stored_df.status .== "fail") == 2
        @test sum(stored_df.severity .== "hard") == 2
        @test sum(stored_df.severity .== "review") == 1
    end

    @testset "3. Grid stage refuses without passing smoke" begin
        store = Harness.InMemoryCheckStore()
        scope = Data.DataScope(name = "test_scope", train_tournaments = [56], target_tournaments = [56], clock_tournaments = [56])
        model = CountModelBuilder(:test_grid_refuse) |>
            add(GlobalInterception()) |>
            add(TimeDecayDynamics(days_half_life = 180.0)) |>
            add(GlobalHomeAdvantage()) |>
            add(PoissonObservation()) |>
            build

        candidate = Harness.Candidate(
            name = "test_grid_refuse",
            model = model,
            scope = scope,
            role = :candidate
        )

        @test Harness.has_passing_smoke(store, candidate) == false

        dummy_matches = DataFrame(
            match_id = [1, 2], tournament_id = [56, 56], season = ["24/25", "24/25"],
            match_date = [Date(2024, 8, 1), Date(2024, 8, 15)],
            match_hour = [15, 15], match_week = [1, 2], match_biweek = [1, 2], match_month = [8, 8],
            home_team = ["A", "B"], away_team = ["B", "A"],
            home_score = [1, 0], away_score = [0, 1], neutral_venue = [false, false]
        )
        empty_df = DataFrame()
        dummy_ds = Data.DataStore(Data.ScottishLower(), dummy_matches, empty_df, empty_df,
                                  empty_df, empty_df, empty_df, empty_df, empty_df)

        # Refusal error must name candidate and smoke
        err = try
            Harness.grid(candidate; ds = dummy_ds, experiment = "test_exp", db = store)
            nothing
        catch e
            e
        end
        @test err isa Exception
        @test occursin("has no passing smoke record", sprint(showerror, err))
        @test occursin("run --stage smoke first", sprint(showerror, err))
    end

    @testset "4. beast_checkout.sh script validation" begin
        script_path = joinpath(@__DIR__, "..", "scripts", "beast_checkout.sh")
        @test isfile(script_path)
        cmd = `bash -n $script_path`
        res = run(cmd)
        @test res.exitcode == 0
    end

    @testset "5. Offline synthetic smoke execution and no-DB grid lock" begin
        store = Harness.InMemoryCheckStore()
        matches = DataFrame(
            match_id = [1, 2, 3, 4],
            tournament_id = [56, 56, 56, 56],
            season = ["23/24", "23/24", "24/25", "24/25"],
            match_date = [Date(2024, 1, 1), Date(2024, 2, 1), Date(2024, 8, 1), Date(2024, 8, 15)],
            match_hour = [15, 15, 15, 15],
            match_week = [1, 2, 1, 2],
            match_biweek = [1, 2, 1, 2],
            match_month = [1, 2, 8, 8],
            home_team = ["A", "B", "A", "B"],
            away_team = ["B", "A", "B", "A"],
            home_score = [1, 0, 2, 1],
            away_score = [0, 1, 1, 1],
            neutral_venue = [false, false, false, false]
        )
        empty_df = DataFrame()
        ds = Data.DataStore(Data.ScottishLower(), matches, empty_df, empty_df,
                            empty_df, empty_df, empty_df, empty_df, empty_df)

        model = CountModelBuilder(:synth_smoke) |>
            add(GlobalInterception()) |>
            add(TimeDecayDynamics(days_half_life = 180.0)) |>
            add(GlobalHomeAdvantage()) |>
            add(PoissonObservation()) |>
            build

        scope = Data.DataScope(
            name = "synth_scope",
            train_tournaments = [56],
            target_tournaments = [56],
            clock_tournaments = [56],
            target_seasons = ["24/25"],
            history_seasons = 1
        )

        # Tiny synthetic sampler
        tiny_sampler = Samplers.QueuedNUTSConfig(
            n_samples = 10,
            n_warmup = 10,
            n_chains = 2,
            accept_rate = 0.65,
            max_depth = 4,
            show_progress = false
        )

        candidate = Harness.Candidate(
            name = "synth_smoke",
            model = model,
            scope = scope,
            sampler = tiny_sampler,
            role = :candidate
        )

        res = Harness.smoke(candidate; ds = ds, experiment = "synth_exp", db = store)
        @test res.fit isa Training.Fit
        @test res.run_id === nothing # No fabricated run_id without PostgresStorage

        # A no-DB smoke records fit_parity as abstain, so has_passing_smoke MUST be false
        @test Harness.has_passing_smoke(store, candidate) == false

        # Grid MUST refuse because smoke has an abstained hard check
        @test_throws ErrorException Harness.grid(candidate; ds = ds, experiment = "synth_exp", db = store)

        stored = Harness.read_checks(store; candidate = "synth_smoke", stage = "smoke")
        @test nrow(stored) >= 5
        @test "gradient" in stored.check
        @test "filtration" in stored.check
        @test "latents" in stored.check
        @test "score_grid_coherence" in stored.check
        @test "fit_parity" in stored.check

        parity_row = only(filter(r -> r.check == "fit_parity", eachrow(stored)))
        @test parity_row.status == "abstain"
        @test parity_row.severity == "hard"
        @test occursin("not a PostgresStorage", parity_row.detail)

        # If and only if a genuine fit_parity pass is recorded, grid is unlocked
        valid_store = Harness.InMemoryCheckStore()
        for r in eachrow(stored)
            if r.check == "fit_parity"
                push!(valid_store.checks, merge(NamedTuple(r), (; status = "pass", detail = "")))
            else
                push!(valid_store.checks, NamedTuple(r))
            end
        end
        @test Harness.has_passing_smoke(valid_store, candidate) == true
    end

end
