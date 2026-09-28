using Test
using BayesianFootball
using DataFrames
using Dates
using UUIDs

# The pre-fix clamp: bare `Float64` bounds put `Real`s in the broadcast, which is what sends
# ReverseDiff to its allocating `tracker_∇broadcast` adjoint (AD guide §10.5).
struct TapeGateScalarClamp <: BayesianFootball.Models.PreGame.Builder.AbstractRateGuard end
BayesianFootball.Models.PreGame.Builder.apply_guard(::TapeGateScalarClamp, η) =
    clamp.(η, -10.0, 10.0)
BayesianFootball.Models.PreGame.Builder.guard_describe(::TapeGateScalarClamp) =
    "scalar clamp (test only)"

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

        fixed_model = CountModelBuilder(:fixed) |>
            add(GlobalInterception()) |>
            add(StaticZeroDynamics()) |>
            add(GlobalHomeAdvantage()) |>
            add(PoissonObservation()) |>
            build

        hierarchical_home_model = CountModelBuilder(:hierarchical_home) |>
            add(GlobalInterception()) |>
            add(StaticZeroDynamics()) |>
            add(HierarchicalTeamHomeAdvantage()) |>
            add(PoissonObservation()) |>
            build

        sampler1 = Samplers.QueuedNUTSConfig(n_samples = 1000, n_warmup = 500, n_chains = 4)
        sampler2 = Samplers.QueuedNUTSConfig(n_samples = 500, n_warmup = 250, n_chains = 2)

        c1 = Harness.Candidate(name = "c1", model = model1, scope = scope1, sampler = sampler1)
        c2 = Harness.Candidate(name = "c2", model = model1, scope = scope1, sampler = sampler2)
        c3 = Harness.Candidate(name = "c3", model = model2, scope = scope1, sampler = sampler1)
        c4 = Harness.Candidate(name = "c4", model = model1, scope = scope2, sampler = sampler1)
        c5 = Harness.Candidate(name = "c5", model = fixed_model, scope = scope1, sampler = sampler1)
        c6 = Harness.Candidate(name = "c6", model = hierarchical_home_model,
                               scope = scope1, sampler = sampler1)

        # Same model and scope but different sampler -> identical recipe_hash
        @test Harness.recipe_hash(c1) == Harness.recipe_hash(c2)
        # Different model -> different recipe_hash
        @test Harness.recipe_hash(c1) != Harness.recipe_hash(c3)
        # Different scope -> different recipe_hash
        @test Harness.recipe_hash(c1) != Harness.recipe_hash(c4)
        # Both TimeDecay and MultiScaleGRW learn hierarchical scales; MAP is limited.
        @test Harness._screen_validity(c1) == "limited"
        @test Harness._screen_validity(c3) == "limited"
        @test Harness._screen_validity(c5) == "ranking_only"
        @test Harness._screen_validity(c6) == "limited"
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
        @test isempty(records[2].value)

        # Expected diagnostic failures carry their metrics through the durable store.
        metrics = (; max_rhat = 1.06, divergences = 12)
        Harness._run_diagnostic!(records, base, "metrics", "review") do
            throw(Harness.DiagnosticFailure(metrics, "R-hat exceeded"))
        end
        Harness.write_checks!(store, records[end:end])
        saved = only(eachrow(Harness.read_checks(store; check = "metrics")))
        @test saved.value == metrics
        @test saved.status == "fail"
        @test saved.detail == "R-hat exceeded"
    end

    @testset "Harness-only convergence rule" begin
        summary = (; max_rhat = 1.04, min_ess_bulk = 300.0, min_ess_tail = 300.0,
                    n_divergent = 0, divergence_rate = 0.0, min_bfmi = 0.2,
                    treedepth_rate = 0.1, thresholds = Training.ConvergenceThresholds(),
                    abstained = String[])
        value = Harness._harness_convergence_value(summary)
        @test value.passed && length(value.notes) == 4
        @test Harness._harness_convergence_value(merge(summary, (; max_rhat = 1.05))).passed
        @test Harness._harness_convergence_value(merge(summary, (; divergence_rate = 0.001))).passed
        for bad in (merge(summary, (; max_rhat = 1.06)),
                    merge(summary, (; divergence_rate = 0.0011)))
            records = NamedTuple[]
            Harness._run_diagnostic!(records, (; run_id = nothing), "convergence", "review") do
                Harness._harness_convergence_value(bad)
            end
            @test only(records).status == "fail"
            @test only(records).value.max_rhat == bad.max_rhat
            @test only(records).value.divergence_rate == bad.divergence_rate
        end
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

        perf_dir = mktempdir()
        res = Harness.smoke(candidate; ds = ds, experiment = "synth_exp", db = store,
                            perf_chains = 2, perf_warmup = 5, perf_samples = 5,
                            log_dir = perf_dir)
        @test res.fit isa Training.Fit
        # The sampling-performance probe ran on the largest fold, wrote its report where it was
        # told to, and recorded an info row with the metrics plus a review row — never hard.
        @test res.perf.status in ("pass", "review")
        @test res.perf.report == joinpath(perf_dir, "smoke_perf", "synth_exp", "synth_smoke.md")
        @test isfile(res.perf.report)
        @test occursin("perf=$(res.perf.status)", Harness.perf_summary(res.perf))
        @test res.run_id === nothing # No fabricated run_id without PostgresStorage

        # A no-DB smoke records fit_parity as abstain, so has_passing_smoke MUST be false
        @test Harness.has_passing_smoke(store, candidate) == false

        # Grid MUST refuse because smoke has an abstained hard check
        @test_throws ErrorException Harness.grid(candidate; ds = ds, experiment = "synth_exp", db = store)

        stored = Harness.read_checks(store; candidate = "synth_smoke", stage = "smoke")
        @test nrow(stored) >= 5
        @test "gradient" in stored.check
        @test "tape_allocation" in stored.check
        tape_row = only(filter(r -> r.check == "tape_allocation", eachrow(stored)))
        @test tape_row.severity == "hard"
        @test tape_row.status == "pass"
        @test tape_row.value.max_allocated_bytes == 0
        @test length(tape_row.value.folds) >= 1
        @test "filtration" in stored.check
        perf_rows = filter(r -> startswith(r.check, "sampling_performance"), eachrow(stored))
        @test Set(r.check for r in perf_rows) ==
              Set(["sampling_performance_metrics", "sampling_performance"])
        @test only(filter(r -> r.check == "sampling_performance", perf_rows)).severity == "review"
        @test only(filter(r -> r.check == "sampling_performance_metrics", perf_rows)).severity == "info"
        @test !("sampling_performance" in Harness.SMOKE_REQUIRED_CHECKS)
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

        # A smoke recorded before the tape_allocation gate existed does not unlock the grid.
        pre_gate_store = Harness.InMemoryCheckStore()
        for r in valid_store.checks
            r.check == "tape_allocation" || push!(pre_gate_store.checks, r)
        end
        @test Harness.has_passing_smoke(pre_gate_store, candidate) == false
        @test "tape_allocation" in Harness.SMOKE_REQUIRED_CHECKS
    end

    @testset "5b. tape_allocation fails on an allocating tape and names the instruction" begin
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
        scope = Data.DataScope(name = "tape_scope", train_tournaments = [56],
            target_tournaments = [56], clock_tournaments = [56],
            target_seasons = ["24/25"], history_seasons = 1)
        model = CountModelBuilder(:tape_gate) |>
            add(GlobalInterception()) |>
            add(TimeDecayDynamics(days_half_life = 180.0)) |>
            add(GlobalHomeAdvantage()) |>
            add(PoissonObservation()) |>
            build
        inputs = Harness._fold_inputs(
            Harness.Candidate(name = "tape_gate", model = model, scope = scope), ds;
            stage = :smoke)

        clean = Harness._tape_allocation_check(model, inputs.feature_sets)
        @test clean.max_allocated_bytes == 0
        @test clean.max_bytes == Harness.TAPE_ALLOCATION_LIMIT_BYTES == 1024

        # The pre-fix ClampGuard: bare Float64 bounds inside the broadcast. The builder only
        # constructs its own guard types, so assemble the model directly, as the l10 prototype did.
        allocating = BayesianFootball.Models.PreGame.Builder.PoissonCountModel(
            model.interception, model.dynamics, model.home_advantage, model.covariates,
            model.observation, TapeGateScalarClamp())
        err = try
            Harness._tape_allocation_check(allocating, inputs.feature_sets; max_bytes = 0)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("allocates", err.msg)
        @test occursin("tracker_∇broadcast[clamp]", err.msg)
        @test occursin("§10.5", err.msg)

        # Through the hard-check wrapper it is a HarnessCheckError with a durable failing row.
        records = NamedTuple[]
        base = (; run_id = nothing, recipe_hash = "h", experiment = "e", candidate = "c",
                  stage = "smoke", git_sha = "g")
        @test_throws Harness.HarnessCheckError Harness._run_hard_check!(
            records, base, "tape_allocation") do
            Harness._tape_allocation_check(allocating, inputs.feature_sets; max_bytes = 0)
        end
        @test only(records).status == "fail"
        @test only(records).severity == "hard"

        # --- the sampling-performance monitor on the same synthetic league -------------------
        fs = first(inputs.feature_sets[end])
        sampler = Samplers.QueuedNUTSConfig(n_samples = 5, n_warmup = 5, n_chains = 2,
                                            max_depth = 4, show_progress = false)
        fixed = Harness.sampling_performance_probe(model, fs, sampler;
                                                   n_chains = 2, n_warmup = 5, n_samples = 5)
        @test fixed.tape_bytes_per_gradient == 0
        @test isempty(fixed.allocating_instructions)
        @test fixed.leapfrogs == sum(fixed.leapfrogs_per_chain) > 0
        @test length(fixed.leapfrogs_per_chain) == 2
        @test fixed.wall_seconds > 0 && fixed.window_seconds > 0
        @test fixed.ms_per_leapfrog isa Real && fixed.efficiency_ratio isa Real
        @test fixed.bytes_per_leapfrog isa Real && fixed.gc_share isa Real
        # rev2: GC from GC_Diff, utilisation and stalls from the statistical profiler.
        @test fixed.gc_pauses >= 0 && fixed.gc_allocd_bytes >= 0
        @test fixed.utilisation_source == "profile"
        @test fixed.sampler_utilisation isa Real && 0.0 <= fixed.sampler_utilisation <= 1.0
        @test fixed.sampler_utilisation_min <= fixed.sampler_utilisation
        @test count(t -> t.pool == "default", fixed.profile_threads) >= 1
        @test fixed.gc_stall_share isa Real && fixed.jit_share isa Real
        @test fixed.top_frames isa AbstractVector && length(fixed.top_frames) <= 15
        # Profile.Allocs on full sampler steps reports sites outside the tape.
        @test !isempty(fixed.non_tape_step_sites)
        @test all(s -> s.source == "nuts_step" && s.est_bytes_per_unit > 0, fixed.non_tape_step_sites)
        @test all(s -> s.source == "gradient", fixed.non_tape_gradient_sites)
        if Sys.islinux()
            @test fixed.sampler_thread_ids_found == Threads.nthreads()
            @test fixed.proc_sampler_utilisation isa Real && fixed.process_utilisation isa Real
        end
        @test !any(f -> f.flag == "tape_allocation", Harness.sampling_performance_flags(fixed))
        healthy = merge(fixed, (; gc_share = 0.02, sampler_utilisation = 0.95,
                                  efficiency_ratio = 1.2, gc_stall_share = 0.0, jit_share = 0.0))
        @test isempty(Harness.sampling_performance_flags(healthy))

        # The deliberately allocating (pre-fix) engine is flagged for review, with the cause named.
        slow = Harness.sampling_performance_probe(allocating, fs, sampler;
                                                  n_chains = 2, n_warmup = 5, n_samples = 5)
        @test slow.tape_bytes_per_gradient > 0
        @test any(r -> occursin("tracker_∇broadcast[clamp]", r.description),
                  slow.allocating_instructions)
        slow_flags = Harness.sampling_performance_flags(
            merge(slow, (; tape_bytes_per_gradient = 2 * Harness.TAPE_ALLOCATION_LIMIT_BYTES)))
        @test any(f -> f.flag == "tape_allocation" && occursin("§10.5", f.diagnosis), slow_flags)

        # Each threshold fires on its own, and the diagnosis reads the other metrics.
        th = Harness.SAMPLING_PERF_THRESHOLDS
        @test (th.max_gc_share, th.min_sampler_utilisation, th.max_efficiency_ratio) ==
              (0.15, 0.75, 2.0)
        gc_only = Harness.sampling_performance_flags(merge(healthy, (; gc_share = 0.30)))
        @test [f.flag for f in gc_only] == ["gc_share"]
        @test occursin("non-tape path", only(gc_only).diagnosis)
        idle = Harness.sampling_performance_flags(merge(healthy, (; sampler_utilisation = 0.40)))
        @test [f.flag for f in idle] == ["sampler_utilisation"]
        @test occursin("too few chains", only(idle).diagnosis)
        contended = Harness.sampling_performance_flags(merge(healthy,
            (; gc_share = 0.40, sampler_utilisation = 0.45, efficiency_ratio = 3.0,
               gc_stall_share = 0.30)))
        @test [f.flag for f in contended] == ["gc_share", "sampler_utilisation", "efficiency_ratio"]
        @test occursin("wait at safepoints", contended[2].diagnosis)
        @test occursin("30.0% of awake sampler-thread samples in GC-stall frames", contended[2].diagnosis)

        # The report carries every section; the allocating one lists its instructions.
        report = Harness.write_sampling_performance_report(
            joinpath(mktempdir(), "nested", "slow.md"), slow, slow_flags;
            title = "Sampling performance — slow", context = (; candidate = "slow"))
        text = read(report, String)
        for section in ("# Sampling performance — slow", "## Verdict", "## Metrics",
                        "## Threads (statistical profiler)", "### Top frames by self samples",
                        "## Allocating tape instructions", "## Allocation outside the tape",
                        "## Diagnosis")
            @test occursin(section, text)
        end
        @test occursin("**review**", text)
        @test occursin("tracker_∇broadcast[clamp]", text)
        @test occursin("None — the compiled tape replays without allocating.",
                       read(Harness.write_sampling_performance_report(
                           joinpath(mktempdir(), "fixed.md"), healthy, NamedTuple[]), String))

        # Profile-thread parsing, checked on a hand-built buffer: thread 1 awake twice (once in a
        # GC stall), asleep once. Blocks are [leaf ip, …, threadid, taskid, clock, sleep+1, 0, 0].
        stall = Base.StackTraces.StackFrame(:jl_safepoint_wait_gc, Symbol("safepoint.c"), 268,
                                            nothing, true, false, UInt64(0))
        work = Base.StackTraces.StackFrame(:gradient!, Symbol("api.jl"), 10,
                                           nothing, false, false, UInt64(0))
        # Julia 1.12's `-t N` adds an interactive thread 1, so take a real default-pool thread id.
        tid = UInt64(fetch(Threads.@spawn :default Threads.threadid()))
        buffer = UInt64[0x10, tid, 7, 99, 1, 0, 0,     # awake, in work
                        0x20, tid, 7, 99, 1, 0, 0,     # awake, in a GC stall
                        0x10, tid, 7, 99, 2, 0, 0]     # asleep
        summary = Harness.profile_thread_summary(buffer,
            Dict(UInt64(0x10) => [work], UInt64(0x20) => [stall]))
        @test only(summary.threads).samples == 3
        @test only(summary.threads).utilisation ≈ 2 / 3
        @test summary.gc_stall_share == 0.5
        @test first(summary.top_frames).self == 1

        # The profiler switched off: utilisation falls back to /proc and says so; with no /proc
        # either (macOS, a container) CPU metrics degrade to "unavailable" and cannot flag.
        proc_only = Harness.sampling_performance_probe(model, fs, sampler; n_chains = 1,
            n_warmup = 3, n_samples = 3, profile = false, allocs = false)
        @test proc_only.utilisation_source == (Sys.islinux() ? "proc" : "unavailable")
        @test proc_only.gc_stall_share == "unavailable"
        @test Harness._proc_cpu_snapshot(joinpath(mktempdir(), "no_proc")) === nothing
        blind = Harness.sampling_performance_probe(model, fs, sampler; n_chains = 1,
            n_warmup = 3, n_samples = 3, profile = false, allocs = false,
            proc_root = joinpath(mktempdir(), "no_proc"))
        @test blind.sampler_utilisation == "unavailable"
        @test blind.utilisation_source == "unavailable"
        @test blind.process_utilisation == "unavailable"
        @test blind.other_cpu_cores == "unavailable"
        @test blind.gc_share isa Real
        @test !any(f -> f.flag == "sampler_utilisation", Harness.sampling_performance_flags(
            merge(blind, (; gc_share = 0.0, efficiency_ratio = 1.0))))
        @test occursin("util=n/a",
            Harness.perf_summary((; status = "pass", metrics = blind, flags = [], report = nothing)))
    end

    @testset "6. Deterministic screen run_id across repeated screen calls" begin
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
        odds_df = DataFrame(
            match_id = [3, 3, 3, 4, 4, 4],
            market_name = ["1X2", "1X2", "1X2", "1X2", "1X2", "1X2"],
            market_line = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            selection = [:home, :draw, :away, :home, :draw, :away],
            is_winner = [true, false, false, false, false, true]
        )
        betfair_odds = DataFrame(
            match_id = [3, 3, 3, 4, 4, 4],
            market_name = ["1X2", "1X2", "1X2", "1X2", "1X2", "1X2"],
            market_line = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            selection = [:home, :draw, :away, :home, :draw, :away],
            odds = [2.0, 3.2, 3.8, 2.1, 3.1, 3.6],
            traded_price = [2.0, 3.2, 3.8, 2.1, 3.1, 3.6],
            minutes_to_kickoff = [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
            tournament_id = [56, 56, 56, 56, 56, 56]
        )
        ds = Data.DataStore(Data.ScottishLower(), matches, empty_df, odds_df,
                            empty_df, empty_df, betfair_odds, empty_df, empty_df)

        model = CountModelBuilder(:synth_screen) |>
            add(GlobalInterception()) |>
            add(TimeDecayDynamics(days_half_life = 180.0)) |>
            add(GlobalHomeAdvantage()) |>
            add(PoissonObservation()) |>
            build

        scope = Data.DataScope(
            name = "screen_scope",
            train_tournaments = [56],
            target_tournaments = [56],
            clock_tournaments = [56],
            target_seasons = ["24/25"],
            history_seasons = 1
        )

        candidate = Harness.Candidate(
            name = "synth_screen",
            model = model,
            scope = scope,
            role = :control
        )

        res1 = Harness.screen([candidate]; ds = ds, experiment = "test_screen_exp", db = nothing, bootstrap_B = 100)
        res2 = Harness.screen([candidate]; ds = ds, experiment = "test_screen_exp", db = nothing, bootstrap_B = 100)

        run_id1 = unique(res1.scores.run_id)
        run_id2 = unique(res2.scores.run_id)

        @test length(run_id1) == 1
        @test length(run_id2) == 1
        # Re-screening the same candidate in the same experiment MUST yield the exact same run_id
        @test run_id1[1] == run_id2[1]
    end

    @testset "7. Multi-control preservation (scoring same run against two controls keeps both delta rows)" begin
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
        odds_df = DataFrame(
            match_id = [3, 3, 3, 4, 4, 4],
            market_name = ["1X2", "1X2", "1X2", "1X2", "1X2", "1X2"],
            market_line = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            selection = [:home, :draw, :away, :home, :draw, :away],
            is_winner = [true, false, false, false, false, true]
        )
        betfair_odds = DataFrame(
            match_id = [3, 3, 3, 4, 4, 4],
            market_name = ["1X2", "1X2", "1X2", "1X2", "1X2", "1X2"],
            market_line = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            selection = [:home, :draw, :away, :home, :draw, :away],
            odds = [2.0, 3.2, 3.8, 2.1, 3.1, 3.6],
            traded_price = [2.0, 3.2, 3.8, 2.1, 3.1, 3.6],
            minutes_to_kickoff = [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0],
            tournament_id = [56, 56, 56, 56, 56, 56]
        )
        ds = Data.DataStore(Data.ScottishLower(), matches, empty_df, odds_df,
                            empty_df, empty_df, betfair_odds, empty_df, empty_df)

        model = CountModelBuilder(:synth_multicontrol) |>
            add(GlobalInterception()) |>
            add(TimeDecayDynamics(days_half_life = 180.0)) |>
            add(GlobalHomeAdvantage()) |>
            add(PoissonObservation()) |>
            build

        scope = Data.DataScope(
            name = "mc_scope",
            train_tournaments = [56],
            target_tournaments = [56],
            clock_tournaments = [56],
            target_seasons = ["24/25"],
            history_seasons = 1
        )

        candidate = Harness.Candidate(
            name = "synth_multicontrol",
            model = model,
            scope = scope,
            role = :candidate
        )

        inputs = Harness._fold_inputs(candidate, ds; stage = :screen)
        fit_cfg = Harness.fit_config(candidate; stage = :screen, experiment = "mc_exp")
        fit = Training.fit_model(fit_cfg; feature_sets = inputs.feature_sets, oos_fixtures = inputs.oos, quiet = true)

        ref_c    = Harness.RunRef("candidate_model", "mc_exp", UUID("11111111-1111-1111-1111-111111111111"), :candidate)
        ref_ctl1 = Harness.RunRef("control_1",       "mc_exp", UUID("22222222-2222-2222-2222-222222222222"), :control)
        ref_ctl2 = Harness.RunRef("control_2",       "mc_exp", UUID("33333333-3333-3333-3333-333333333333"), :control)

        tiers = Harness.club_season_tiers(ds)

        scores1 = Harness.score_fits([ref_ctl1 => fit, ref_c => fit]; ds = ds, tiers = tiers, control = ref_ctl1, target_seasons = ["24/25"], bootstrap_B = 100)
        scores2 = Harness.score_fits([ref_ctl2 => fit, ref_c => fit]; ds = ds, tiers = tiers, control = ref_ctl2, target_seasons = ["24/25"], bootstrap_B = 100)

        delta1 = filter(r -> r.run_id == ref_c.run_id && r.metric == "delta_logloss_vs_control", scores1)
        @test all(r -> r.control_run_id == ref_ctl1.run_id, eachrow(delta1))

        delta2 = filter(r -> r.run_id == ref_c.run_id && r.metric == "delta_logloss_vs_control", scores2)
        @test all(r -> r.control_run_id == ref_ctl2.run_id, eachrow(delta2))

        combined = vcat(scores1, scores2)
        deltas = filter(r -> r.run_id == ref_c.run_id && r.subset == "target" && r.market == "all" && r.metric == "delta_logloss_vs_control", combined)
        @test nrow(deltas) == 2
        @test Set(deltas.control_run_id) == Set([ref_ctl1.run_id, ref_ctl2.run_id])
    end

    @testset "Screen CLI test-database opt-in" begin
        script = Module(:CandidatesScriptTest)
        Base.include(script, joinpath(@__DIR__, "..", "scripts", "run_candidates.jl"))
        parsed = script.parse_args(["candidates.jl", "--stage", "screen", "--test-db"])
        @test parsed.test_db
        @test !script.parse_args(["candidates.jl", "--stage", "screen"]).test_db
    end

    @testset "8. Run CSV control groups and panel CLI" begin
        script = Module(:ScoreRunsScriptTest)
        Base.include(script, joinpath(@__DIR__, "..", "scripts", "score_runs.jl"))
        w0_path = joinpath(@__DIR__, "..", "docs", "architecture",
                           "harness_klm_W0_rescore_v12.csv")
        w0_refs = script.load_run_refs(w0_path)
        w0_groups = script.load_run_groups(w0_path)
        @test length(w0_refs) == 14
        @test length(w0_groups) == 2
        @test sum(length(group.refs) for group in w0_groups) == 14
        @test Set(group.control.label for group in w0_groups) ==
              Set(["m12_td", "g2_harness_repro"])
        parsed = script.parse_args(["runs.csv", "--target-seasons", "23/24,24/25,25/26",
                                    "--expected-fixtures", "none"])
        @test parsed.target_seasons == ["23/24", "24/25", "25/26"]
        @test parsed.expected_fixtures === nothing
        @test script.parse_args(["runs.csv"]).expected_fixtures == 710
        mktemp() do path, io
            write(io, "label,experiment,run_id,role,control\n" *
                "td_base,synth,11111111-1111-1111-1111-111111111111,control,\n" *
                "td_arm,synth,22222222-2222-2222-2222-222222222222,candidate,td_base\n" *
                "grw_base,synth,33333333-3333-3333-3333-333333333333,control,\n" *
                "grw_arm,synth,44444444-4444-4444-4444-444444444444,candidate,33333333-3333-3333-3333-333333333333\n")
            flush(io)
            groups = script.load_run_groups(path)
            @test length(groups) == 2
            @test [[ref.label for ref in group.refs] for group in groups] ==
                  [["td_base", "td_arm"], ["grw_base", "grw_arm"]]
            @test [group.control.label for group in groups] == ["td_base", "grw_base"]
            arm = DataFrame(match_id = collect(1:4), selection = fill(:home, 4),
                            family = fill("1X2", 4), ll_model = fill(1.5, 4))
            base_loss = Dict("td_base" => 1.0, "grw_base" => 2.0)
            deltas = [Harness._paired_bootstrap(arm,
                          DataFrame(match_id = collect(1:4), selection = fill(:home, 4),
                                    family = fill("1X2", 4),
                                    ll_model = fill(base_loss[group.control.label], 4));
                          B = 100).delta for group in groups]
            @test deltas == [0.5, -0.5]
        end
    end

    @testset "9. Sequential --only register merges preserve both runs" begin
        first_id = "11111111-1111-1111-1111-111111111111"
        second_id = "22222222-2222-2222-2222-222222222222"
        run_ids = Harness._merge_experiment_run_ids("", first_id)
        run_ids = Harness._merge_experiment_run_ids(run_ids, second_id)
        @test Set(split(run_ids, ',')) == Set([first_id, second_id])
        @test Harness._merge_experiment_run_ids(run_ids, first_id) == run_ids
        # SQL upsert merges JSONB commit metadata by UUID, rather than replacing it.
        first_commit = Dict(first_id => (candidate = "td_base", git_sha = "sha1"))
        second_commit = Dict(second_id => (candidate = "grw_base", git_sha = "sha2"))
        @test length(merge(first_commit, second_commit)) == 2
    end

end
