# ==============================================================================
# r01 — SlateDrawdown lambda sweep on GRW baseline and smile-spine posteriors
# ==============================================================================
#
# WHAT THIS ANSWERS
#
# The GRW portfolios compound strongly at SlateDrawdown(8) but lose roughly 43% peak to
# trough. This runner changes the sole active tail-risk dial, lambda, and maps realised
# exposure, terminal growth, annual Sharpe, and max drawdown at the close and at T−25.
#
# WHAT IS HELD FIXED
#
# Immutable posterior runs; Scottish Lower 24/25 + 25/26 fold-held-out fixtures; the exact
# operational Option-B trust vector, pricing, 30% Kelly shrinkage, execution cost, 25% cap and
# daily grouping. Ticket T012 is handled by market-level `:excise_pruned`: only markets with an
# active Option-B direction enter the payoff matrix. No MCMC is launched.
#
# ENVIRONMENTS
#
#   close/raw       de-vigged Betfair TWA(-20, 0] close
#   t25/raw         tradeable point-in-time T−25 book
#   t25/l2_calibrated  InverseGaussianLaw(0.25, 0.35), with phi dropped at pricing time
#
# Smile books use Task 016's anti-diagonal reweighted score-grid route (Ticket T011). The L2
# smile control is deliberately CountLatents after calibration, matching Task 016's validated
# `t25_inv_grid` definition.
#
# OUTPUTS
#
# current_development/grw_risk_sweep/results/
#   lambda_sweep_summary.csv, pareto_frontier.csv, overshoot_calibration.csv,
#   LAMBDA_RISK_SWEEP_REPORT.md
#
# USAGE (mcmc-beast; no sampling)
#
#   julia --project -t 16 current_development/grw_risk_sweep/r01_lambda_sweep.jl
# ==============================================================================

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball
using DataFrames
using Printf
using UUIDs

# Defines the prototype model type needed to deserialize/rebuild the detached SmileLatents
# panel, plus the T011-correct score-grid builder.
include(joinpath(@__DIR__, "..", "grw_smile_spine", "l02_evaluation.jl"))
include(joinpath(@__DIR__, "l01_risk_sweep.jl"))
using .GRWRiskSweep

const R01_DATA = BayesianFootball.Data
const R01_MODELS = BayesianFootball.Models
const R01_PORTFOLIO = BayesianFootball.Portfolio
const R01_CAL = BayesianFootball.Calibration

# %%
# ===================================================================
# 2. Immutable configuration
# ===================================================================
const R01_CONFIG = RiskSweepConfig(
    lambdas = [8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 23.0, 28.0, 35.0, 45.0],
    beta = 0.01,
    target_drawdown_pct = 20.0,
    initial_bankroll = 1_000.0,
    t012_mode = :excise_pruned,
)
const R01_TARGET_SEASONS = ["24/25", "25/26"]
const R01_EXPECTED_LATENT_PANEL = 710
const R01_EXPECTED_CLOSE_PANEL = 628
const R01_OUTPUT_DIR = joinpath(@__DIR__, "results")
const R01_LATENT_DIR = joinpath(@__DIR__, "..", "grw_smile_spine", "results", "latents")

const R01_REFERENCES = [
    RunReference(
        "m05_joint_grw_smile_spine_w040",
        "m05_joint_grw_smile_spine_w040",
        "scottish_lower_grw_smile_spine",
        UUID("582035c0-e145-44f7-9f40-89e25388e79a"),
        :smile,
    ),
    RunReference(
        "m05_joint_grw_baseline",
        "m05_wealth_grw",
        "scottish_lower_grw_player_hybrid",
        UUID("b0961bc4-c40c-4dbe-9c05-57df7ae0839e"),
        :count,
    ),
]

function r01_git_label()
    commit = get(ENV, "GRW_SWEEP_GIT_COMMIT") do
        try
            readchomp(pipeline(`git rev-parse --short HEAD`; stderr = devnull))
        catch
            "unknown"
        end
    end
    dirty = if haskey(ENV, "GRW_SWEEP_DIRTY")
        lowercase(ENV["GRW_SWEEP_DIRTY"]) in ("1", "true", "yes")
    else
        try
            !isempty(readchomp(pipeline(`git status --porcelain`; stderr = devnull)))
        catch
            true  # rsynced execution tree has no .git metadata, so it cannot claim clean provenance
        end
    end
    return dirty && !endswith(commit, "-dirty") ? commit * "-dirty" : commit
end

const R01_GIT = r01_git_label()

# Task 018's exact :excise_pruned close result on the 628-fixture panel.
const R01_SPINE_LAMBDA8_REFERENCE = (
    n_panel = 628,
    n_bets = 1188,
    terminal_return_pct = 576.1331019902052,
    max_drawdown_pct = -42.81333765155565,
)

# %%
# ===================================================================
# 3. Loading helpers and panel contract
# ===================================================================
function r01_target_panel(ds, sources)
    season_of = Dict(Int(row.match_id) => String(row.season) for row in eachrow(ds.matches))
    panels = [Set(Int(match_id) for match_id in R01_MODELS.latent_match_ids(latents)
                  if get(season_of, Int(match_id), "") in R01_TARGET_SEASONS)
              for latents in values(sources)]
    panel = sort!(collect(reduce(intersect, panels)))
    length(panel) == R01_EXPECTED_LATENT_PANEL || error(
        "latent intersection has $(length(panel)) fixtures; expected $R01_EXPECTED_LATENT_PANEL")
    return panel
end

function r01_load_sources(ds)
    inventory = DataFrame(verified_run_inventory.(R01_REFERENCES))
    sources = Dict{String,Any}()

    baseline_ref = only(filter(ref -> ref.latent_kind === :count, R01_REFERENCES))
    sources[baseline_ref.model] = load_run_latents(baseline_ref, ds)

    spine_ref = only(filter(ref -> ref.latent_kind === :smile, R01_REFERENCES))
    sources[spine_ref.model] = load_run_latents(spine_ref, ds; smile_loader = (ref, store) -> begin
        splitter = gph_splitter(gss_config().extension_seasons)
        arm = GMSArm(ref.model, ref.experiment, ref.run_id, "candidate")
        fit = gms_load_arm(arm, store; splitter, latent_dir = R01_LATENT_DIR)
        fit.latents
    end)

    for row in eachrow(inventory)
        source = sources[String(row.model)]
        row.latent_kind = string(nameof(typeof(source)))
    end
    return sources, inventory
end

function r01_build_books(spec, source, odds, ds;
                         route::Symbol, audited_converged::Bool)
    audited_converged || error("refusing to build books from an unaudited latent source")
    route in (:native_smile, :native_count) || error("unknown book route $route")
    route === :native_smile && !(source isa R01_MODELS.SmileLatents) && error(
        "native_smile route requires SmileLatents, got $(typeof(source))")
    return R01_PORTFOLIO.build_books_reported(
        spec, source, odds, ds; converged = audited_converged,
        gated = audited_converged, quiet = true)
end

# SmileScoreGrid now performs the anti-diagonal reweighting inside the production zero-allocation
# kernel. Task 016's standalone gss builder predates that graduation and must not be layered on top.
r01_route(source) = source isa R01_MODELS.SmileLatents ? :native_smile : :native_count

function r01_buildable_panel(spec, sources, audited, odds, ds, target_panel)
    buildable = Set(target_panel)
    details = NamedTuple[]
    for (model, source) in sort!(collect(sources); by = first)
        restricted = gms_restrict_latents(source, target_panel)
        books, report = r01_build_books(
            spec, restricted, odds, ds; route = r01_route(restricted),
            audited_converged = audited[model])
        isempty(report.errored) || error("$model book errors: $(report.errored)")
        ids = Set(Int(book.m_id) for book in books)
        intersect!(buildable, ids)
        push!(details, (model, n_built = length(ids), n_skipped = R01_PORTFOLIO.n_skipped(report)))
    end
    panel = sort!(collect(buildable))
    isempty(panel) && error("environment has no commonly buildable fixtures")
    return panel, DataFrame(details)
end

function r01_case(model, environment, variant, run_id, spec, source, odds, ds, panel;
                  audited_converged::Bool)
    restricted = gms_restrict_latents(source, panel)
    route = r01_route(restricted)
    books, report = r01_build_books(
        spec, restricted, odds, ds; route, audited_converged)
    R01_PORTFOLIO.n_skipped(report) == 0 || error(
        "$model/$environment/$variant skipped $(R01_PORTFOLIO.n_skipped(report)) frozen-panel fixtures")
    length(books) == length(panel) || error(
        "$model/$environment/$variant built $(length(books)) books for $(length(panel)) fixtures")
    return SweepCase(model, environment, variant, String(route), string(run_id),
                     length(panel), books, report)
end

"Task-016 L2 control: calibrate home/away draws and discard the fitted smile phi at pricing."
function r01_calibrate_drop_phi(calibrator, source, rates)
    count_source = if source isa R01_MODELS.SmileLatents
        R01_MODELS.CountLatents(source.match_ids, source.λ_home, source.λ_away, nothing)
    elseif source isa R01_MODELS.CountLatents
        source
    else
        error("unsupported latent family $(typeof(source))")
    end
    calibrated, diagnostics = R01_CAL.calibrate_latents(calibrator, count_source, rates)
    calibrated isa R01_MODELS.CountLatents || error(
        "calibration returned $(typeof(calibrated)); expected CountLatents")
    return calibrated, diagnostics
end

# %%
# ===================================================================
# 4. Zero-allocation pricing-kernel gate
# ===================================================================
function r01_pricing_allocation_row(label, spec, source)
    workspace = R01_PORTFOLIO.BookWorkspace(spec, source; quiet = true)
    R01_PORTFOLIO.price_fixture!(workspace, source, 1)
    bytes = @allocated R01_PORTFOLIO.price_fixture!(workspace, source, 1)
    latent_type = source isa R01_MODELS.SmileLatents ? "SmileLatents" : "CountLatents"
    return (case = String(label), latent_type,
            kernel = "Portfolio.price_fixture!", allocated_bytes = bytes,
            pass = bytes == 0)
end

# %%
# ===================================================================
# 5. End-to-end execution
# ===================================================================
function r01_execute()
    mkpath(R01_OUTPUT_DIR)
    println("\n", "="^106)
    println("  TASK 020 — GRW SLATEDRAWDOWN LAMBDA SWEEP")
    println("  lambda: ", join(R01_CONFIG.lambdas, ", "))
    println("  mode: ", R01_CONFIG.t012_mode,
            "  host: ", gethostname(), "  threads: ", Threads.nthreads(), "  git: ", R01_GIT)
    println("="^106)

    println("\n[1/9] Loading cached Scottish Lower data and immutable posterior panels...")
    ds = R01_DATA.load_datastore_cached(R01_DATA.ScottishLower(); max_age_hours = 100_000)
    sources, inventory = r01_load_sources(ds)
    audited = Dict(String(row.model) => Int(row.converged_folds) == Int(row.n_folds)
                   for row in eachrow(inventory))
    all(values(audited)) || error("at least one immutable run failed its convergence audit")
    target_panel = r01_target_panel(ds, sources)
    for row in eachrow(inventory)
        @printf("  %-36s folds %d/%d  %-13s  %s\n",
                row.model, row.converged_folds, row.n_folds, row.latent_kind, row.run_id)
    end

    println("\n[2/9] Freezing the T012-correct Option-B book and environment panels...")
    system = excised_option_b_system()
    spec = system.book
    base_policy = system.policy
    length(spec.markets.markets) == 3 || error(
        "excised Option-B book has $(length(spec.markets.markets)) markets; expected 3")

    close_odds = gms_betfair_closing_odds(ds)
    t25_odds, t25_refusals = R01_CAL.point_in_time_book(
        ds; config = R01_CAL.PointInTimeBookConfig(as_of_minutes = -25.0))
    t25_book_instant_pass = try
        R01_CAL.assert_book_as_of(t25_odds, -25.0)
        true
    catch err
        @error "T−25 book instant gate failed" exception = (err, catch_backtrace())
        false
    end
    t25_book_instant_pass || error("T−25 book does not carry the required -25 minute instant")

    close_panel, close_detail = r01_buildable_panel(
        spec, sources, audited, close_odds, ds, target_panel)
    t25_panel, t25_detail = r01_buildable_panel(
        spec, sources, audited, t25_odds, ds, target_panel)
    length(close_panel) == R01_EXPECTED_CLOSE_PANEL || error(
        "close panel has $(length(close_panel)) fixtures; expected $R01_EXPECTED_CLOSE_PANEL")

    target_set = Set(target_panel)
    panels = DataFrame([
        (environment = "close", variant = "raw",
         n_walk_forward = length(target_panel),
         n_quoted = length(intersect(target_set, Set(Int.(close_odds.match_id)))),
         n_buildable = length(close_panel), n_excluded = length(target_panel) - length(close_panel)),
        (environment = "t25", variant = "raw_and_l2_calibrated",
         n_walk_forward = length(target_panel),
         n_quoted = length(intersect(target_set, Set(Int.(t25_odds.match_id)))),
         n_buildable = length(t25_panel), n_excluded = length(target_panel) - length(t25_panel)),
    ])
    println("  close: ", length(close_panel), " buildable (model detail ", close_detail, ")")
    println("  T−25 : ", length(t25_panel), " buildable (model detail ", t25_detail, ")")
    println("  T−25 refusals from book construction: ", nrow(t25_refusals))

    println("\n[3/9] Inverting T−25 and constructing phi-dropped L2 sources...")
    calibrator = R01_CAL.GenerativeRateCalibrator(
        name = "scot_lower_t25_inv",
        law = R01_CAL.InverseGaussianLaw(w_base = 0.25, sigma = 0.35),
        book_as_of_minutes = -25.0,
    )
    rates = R01_CAL.invert_market_rates(calibrator, t25_odds; match_ids = t25_panel)
    coverage = R01_CAL.inversion_coverage(rates, t25_panel)
    coverage.n_accepted > 0 || error("T−25 inversion accepted no frozen fixtures")
    @printf("  inversion: %d/%d accepted; %d use the calibrator's documented identity fallback\n",
            coverage.n_accepted, length(t25_panel), length(t25_panel) - coverage.n_accepted)

    calibrated = Dict{String,Any}()
    calibration_rows = NamedTuple[]
    for ref in R01_REFERENCES
        raw = gms_restrict_latents(sources[ref.model], t25_panel)
        shifted, diagnostics = r01_calibrate_drop_phi(calibrator, raw, rates)
        calibrated[ref.model] = shifted
        n_shifted = count(diagnostics.inverted)
        push!(calibration_rows, (
            model = ref.model,
            n_panel = length(t25_panel),
            n_inverted = n_shifted,
            coverage_pct = 100.0 * n_shifted / length(t25_panel),
            book_as_of_minutes = calibrator.book_as_of_minutes,
            law = R01_CAL.law_label(calibrator.law),
            phi_pricing = "dropped",
            identity_fallback = length(t25_panel) - n_shifted,
            pass = n_shifted == coverage.n_accepted,
        ))
    end
    calibration_coverage = DataFrame(calibration_rows)

    println("\n[4/9] Building each model/environment book once...")
    cases = SweepCase[]
    for ref in R01_REFERENCES
        push!(cases,
            r01_case(ref.model, "close", "raw", ref.run_id, spec,
                     sources[ref.model], close_odds, ds, close_panel;
                     audited_converged = audited[ref.model]),
            r01_case(ref.model, "t25", "raw", ref.run_id, spec,
                     sources[ref.model], t25_odds, ds, t25_panel;
                     audited_converged = audited[ref.model]),
            r01_case(ref.model, "t25", "l2_calibrated", ref.run_id, spec,
                     calibrated[ref.model], t25_odds, ds, t25_panel;
                     audited_converged = audited[ref.model]),
        )
    end
    @printf("  %d cases; %d lambda cells\n", length(cases),
            length(cases) * length(R01_CONFIG.lambdas))

    println("\n[5/9] Gate A — steady-state pricing kernels allocate zero heap bytes...")
    allocation_gates = DataFrame([
        r01_pricing_allocation_row("baseline/raw", spec,
            gms_restrict_latents(sources["m05_joint_grw_baseline"], close_panel)),
        r01_pricing_allocation_row("spine/raw-reweighted", spec,
            gms_restrict_latents(sources["m05_joint_grw_smile_spine_w040"], close_panel)),
        r01_pricing_allocation_row("baseline/t25-l2", spec,
            calibrated["m05_joint_grw_baseline"]),
        r01_pricing_allocation_row("spine/t25-l2-phi-dropped", spec,
            calibrated["m05_joint_grw_smile_spine_w040"]),
    ])
    all(allocation_gates.pass) || error("zero-allocation pricing gate failed")
    show(allocation_gates; allrows = true, allcols = true)
    println()

    println("\n[6/9] Running the complete sequential reference sweep...")
    t_sequential = @elapsed sequential = run_lambda_sweep(
        cases, base_policy, R01_CONFIG; threaded = false)
    @printf("  sequential %.2f s\n", t_sequential)

    println("\n[7/9] Running the complete threaded sweep and exact comparison...")
    t_threaded = @elapsed threaded = run_lambda_sweep(
        cases, base_policy, R01_CONFIG; threaded = true)
    exact = verify_exact_reproduction(sequential, threaded)
    @printf("  threaded %.2f s; %d/%d cells bit-identical\n",
            t_threaded, exact.n_cells, exact.n_cells)

    summary = threaded.summary
    expected_rows = length(R01_REFERENCES) * 3 * length(R01_CONFIG.lambdas)
    nrow(summary) == expected_rows || error(
        "summary has $(nrow(summary)) rows; expected $expected_rows")

    println("\n[8/9] Gates B–D — reproduction, exposure monotonicity, and completeness...")
    spine8 = only(filter(row -> row.model == "m05_joint_grw_smile_spine_w040" &&
                               row.environment == "close" && row.variant == "raw" &&
                               row.lambda == 8.0, eachrow(summary)))
    reproduction_pass = spine8.n_panel == R01_SPINE_LAMBDA8_REFERENCE.n_panel &&
        spine8.n_bets == R01_SPINE_LAMBDA8_REFERENCE.n_bets &&
        abs(spine8.terminal_return_pct -
            R01_SPINE_LAMBDA8_REFERENCE.terminal_return_pct) <= 1.0e-9 &&
        abs(spine8.max_drawdown_pct -
            R01_SPINE_LAMBDA8_REFERENCE.max_drawdown_pct) <= 1.0e-9
    reproduction_pass || error(
        "lambda=8 spine close row did not reproduce Task 018's excised Option-B reference")

    monotone_pass = true
    worst_exposure_increase = 0.0
    for group in groupby(summary, [:model, :environment, :variant])
        ordered = sort(DataFrame(group), :lambda)
        increases = diff(ordered.mean_slate_exposure)
        local_worst = isempty(increases) ? 0.0 : maximum(increases)
        worst_exposure_increase = max(worst_exposure_increase, local_worst)
        monotone_pass &= local_worst <= eps(Float64) * 64
    end
    monotone_pass || error(
        "mean exposure increased with lambda; worst increase $worst_exposure_increase")

    gates = DataFrame([
        (gate = "immutable runs completed and converged", pass = all(values(audited)),
         detail = "$(sum(inventory.converged_folds))/$(sum(inventory.n_folds)) folds"),
        (gate = "latent panel", pass = length(target_panel) == R01_EXPECTED_LATENT_PANEL,
         detail = "$(length(target_panel))/$R01_EXPECTED_LATENT_PANEL fixtures"),
        (gate = "close common buildable panel", pass = length(close_panel) == R01_EXPECTED_CLOSE_PANEL,
         detail = "$(length(close_panel))/$R01_EXPECTED_CLOSE_PANEL fixtures"),
        (gate = "T012 market-level excision", pass = length(spec.markets.markets) == 3,
         detail = "3 active markets; all-zero markets absent"),
        (gate = "T−25 book instant", pass = t25_book_instant_pass,
         detail = "assert_book_as_of(-25.0)"),
        (gate = "T−25 inversion and identity fallback accounted", pass =
             all(calibration_coverage.pass),
         detail = "$(coverage.n_accepted)/$(length(t25_panel)) shifted; " *
                  "$(length(t25_panel) - coverage.n_accepted) documented identity fallbacks"),
        (gate = "zero-allocation pricing", pass = all(allocation_gates.pass),
         detail = "max $(maximum(allocation_gates.allocated_bytes)) bytes"),
        (gate = "lambda=8 published excised reproduction", pass = reproduction_pass,
         detail = @sprintf("spine close %.12f%%, %d bets, MDD %.12f%%",
                           spine8.terminal_return_pct, spine8.n_bets,
                           spine8.max_drawdown_pct)),
        (gate = "sequential/threaded bit identity", pass = exact.pass,
         detail = "$(exact.n_cells) canonical result payloads and summary rows"),
        (gate = "mean exposure non-increasing in lambda", pass = monotone_pass,
         detail = @sprintf("worst adjacent increase %.3e", worst_exposure_increase)),
        (gate = "grid completeness", pass = nrow(summary) == expected_rows,
         detail = "$(nrow(summary))/$expected_rows cells"),
    ])
    all(gates.pass) || error("at least one verification gate failed")

    println("\n[9/9] Computing Pareto sets, target-risk choices, and writing artifacts...")
    marked = mark_pareto(summary)
    # The deliverable called `pareto_frontier.csv` is the operational two-objective frontier.
    # The four-objective diagnostic is retained as a flag on these rows, not used to refill the
    # file with the whole return-vs-risk trade-off grid.
    frontier = marked[marked.pareto_drawdown_sharpe, :]
    sort!(frontier, [:environment, :variant, :model, :lambda])
    overshoot = overshoot_frame(summary)
    optima = target_drawdown_optima(summary, R01_CONFIG.target_drawdown_pct)

    report = write_sweep_outputs(
        R01_OUTPUT_DIR;
        summary,
        frontier,
        overshoot,
        optima,
        panels,
        inventory,
        gates,
        allocation_gates,
        calibration_coverage,
        git_commit = R01_GIT,
        host = gethostname(),
        n_threads = Threads.nthreads(),
    )

    println("\n  Target-drawdown choices:")
    show(optima; allrows = true, allcols = true)
    println("\n\n  wrote ", join([
        "lambda_sweep_summary.csv",
        "pareto_frontier.csv",
        "overshoot_calibration.csv",
        basename(report),
    ], ", "))
    return (; summary, frontier, overshoot, optima, panels, inventory, gates,
            allocation_gates, calibration_coverage, sequential_seconds = t_sequential,
            threaded_seconds = t_threaded)
end

# %%
# ===================================================================
# 6. Final report
# ===================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    R01_RESULT = r01_execute()
    println("\nR01_GRW_LAMBDA_SWEEP_DONE")
end
