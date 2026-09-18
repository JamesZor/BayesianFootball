module GRWRiskSweep

# Reusable Task-020 machinery. Definitions only: r01_lambda_sweep.jl owns data access,
# immutable run selection, posterior reconstruction, execution, and scientific interpretation.

import BayesianFootball
import CSV
import DataFrames
import Dates
import LibPQ
import SHA
import Serialization
import Statistics
import UUIDs

const BF = BayesianFootball
const Data = BF.Data
const Models = BF.Models
const Portfolio = BF.Portfolio

export RiskSweepConfig, RunReference, SweepCase,
       excised_option_b_system, policy_at_lambda,
       verified_run_inventory, load_relational_count_latents, load_run_latents,
       run_lambda_sweep, exact_result_digest, verify_exact_reproduction,
       mark_pareto, target_drawdown_optima, overshoot_frame,
       write_sweep_outputs, markdown_table

Base.@kwdef struct RiskSweepConfig
    lambdas::Vector{Float64} = [8.0, 10.0, 12.0, 15.0, 18.0,
                                20.0, 23.0, 28.0, 35.0, 45.0]
    beta::Float64 = 0.01
    target_drawdown_pct::Float64 = 20.0
    initial_bankroll::Float64 = 1_000.0
    t012_mode::Symbol = :excise_pruned
end

struct RunReference
    model::String
    persisted_run_name::String
    experiment::String
    run_id::UUIDs.UUID
    latent_kind::Symbol
end

"Prebuilt books for one model × market environment. A lambda sweep never rebuilds them."
struct SweepCase{B,R}
    model::String
    environment::String
    variant::String
    route::String
    run_id::String
    n_panel::Int
    books::B
    report::R
end

"""
    excised_option_b_system() -> PortfolioSystem

The exact operational Option-B trust vector, with all-zero markets removed from the BookSpec.
The resulting priced markets are 1X2, O/U 1.5 (Over active), and O/U 2.5 (Under active).
This is the market-level meaning of `:excise_pruned` established by Ticket T012: a two-sided
market remains whole whenever either direction has positive trust.
"""
function excised_option_b_system()
    base = BF.MatchDay.option_b_system()
    markets = Data.MarketConfig(Data.Markets.AbstractMarket[
        Data.Market1X2(),
        Data.MarketOverUnder(1.5),
        Data.MarketOverUnder(2.5),
    ])
    book = Portfolio.BookSpec(
        markets = markets,
        price = base.book.price,
        allocator = base.book.allocator,
        shrink = base.book.shrink,
        exec = base.book.exec,
    )
    return Portfolio.PortfolioSystem(book, base.policy)
end

"Clone a policy while changing only SlateDrawdown.lambda."
function policy_at_lambda(base::Portfolio.PolicySpec, lambda::Real)
    risk = base.risk
    risk isa Portfolio.SlateDrawdown || error(
        "risk sweep requires SlateDrawdown, got $(typeof(risk))")
    lambda > 0 || error("lambda must be positive, got $lambda")
    return Portfolio.PolicySpec(
        trust = base.trust,
        risk = Portfolio.SlateDrawdown(
            lambda = Float64(lambda),
            mode = risk.mode,
            joint_draws = risk.joint_draws,
            seed = risk.seed,
        ),
        cap = base.cap,
        filter = base.filter,
        grouping = base.grouping,
    )
end

"Verify one immutable run address and all persisted fold convergence flags."
function verified_run_inventory(ref::RunReference)
    storage = BF.Training.PostgresStorage(ref.experiment)
    conn = BF.Training.Inference._db_connect(storage)
    try
        query = LibPQ.execute(conn, """
            SELECT r.experiment_name, r.name, r.status,
                   COUNT(fr.*) AS n_folds,
                   COUNT(*) FILTER (WHERE fr.converged) AS converged_folds
            FROM runs r
            LEFT JOIN fold_results fr ON fr.run_id = r.run_id
            WHERE r.run_id = \$1::uuid
            GROUP BY r.experiment_name, r.name, r.status;
        """, (string(ref.run_id),))
        frame = DataFrames.DataFrame(query)
        close(query)
        DataFrames.nrow(frame) == 1 || error("run $(ref.run_id) did not resolve exactly once")
        row = frame[1, :]
        String(row.experiment_name) == ref.experiment || error(
            "run $(ref.run_id) resolved in $(row.experiment_name), expected $(ref.experiment)")
        String(row.name) == ref.persisted_run_name || error(
            "run $(ref.run_id) is named $(row.name), expected $(ref.persisted_run_name)")
        String(row.status) == "completed" || error("$(ref.model) status is $(row.status)")
        n_folds = Int(row.n_folds)
        converged_folds = Int(row.converged_folds)
        n_folds > 0 && converged_folds == n_folds || error(
            "$(ref.model) converged $converged_folds/$n_folds folds")
        return (model = ref.model, persisted_run_name = ref.persisted_run_name,
                experiment = ref.experiment, run_id = string(ref.run_id),
                n_folds, converged_folds, latent_kind = String(ref.latent_kind))
    finally
        close(conn)
    end
end

"""
    load_relational_count_latents(ref) -> CountLatents

Recover a legacy run's exact relational latent panel without deserialising its obsolete model
blob. This is the same reconstruction helper used by `load_fit` after artefact loading.
"""
function load_relational_count_latents(ref::RunReference)
    storage = BF.Training.PostgresStorage(ref.experiment)
    conn = BF.Training.Inference._db_connect(storage)
    try
        latents = BF.Training.Inference._db_load_count_latents(conn, ref.run_id)
        latents === nothing && error("$(ref.model) has no relational match_latents")
        latents isa Models.CountLatents || error(
            "$(ref.model) relational panel is $(typeof(latents)), expected CountLatents")
        return latents
    finally
        close(conn)
    end
end

"""
    load_run_latents(ref, ds; smile_loader) -> AbstractPosteriorLatents

Resolve the latent family declared by an immutable run reference. Historical CountLatents use
relational reconstruction. SmileLatents require the caller's prototype-aware reconstruction
callback because the persisted fit names a model type defined outside `src`; the returned family
is validated here so a detached or wrong-family panel cannot enter the sweep.
"""
function load_run_latents(ref::RunReference, ds; smile_loader = nothing)
    if ref.latent_kind === :count
        return load_relational_count_latents(ref)
    elseif ref.latent_kind === :smile
        smile_loader === nothing && error(
            "$(ref.model) requires a prototype-aware SmileLatents reconstruction callback")
        latents = smile_loader(ref, ds)
        latents isa Models.SmileLatents || error(
            "$(ref.model) reconstruction returned $(typeof(latents)); expected SmileLatents")
        return latents
    end
    error("$(ref.model) declares unknown latent kind $(ref.latent_kind)")
end

function _summary_row(case::SweepCase, lambda::Float64, beta::Float64,
                      t012_mode::Symbol, result)
    summary = result.summary
    nominal_floor = exp(log(beta) / lambda)
    nominal_drawdown_pct = 100.0 * (1.0 - nominal_floor)
    realised_drawdown_pct = abs(summary.mdd)
    return (
        model = case.model,
        environment = case.environment,
        variant = case.variant,
        route = case.route,
        run_id = case.run_id,
        t012_mode = String(t012_mode),
        lambda = lambda,
        beta = beta,
        nominal_floor = nominal_floor,
        nominal_drawdown_pct = nominal_drawdown_pct,
        n_panel = case.n_panel,
        n_slates = summary.n_slates,
        n_bets = summary.n_bets,
        terminal_return_pct = summary.total_return_pct,
        cagr_pct = 100.0 * summary.cagr,
        roi_pct = summary.roi,
        sharpe_ann = summary.sharpe_ann,
        sortino = summary.sortino,
        calmar = summary.calmar,
        max_drawdown_pct = summary.mdd,
        realised_drawdown_pct = realised_drawdown_pct,
        win_rate_pct = 100.0 * summary.win_rate,
        total_turnover = summary.total_stake,
        mean_slate_exposure = summary.mean_exposure,
        max_slate_exposure = summary.max_exposure,
        mean_k_risk = summary.mean_k_risk,
        n_capped = summary.n_capped,
        overshoot_ratio = nominal_drawdown_pct > 0.0 ?
                          realised_drawdown_pct / nominal_drawdown_pct : NaN,
    )
end

function _sweep_jobs(cases, lambdas)
    return [(case_index = case_index, lambda = Float64(lambda))
            for case_index in eachindex(cases) for lambda in lambdas]
end

function _run_job(case::SweepCase, base_policy, config::RiskSweepConfig, lambda::Float64)
    policy = policy_at_lambda(base_policy, lambda)
    result = Portfolio.simulate_portfolio(
        policy,
        case.books,
        case.report;
        initial_bankroll = config.initial_bankroll,
        bootstrap = false,
    )
    return _summary_row(case, lambda, config.beta, config.t012_mode, result), result
end

"""
    run_lambda_sweep(cases, base_policy, config; threaded)

Evaluate every case × lambda cell. Books are immutable and shared; only the deterministic staking
solve is repeated. Threaded writes target preallocated indices and therefore has no shared push!.
"""
function run_lambda_sweep(cases::AbstractVector{<:SweepCase},
                          base_policy::Portfolio.PolicySpec,
                          config::RiskSweepConfig;
                          threaded::Bool)
    config.t012_mode === :excise_pruned || error(
        "Task 020 requires :excise_pruned, got $(config.t012_mode)")
    isempty(config.lambdas) && error("lambda grid is empty")
    length(unique(config.lambdas)) == length(config.lambdas) || error(
        "lambda grid contains duplicates")
    all(>(0.0), config.lambdas) || error("every lambda must be positive")

    jobs = _sweep_jobs(cases, config.lambdas)
    rows = Vector{Any}(undef, length(jobs))
    results = Vector{Any}(undef, length(jobs))

    if threaded
        Threads.@threads for index in eachindex(jobs)
            job = jobs[index]
            rows[index], results[index] = _run_job(
                cases[job.case_index], base_policy, config, job.lambda)
        end
    else
        for index in eachindex(jobs)
            job = jobs[index]
            rows[index], results[index] = _run_job(
                cases[job.case_index], base_policy, config, job.lambda)
        end
    end

    frame = DataFrames.DataFrame(rows)
    permutation = sortperm(frame, [:environment, :variant, :model, :lambda])
    return (summary = frame[permutation, :],
            results = results[permutation], jobs = jobs[permutation])
end

_struct_payload(value) = NamedTuple{fieldnames(typeof(value))}(
    getfield(value, name) for name in fieldnames(typeof(value)))

function _frame_payload(frame::DataFrames.AbstractDataFrame)
    names_ = Tuple(Symbol.(DataFrames.names(frame)))
    columns = Tuple(Tuple(frame[!, name]) for name in DataFrames.names(frame))
    return (names = names_, columns)
end

"Canonical field payload for exact comparisons; excludes object-identity/backreference layout."
function _result_payload(result)
    trajectory = result.trajectory
    trajectory_payload = (
        bankroll = Tuple(trajectory.bankroll),
        dates = Tuple(trajectory.dates),
        slate_pl = Tuple(trajectory.slate_pl),
        k_risk = Tuple(trajectory.k_risk),
        exposure = Tuple(trajectory.exposure),
        n_capped = trajectory.n_capped,
        total_stake = trajectory.total_stake,
        total_pl = trajectory.total_pl,
        bets = _frame_payload(trajectory.bets),
    )
    return (
        daily_states = Tuple(_struct_payload(state) for state in result.daily_states),
        summary = _struct_payload(result.summary),
        metrics = result.metrics,
        bootstrap_ci = result.bootstrap_ci,
        trajectory = trajectory_payload,
        attribution = _frame_payload(result.attribution),
        converged = result.converged,
        failed_gates = Tuple(result.failed_gates),
    )
end

"SHA-256 of every scientific field in a PortfolioResult, in canonical column/value order."
function exact_result_digest(result)
    io = IOBuffer()
    Serialization.serialize(io, _result_payload(result))
    return bytes2hex(SHA.sha256(take!(io)))
end

"Require identical rows and every canonical PortfolioResult field across execution paths."
function verify_exact_reproduction(sequential, threaded)
    isequal(sequential.summary, threaded.summary) || error(
        "sequential and threaded summary frames are not bit-identical")
    length(sequential.results) == length(threaded.results) || error(
        "sequential and threaded result counts differ")
    seq_payload = _result_payload.(sequential.results)
    threaded_payload = _result_payload.(threaded.results)
    isequal(seq_payload, threaded_payload) || begin
        bad = [i for i in eachindex(seq_payload)
               if !isequal(seq_payload[i], threaded_payload[i])]
        error("sequential and threaded PortfolioResult fields differ at job indices $(bad)")
    end
    seq_hash = exact_result_digest.(sequential.results)
    threaded_hash = exact_result_digest.(threaded.results)
    seq_hash == threaded_hash || error(
        "canonical result payloads compare equal but their SHA-256 digests differ")
    return (pass = true, n_cells = length(seq_hash), digests = seq_hash)
end

function _dominates(frame, j::Int, i::Int; all_metrics::Bool)
    sharpe_better = frame.sharpe_ann[j] >= frame.sharpe_ann[i]
    drawdown_better = frame.realised_drawdown_pct[j] <= frame.realised_drawdown_pct[i]
    if !all_metrics
        strict = frame.sharpe_ann[j] > frame.sharpe_ann[i] ||
                 frame.realised_drawdown_pct[j] < frame.realised_drawdown_pct[i]
        return sharpe_better && drawdown_better && strict
    end
    return_better = frame.terminal_return_pct[j] >= frame.terminal_return_pct[i]
    exposure_better = frame.mean_slate_exposure[j] <= frame.mean_slate_exposure[i]
    strict = frame.sharpe_ann[j] > frame.sharpe_ann[i] ||
             frame.realised_drawdown_pct[j] < frame.realised_drawdown_pct[i] ||
             frame.terminal_return_pct[j] > frame.terminal_return_pct[i] ||
             frame.mean_slate_exposure[j] < frame.mean_slate_exposure[i]
    return sharpe_better && drawdown_better && return_better && exposure_better && strict
end

"Mark drawdown-vs-Sharpe and four-metric non-dominated cells within each model/environment."
function mark_pareto(summary::DataFrames.AbstractDataFrame)
    frame = DataFrames.DataFrame(summary)
    frame.pareto_drawdown_sharpe = falses(DataFrames.nrow(frame))
    frame.pareto_all_metrics = falses(DataFrames.nrow(frame))
    groups = DataFrames.groupby(frame, [:model, :environment, :variant])
    labels = DataFrames.groupindices(groups)
    for label in unique(labels)
        rows = findall(==(label), labels)
        for i in rows
            finite_two = isfinite(frame.sharpe_ann[i]) &&
                         isfinite(frame.realised_drawdown_pct[i])
            finite_four = finite_two && isfinite(frame.terminal_return_pct[i]) &&
                          isfinite(frame.mean_slate_exposure[i])
            frame.pareto_drawdown_sharpe[i] = finite_two && !any(
                j != i && isfinite(frame.sharpe_ann[j]) &&
                isfinite(frame.realised_drawdown_pct[j]) &&
                _dominates(frame, j, i; all_metrics = false) for j in rows)
            # Return falls as tighter risk reduces drawdown/exposure, so this four-objective
            # frontier can legitimately retain the whole grid. It is diagnostic only; the
            # exported operational frontier is drawdown-vs-Sharpe.
            frame.pareto_all_metrics[i] = finite_four && !any(
                j != i && isfinite(frame.sharpe_ann[j]) &&
                isfinite(frame.realised_drawdown_pct[j]) &&
                isfinite(frame.terminal_return_pct[j]) &&
                isfinite(frame.mean_slate_exposure[j]) &&
                _dominates(frame, j, i; all_metrics = true) for j in rows)
        end
    end
    return frame
end

"Highest-Sharpe cell at or below the realised drawdown target, one per model/environment."
function target_drawdown_optima(summary::DataFrames.AbstractDataFrame,
                                 target_drawdown_pct::Real)
    rows = NamedTuple[]
    for group in DataFrames.groupby(summary, [:model, :environment, :variant])
        feasible = group[(group.realised_drawdown_pct .<= target_drawdown_pct) .&
                         isfinite.(group.sharpe_ann), :]
        if DataFrames.nrow(feasible) == 0
            best = group[argmin(group.realised_drawdown_pct), :]
            status = "no lambda met target; minimum drawdown shown"
        else
            best = feasible[argmax(feasible.sharpe_ann), :]
            status = "target met; maximum Sharpe among feasible cells"
        end
        push!(rows, (
            model = String(best.model),
            environment = String(best.environment),
            variant = String(best.variant),
            target_drawdown_pct = Float64(target_drawdown_pct),
            lambda = Float64(best.lambda),
            realised_drawdown_pct = Float64(best.realised_drawdown_pct),
            sharpe_ann = Float64(best.sharpe_ann),
            terminal_return_pct = Float64(best.terminal_return_pct),
            mean_slate_exposure = Float64(best.mean_slate_exposure),
            status,
        ))
    end
    return DataFrames.sort!(DataFrames.DataFrame(rows), [:environment, :variant, :model])
end

function overshoot_frame(summary::DataFrames.AbstractDataFrame)
    cols = [:model, :environment, :variant, :lambda, :beta, :nominal_floor,
            :nominal_drawdown_pct, :realised_drawdown_pct, :overshoot_ratio,
            :mean_slate_exposure, :max_slate_exposure]
    frame = DataFrames.select(summary, cols)
    frame.overshoot_vs_1_15 = frame.overshoot_ratio .- 1.15
    return DataFrames.sort!(frame, [:environment, :variant, :model, :lambda])
end

function markdown_table(io::IO, frame::DataFrames.AbstractDataFrame)
    names_ = String.(DataFrames.names(frame))
    println(io, "| ", join(names_, " | "), " |")
    println(io, "|", join(fill("---", length(names_)), "|"), "|")
    for row in DataFrames.eachrow(frame)
        values = [replace(string(row[name]), "|" => "\\|") for name in DataFrames.names(frame)]
        println(io, "| ", join(values, " | "), " |")
    end
end

function _rounded(frame, columns; digits::Int = 4)
    out = DataFrames.DataFrame(frame)
    for column in columns
        column in propertynames(out) || continue
        out[!, column] = [value isa AbstractFloat && isfinite(value) ?
                          round(value; digits) : value for value in out[!, column]]
    end
    return out
end

function write_sweep_outputs(output_dir::AbstractString;
                             summary, frontier, overshoot, optima,
                             panels, inventory, gates, allocation_gates,
                             calibration_coverage, git_commit, host, n_threads)
    mkpath(output_dir)
    CSV.write(joinpath(output_dir, "lambda_sweep_summary.csv"), summary)
    CSV.write(joinpath(output_dir, "pareto_frontier.csv"), frontier)
    CSV.write(joinpath(output_dir, "overshoot_calibration.csv"), overshoot)

    targets = unique(Float64.(optima.target_drawdown_pct))
    length(targets) == 1 || error("report requires one drawdown target, got $targets")
    target = only(targets)
    betas = unique(Float64.(summary.beta))
    length(betas) == 1 || error("report requires one beta, got $betas")
    beta = only(betas)
    model_names = join(["`" * String(name) * "`" for name in unique(inventory.model)], " and ")

    report = joinpath(output_dir, "LAMBDA_RISK_SWEEP_REPORT.md")
    open(report, "w") do io
        println(io, "# GRW SlateDrawdown lambda risk sweep\n")
        println(io, "Generated `", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"),
                "` at Git `", git_commit, "` on `", host, "` with ", n_threads,
                " Julia threads. No MCMC sampling was performed.\n")
        println(io, "## Contract\n")
        println(io, "- Models: ", model_names, ", loaded by immutable UUID.")
        println(io, "- Policy: operational Option B trust, `SlateDrawdown(lambda)`, `FixedCap(0.25)`, daily slates.")
        println(io, "- T012: `:excise_pruned`; the priced book contains 1X2, O/U 1.5 (Over active), and O/U 2.5 (Under active). Excision is market-level.")
        println(io, "- Smile route: anti-diagonal reweighted score grid. T−25 L2 uses the validated φ-dropped CountLatents control; that arm is de-smiled and calibrated, so only the count baseline isolates calibration alone.")
        println(io, "- Nominal floor: `D = exp(log(", beta, ")/lambda)`; overshoot is realised max drawdown divided by `100(1-D)`.\n")

        println(io, "## Data panels\n")
        markdown_table(io, panels)
        println(io, "\nEnvironment coverage is reported rather than silently padding absent T−25 quotes. Cross-environment returns are therefore descriptive, not paired fixture-for-fixture unless panel counts match.\n")

        println(io, "## Immutable run inventory\n")
        markdown_table(io, inventory)
        println(io, "\n## Verification gates\n")
        markdown_table(io, gates)
        println(io, "\n### Zero-allocation pricing kernels\n")
        markdown_table(io, allocation_gates)
        println(io, "\n### T−25 calibration coverage\n")
        markdown_table(io, calibration_coverage)

        println(io, "\n## First lambda that meets realised max drawdown ≤ ", target, "%\n")
        threshold_rows = NamedTuple[]
        for group in DataFrames.groupby(summary, [:model, :environment, :variant])
            feasible = group[(group.realised_drawdown_pct .<= target) .&
                             isfinite.(group.sharpe_ann), :]
            DataFrames.nrow(feasible) == 0 && continue
            first_feasible = feasible[argmin(feasible.lambda), :]
            push!(threshold_rows, (
                model = String(first_feasible.model),
                environment = String(first_feasible.environment),
                variant = String(first_feasible.variant),
                lambda = Float64(first_feasible.lambda),
                realised_drawdown_pct = Float64(first_feasible.realised_drawdown_pct),
                sharpe_ann = Float64(first_feasible.sharpe_ann),
                terminal_return_pct = Float64(first_feasible.terminal_return_pct),
                mean_slate_exposure = Float64(first_feasible.mean_slate_exposure),
            ))
        end
        if isempty(threshold_rows)
            println(io, "No lambda in the tested grid met the target in any cell.\n")
        else
            thresholds = DataFrames.sort!(DataFrames.DataFrame(threshold_rows),
                                          [:environment, :variant, :model])
            markdown_table(io, _rounded(thresholds,
                [:lambda, :realised_drawdown_pct, :sharpe_ann, :terminal_return_pct,
                 :mean_slate_exposure]; digits = 4))
            threshold_choices = join([
                "$(row.model)/$(row.environment)/$(row.variant): λ=$(row.lambda)"
                for row in DataFrames.eachrow(thresholds)], "; ")
            println(io, "\nFirst-feasible choices: ", threshold_choices,
                    ". Calibrated and raw cells are separate regimes.\n")
        end

        println(io, "## Strict maximum-Sharpe choice under the ", target, "% ceiling\n")
        markdown_table(io, _rounded(optima,
            [:lambda, :realised_drawdown_pct, :sharpe_ann, :terminal_return_pct,
             :mean_slate_exposure]; digits = 4))
        strict_choices = join([
            "$(row.model)/$(row.environment)/$(row.variant): λ=$(row.lambda)"
            for row in DataFrames.eachrow(optima)], "; ")
        println(io, "\nMaximum-Sharpe feasible choices: ", strict_choices,
                ". This criterion need not maximise growth; compare it with the first-feasible table. These are in-sample risk-policy choices on the held-out prediction panel, not fresh predictive-model promotion tests.\n")

        println(io, "## Drawdown–Sharpe Pareto frontier\n")
        shown = frontier[frontier.pareto_drawdown_sharpe, :]
        columns = [:model, :environment, :variant, :lambda, :terminal_return_pct,
                   :sharpe_ann, :realised_drawdown_pct, :mean_slate_exposure,
                   :overshoot_ratio, :pareto_all_metrics]
        markdown_table(io, _rounded(DataFrames.select(shown, columns),
            [:lambda, :terminal_return_pct, :sharpe_ann, :realised_drawdown_pct,
             :mean_slate_exposure, :overshoot_ratio]; digits = 4))

        println(io, "\n## Overshoot calibration\n")
        by_lambda = DataFrames.combine(DataFrames.groupby(overshoot, :lambda),
            :overshoot_ratio => Statistics.mean => :mean_overshoot_ratio,
            :overshoot_ratio => Statistics.minimum => :min_overshoot_ratio,
            :overshoot_ratio => Statistics.maximum => :max_overshoot_ratio)
        markdown_table(io, _rounded(by_lambda,
            [:lambda, :mean_overshoot_ratio, :min_overshoot_ratio,
             :max_overshoot_ratio]; digits = 4))
        raw_ratios = overshoot.overshoot_ratio[overshoot.variant .== "raw"]
        calibrated_ratios = overshoot.overshoot_ratio[overshoot.variant .== "l2_calibrated"]
        raw_range = isempty(raw_ratios) ? "unavailable" :
            string(round(minimum(raw_ratios); digits = 3), "–",
                   round(maximum(raw_ratios); digits = 3))
        calibrated_range = isempty(calibrated_ratios) ? "unavailable" :
            string(round(minimum(calibrated_ratios); digits = 3), "–",
                   round(maximum(calibrated_ratios); digits = 3))
        println(io, "\nThe 1.15 constant is a historical empirical rule, not a gate. Observed raw-regime ratios span ",
                raw_range, " and calibrated-regime ratios span ", calibrated_range,
                ". Pooling regimes into one universal correction would therefore be wrong.\n")

        println(io, "## Interpretation guardrails\n")
        println(io, "- The sweep reuses one set of prebuilt posterior books per model/environment; lambda changes staking only.")
        println(io, "- Terminal return is path-dependent and should not be maximised without the drawdown and Sharpe columns.")
        println(io, "- The close and T−25 books are different price instants. The L2 recipe is applied only to T−25, its declared instant.")
        println(io, "- `bit-identical` means every summary value and every canonical `PortfolioResult` field (including ledger columns and daily states) matched exactly between sequential and threaded execution; SHA-256 digests of those canonical payloads also matched.")
    end
    return report
end

end # module
