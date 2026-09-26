# ==============================================================================
# 01 — Cross-tier Stage-1 smoke-test helpers
# ==============================================================================
#
# Technical machinery for r01_smoke_test.jl.  These helpers deliberately do not
# construct models or launch sampling; the runner owns the scientific workflow.
# ============================================================================

if !isdefined(@__MODULE__, :CrossTierSmokeGate)

import BayesianFootball
import BayesianFootball.Training
import DataFrames
import Dates
import DynamicPPL
import ForwardDiff
import LibPQ
import LogDensityProblems
import LinearAlgebra
import MCMCChains
import ReverseDiff
import Serialization
import SHA
import Statistics

"One of the seven required smoke gates, including an honest abstention reason."
Base.@kwdef struct CrossTierSmokeGate
    name::String
    status::Symbol = :abstained
    detail::String = "not attempted"
    metrics::NamedTuple = NamedTuple()
end

"A durable per-candidate record; `production_blocked` is always derived from the seven gates."
Base.@kwdef struct CrossTierSmokeReport
    candidate::String
    started_at::Dates.DateTime
    completed_at::Dates.DateTime = Dates.now()
    gates::Vector{CrossTierSmokeGate} = CrossTierSmokeGate[]
    production_blocked::Bool = true
end

"Execute a gate without converting an error into a pass."
function ct_smoke_gate(name::AbstractString, f::Function)
    try
        detail, metrics = f()
        return CrossTierSmokeGate(name = String(name), status = :passed,
                                  detail = String(detail), metrics = metrics)
    catch err
        return CrossTierSmokeGate(
            name = String(name),
            status = :failed,
            detail = sprint(showerror, err, catch_backtrace()),
        )
    end
end

"`do` blocks pass their anonymous function first; retain the readable runner call form."
ct_smoke_gate(f::Function, name::AbstractString) = ct_smoke_gate(name, f)

"Record an unavailable gate explicitly; it is neither a pass nor hidden from the report."
ct_smoke_abstained(name::AbstractString, detail::AbstractString) =
    CrossTierSmokeGate(name = String(name), status = :abstained, detail = String(detail))

"All seven gates must pass before the candidate is eligible for production."
ct_smoke_production_blocked(gates::Vector{CrossTierSmokeGate}) =
    length(gates) != 7 || any(gate -> gate.status !== :passed, gates)

"Exact recipe hash matching `Training.config_hash(fit, db)` before a `Fit` exists."
function ct_smoke_config_hash(db::Training.PostgresStorage, config::Training.FitConfig;
                              convergence_failed::Bool = false)
    tags = copy(config.tags)
    convergence_failed && push!(tags, "convergence:FAIL")
    tags = filter(tags) do tag
        !any(prefix -> startswith(tag, prefix), ("time:", "folds_failed:", "latents:"))
    end
    canonical = join((db.experiment_name, config.name, string(config.model),
                      string(config.splitter), string(config.sampler),
                      string(config.execution), join(tags, "\u001f"),
                      config.description), "\u001e")
    return bytes2hex(SHA.sha256(canonical))
end

"Register every canonical recipe and either return a reusable completed Fit or authorise sampling."
function ct_smoke_register_and_preflight!(db::Training.PostgresStorage,
                                          candidate, description::AbstractString,
                                          tags::Vector{String})
    Training.ensure_schema!(db)
    Training.Inference._save_truth_config(db, candidate.name * "_model", candidate.model;
                                          description = description, tags = tags)
    Training.Inference._save_truth_config(db, candidate.name * "_splitter", candidate.splitter;
                                          description = description, tags = tags)
    Training.Inference._save_truth_config(db, candidate.name * "_smoke_nuts", candidate.fit_config.sampler;
                                          description = "Stage-1 smoke: exactly two chains × 100 retained draws.",
                                          tags = tags)
    Training.save_config(db, candidate.name * "_smoke_fit", candidate.fit_config;
                         description = description, tags = tags)

    base_hash = ct_smoke_config_hash(db, candidate.fit_config)
    failed_hash = ct_smoke_config_hash(db, candidate.fit_config; convergence_failed = true)
    conn = LibPQ.Connection(db.conn_str)
    try
        result = LibPQ.execute(conn, """
            SELECT c.config_hash, r.id, r.run_id, r.status
            FROM configs AS c
            JOIN runs AS r ON r.run_id = c.config_id
            WHERE c.config_hash = ANY(\$1::varchar[])
            ORDER BY CASE WHEN c.config_hash = \$2 THEN 0 ELSE 1 END
            LIMIT 1;
        """, ([base_hash, failed_hash], base_hash))
        try
            rows = DataFrames.DataFrame(result)
            DataFrames.nrow(rows) == 0 && return (; base_hash, failed_hash, existing_run_id = nothing)
            String(rows.status[1]) == "completed" || error(
                "Preflight refused $(candidate.name): recipe hash $(rows.config_hash[1]) already has " *
                "run #$(rows.id[1]) ($(rows.run_id[1]), status=$(rows.status[1])).")
            return (; base_hash, failed_hash, existing_run_id = string(rows.run_id[1]))
        finally
            close(result)
        end
    finally
        close(conn)
    end
end

ct_smoke_relerr(a, b) = LinearAlgebra.norm(a - b) /
                         max(LinearAlgebra.norm(a), LinearAlgebra.norm(b), 1.0)

"Build a compiled ReverseDiff tape; correctness failures are distinguished from performance failures."
function ct_smoke_tape_gate(model, feature_set)
    turing_model = BayesianFootball.Models.PreGame.build_turing_model(model, feature_set)
    vi = DynamicPPL.VarInfo(turing_model)
    turing_model(vi)
    linked_vi = DynamicPPL.link(vi, turing_model)
    # Match Turing's NUTS Hamiltonian: unconstrained coordinates INCLUDING Jacobian.
    ldf = DynamicPPL.LogDensityFunction(turing_model, DynamicPPL.getlogjoint_internal, linked_vi)
    theta = copy(DynamicPPL.getparams(ldf))
    f = x -> LogDensityProblems.logdensity(ldf, x)
    raw = ReverseDiff.GradientTape(f, theta)
    tape = ReverseDiff.compile(raw)
    gradient = similar(theta)
    for _ in 1:50
        ReverseDiff.gradient!(gradient, tape, theta)
    end
    elapsed = minimum([@elapsed ReverseDiff.gradient!(gradient, tape, theta) for _ in 1:400])
    allocation = @allocated ReverseDiff.gradient!(gradient, tape, theta)
    fresh = ReverseDiff.gradient(f, theta)
    forward = ForwardDiff.gradient(f, theta)
    ct_smoke_relerr(gradient, fresh) <= 1e-8 || error("compiled ReverseDiff gradient differs from fresh ReverseDiff")
    ct_smoke_relerr(gradient, forward) <= 1e-6 || error("compiled ReverseDiff gradient differs from ForwardDiff")
    for delta in (0.001, -0.002, 0.003, 0.8, -0.8)
        theta_perturbed = theta .+ delta .* sin.(collect(eachindex(theta)))
        compiled_perturbed = similar(theta_perturbed)
        ReverseDiff.gradient!(compiled_perturbed, tape, theta_perturbed)
        fresh_perturbed = ReverseDiff.gradient(f, theta_perturbed)
        ct_smoke_relerr(compiled_perturbed, fresh_perturbed) <= 1e-8 || error(
            "compiled tape diverges from fresh ReverseDiff at perturbation $delta")
    end
    performance_ok = elapsed < 0.00005 && allocation == 0
    return (; correctness_ok = true, performance_ok, elapsed, allocation,
            instructions = length(raw.tape))
end

"Translate the tape result into the required strict Gate-1 status without hiding the metric."
function ct_smoke_tape_gate_result(tape_result)
    milliseconds = tape_result.elapsed * 1e3
    tape_result.performance_ok && return (
        "$(round(milliseconds, digits = 4)) ms, zero allocations; AD parity passed",
        (; milliseconds, allocation = tape_result.allocation, instructions = tape_result.instructions),
    )
    error("AD correctness passed, but compiled gradient replay was $(milliseconds) ms and " *
          "allocated $(tape_result.allocation) bytes; Gate 1 requires < 0.05 ms and zero allocation")
end

"Require the requested retained-draw and chain shape; never silently enlarge the run."
function ct_smoke_chain_shape(fit)
    length(fit.folds) == 1 || error("smoke fit has $(length(fit.folds)) folds; expected exactly one")
    chain = fit.folds[1].chain
    chain isa MCMCChains.Chains || error("smoke sampler returned $(typeof(chain)), not MCMCChains.Chains")
    n_samples, _, n_chains = size(chain)
    n_samples == 100 || error("smoke chain has $n_samples retained draws; required exactly 100")
    n_chains == 2 || error("smoke chain has $n_chains chains; required exactly 2")
    return "2 chains × 100 retained draws completed", (; n_samples, n_chains)
end

"The unweakened six-part smoke audit required by the work package."
function ct_smoke_convergence_gate(fit)
    thresholds = Training.ConvergenceThresholds(
        max_rhat = 1.05,
        min_ess = 100.0,
        max_divergence_rate = eps(),
        min_bfmi = 0.30,
        max_treedepth_rate = 0.05,
    )
    summary = Training.audit_convergence(fit; thresholds,
                                         max_depth = fit.config.sampler.max_depth)
    summary.passed || error("six-part audit failed: " * join(summary.failures, " | ") *
                            (isempty(summary.abstained) ? "" :
                             " | absent diagnostics: " * join(summary.abstained, ", ")))
    return "six-part convergence audit passed", (; max_rhat = summary.max_rhat,
        min_ess_bulk = summary.min_ess_bulk, min_ess_tail = summary.min_ess_tail,
        divergences = summary.n_divergent, min_bfmi = summary.min_bfmi,
        treedepth_rate = summary.treedepth_rate)
end

"Check the CountLatents contract rather than accepting an arbitrary extraction result."
function ct_smoke_latent_gate(fit)
    latents = fit.latents
    latents isa BayesianFootball.Models.CountLatents || error(
        "latent extraction returned $(typeof(latents)); required CountLatents")
    BayesianFootball.Models.n_matches(latents) > 0 || error("CountLatents has no OOS fixtures")
    BayesianFootball.Models.n_draws(latents) == 200 || error(
        "CountLatents has $(BayesianFootball.Models.n_draws(latents)) draws; expected 200")
    return "CountLatents extracted", (; n_matches = BayesianFootball.Models.n_matches(latents),
                                       n_draws = BayesianFootball.Models.n_draws(latents))
end

"Verify one score grid and all required market pricing routes."
function ct_smoke_score_grid_gate(fit)
    latents = fit.latents
    n_fixtures = BayesianFootball.Models.n_matches(latents)
    n_draws = BayesianFootball.Models.n_draws(latents)
    markets = (BayesianFootball.Data.Market1X2(), BayesianFootball.Data.MarketOverUnder(2.5),
               BayesianFootball.Data.MarketBTTS())
    min_mass = Inf
    for fixture in 1:n_fixtures
        grid = BayesianFootball.Predictions.compute_score_grid(latents, fixture)
        size(grid) == (12, 12, n_draws) || error(
            "fixture $fixture score grid has size $(size(grid)); expected (12, 12, $n_draws)")
        prices = [BayesianFootball.Predictions.price_market(grid, market) for market in markets]
        for (market, price) in zip(markets, prices)
            !isempty(price) || error("fixture $fixture $(typeof(market)) pricing returned no selections")
            all(draw_probs -> all(isfinite, draw_probs) && all(p -> 0.0 <= p <= 1.0, draw_probs),
                values(price)) || error("fixture $fixture $(typeof(market)) pricing produced invalid probabilities")
        end
        for draw in 1:n_draws
            total_mass = sum(@view grid[:, :, draw])
            min_mass = min(min_mass, total_mass)
            abs(sum(v[draw] for v in values(prices[1])) - total_mass) <= 1e-12 ||
                error("fixture $fixture 1X2 partition does not equal grid mass for draw $draw")
            abs(sum(v[draw] for v in values(prices[2])) - total_mass) <= 1e-12 ||
                error("fixture $fixture O/U 2.5 partition does not equal grid mass for draw $draw")
            abs(sum(v[draw] for v in values(prices[3])) - total_mass) <= 1e-12 ||
                error("fixture $fixture BTTS partition does not equal grid mass for draw $draw")
            0.999 <= total_mass <= 1.0 + 1e-12 || error(
                "fixture $fixture grid mass $total_mass is below the 0.999 truncation-tolerance floor")
        end
    end
    return "all $n_fixtures score grids and coherent 1X2 / totals / BTTS pricing succeeded",
           (; n_fixtures, n_draws, min_mass)
end

"Structural equality that compares mutable containers by contents, not object identity."
function ct_smoke_structural_equal(a, b)
    typeof(a) === typeof(b) || return false
    a isa Union{Nothing,Missing,Number,AbstractString,Symbol,Char,Dates.Date,Dates.DateTime} && return isequal(a, b)
    a isa AbstractArray && return size(a) == size(b) && all(
        ct_smoke_structural_equal(a[i], b[i]) for i in eachindex(a))
    a isa Tuple && return length(a) == length(b) && all(
        ct_smoke_structural_equal(a[i], b[i]) for i in eachindex(a))
    a isa NamedTuple && return keys(a) == keys(b) && all(
        ct_smoke_structural_equal(getproperty(a, key), getproperty(b, key)) for key in keys(a))
    a isa AbstractDict && return length(a) == length(b) && all(
        haskey(b, key) && ct_smoke_structural_equal(value, b[key]) for (key, value) in a)
    fieldcount(typeof(a)) == 0 && return isequal(a, b)
    return all(ct_smoke_structural_equal(getfield(a, field), getfield(b, field))
               for field in fieldnames(typeof(a)))
end

"Exact Fit parity after PostgreSQL serialisation and relational CountLatents reconstruction."
function ct_smoke_fit_parity(original, recovered)
    for field in fieldnames(typeof(original.config))
        ct_smoke_structural_equal(getfield(original.config, field), getfield(recovered.config, field)) || error(
            "FitConfig.$field changed on reload")
    end
    length(original.folds) == length(recovered.folds) || error("fold count changed on reload")
    for i in eachindex(original.folds)
        original.folds[i].fold == recovered.folds[i].fold || error("fold index changed on reload")
        Array(original.folds[i].chain) == Array(recovered.folds[i].chain) || error(
            "posterior chain changed on reload for fold $(original.folds[i].fold)")
    end
    for field in fieldnames(typeof(original.diagnostics))
        ct_smoke_structural_equal(getfield(original.diagnostics, field), getfield(recovered.diagnostics, field)) || error(
            "convergence diagnostics.$field changed on reload")
    end
    lhs = original.latents
    rhs = recovered.latents
    lhs isa BayesianFootball.Models.CountLatents || error("original Fit lacks CountLatents")
    rhs isa BayesianFootball.Models.CountLatents || error("reloaded Fit lacks CountLatents")
    lhs.match_ids == rhs.match_ids || error("latent match IDs changed on reload")
    lhs.λ_home == rhs.λ_home || error("home-rate draws changed on reload")
    lhs.λ_away == rhs.λ_away || error("away-rate draws changed on reload")
    lhs.observation_params == rhs.observation_params || error("observation draws changed on reload")
    return nothing
end

"Exact result parity for the persisted portfolio artefact."
function ct_smoke_portfolio_parity(original, recovered)
    for field in fieldnames(typeof(original))
        ct_smoke_structural_equal(getfield(original, field), getfield(recovered, field)) || error(
            "PortfolioResult.$field changed on reload")
    end
    return nothing
end

"Write both machine-readable and human-readable records, including failed/abstained gates."
function ct_smoke_write_report(report::CrossTierSmokeReport, output_dir::AbstractString)
    mkpath(output_dir)
    stem = joinpath(output_dir, report.candidate * "_smoke_report")
    open(stem * ".jls", "w") do io
        Serialization.serialize(io, report)
    end
    open(stem * ".md", "w") do io
        println(io, "# Stage-1 smoke report — `", report.candidate, "`")
        println(io)
        println(io, "- Started: `", report.started_at, "`")
        println(io, "- Completed: `", report.completed_at, "`")
        println(io, "- Production blocked: **", report.production_blocked, "**")
        println(io)
        println(io, "| Gate | Status | Detail |")
        println(io, "|---|---|---|")
        for gate in report.gates
            detail = replace(gate.detail, "|" => "\\|", "\n" => "<br>")
            println(io, "| ", gate.name, " | ", uppercase(String(gate.status)), " | ", detail, " |")
        end
    end
    return stem
end

end # include guard
