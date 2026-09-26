# Read-only audit of persisted smoke Fits; no sampling and no database writes.
# Usage: julia --project audit_saved_smoke.jl results/smoke/<execution>
using BayesianFootball
import JSON3
import Serialization
include("l01_cross_tier_loader.jl")
include("l02_smoke_helpers.jl")

function audit_saved_smoke(directory)
    db = PostgresStorage("scotland_cross_tier_time_decay")
    ds = Data.load_datastore_cached(Data.ScottishAll(); max_age_hours = 100_000)
    thresholds = ConvergenceThresholds(max_rhat = 1.05, min_ess = 100.0,
        max_divergence_rate = eps(), min_bfmi = 0.30, max_treedepth_rate = 0.05)
    records = []
    for candidate in cross_tier_smoke_candidates()
        report = Serialization.deserialize(joinpath(directory, candidate.name * "_smoke_report.jls"))
        run_id = report.gates[6].metrics.run_id
        fit = load_fit(db, run_id)
        audit = audit_convergence(fit; thresholds, max_depth = fit.config.sampler.max_depth)
        prepared = cross_tier_smoke_prepare_one_fold(candidate, ds)
        record = (; candidate = candidate.name, run_id,
            config_hash = Training.config_hash(fit, db),
            samples = size(fit.folds[1].chain, 1), chains = size(fit.folds[1].chain, 3),
            max_rhat = audit.max_rhat, min_ess_bulk = audit.min_ess_bulk,
            min_ess_tail = audit.min_ess_tail, min_bfmi = audit.min_bfmi,
            divergences = audit.n_divergent, treedepth_rate = audit.treedepth_rate,
            converged = audit.passed, failures = audit.failures, abstained = audit.abstained,
            n_fitted = length(prepared.fitted_ids), heldout_ids = prepared.heldout_ids,
            prior_metrics = prepared.prior_metrics,
            gates = [(; name = g.name, status = g.status, metrics = g.metrics) for g in report.gates])
        push!(records, record)
        println(candidate.name, " ", run_id, " Rhat=", audit.max_rhat,
            " ESSbulk=", audit.min_ess_bulk, " ESStail=", audit.min_ess_tail,
            " BFMI=", audit.min_bfmi, " divergences=", audit.n_divergent)
    end
    output = joinpath(directory, "saved_fit_audit.json")
    isfile(output) && error("refusing to overwrite immutable audit $output")
    open(output, "w") do io
        JSON3.pretty(io, records)
    end
    return output
end

length(ARGS) == 1 || error("supply the immutable smoke execution directory")
audit_saved_smoke(only(ARGS))
