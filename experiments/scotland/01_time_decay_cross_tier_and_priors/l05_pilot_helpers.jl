# Definitions only: separate, user-approved longer-chain convergence pilot.
if !isdefined(@__MODULE__, :ct_pilot_register_preflight)
import LibPQ
import DataFrames
import JSON3

"Register a distinct pilot recipe; reuse an exact completed/failed recipe, never retry its seed."
function ct_pilot_register_preflight(db, config)
    Training.ensure_schema!(db)
    prefix = config.name
    description = config.description
    tags = config.tags
    for (label, value) in (("model", config.model), ("splitter", config.splitter),
                           ("sampler", config.sampler))
        # Public save_model is restricted to composable models; these experiment
        # model types use the same underlying typed configuration registry.
        Training.Inference._save_truth_config(db, prefix * "_" * label, value;
                                              description, tags)
    end
    Training.save_config(db, prefix * "_fit", config; description, tags)
    base_hash = ct_smoke_config_hash(db, config)
    failed_hash = ct_smoke_config_hash(db, config; convergence_failed = true)
    conn = LibPQ.Connection(db.conn_str)
    try
        result = LibPQ.execute(conn, """
            SELECT r.run_id, r.status FROM configs c JOIN runs r ON r.run_id = c.config_id
            WHERE c.config_hash IN (\$1, \$2)
            ORDER BY CASE WHEN c.config_hash = \$1 THEN 0 ELSE 1 END LIMIT 1;
        """, (base_hash, failed_hash))
        try
            rows = DataFrames.DataFrame(result)
            isempty(rows) && return (; base_hash, failed_hash, run_id = nothing)
            String(rows.status[1]) in ("completed", "blocked") ||
                error("pilot recipe already exists with nonterminal status $(rows.status[1])")
            return (; base_hash, failed_hash, run_id = string(rows.run_id[1]))
        finally
            close(result)
        end
    finally
        close(conn)
    end
end

"Atomic progress record; the containing execution directory is unique and never reused."
function ct_pilot_write_progress(path, records)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    open(temporary, "w") do io
        JSON3.pretty(io, records)
    end
    mv(temporary, path; force = true)
    return path
end

"Report every convergence diagnostic, including abstentions; no silent threshold changes."
function ct_pilot_diagnostics(fit, thresholds)
    audit = Training.audit_convergence(fit; thresholds, max_depth = fit.config.sampler.max_depth)
    return (; passed = audit.passed, max_rhat = audit.max_rhat,
              min_ess_bulk = audit.min_ess_bulk, min_ess_tail = audit.min_ess_tail,
              min_bfmi = audit.min_bfmi, divergences = audit.n_divergent,
              treedepth_rate = audit.treedepth_rate,
              failures = audit.failures, abstained = audit.abstained)
end
end
