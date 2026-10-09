# Read-only persistence diagnosis; no sampling, retries, writes or gate alteration.
module Wave2ParityAudit
import BayesianFootball, Serialization, CSV, DataFrames
const H = BayesianFootball.Harness
const T = BayesianFootball.Training

function clone(value)
    io = IOBuffer()
    Serialization.serialize(io, value)
    seekstart(io)
    return Serialization.deserialize(io)
end

function audit(candidate, experiment; output)
    template = H.fit_config(candidate; stage = :smoke, experiment)
    restored = clone(template)
    config_fields = [(; field = string(f), structural_equal = H._structural_equal(
        getfield(template, f), getfield(restored, f))) for f in fieldnames(typeof(template))]
    CSV.write(joinpath(output, "parity_config_fields.csv"), DataFrames.DataFrame(config_fields))
    original = candidate.model.observation.feature.rates
    copied = restored.model.observation.feature.rates
    db = T.PostgresStorage(experiment * "_smoke")
    conn = T.Inference._db_connect(db)
    runs = try
        T.Inference._db_rows(conn, "SELECT run_id, name, status FROM runs WHERE experiment_name = \$1 AND name = \$2 ORDER BY id DESC LIMIT 1;",
            (db.experiment_name, candidate.name))
    finally
        close(conn)
    end
    DataFrames.nrow(runs) == 1 || error("expected one saved market-observation smoke")
    recovered = T.load_fit(db, string(runs.run_id[1]))
    persisted = recovered.config.model.observation.feature.rates
    checks = H.read_checks(T.PostgresStorage(experiment); candidate = candidate.name, stage = "smoke")
    CSV.write(joinpath(output, "smoke_checks_$(candidate.name)_failed.csv"), checks)
    facts = (; candidate = candidate.name, run_id = string(runs.run_id[1]),
        saved_run_status = string(runs.status[1]), entries = length(original),
        cloned_config_structural = H._structural_equal(template, restored),
        cloned_dictionary_structural = H._structural_equal(original, copied),
        cloned_dictionary_value_equal = isequal(original, copied),
        persisted_dictionary_structural = H._structural_equal(original, persisted),
        persisted_dictionary_value_equal = isequal(original, persisted),
        persisted_model_structural = H._structural_equal(candidate.model, recovered.config.model),
        other_config_fields_unchanged = all(r -> r.field == "model" || r.structural_equal, config_fields))
    CSV.write(joinpath(output, "parity_audit.csv"), DataFrames.DataFrame([facts]))
    println("[PARITY_DIAGNOSIS] ", facts)
    return facts
end
end
