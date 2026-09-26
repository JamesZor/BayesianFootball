# ==============================================================================
# 04 — Cross-tier Stage-1 production-grid helpers
# ==============================================================================
#
# Durable smoke-manifest validation and production-recipe preflight for r02.
# The smoke runner writes checksum-linked JSON records; this file consumes them.
# ============================================================================

if !isdefined(@__MODULE__, :CrossTierProductionManifest)

import BayesianFootball
import BayesianFootball.Training
import DataFrames
import JSON3
import LibPQ
import SHA

"The immutable, smoke-approved linkage consumed by one production candidate."
struct CrossTierProductionManifest
    candidate::String
    source_digest::String
    loader_digest::String
    helper_digest::String
    core_recipe_digest::String
    heldout_ids::Vector{Int}
    heldout_count::Int
    heldout_ids_sha256::String
    data_manifest
    smoke_report_sha256::String
end

ct02_sha256(text::AbstractString) = bytes2hex(SHA.sha256(codeunits(String(text))))
ct02_file_digest(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))
ct02_ids_digest(ids::AbstractVector{<:Integer}) =
    ct02_sha256(join(string.(sort!(unique(Int.(ids)))), "\u001f"))

function ct02_manifest_field(record, field::Symbol)
    haskey(record, field) || error("smoke manifest is missing required `$field`")
    return getproperty(record, field)
end

"Read and validate the approved smoke schema without treating an absent field as a pass."
function ct02_load_smoke_manifest(path::AbstractString)
    isfile(path) || error("required smoke manifest is absent: $path")
    raw = JSON3.read(read(path, String))
    Int(ct02_manifest_field(raw, :schema_version)) == 1 ||
        error("$path has unsupported smoke manifest schema")
    ct02_manifest_field(raw, :production_blocked) === false ||
        error("$path records production_blocked=true")
    report_path = replace(path, "_production_manifest.json" => "_smoke_report.md")
    isfile(report_path) || error("$path has no companion immutable smoke report: $report_path")
    String(ct02_manifest_field(raw, :smoke_report_sha256)) == ct02_file_digest(report_path) ||
        error("$path companion smoke-report checksum mismatch")
    gates = ct02_manifest_field(raw, :gates)
    length(gates) == 7 || error("$path has $(length(gates)) gates; expected all seven")
    all(String(ct02_manifest_field(gate, :status)) == "passed" for gate in gates) ||
        error("$path contains a failed or unavailable verification gate")
    ids = Int.(collect(ct02_manifest_field(raw, :heldout_match_ids)))
    isempty(ids) && error("$path has an empty smoke held-out manifest")
    digest = String(ct02_manifest_field(raw, :heldout_ids_sha256))
    length(unique(ids)) == length(ids) || error("$path held-out IDs are not unique")
    ct02_ids_digest(ids) == digest || error("$path held-out ID digest does not match its IDs")
    count = Int(ct02_manifest_field(raw, :heldout_count))
    length(ids) == count || error("$path held-out count does not match its IDs")
    smoke_sampler = ct02_manifest_field(raw, :smoke_sampler)
    Int(ct02_manifest_field(smoke_sampler, :n_samples)) == 100 || error("$path smoke retained draws are not 100")
    Int(ct02_manifest_field(smoke_sampler, :n_chains)) == 2 || error("$path smoke chains are not 2")
    Int(ct02_manifest_field(smoke_sampler, :n_warmup)) == 100 || error("$path smoke adaptation is not 100")
    Int(ct02_manifest_field(raw, :history_seasons)) == 2 || error("$path smoke history is not two seasons")
    Int(ct02_manifest_field(raw, :warmup_period)) == 0 || error("$path smoke biweek warmup is not zero")
    fitted_ids = Int.(collect(ct02_manifest_field(raw, :fitted_match_ids)))
    isempty(fitted_ids) && error("$path has no fitted-ID audit")
    isempty(intersect(fitted_ids, ids)) || error("$path smoke fitted and held-out IDs overlap")
    return CrossTierProductionManifest(
        String(ct02_manifest_field(raw, :candidate)),
        String(ct02_manifest_field(raw, :source_digest)),
        String(ct02_manifest_field(raw, :loader_digest)),
        String(ct02_manifest_field(raw, :helper_digest)),
        String(ct02_manifest_field(raw, :core_recipe_digest)),
        sort!(unique(ids)), count, digest, ct02_manifest_field(raw, :data_manifest),
        String(ct02_manifest_field(raw, :smoke_report_sha256)),
    )
end

"Stable digest over a DataFrame's ordered columns and rows; it is a data snapshot identity."
function ct02_dataframe_digest(frame::DataFrames.AbstractDataFrame)
    io = IOBuffer()
    for name in names(frame)
        print(io, name, '\u001e')
        column = frame[!, name]
        for value in column
            print(io, ismissing(value) ? "<missing>" : repr(value), '\u001f')
        end
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

"The source snapshot fields that must match the completed smoke exactly."
function ct02_data_manifest(ds)
    return (; matches_sha256 = ct02_dataframe_digest(ds.matches),
              odds_sha256 = ct02_dataframe_digest(ds.odds),
              match_count = DataFrames.nrow(ds.matches),
              odds_count = DataFrames.nrow(ds.odds))
end

function ct02_data_manifest_matches(expected, observed)
    String(ct02_manifest_field(expected, :matches_sha256)) == observed.matches_sha256 || return false
    String(ct02_manifest_field(expected, :odds_sha256)) == observed.odds_sha256 || return false
    Int(ct02_manifest_field(expected, :match_count)) == observed.match_count || return false
    Int(ct02_manifest_field(expected, :odds_count)) == observed.odds_count || return false
    return true
end

"Core identity intentionally excludes sampler and runner source: it binds model/split/data."
function ct02_core_recipe_digest(candidate, data_manifest)
    canonical = join((candidate.name, string(typeof(candidate.model)), string(candidate.model),
                      string(candidate.splitter), string(typeof(candidate.segment)),
                      data_manifest.matches_sha256, data_manifest.odds_sha256,
                      string(data_manifest.match_count), string(data_manifest.odds_count)), "\u001e")
    return ct02_sha256(canonical)
end

"Write the immutable v1 smoke promotion record only after all seven gates passed."
function ct02_write_smoke_promotion_manifest(candidate, prepared, ds, report,
                                              report_path::AbstractString;
                                              source_digest::AbstractString,
                                              loader_path::AbstractString,
                                              helper_path::AbstractString,
                                              output_dir::AbstractString = dirname(report_path))
    report.production_blocked && error(
        "$(candidate.name): refusing promotion manifest because smoke blocks production")
    length(report.gates) == 7 || error(
        "$(candidate.name): refusing promotion manifest without all seven gates")
    all(gate -> gate.status === :passed, report.gates) || error(
        "$(candidate.name): refusing promotion manifest with a failed or abstained gate")
    isfile(report_path) || error("$(candidate.name): smoke report is missing at $report_path")
    heldout_ids = sort!(unique(Int.(prepared.heldout_ids)))
    observed_data = ct02_data_manifest(ds)
    gate_records = [(; name = gate.name, status = String(gate.status),
                      metrics = gate.metrics) for gate in report.gates]
    isfile(joinpath(output_dir, candidate.name * "_production_manifest.json")) && error(
        "$(candidate.name): promotion manifest already exists; immutable records are never overwritten")
    smoke_sampler = candidate.fit_config.sampler
    hasproperty(smoke_sampler, :n_samples) && smoke_sampler.n_samples == 100 || error(
        "$(candidate.name): promotion requires the exact smoke retained-draw budget")
    hasproperty(smoke_sampler, :n_chains) && smoke_sampler.n_chains == 2 || error(
        "$(candidate.name): promotion requires the exact smoke chain budget")
    hasproperty(smoke_sampler, :n_warmup) && smoke_sampler.n_warmup == 100 || error(
        "$(candidate.name): promotion requires the exact smoke adaptation budget")
    record = (; schema_version = 1,
                candidate = candidate.name,
                smoke_report_sha256 = ct02_file_digest(report_path),
                smoke_started_at = string(report.started_at),
                smoke_completed_at = string(report.completed_at),
                production_blocked = false,
                gates = gate_records,
                source_digest = String(source_digest),
                loader_digest = ct02_file_digest(loader_path),
                helper_digest = ct02_file_digest(helper_path),
                core_recipe_digest = ct02_core_recipe_digest(candidate, observed_data),
                segment = string(typeof(candidate.segment)),
                model_type = string(typeof(candidate.model)),
                model_repr = string(candidate.model),
                splitter_repr = string(candidate.splitter),
                target_seasons = candidate.splitter.target_seasons,
                history_seasons = candidate.splitter.history_seasons,
                warmup_period = candidate.splitter.warmup_period,
                smoke_sampler = (; n_samples = smoke_sampler.n_samples,
                                   n_warmup = smoke_sampler.n_warmup,
                                   n_chains = smoke_sampler.n_chains,
                                   max_depth = smoke_sampler.max_depth),
                fitted_match_ids = sort!(unique(Int.(prepared.fitted_ids))),
                heldout_match_ids = heldout_ids,
                heldout_count = length(heldout_ids),
                heldout_ids_sha256 = ct02_ids_digest(heldout_ids),
                data_manifest = observed_data)
    target = joinpath(output_dir, candidate.name * "_production_manifest.json")
    mkpath(dirname(target))
    isfile(target) && error("$(candidate.name): promotion manifest appeared during write")
    open(target, "x") do io
        JSON3.pretty(io, record)
    end
    return target
end

function ct02_validate_smoke_manifest(manifest::CrossTierProductionManifest, candidate,
                                      loader_digest::AbstractString, helper_digest::AbstractString,
                                      data_manifest)
    manifest.candidate == candidate.name || error(
        "smoke manifest candidate $(manifest.candidate) cannot authorise $(candidate.name)")
    manifest.loader_digest == loader_digest || error(
        "$(candidate.name): smoke is stale because l01_cross_tier_loader.jl changed")
    manifest.helper_digest == helper_digest || error(
        "$(candidate.name): smoke is stale because l02_smoke_helpers.jl changed")
    ct02_data_manifest_matches(manifest.data_manifest, data_manifest) || error(
        "$(candidate.name): smoke is stale because the ScottishAll data snapshot changed")
    manifest.core_recipe_digest == ct02_core_recipe_digest(candidate, data_manifest) || error(
        "$(candidate.name): smoke is stale because its model/splitter/data recipe changed")
    return nothing
end

"Validate the fixed production sampling contract without sampling."
function ct02_validate_production_execution(config::Training.FitConfig)
    sampler = config.sampler
    nameof(typeof(sampler)) === :QueuedNUTSConfig || error(
        "$(config.name): production requires QueuedNUTSConfig, got $(typeof(sampler))")
    sampler.n_samples == 1_000 || error("$(config.name): requires 1,000 retained draws")
    sampler.n_chains == 4 || error("$(config.name): requires four chains")
    sampler.n_warmup == 500 || error("$(config.name): requires 500 adaptation draws")
    config.splitter.warmup_period == 0 || error("$(config.name): requires biweek warmup_period = 0")
    sampler.accept_rate == 0.65 || error("$(config.name): requires user-approved accept_rate = 0.65")
    sampler.max_depth == 10 || error("$(config.name): requires max_depth = 10")
    config.execution isa Training.QueuedExecution || error(
        "$(config.name): production requires explicit QueuedExecution")
    config.execution.max_concurrent_tasks == 16 || error(
        "$(config.name): requires 16 queued concurrent tasks")
    return nothing
end

"Production's exact Fit hash, matching `Training.config_hash` before a Fit exists."
function ct02_config_hash(db::Training.PostgresStorage, config::Training.FitConfig)
    tags = filter(config.tags) do tag
        !any(prefix -> startswith(tag, prefix), ("time:", "folds_failed:", "latents:"))
    end
    canonical = join((db.experiment_name, config.name, string(config.model),
                      string(config.splitter), string(config.sampler), string(config.execution),
                      join(tags, "\u001f"), config.description), "\u001e")
    return ct02_sha256(canonical)
end

"The engine appends this tag after a failed audit; its persisted hash must also block a refit."
"Durable checkpoint namespace for the exact recipe, including data/model/sampler provenance tags."
function ct02_checkpoint_key(db::Training.PostgresStorage, config::Training.FitConfig)
    return ct02_config_hash(db, config)
end

"Persist an unconverged Fit for audit, then mark it blocked so it can never be promoted."
function ct02_save_blocked_fit!(fit, db::Training.PostgresStorage)
    run_id = Training.save_fit(fit, db)
    conn = LibPQ.Connection(db.conn_str)
    try
        result = LibPQ.execute(conn, "UPDATE runs SET status = 'blocked' WHERE run_id = \$1::uuid;",
                               (string(run_id),))
        close(result)
    finally
        close(conn)
    end
    return run_id
end

"Write one durable, replaceable progress record; immutable Fits and manifests remain separate."
function ct02_write_production_outcome(path::AbstractString, outcome)
    tmp = String(path) * ".tmp." * string(rand(UInt64), base = 16)
    mkpath(dirname(path))
    try
        open(tmp, "w") do io
            JSON3.pretty(io, outcome)
        end
        mv(tmp, path; force = true)
    catch
        isfile(tmp) && rm(tmp; force = true)
        rethrow()
    end
    return path
end

function ct02_failed_config_hash(db::Training.PostgresStorage, config::Training.FitConfig)
    failed = Training.FitConfig(
        name = config.name, model = config.model, splitter = config.splitter,
        sampler = config.sampler, execution = config.execution,
        tags = [config.tags; "convergence:FAIL"], description = config.description,
        save_dir = config.save_dir,
    )
    return ct02_config_hash(db, failed)
end

"Register immutable canonical recipes and reuse or report either exact production recipe state."
function ct02_register_and_preflight!(db::Training.PostgresStorage, candidate,
                                      config::Training.FitConfig, description::AbstractString,
                                      tags::Vector{String})
    Training.ensure_schema!(db)
    Training.Inference._save_truth_config(db, candidate.name * "_model", candidate.model;
                                          description, tags)
    Training.Inference._save_truth_config(db, candidate.name * "_splitter", candidate.splitter;
                                          description, tags)
    Training.Inference._save_truth_config(db, candidate.name * "_production_nuts", config.sampler;
                                          description = "TODO028 production NUTS recipe.", tags)
    Training.save_config(db, candidate.name * "_production_fit", config; description, tags)
    hash = ct02_config_hash(db, config)
    failed_hash = ct02_failed_config_hash(db, config)
    conn = LibPQ.Connection(db.conn_str)
    try
        rows = DataFrames.DataFrame(LibPQ.execute(conn, """
            SELECT r.id, r.run_id, r.status, c.config_hash
            FROM configs AS c JOIN runs AS r ON r.run_id = c.config_id
            WHERE c.config_hash = ANY(\$1::varchar[])
            ORDER BY CASE WHEN c.config_hash = \$2 THEN 0 ELSE 1 END
            LIMIT 1;
        """, ([hash, failed_hash], hash)))
        isempty(rows) && return (; hash, failed_hash, existing_run_id = nothing,
                                  prior_convergence_failed = false)
        prior_failed = String(rows.config_hash[1]) == failed_hash
        status = String(rows.status[1])
        prior_failed && status == "blocked" && return (
            ; hash, failed_hash, existing_run_id = string(rows.run_id[1]),
              prior_convergence_failed = true)
        status == "completed" || error(
            "$(candidate.name): exact production recipe $(rows.config_hash[1]) has " *
            "non-reusable status=$status in run $(rows.run_id[1])")
        return (; hash, failed_hash, existing_run_id = string(rows.run_id[1]),
                prior_convergence_failed = prior_failed)
    finally
        close(conn)
    end
end

end # include guard
