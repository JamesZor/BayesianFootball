"""
    ensure_harness_schema!(db)

Create the additive, versioned scorecard and experiment-register tables.  These tables are
intentionally separate from the inference lifecycle schema: the harness may only replace its
own scorecard/register rows and never changes historical inference records.
"""
function ensure_harness_schema!(db::Training.PostgresStorage)
    conn = Training.Inference._db_connect(db)
    try
        Training.Inference._db_exec(conn, """
            CREATE TABLE IF NOT EXISTS harness_scores (
                run_id UUID NOT NULL,
                model TEXT NOT NULL,
                stage TEXT NOT NULL,
                scorecard_version TEXT NOT NULL DEFAULT 'v1',
                subset TEXT NOT NULL,
                market TEXT NOT NULL,
                metric TEXT NOT NULL,
                value DOUBLE PRECISION NOT NULL,
                lo DOUBLE PRECISION,
                hi DOUBLE PRECISION,
                n_obs INTEGER,
                n_fixtures INTEGER,
                reference TEXT NOT NULL,
                control_run_id UUID
            );
        """)
        Training.Inference._db_exec(conn, """
            ALTER TABLE harness_scores ADD COLUMN IF NOT EXISTS control_run_id UUID;
        """)
        Training.Inference._db_exec(conn, """
            ALTER TABLE harness_scores DROP CONSTRAINT IF EXISTS harness_scores_pkey;
        """)
        Training.Inference._db_exec(conn, """
            CREATE UNIQUE INDEX IF NOT EXISTS harness_scores_unique_idx
            ON harness_scores (run_id, scorecard_version, subset, market, metric, COALESCE(control_run_id, '00000000-0000-0000-0000-000000000000'::uuid));
        """)
        Training.Inference._db_exec(conn, """
            CREATE INDEX IF NOT EXISTS harness_scores_lookup_idx
            ON harness_scores (scorecard_version, subset, market, metric);
        """)
        Training.Inference._db_exec(conn, """
            CREATE TABLE IF NOT EXISTS harness_experiments (
                id TEXT PRIMARY KEY,
                date DATE NOT NULL,
                todo INTEGER,
                question TEXT NOT NULL,
                dimension TEXT NOT NULL,
                status TEXT NOT NULL,
                decision TEXT NOT NULL,
                run_ids TEXT NOT NULL,
                readme TEXT NOT NULL
            );
        """)
        Training.Inference._db_exec(conn, """
            CREATE TABLE IF NOT EXISTS harness_checks (
                id BIGSERIAL PRIMARY KEY,
                run_id UUID,
                recipe_hash TEXT NOT NULL,
                experiment TEXT NOT NULL,
                candidate TEXT NOT NULL,
                stage TEXT NOT NULL,
                "check" TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                value JSONB,
                detail TEXT NOT NULL DEFAULT '',
                git_sha TEXT,
                at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        Training.Inference._db_exec(conn, """
            CREATE INDEX IF NOT EXISTS harness_checks_recipe_stage_idx
            ON harness_checks (recipe_hash, stage);
        """)
        Training.Inference._db_exec(conn, """
            CREATE INDEX IF NOT EXISTS harness_checks_run_idx
            ON harness_checks (run_id);
        """)
    finally
        close(conn)
    end
    return db
end

const _HARNESS_SCORE_COLUMNS = (:run_id, :model, :stage, :scorecard_version, :subset,
                                :market, :metric, :value, :lo, :hi, :n_obs,
                                :n_fixtures, :reference)
const _HARNESS_EXPERIMENT_COLUMNS = (:id, :date, :todo, :question, :dimension, :status,
                                     :decision, :run_ids, :readme)

function _harness_require_columns(df::AbstractDataFrame, columns::Tuple, name::AbstractString)
    present = Set(propertynames(df))
    missing_columns = Symbol[column for column in columns if column ∉ present]
    isempty(missing_columns) || error(
        "$name is missing required columns $(join(string.(missing_columns), ", ")).")
    return nothing
end

_harness_nullable(value) = ismissing(value) || value === nothing ? missing : value
_harness_score_version(value) =
    ismissing(value) || value === nothing ? SCORECARD_VERSION : String(value)

"""
    write_scores!(db, df)

Replace every row for each `(run_id, scorecard_version, control_run_id, stage)` represented
in `df`, then insert the long scorecard rows in one transaction. A finalist portfolio write
cannot erase grid scores, and a partial re-score cannot leave a mixed stage scorecard.
"""
function write_scores!(db::Training.PostgresStorage, df::AbstractDataFrame)
    scores = if :scorecard_version in propertynames(df)
        df
    else
        copy(df)
        # `write_scores!` must not mutate its caller merely to apply the current default.
    end
    if scores !== df
        scores.scorecard_version = fill(SCORECARD_VERSION, nrow(scores))
    end
    if !(:control_run_id in propertynames(scores))
        scores.control_run_id = fill(missing, nrow(scores))
    end
    _harness_require_columns(scores, _HARNESS_SCORE_COLUMNS, "harness score frame")
    isempty(scores) && return df

    versions = [_harness_score_version(value) for value in scores.scorecard_version]
    all(in(("v1", "v1.1")), versions) || error(
        "write_scores!: supported scorecards are v1 and v1.1.")
    control_ids = [_harness_nullable(scores.control_run_id[i]) for i in eachindex(scores.control_run_id)]
    run_control_versions = unique([
        (string(scores.run_id[i]),
         versions[i],
         control_ids[i] === missing ? missing : string(control_ids[i]),
         String(scores.stage[i]))
        for i in eachindex(versions)
    ])
    conn = Training.Inference._db_connect(db)
    try
        Training.Inference._db_exec(conn, "BEGIN;")
        try
            for (run_id, scorecard_version, control_run_id, stage) in run_control_versions
                Training.Inference._db_exec(conn, """
                    DELETE FROM harness_scores
                    WHERE run_id = \$1::uuid
                      AND scorecard_version = \$2
                      AND (\$3::uuid IS NULL AND control_run_id IS NULL OR control_run_id = \$3::uuid)
                      AND stage = \$4;
                """, (run_id, scorecard_version, control_run_id, stage))
            end
            for i in axes(scores, 1)
                c_id = control_ids[i] === missing ? missing : string(control_ids[i])
                Training.Inference._db_exec(conn, """
                    INSERT INTO harness_scores (
                        run_id, model, stage, scorecard_version, subset, market, metric,
                        value, lo, hi, n_obs, n_fixtures, reference, control_run_id
                    ) VALUES (
                        \$1::uuid, \$2, \$3, \$4, \$5, \$6, \$7, \$8, \$9, \$10,
                        \$11, \$12, \$13, \$14::uuid
                    );
                """, (string(scores.run_id[i]), String(scores.model[i]), String(scores.stage[i]), versions[i],
                      String(scores.subset[i]), String(scores.market[i]), String(scores.metric[i]),
                      Float64(scores.value[i]), _harness_nullable(scores.lo[i]),
                      _harness_nullable(scores.hi[i]), _harness_nullable(scores.n_obs[i]),
                      _harness_nullable(scores.n_fixtures[i]), String(scores.reference[i]),
                      c_id))
            end
            Training.Inference._db_exec(conn, "COMMIT;")
        catch
            try
                Training.Inference._db_exec(conn, "ROLLBACK;")
            catch
            end
            rethrow()
        end
    finally
        close(conn)
    end
    return df
end

"""
    read_scores(db; run_id=nothing, scorecard_version=nothing, subset=nothing,
                market=nothing, metric=nothing)

Read harness scorecard rows, with every filter optional.  Filter values are bound parameters,
never interpolated into SQL.
"""
function read_scores(db::Training.PostgresStorage; run_id = nothing,
                     scorecard_version = SCORECARD_VERSION,
                     subset = nothing, market = nothing, metric = nothing,
                     control_run_id = nothing)
    conn = Training.Inference._db_connect(db)
    try
        return Training.Inference._db_rows(conn, """
            SELECT run_id, model, stage, scorecard_version, subset, market, metric, value,
                   lo, hi, n_obs, n_fixtures, reference, control_run_id
            FROM harness_scores
            WHERE (\$1::uuid IS NULL OR run_id = \$1::uuid)
              AND (\$2::text IS NULL OR scorecard_version = \$2)
              AND (\$3::text IS NULL OR subset = \$3)
              AND (\$4::text IS NULL OR market = \$4)
              AND (\$5::text IS NULL OR metric = \$5)
              AND (\$6::uuid IS NULL OR control_run_id = \$6::uuid)
            ORDER BY run_id, scorecard_version, subset, market, metric;
        """, (run_id === nothing ? missing : string(run_id),
               scorecard_version === nothing ? missing : string(scorecard_version),
               subset === nothing ? missing : string(subset),
               market === nothing ? missing : string(market),
               metric === nothing ? missing : string(metric),
               control_run_id === nothing ? missing : string(control_run_id)))
    finally
        close(conn)
    end
end

function _harness_row_value(row, column::Symbol)
    column in propertynames(row) || error("experiment row is missing required column $column.")
    return getproperty(row, column)
end

"""
    write_experiment!(db, row)

Upsert one static register row by its stable textual ID.  `run_ids` deliberately remains text:
older EDA-only suites and a missing suite 09 have no model-run UUID to invent.
"""
function write_experiment!(db::Training.PostgresStorage, row)
    values = NamedTuple{_HARNESS_EXPERIMENT_COLUMNS}(Tuple(
        _harness_row_value(row, column) for column in _HARNESS_EXPERIMENT_COLUMNS))
    ismissing(values.id) && error("experiment row id must not be missing.")
    ismissing(values.date) && error("experiment row date must not be missing.")
    conn = Training.Inference._db_connect(db)
    try
        Training.Inference._db_exec(conn, """
            INSERT INTO harness_experiments (
                id, date, todo, question, dimension, status, decision, run_ids, readme
            ) VALUES (\$1, \$2::date, \$3, \$4, \$5, \$6, \$7, \$8, \$9)
            ON CONFLICT (id) DO UPDATE SET
                date = EXCLUDED.date,
                todo = EXCLUDED.todo,
                question = EXCLUDED.question,
                dimension = EXCLUDED.dimension,
                status = EXCLUDED.status,
                decision = EXCLUDED.decision,
                run_ids = EXCLUDED.run_ids,
                readme = EXCLUDED.readme;
        """, (string(values.id), string(values.date), _harness_nullable(values.todo),
              String(values.question), String(values.dimension), String(values.status),
              String(values.decision), String(values.run_ids), String(values.readme)))
    finally
        close(conn)
    end
    return row
end

"Read the static harness experiment register in stable ID order."
function read_experiments(db::Training.PostgresStorage)
    conn = Training.Inference._db_connect(db)
    try
        return Training.Inference._db_rows(conn, """
            SELECT id, date, todo, question, dimension, status, decision, run_ids, readme
            FROM harness_experiments
            ORDER BY CASE WHEN id ~ '^[0-9]+\$' THEN length(id) ELSE 99 END, id;
        """)
    finally
        close(conn)
    end
end

"""
    write_checks!(db, records)

Append check records to the `harness_checks` table.
"""
function write_checks!(db::Training.PostgresStorage, records)
    isempty(records) && return records
    conn = Training.Inference._db_connect(db)
    try
        Training.Inference._db_exec(conn, "BEGIN;")
        try
            for rec in records
                run_id = hasproperty(rec, :run_id) ? rec.run_id : nothing
                val = hasproperty(rec, :value) ? rec.value : NamedTuple()
                val_json = val === nothing || ismissing(val) ? "{}" : JSON3.write(val)
                detail = hasproperty(rec, :detail) ? rec.detail : ""
                git_sha = hasproperty(rec, :git_sha) ? rec.git_sha : nothing
                at_val = hasproperty(rec, :at) ? rec.at : now()
                check_name = hasproperty(rec, :check) ? rec.check : rec.check_name
                Training.Inference._db_exec(conn, """
                    INSERT INTO harness_checks (
                        run_id, recipe_hash, experiment, candidate, stage, "check",
                        severity, status, value, detail, git_sha, at
                    ) VALUES (
                        \$1::uuid, \$2, \$3, \$4, \$5, \$6, \$7, \$8, \$9::jsonb, \$10, \$11, \$12::timestamptz
                    );
                """, (run_id === nothing || ismissing(run_id) ? missing : string(run_id),
                      String(rec.recipe_hash),
                      String(rec.experiment),
                      String(rec.candidate),
                      String(rec.stage),
                      String(check_name),
                      String(rec.severity),
                      String(rec.status),
                      val_json,
                      String(detail),
                      git_sha === nothing || ismissing(git_sha) ? missing : String(git_sha),
                      string(at_val)))
            end
            Training.Inference._db_exec(conn, "COMMIT;")
        catch
            try
                Training.Inference._db_exec(conn, "ROLLBACK;")
            catch
            end
            rethrow()
        end
    finally
        close(conn)
    end
    return records
end

"""
    read_checks(db; run_id=nothing, recipe_hash=nothing, experiment=nothing,
                candidate=nothing, stage=nothing, check=nothing, status=nothing)

Read check rows from `harness_checks`, with every filter optional.
"""
function read_checks(db::Training.PostgresStorage;
                     run_id = nothing,
                     recipe_hash = nothing,
                     experiment = nothing,
                     candidate = nothing,
                     stage = nothing,
                     check = nothing,
                     status = nothing)
    conn = Training.Inference._db_connect(db)
    try
        return Training.Inference._db_rows(conn, """
            SELECT id, run_id, recipe_hash, experiment, candidate, stage,
                   "check", severity, status, value, detail, git_sha, at
            FROM harness_checks
            WHERE (\$1::uuid IS NULL OR run_id = \$1::uuid)
              AND (\$2::text IS NULL OR recipe_hash = \$2)
              AND (\$3::text IS NULL OR experiment = \$3)
              AND (\$4::text IS NULL OR candidate = \$4)
              AND (\$5::text IS NULL OR stage = \$5)
              AND (\$6::text IS NULL OR "check" = \$6)
              AND (\$7::text IS NULL OR status = \$7)
            ORDER BY at ASC, id ASC;
        """, (run_id === nothing ? missing : string(run_id),
              recipe_hash === nothing ? missing : string(recipe_hash),
              experiment === nothing ? missing : string(experiment),
              candidate === nothing ? missing : string(candidate),
              stage === nothing ? missing : string(stage),
              check === nothing ? missing : string(check),
              status === nothing ? missing : string(status)))
    finally
        close(conn)
    end
end

"""
    has_passing_smoke(db, candidate_or_hash) -> Bool

True if `harness_checks` holds a completed smoke stage for this recipe where all hard checks passed.
"""
function has_passing_smoke(db::Training.PostgresStorage, candidate_or_hash)
    hash = candidate_or_hash isa Candidate ? recipe_hash(candidate_or_hash) : String(candidate_or_hash)
    rows = read_checks(db; recipe_hash = hash, stage = "smoke")
    nrow(rows) == 0 && return false
    any(r -> r.severity == "hard" && r.status != "pass", eachrow(rows)) && return false
    hard_passes = Set(r.check for r in eachrow(rows) if r.severity == "hard" && r.status == "pass")
    required = Set(["gradient", "filtration", "latents", "score_grid_coherence", "fit_parity"])
    return issubset(required, hard_passes)
end

function find_completed_run(db::Training.PostgresStorage, config::Training.FitConfig)
    canonical = join((db.experiment_name, config.name,
                      string(config.model), string(config.splitter),
                      string(config.sampler), string(config.execution),
                      join(Training.Inference._db_recipe_tags(config.tags), "\u001f"),
                      config.description), "\u001e")
    hash = bytes2hex(SHA.sha256(canonical))
    conn = Training.Inference._db_connect(db)
    try
        rows = Training.Inference._db_rows(conn, """
            SELECT r.run_id
            FROM configs AS c
            JOIN runs AS r ON r.run_id = c.config_id
            WHERE c.config_hash = \$1 AND r.status = 'completed'
            LIMIT 1;
        """, (hash,))
        return nrow(rows) == 0 ? nothing : UUID(string(rows.run_id[1]))
    finally
        close(conn)
    end
end
find_completed_run(::Any, ::Training.FitConfig) = nothing

"""In-memory check store for offline and unit test execution."""
mutable struct InMemoryCheckStore
    checks::Vector{NamedTuple}
    InMemoryCheckStore() = new(NamedTuple[])
end

ensure_harness_schema!(::InMemoryCheckStore) = nothing

function write_checks!(store::InMemoryCheckStore, records)
    for rec in records
        check_name = hasproperty(rec, :check) ? rec.check : rec.check_name
        row = (;
            id = length(store.checks) + 1,
            run_id = hasproperty(rec, :run_id) ? rec.run_id : nothing,
            recipe_hash = String(rec.recipe_hash),
            experiment = String(rec.experiment),
            candidate = String(rec.candidate),
            stage = String(rec.stage),
            check = String(check_name),
            severity = String(rec.severity),
            status = String(rec.status),
            value = hasproperty(rec, :value) ? rec.value : NamedTuple(),
            detail = hasproperty(rec, :detail) ? String(rec.detail) : "",
            git_sha = hasproperty(rec, :git_sha) ? rec.git_sha : nothing,
            at = hasproperty(rec, :at) ? rec.at : now()
        )
        push!(store.checks, row)
    end
    return records
end

function read_checks(store::InMemoryCheckStore;
                     run_id = nothing,
                     recipe_hash = nothing,
                     experiment = nothing,
                     candidate = nothing,
                     stage = nothing,
                     check = nothing,
                     status = nothing)
    matches = filter(store.checks) do r
        run_id !== nothing && string(r.run_id) != string(run_id) && return false
        recipe_hash !== nothing && string(r.recipe_hash) != string(recipe_hash) && return false
        experiment !== nothing && string(r.experiment) != string(experiment) && return false
        candidate !== nothing && string(r.candidate) != string(candidate) && return false
        stage !== nothing && string(r.stage) != string(stage) && return false
        check !== nothing && string(r.check) != string(check) && return false
        status !== nothing && string(r.status) != string(status) && return false
        return true
    end
    return DataFrame(matches)
end

function has_passing_smoke(store::InMemoryCheckStore, candidate_or_hash)
    hash = candidate_or_hash isa Candidate ? recipe_hash(candidate_or_hash) : String(candidate_or_hash)
    rows = read_checks(store; recipe_hash = hash, stage = "smoke")
    nrow(rows) == 0 && return false
    any(r -> r.severity == "hard" && r.status != "pass", eachrow(rows)) && return false
    hard_passes = Set(r.check for r in eachrow(rows) if r.severity == "hard" && r.status == "pass")
    required = Set(["gradient", "filtration", "latents", "score_grid_coherence", "fit_parity"])
    return issubset(required, hard_passes)
end

