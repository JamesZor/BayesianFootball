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
                PRIMARY KEY (run_id, scorecard_version, subset, market, metric)
            );
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
_harness_score_version(value) = ismissing(value) || value === nothing ? "v1" : String(value)

"""
    write_scores!(db, df)

Replace every row for each `(run_id, scorecard_version)` represented in `df`, then insert the
long scorecard rows in one transaction.  A partial re-score cannot leave a mixed scorecard.
"""
function write_scores!(db::Training.PostgresStorage, df::AbstractDataFrame)
    scores = if :scorecard_version in propertynames(df)
        df
    else
        copy(df)
        # `write_scores!` must not mutate its caller merely to apply the v1 default.
    end
    if scores !== df
        scores.scorecard_version = fill("v1", nrow(scores))
    end
    _harness_require_columns(scores, _HARNESS_SCORE_COLUMNS, "harness score frame")
    isempty(scores) && return df

    versions = [_harness_score_version(value) for value in scores.scorecard_version]
    run_versions = unique([(string(scores.run_id[i]), versions[i]) for i in eachindex(versions)])
    conn = Training.Inference._db_connect(db)
    try
        Training.Inference._db_exec(conn, "BEGIN;")
        try
            for (run_id, scorecard_version) in run_versions
                Training.Inference._db_exec(conn, """
                    DELETE FROM harness_scores
                    WHERE run_id = \$1::uuid AND scorecard_version = \$2;
                """, (run_id, scorecard_version))
            end
            for i in axes(scores, 1)
                Training.Inference._db_exec(conn, """
                    INSERT INTO harness_scores (
                        run_id, model, stage, scorecard_version, subset, market, metric,
                        value, lo, hi, n_obs, n_fixtures, reference
                    ) VALUES (
                        \$1::uuid, \$2, \$3, \$4, \$5, \$6, \$7, \$8, \$9, \$10,
                        \$11, \$12, \$13
                    );
                """, (string(scores.run_id[i]), String(scores.model[i]), String(scores.stage[i]), versions[i],
                      String(scores.subset[i]), String(scores.market[i]), String(scores.metric[i]),
                      Float64(scores.value[i]), _harness_nullable(scores.lo[i]),
                      _harness_nullable(scores.hi[i]), _harness_nullable(scores.n_obs[i]),
                      _harness_nullable(scores.n_fixtures[i]), String(scores.reference[i])))
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
function read_scores(db::Training.PostgresStorage; run_id = nothing, scorecard_version = "v1",
                     subset = nothing, market = nothing, metric = nothing)
    conn = Training.Inference._db_connect(db)
    try
        return Training.Inference._db_rows(conn, """
            SELECT run_id, model, stage, scorecard_version, subset, market, metric, value,
                   lo, hi, n_obs, n_fixtures, reference
            FROM harness_scores
            WHERE (\$1::uuid IS NULL OR run_id = \$1::uuid)
              AND (\$2::text IS NULL OR scorecard_version = \$2)
              AND (\$3::text IS NULL OR subset = \$3)
              AND (\$4::text IS NULL OR market = \$4)
              AND (\$5::text IS NULL OR metric = \$5)
            ORDER BY run_id, scorecard_version, subset, market, metric;
        """, (run_id === nothing ? missing : string(run_id),
               scorecard_version === nothing ? missing : string(scorecard_version),
               subset === nothing ? missing : string(subset),
               market === nothing ? missing : string(market),
               metric === nothing ? missing : string(metric)))
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
