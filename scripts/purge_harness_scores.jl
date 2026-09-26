#!/usr/bin/env julia

using BayesianFootball
using UUIDs

function parse_args(args)
    isempty(args) && error(
        "usage: julia --project scripts/purge_harness_scores.jl <run_id> [--stage screen]")
    startswith(first(args), "-") && error("an explicit run_id is required")
    run_id = try
        UUID(first(args))
    catch
        error("run_id must be an explicit UUID; got $(repr(first(args)))")
    end
    stage = "screen"
    i = 2
    while i <= length(args)
        if args[i] == "--stage"
            i + 1 <= length(args) || error("--stage requires a value")
            stage = String(args[i + 1])
            i += 2
        elseif startswith(args[i], "--stage=")
            stage = split(args[i], "=", limit = 2)[2]
            i += 1
        else
            error("unknown argument: $(args[i])")
        end
    end
    isempty(stage) && error("stage must be explicit and non-empty")
    return (; run_id, stage)
end

function main(args = ARGS)
    parsed = parse_args(args)
    db = Training.PostgresStorage("harness")
    Harness.ensure_harness_schema!(db)
    conn = Training.Inference._db_connect(db)
    try
        rows = Training.Inference._db_rows(conn, """
            SELECT count(*)::bigint AS n
            FROM harness_scores
            WHERE run_id = \$1::uuid AND stage = \$2;
        """, (string(parsed.run_id), parsed.stage))
        before = Int(only(rows.n))
        println("BEFORE run_id=$(parsed.run_id) stage=$(parsed.stage) rows=$before")
        Training.Inference._db_exec(conn, "BEGIN;")
        try
            Training.Inference._db_exec(conn, """
                DELETE FROM harness_scores
                WHERE run_id = \$1::uuid AND stage = \$2;
            """, (string(parsed.run_id), parsed.stage))
            Training.Inference._db_exec(conn, "COMMIT;")
        catch
            try
                Training.Inference._db_exec(conn, "ROLLBACK;")
            catch
            end
            rethrow()
        end
        after_rows = Training.Inference._db_rows(conn, """
            SELECT count(*)::bigint AS n
            FROM harness_scores
            WHERE run_id = \$1::uuid AND stage = \$2;
        """, (string(parsed.run_id), parsed.stage))
        after = Int(only(after_rows.n))
        println("AFTER run_id=$(parsed.run_id) stage=$(parsed.stage) rows=$after deleted=$(before - after)")
    finally
        close(conn)
    end
    return 0
end

exit(main())
