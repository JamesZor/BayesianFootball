# Authorised D14 cleanup: archive first; delete only id832 after archive commit.
using BayesianFootball, CSV, DataFrames
const S3_HASH = "f75623499919dd9e2e045595bdbe3735c1b2e7748b0655602844ba68a0f46ba7"
const S3_OUT = @__DIR__
s3db = Training.PostgresStorage("scottish_lower_qs_transition_2326")
s3conn = Training.Inference._db_connect(s3db)
try
    rows = Training.Inference._db_rows(s3conn, "SELECT * FROM harness_checks WHERE id=832;")
    nrow(rows)==1 || error("Expected exactly row832")
    only(rows.recipe_hash)==S3_HASH || error("Recipe mismatch")
    only(rows.status)=="fail" && only(rows.severity)=="hard" || error("Not stale hard failure")
    only(rows.check)=="tape_allocation" || error("Wrong check")
    ismissing(only(rows.run_id)) || error("Unexpected run_id")
    if isempty(ARGS) || only(ARGS)=="archive"
        CSV.write(joinpath(S3_OUT,"archived_harness_check_832.csv"),rows)
        println("ARCHIVED row832; no DB mutation")
    elseif only(ARGS)=="cleanup"
        archived=CSV.read(joinpath(S3_OUT,"archived_harness_check_832.csv"),DataFrame)
        isequal(rows,archived) || error("Archive differs from current row")
        Training.Inference._db_exec(s3conn,"BEGIN;")
        try
            before=Training.Inference._db_rows(s3conn,"SELECT count(*) AS n FROM harness_checks WHERE recipe_hash='$S3_HASH';")
            total_before=Training.Inference._db_rows(s3conn,"SELECT count(*) AS n FROM harness_checks;")
            deleted=Training.Inference._db_rows(s3conn,"DELETE FROM harness_checks WHERE id=832 AND recipe_hash='$S3_HASH' AND status='fail' AND severity='hard' AND run_id IS NULL RETURNING *;")
            isequal(deleted,rows) || error("Delete mismatch")
            after=Training.Inference._db_rows(s3conn,"SELECT count(*) AS n FROM harness_checks WHERE recipe_hash='$S3_HASH';")
            total_after=Training.Inference._db_rows(s3conn,"SELECT count(*) AS n FROM harness_checks;")
            only(before.n)-only(after.n)==1 || error("Recipe count mismatch")
            only(total_before.n)-only(total_after.n)==1 || error("Total count mismatch")
            Training.Inference._db_exec(s3conn,"COMMIT;")
            CSV.write(joinpath(S3_OUT,"stale_check_counts.csv"),DataFrame(recipe_hash=[S3_HASH],deleted_id=[832],before=[only(before.n)],after=[only(after.n)],total_before=[only(total_before.n)],total_after=[only(total_after.n)]))
            println("DELETED only832; recipe count $(only(before.n)) -> $(only(after.n))")
        catch
            Training.Inference._db_exec(s3conn,"ROLLBACK;")
            rethrow()
        end
    else
        error("Expected archive or cleanup")
    end
finally
    close(s3conn)
end
