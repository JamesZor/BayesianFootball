# One-off manual cleanup AUTHORISED by user after full CSV backup. Never auto-run.
# Only stale null-run hard failure890 for this exact pooled recipe is eligible.
using BayesianFootball, CSV, DataFrames, LibPQ, SHA, Test
const CLEAN_HASH="0db99f621d0e430127f62404c530f07bdbdb22a459f8a345856193e0e09df887"
const CLEAN_RUN="5cb867cd-f6b8-43ef-a4d7-3d5da9c47ca2"
clean_db=Training.PostgresStorage("scottish_lower_qs_transition_2326")
clean_rows=Harness.read_checks(clean_db;recipe_hash=CLEAN_HASH,stage="smoke")
clean_old=filter(r->r.id==890,clean_rows)
@test nrow(clean_old)==1
@test ismissing(only(clean_old.run_id))
clean_latest=filter(r->!ismissing(r.run_id) && string(r.run_id)==CLEAN_RUN,clean_rows)
clean_hard=filter(r->r.severity=="hard",clean_latest)
@test all(r.status=="pass" for r in eachrow(clean_hard))
@test issubset(Set(Harness.SMOKE_REQUIRED_CHECKS),Set(clean_hard.check))
clean_archive=joinpath(@__DIR__,"authorised_archive_check890.csv")
CSV.write(clean_archive,clean_old)
clean_sha=bytes2hex(open(SHA.sha256,clean_archive))
clean_conn=Training.Inference._db_connect(clean_db)
try
    LibPQ.execute(clean_conn,"BEGIN")
    clean_deleted=Training.Inference._db_rows(clean_conn,"""
        DELETE FROM harness_checks
        WHERE id=890 AND run_id IS NULL AND recipe_hash=\$1
          AND experiment='scottish_lower_qs_transition_2326'
          AND candidate='qs_pooled_a2' AND stage='smoke'
          AND "check"='tape_allocation' AND severity='hard' AND status='fail'
          AND detail=\$2
        RETURNING id;
        """,(CLEAN_HASH,only(clean_old.detail)))
    @test nrow(clean_deleted)==1 && only(clean_deleted.id)==890
    LibPQ.execute(clean_conn,"COMMIT")
    CSV.write(joinpath(@__DIR__,"authorised_cleanup_receipt.csv"),DataFrame([(
        deleted_id=890,deleted_rows=nrow(clean_deleted),recipe_hash=CLEAN_HASH,
        latest_passing_run=CLEAN_RUN,archive_sha256=clean_sha,authorisation="ask_user_question: delete only backed-up row890")]))
    println("POOL_AUTHORISED_CLEANUP deleted_only=890 archive_sha256=$clean_sha")
catch
    LibPQ.execute(clean_conn,"ROLLBACK")
    rethrow()
finally
    close(clean_conn)
end
@test Harness.has_passing_smoke(clean_db,CLEAN_HASH)
