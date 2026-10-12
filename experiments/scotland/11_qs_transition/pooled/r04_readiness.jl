# Read-only final readiness receipt; no sampling or registry writes.
using BayesianFootball, CSV, DataFrames, Test
isdefined(Main,:PoolCandidates) || error("run panel setup first")
ready_db=Training.PostgresStorage(PoolCandidates.EXPERIMENT)
ready_hash=Harness.recipe_hash(pool_candidate)
ready_summary=only(eachrow(CSV.read(joinpath(@__DIR__,"smoke_summary_qs_pooled_a2.csv"),DataFrame)))
@test ready_summary.recipe_hash==ready_hash
ready_checks=Harness.read_checks(ready_db;recipe_hash=ready_hash,stage="smoke")
ready_failures=filter(r->r.severity=="hard" && r.status!="pass",ready_checks)
CSV.write(joinpath(@__DIR__,"harness_history_current_qs_pooled_a2.csv"),ready_checks)
CSV.write(joinpath(@__DIR__,"historical_failed_hard_current_qs_pooled_a2.csv"),ready_failures)
ready=Harness.has_passing_smoke(ready_db,pool_candidate)
CSV.write(joinpath(@__DIR__,"readiness.csv"),DataFrame([(
    candidate=pool_candidate.name,recipe_hash=ready_hash,run_id=ready_summary.run_id,
    has_passing_smoke=ready,historical_failed_hard_checks=nrow(ready_failures),
    failed_check_ids=join(ready_failures.id,";"),source_folds="1;60",first_target_season="23/24",
    target_fixtures=1070,src_changed=false,git_sha=Training.git_commit_id())]))
println("POOL_READY hard_smoke=$ready recipe=$ready_hash run=$(ready_summary.run_id)")
