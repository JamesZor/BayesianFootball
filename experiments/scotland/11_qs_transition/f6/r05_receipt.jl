# Read-only final smoke/DB gate receipt, including the brief's stricter exactly-0-B check.
using BayesianFootball, CSV, DataFrames
const F6ReceiptCandidates = Module(:F6ReceiptCandidates)
Base.include(F6ReceiptCandidates, joinpath(@__DIR__, "..", "candidates.jl"))
candidate = only(filter(c->c.name=="qs_weekly", F6ReceiptCandidates.CANDIDATES))
checks = CSV.read(joinpath(@__DIR__,"smoke_checks_qs_weekly.csv"), DataFrame)
summary = only(eachrow(CSV.read(joinpath(@__DIR__,"smoke_summary_qs_weekly.csv"),DataFrame)))
all(row.status=="pass" for row in eachrow(checks) if row.severity=="hard") || error("hard smoke failure")
issubset(Set(Harness.SMOKE_REQUIRED_CHECKS), Set(checks.check[checks.status.=="pass"])) || error("missing required check")
tape = only(eachrow(checks[checks.check.=="tape_allocation",:]))
allocation = match(r"max_allocated_bytes\s*=\s*(\d+)",tape.value)
allocation === nothing && error("tape result cannot be read")
bytes = parse(Int,only(allocation.captures))
bytes == 0 || error("brief requires exactly 0 B, not just harness allowance")
summary.recipe_hash == Harness.recipe_hash(candidate) || error("recipe changed after smoke")
all(==(summary.run_id), checks.run_id) || error("run UUID mismatch")
db = Training.PostgresStorage(F6ReceiptCandidates.EXPERIMENT)
Harness.has_passing_smoke(db,candidate) || error("DB does not accept smoke for this recipe")
CSV.write(joinpath(@__DIR__,"smoke_gate.csv"),DataFrame([(; pass=true,
    run_id=summary.run_id, recipe_hash=summary.recipe_hash, hard_checks=count(==("hard"),checks.severity),
    tape_bytes=bytes, db_has_passing_smoke=true)]))
println("F6_SMOKE_GATE_PASS uuid=$(summary.run_id) tape_bytes=$bytes DB_ready=true")
