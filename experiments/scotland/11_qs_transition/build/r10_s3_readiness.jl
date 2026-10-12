# Read-only S3 harness-history audit; never deletes/updates checks or starts a grid.
using BayesianFootball, CSV, DataFrames, LinearAlgebra, ThreadPinning
strip(read(`hostname`,String))=="mcmc-beast" && Threads.nthreads()==16 || error("beast16 only")
pinthreads(:cores)
BLAS.set_num_threads(1)
const R10C=Module(:R10Candidates)
Base.include(R10C,joinpath(@__DIR__,"..","candidates.jl"))
r10db=Training.PostgresStorage(R10C.EXPERIMENT)
r10summary=NamedTuple[]
for candidate in R10C.CANDIDATES
    hash=Harness.recipe_hash(candidate)
    rows=Harness.read_checks(r10db;recipe_hash=hash,stage="smoke")
    failures=filter(r -> r.severity=="hard" && r.status!="pass",rows)
    CSV.write(joinpath(@__DIR__,"harness_history_$(candidate.name).csv"),rows)
    CSV.write(joinpath(@__DIR__,"historical_failed_hard_$(candidate.name).csv"),failures)
    ready=Harness.has_passing_smoke(r10db,candidate)
    push!(r10summary,(;candidate=candidate.name,recipe_hash=hash,has_passing_smoke=ready,
        smoke_check_rows=nrow(rows),historical_failed_hard_checks=nrow(failures),
        failed_check_ids=join(failures.id,";")))
    println("S2_S3_READINESS arm=$(candidate.name) ready=$ready historical_failed_hard=$(nrow(failures))")
end
CSV.write(joinpath(@__DIR__,"s3_readiness.csv"),DataFrame(r10summary))
println("S2_READ_ONLY_S3_HISTORY_AUDIT_DONE")
