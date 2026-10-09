using BayesianFootball, CSV, DataFrames, LinearAlgebra, ThreadPinning
pinthreads(:cores)
BLAS.set_num_threads(1)
const EC = Module(:Wave2EligibilityCandidates)
Base.include(EC, "/root/BF_runs/qs_experiment/experiments/scotland/06_qs_joint_and_market_observation/candidates.jl")
const EH = BayesianFootball.Harness
const ED = Training.PostgresStorage("scottish_lower_qs_wave2_2426")
rows = [(; candidate = c.name, recipe_hash = EH.recipe_hash(c), eligible = EH.has_passing_smoke(ED, c)) for c in EC.CANDIDATES]
CSV.write("/root/BF_runs/qs_experiment_w2_out/phase2_arrays/smoke_eligibility.csv", DataFrame(rows))
@assert all(row -> row.eligible, rows)
println("SMOKE_ELIGIBILITY_READONLY_PASS: all four recipes; no sampling or database writes")
