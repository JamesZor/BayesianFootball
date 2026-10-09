# Add the missing harness fixture-clustered 95% intervals to the saved goal-score pairs.
# Uses frozen per-fixture scores only; the block intervals and classifications stay unchanged.
# Include in an owned beast REPL; no sampling or database writes.
using BayesianFootball, CSV, DataFrames
strip(read(`hostname`, String)) == "mcmc-beast" || error("goal intervals only on mcmc-beast")
gc_out = joinpath(get(ENV, "QSX_OUT", "/root/BF_runs/qs_experiment_out"), "phase5")
gc_fixtures = CSV.read(joinpath(gc_out, "goal_logscore_fixtures.csv"), DataFrame)
gc_pairs = CSV.read(joinpath(gc_out, "paired_goal_logscore.csv"), DataFrame)
gc_lo, gc_hi = Float64[], Float64[]
for r in eachrow(gc_pairs)
    channel = Symbol(r.channel)
    a = filter(:arm => ==(r.arm), gc_fixtures)
    b = filter(:arm => ==(r.reference), gc_fixtures)
    Set(a.match_id) == Set(b.match_id) && nrow(a) == nrow(b) == 710 ||
        error("goal-score fixture panel mismatch")
    fa = DataFrame(match_id = a.match_id, selection = fill(channel, nrow(a)),
                   ll_model = -a[!, channel])
    fb = DataFrame(match_id = b.match_id, selection = fill(channel, nrow(b)),
                   ll_model = -b[!, channel])
    paired = Harness._paired_bootstrap(fa, fb; B = 10_000)
    isapprox(paired.delta, r.delta_neg_logscore; atol = 1e-12, rtol = 0) ||
        error("goal-score point estimate mismatch")
    push!(gc_lo, paired.lo)
    push!(gc_hi, paired.hi)
end
gc_pairs.clustered_lo95 = gc_lo
gc_pairs.clustered_hi95 = gc_hi
CSV.write(joinpath(gc_out, "paired_goal_logscore.csv"), gc_pairs)
println("GOAL_CLUSTER_DONE pairs=$(nrow(gc_pairs)) fixtures=710 all_point_estimates_match=true")
