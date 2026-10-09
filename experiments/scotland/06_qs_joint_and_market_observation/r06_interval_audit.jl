# Additional circular sensitivity for BTTS/all and goal channels, using the exact
# wave-1 bc_bootstrap function already loaded by r06_score. No fitting or DB writes.
using CSV, DataFrames, Statistics
ia_rows = NamedTuple[]
for (tier, a, b) in s5_pairs_spec, family in ("BTTS", "all")
    fa, fb = s5_ll(s5_frames[a], family), s5_ll(s5_frames[b], family)
    j = innerjoin(select(fa, :match_id, :selection, :ll_model => :la),
                  select(fb, :match_id, :selection, :ll_model => :lb); on=[:match_id, :selection])
    nrow(j) == nrow(fa) == nrow(fb) || error("observation set mismatch")
    j.d = j.la .- j.lb
    pf = combine(groupby(j, :match_id), :d => sum => :d, nrow => :n)
    nc = bc_bootstrap(pf.match_id, pf.d, pf.n; circular=false)
    cc = bc_bootstrap(pf.match_id, pf.d, pf.n; circular=true)
    push!(ia_rows, (; tier, arm=a, reference=b, market=family, point=mean(j.d),
        noncircular_mean=nc.mean, noncircular_lo90=nc.lo, noncircular_hi90=nc.hi,
        noncircular_class=classify(nc.lo, nc.hi), circular_mean=cc.mean,
        circular_lo90=cc.lo, circular_hi90=cc.hi, circular_class=classify(cc.lo, cc.hi)))
end
ia_blocks = vcat(bc, DataFrame(ia_rows))
ia_blocks.circular_contradicts = ia_blocks.noncircular_class .!= ia_blocks.circular_class
CSV.write(joinpath(S5_OUT, "block_bootstrap_all_markets.csv"), ia_blocks)

ia_goals = CSV.read(joinpath(S5_OUT, "paired_goal_logscore.csv"), DataFrame)
ia_circ = NamedTuple[]
for r in eachrow(ia_goals)
    j = innerjoin(filter(:arm => ==(r.arm), s5_goal),
                  filter(:arm => ==(r.reference), s5_goal); on=:match_id, makeunique=true)
    ch = Symbol(r.channel)
    d = -(j[!, ch] .- j[!, Symbol(ch, "_1")])
    nc = bc_bootstrap(j.match_id, d, ones(length(d)); circular=false)
    nc.lo == r.block8_lo90 && nc.hi == r.block8_hi90 || error("goal block parity")
    cc = bc_bootstrap(j.match_id, d, ones(length(d)); circular=true)
    push!(ia_circ, (; tier=r.tier, arm=r.arm, reference=r.reference, channel=r.channel,
        point=r.delta_neg_logscore, noncircular_mean=nc.mean,
        circular_mean=cc.mean, circular_lo90=cc.lo, circular_hi90=cc.hi,
        circular_class=classify(cc.lo, cc.hi),
        circular_contradicts=classify(cc.lo, cc.hi) != r.class_block8))
end
CSV.write(joinpath(S5_OUT, "goal_circular_sensitivity.csv"), DataFrame(ia_circ))
# Freeze full observations to make every pair independently auditable.
CSV.write(joinpath(S5_OUT, "observation_frames.csv"), vcat(values(s5_frames)...))
include(joinpath(@__DIR__, "r06_figures.jl"))
println("PHASE4_INTERVAL_AUDIT_PASS market_pairs=$(nrow(ia_blocks)) goal_pairs=$(nrow(ia_goals))")
