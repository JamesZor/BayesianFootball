# Two-fold normal harness smoke, with the 23/24 opening fold included.
# %% 1. Packages / runtime
using BayesianFootball, LinearAlgebra, ThreadPinning, Serialization, CSV, DataFrames
strip(read(`hostname`,String)) == "mcmc-beast" && Threads.nthreads()==16 || error("beast16 only")
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"l02_smoke.jl"))
const S2C = Module(:S2Candidates)
Base.include(S2C,joinpath(@__DIR__,"..","candidates.jl"))

# %% 2. Frozen W2 data and panel verification
s2ds = Serialization.deserialize("/root/BF_runs/a76a65df/.cache/datastore_ScottishPyramid.jls")
s2panel = Data.apply_scope(s2ds,S2C.SCOPE)
s2splitter = Data.ScopedWalkForwardCV(S2C.SCOPE)
s2boundaries = Data.create_id_boundaries(s2panel,s2splitter)
length(s2boundaries)==60 || error("W2 fold count changed")
s2oos_ids = vcat([Int.(Data.get_next_matches(s2panel,b,s2splitter).match_id) for b in s2boundaries]...)
length(s2oos_ids)==length(unique(s2oos_ids))==1070 || error("W2 target fixture count changed")
CSV.write(joinpath(@__DIR__,"panel.csv"),DataFrame([(; folds=60,fixtures=1070,
    first_season=last(first(s2boundaries)).target_season,last_season=last(last(s2boundaries)).target_season)]))

# %% 3. Model / training / gates / diagnostics / persistence
s2only = get(ENV,"S2_ONLY","qs_control")
s2candidate = only(filter(c -> c.name==s2only, S2C.CANDIDATES))
S2Smoke.run(s2candidate,s2ds,S2C.EXPERIMENT)
