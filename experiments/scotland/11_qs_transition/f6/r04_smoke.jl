# Normal two-fold harness smoke, four threads on the shared beast. No grid.
# %% 1. Runtime
using BayesianFootball, LinearAlgebra, ThreadPinning, Serialization, CSV, DataFrames
strip(read(`hostname`,String)) == "mcmc-beast" && Threads.nthreads()==4 || error("beast4 only")
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"l04_smoke.jl"))
const F6SmokeCandidates = Module(:F6SmokeCandidates)
Base.include(F6SmokeCandidates,joinpath(@__DIR__,"..","candidates.jl"))

# %% 2. Frozen inputs / real non-empty fold count
f6ds = deserialize(joinpath(Base.pkgdir(BayesianFootball),".cache","datastore_ScottishPyramid.jls"))
f6candidate = only(filter(c -> c.name=="qs_weekly", F6SmokeCandidates.CANDIDATES))
f6panel = Data.apply_scope(f6ds, f6candidate.scope)
f6splitter = Data.ScopedWalkForwardCV(f6candidate.scope)
f6boundaries = Data.create_id_boundaries(f6panel,f6splitter)
f6expected = CSV.read(joinpath(@__DIR__,"weekly_panel.csv"),DataFrame).folds[1]
length(f6boundaries)==f6expected || error("weekly fold count changed")

# %% 3. Training / hard checks / diagnostics / persistence
F6Smoke.run(f6candidate,f6ds,F6SmokeCandidates.EXPERIMENT; folds=length(f6boundaries))
