# %% 1. Packages / fixed runtime
using BayesianFootball, LinearAlgebra, ThreadPinning
strip(read(`hostname`,String))=="mcmc-beast" && Threads.nthreads()==4 || error("pool/beast4 only")
pinthreads(:cores); BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"l02_recovery.jl"))

# %% 2. Candidate and independent synthetic data / training / recovery gates
isdefined(Main,:PoolCandidates) || error("run r01_panel.jl first")
PooledRecovery.run(pool_candidate,@__DIR__)
