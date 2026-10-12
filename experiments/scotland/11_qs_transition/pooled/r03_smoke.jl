# %% 1. Packages / fixed runtime
using BayesianFootball, CSV, DataFrames, LinearAlgebra, ThreadPinning, Test
strip(read(`hostname`,String))=="mcmc-beast" && Threads.nthreads()==4 || error("pool/beast4 only")
pinthreads(:cores); BLAS.set_num_threads(1)
isdefined(Main,:PoolCandidates) || error("run r01_panel.jl first")
@test CSV.read(joinpath(@__DIR__,"recovery_gate.csv"),DataFrame).passed==[true,true]
include(joinpath(@__DIR__,"l03_smoke.jl"))

# %% 2. Normal harness registration, opener/final fold smoke and hard checks
pool_smoke=PooledSmoke.run(pool_candidate,pool_raw,PoolCandidates.EXPERIMENT;output=@__DIR__)

# %% 3. Explicit sixteen-thread estimate (not a grid run)
# Existing smoke helper projects fixed4-thread throughput. Divide by4 ONLY as ideal scaling.
CSV.write(joinpath(@__DIR__,"timing_projection.csv"),DataFrame([(
    measured_threads=4,estimated_threads=16,folds=60,chains_per_fold=4,warmup=500,samples=1000,
    probe_window_seconds=pool_smoke.probe_window_seconds,
    four_thread_grid_seconds=pool_smoke.projected_grid_seconds,
    ideal_sixteen_thread_grid_seconds=pool_smoke.projected_grid_seconds/4,
    assumption="Ideal 4x throughput from4 to16 threads; excludes overhead; shared S3 load may inflate probe")]))
println("POOL_SMOKE_PASS ",repr(pool_smoke))
