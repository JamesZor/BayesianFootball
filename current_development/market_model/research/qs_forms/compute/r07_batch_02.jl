# Batch02 diagnostics: reuse batch01; no MCMC, SQL, new simulations or package changes.
# R09 scoring partition canceled by manager: C0 common mode has zero loading.
# Run requests separately in owned beast REPL; commit each completed result.
# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using BayesianFootball, LinearAlgebra, ThreadPinning
if !isdefined(@__MODULE__, :QSFormsBatch02)
    include(joinpath(@__DIR__,"q07_batch_02.jl"))
end
# %%
# ===================================================================
# 2. Runtime, immutable inputs and request
# ===================================================================
strip(read(`hostname`,String))=="mcmc-beast" || error("Julia only on beast")
Threads.nthreads()==16 || error("requires16 threads")
pinthreads(:cores); BLAS.set_num_threads(1)
mkpath(QSFormsBatch02.OUT)
for (_,_,segment) in QSFormsBatch02.QS.LEAGUES
    name=string(nameof(typeof(segment)))
    path=joinpath(dirname(dirname(QSFormsBatch02.P)),".cache","datastore_$(name).jls")
    @assert QSFormsBatch02.sha(path)==QSFormsBatch02.QS.SNAPSHOT_HASHES[name]
    QSFormsBatch02.input(path)
end
qsf2_request=get(ENV,"QSF2_REQUEST","R09")
qsf2_commit=strip(read(`git -C $(QSFormsBatch02.P) rev-parse HEAD`,String))
for name in filter(n->startswith(n,"q07_batch_02")||n=="r07_batch_02.jl",readdir(@__DIR__))
    QSFormsBatch02.record!(qsf2_request,"code",joinpath(@__DIR__,name);detail="git=$qsf2_commit; owned pane=%304; seed610802")
end
QSFormsBatch02.flush!()
# %%
# ===================================================================
# 3. Selected diagnostic and sentinel
# ===================================================================
Base.invokelatest(getfield(QSFormsBatch02,Symbol(lowercase(qsf2_request))))
println("R07_BATCH_02_DONE")
