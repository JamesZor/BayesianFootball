# Existing-fit diagnostics only: no fitting, SQL, package changes or source output writes.
# Run each request explicitly in the owned persistent beast REPL. Large tables stay there.
# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
using BayesianFootball, LinearAlgebra, ThreadPinning
if !isdefined(@__MODULE__, :QSFormsBatch01)
    include(joinpath(@__DIR__, "q07_batch_01.jl"))
end
const Q07 = QSFormsBatch01

# %%
# ===================================================================
# 2. Runtime and input contract
# ===================================================================
strip(read(`hostname`, String)) == "mcmc-beast" || error("Julia only on mcmc-beast")
Threads.nthreads() == 16 || error("requires 16 threads")
pinthreads(:cores)
BLAS.set_num_threads(1)
mkpath(Q07.OUT)
Q07.record!("shared", "code", "git_commit", "";
    detail=strip(read(`git -C $(Q07.P) rev-parse HEAD`,String)))
for (_,_,segment) in Q07.QS.LEAGUES
    name=string(nameof(typeof(segment)))
    path=joinpath(dirname(dirname(Q07.P)),".cache","datastore_$(name).jls")
    @assert Q07.sha(path)==Q07.QS.SNAPSHOT_HASHES[name]
    Q07.record!("shared","input_cache",name,path;detail="read-only pinned datastore; verified SHA")
end
Q07.flush_manifest!()

# %%
# ===================================================================
# 3. Request selection (call later requests explicitly after their gates)
# ===================================================================
q07_request=get(ENV,"QSF_REQUEST","R01")
q07_log="/root/BF_runs/logs/market_model_qsf/$(lowercase(q07_request)).log"
Q07.record!(q07_request,"code","loader",joinpath(@__DIR__,"q07_batch_01.jl");
    detail="git="*strip(read(`git -C $(Q07.P) rev-parse HEAD`,String))*"; ENV[QSF_REQUEST]=$q07_request; include(r07_batch_01.jl); owned pane=%304; threads16; BLAS1")
Q07.flush_manifest!()
open(q07_log,"w") do io
    println(io,"RUN $q07_request via r07_batch_01.jl; 16 threads, BLAS1")
    redirect_stdout(io) do
        redirect_stderr(io) do
            try
                getfield(Q07,Symbol(lowercase(q07_request)))()
            catch err
                showerror(io,err,catch_backtrace());println(io);flush(io)
                rethrow()
            end
        end
    end
    println(io,"R07_BATCH_01_DONE");flush(io)
end
println("R07_BATCH_01_DONE")
