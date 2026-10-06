# Revision 8 permits C1 production sampling in parallel with synthetic recovery.
# This runner NEVER promotes or interprets C1; recovery is still pending.
# One protocol per own session, frozen priors/seeds/4x(2000+3000), thin1.
# Full books only. 10b fits ONLY honest_train. No inference or outcome scoring.
# The saved draws must later pass BOTH recovery and production convergence gates
# before a separate accepted-inference workflow may use them.
# Fresh output only; no resume/overwrite; load l05_pooled before deserializing.
# Persistent OWNED beast REPL, 16 threads/core pinning/BLAS1; no one-shot Julia.

# %%
# ===================================================================
# 1. Packages and frozen implementation
# ===================================================================
using BayesianFootball, LinearAlgebra, CSV, DataFrames, SHA, TOML, ThreadPinning, Serialization
if !isdefined(@__MODULE__, :PooledMarket)
    include(joinpath(@__DIR__,"l05_pooled.jl"))
end
const C08C = PooledMarket

# %%
# ===================================================================
# 2. Protocol, immutable output and runtime
# ===================================================================
const C08C_CONFIG = C08C.MM.scottish_lower_2425_2526()
const C08C_SEEDS = TOML.parsefile(joinpath(@__DIR__,"fullbook_seeds.toml"))
const C08C_PROTOCOL = get(ENV,"C08_C1_PROTOCOL","")
C08C_PROTOCOL in ("10a","10b") || error("C08_C1_PROTOCOL must be 10a or 10b")
const C08C_OUT = get(ENV,"C08_C1_OUT",joinpath(@__DIR__,"results","C","v8_c1_$(C08C_PROTOCOL)"))
strip(read(`hostname`,String)) == "mcmc-beast" || error("sampling requires mcmc-beast")
Threads.nthreads() == 16 || error("C1 requires 16 Julia threads")
pinthreads(:cores)
BLAS.set_num_threads(1)
isdir(C08C_OUT) && !isempty(readdir(C08C_OUT)) && error("output must be fresh")
mkpath(C08C_OUT)
write(joinpath(C08C_OUT,"PROMOTION_PENDING.txt"),"C1 production is NOT promoted. Require accepted frozen synthetic recovery plus all production convergence gates before inference.\n")

# %%
# ===================================================================
# 3. Pinned snapshot and prescribed training restriction
# ===================================================================
c08c_cache = joinpath(pwd(),".cache","datastore_ScottishLower.jls")
open(io -> bytes2hex(SHA.sha256(io)),c08c_cache) ==
    "c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4" || error("wrong pinned cache")
c08c_ds = BayesianFootball.Data.load_datastore_cached(C08C_CONFIG.segment; max_age_hours=10^6)
c08c_full = C08C.CM.TB.phase_b_panel(c08c_ds; config=C08C_CONFIG).panel
c08c_books = C08C.fullbook_markets(c08c_ds,c08c_full,C08C_CONFIG)
c08c_panel,c08c_markets = C08C_PROTOCOL == "10a" ? (c08c_full,c08c_books) :
    C08C.restrict_fullbook(c08c_full,c08c_books,C08C_CONFIG.honest_train)
println("C1 $(C08C_PROTOCOL) UNPROMOTED population=$(length(c08c_markets)); chains=$(C08C_SEEDS["C1"][C08C_PROTOCOL]); 2000+3000, thin1")

# %%
# ===================================================================
# 4. Regenerated full-book Gate1 / state gates before sampling
# ===================================================================
C08C.reset_newton_accounting!()
try
    global c08c_gate = C08C.laplace_gate(c08c_ds,C08C_CONFIG,C08C_OUT; kinds=("full",))
    all(c08c_gate.summary.gate_pass) || error("full-book Gate1 failed")
    global c08c_engine = C08C.fullbook_engine_gates()
    CSV.write(joinpath(C08C_OUT,"engine_gates_c.csv"),c08c_engine)
    all(c08c_engine.pass) || error("engine gates failed")
finally
    C08C.write_newton_accounting(C08C_OUT; run="C1_pending_preflight")
end

# %%
# ===================================================================
# 5. Prescribed sampling, immutable draws, convergence (NOT promotion)
# ===================================================================
c08c_result = C08C.fit_fullbook(C08C.FullBookRung(:C1),c08c_panel;
    markets=c08c_markets,seeds=C08C_SEEDS["C1"][C08C_PROTOCOL],
    accounting_out=C08C_OUT,accounting_run="C1_$(C08C_PROTOCOL)_unpromoted")
Serialization.serialize(joinpath(C08C_OUT,"C1_$(C08C_PROTOCOL).jls"),c08c_result)
c08c_diag = C08C.fullbook_diagnostics(c08c_result.fit; protocol=C08C_PROTOCOL,
    seed=first(C08C_SEEDS["C1"][C08C_PROTOCOL]))
c08c_diag.sha .= strip(read(`git rev-parse HEAD`,String))
c08c_diag.chain_seeds .= join(C08C_SEEDS["C1"][C08C_PROTOCOL],";")
CSV.write(joinpath(C08C_OUT,"convergence_c.csv"),c08c_diag)
all(c08c_diag.gate_pass) || error("C1 production convergence failed; no promotion")
println("R05_C1_PENDING_DRAWS_DONE") # never R05_DONE: recovery and interpretation still gated
