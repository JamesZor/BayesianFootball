# Freeze the market-rate observation table from the pinned wave-1 inversion panel.
# No reference refits, SQL writes, model fitting or scoring.
# Include on mcmc-beast in a fresh persistent 16-thread REPL.

# %% 1. Packages and loader
using BayesianFootball, CSV, SHA, LinearAlgebra, ThreadPinning
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l01_market_table.jl"))

# %% 2. Pinned datastore and unchanged full-book inversion adapter
w2_table_ds = Wave2MarketTable.QSMarketArm.checked_datastore()
w2_table = Wave2MarketTable.prepare(w2_table_ds)

# %% 3. Freeze outside checkout; copy CSV and digest into results/ before tests
const W2_TABLE_OUT = "/root/BF_runs/qs_experiment_w2_out/phase1"
mkpath(W2_TABLE_OUT)
w2_table_path = joinpath(W2_TABLE_OUT, "market_rates.csv")
isfile(w2_table_path) && error("table already frozen; do not overwrite")
CSV.write(w2_table_path, w2_table)
w2_table_digest = bytes2hex(open(SHA.sha256, w2_table_path))
write(joinpath(W2_TABLE_OUT, "market_rates.sha256"), w2_table_digest * "\n")
println("MARKET_TABLE rows=", size(w2_table, 1), " full_book=", count(w2_table.full_book),
    " sha256=", w2_table_digest)
println("PHASE1_TABLE_DONE")
