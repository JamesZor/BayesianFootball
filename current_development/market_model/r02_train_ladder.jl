# Optional training-only entry point. No inference is promoted here.
# The full scientific notebook is r02_two_stage_ladder.jl.
# Run only in the dedicated beast tmux REPL, -t 16.
using BayesianFootball, LinearAlgebra, ThreadPinning, CSV, DataFrames
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__, "l02_two_stage.jl"))
mm_b_train_ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower(); max_age_hours=10^6)
mm_b_train_panel = TwoStageMarket.phase_b_panel(mm_b_train_ds).panel
mm_b_train_gates = vcat(TwoStageMarket.engine_gates(), TwoStageMarket.rbpf_gate())
all(mm_b_train_gates.pass) || error("Phase B engine gate failed")
mm_b_train_out = joinpath(@__DIR__, "results", "B")
mkpath(mm_b_train_out)
CSV.write(joinpath(mm_b_train_out,"engine_gates.csv"),mm_b_train_gates)
TwoStageMarket.train_ladder(mm_b_train_panel,mm_b_train_out)
println("R02_FITS_DONE")
