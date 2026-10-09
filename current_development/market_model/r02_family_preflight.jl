# Full-budget Gamma-only preflight after the first uniform-initialisation failure.
# Not the family comparison; no winner or Phase B completion is asserted here.
# Dedicated beast tmux REPL, -t 16; no database writes.
using BayesianFootball, LinearAlgebra, ThreadPinning, CSV, DataFrames, Serialization
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"l02_two_stage.jl"))
const MB_PF_CONFIG = TwoStageMarket.MM.scottish_lower_2425_2526()
mb_pf_ds = BayesianFootball.Data.load_datastore_cached(MB_PF_CONFIG.segment; max_age_hours=10^6)
mb_pf_panel = TwoStageMarket.phase_b_panel(mb_pf_ds; config=MB_PF_CONFIG).panel
mb_pf_data = TwoStageMarket.family_data(mb_pf_panel)
mb_pf_chain,mb_pf_seconds = TwoStageMarket.fit_family(mb_pf_data,:gamma,trues(length(mb_pf_data.y)); seed=4009)
mb_pf_diag,mb_pf_nuts = TwoStageMarket.family_diagnostics(mb_pf_chain,:gamma,0)
mb_pf_out = joinpath(@__DIR__,"results","B")
serialize(joinpath(mb_pf_out,"gamma_preflight.jls"),mb_pf_chain)
CSV.write(joinpath(mb_pf_out,"gamma_preflight_diagnostics.csv"),mb_pf_diag)
CSV.write(joinpath(mb_pf_out,"gamma_preflight_nuts.csv"),DataFrame([mb_pf_nuts]))
open(joinpath(mb_pf_out,"RUN_PROVENANCE.md"),"a") do io
    println(io,"- Gamma preflight SHA `",strip(read(`git rev-parse HEAD`,String)),"`; wall time ",mb_pf_seconds," seconds; 4 × (1000 warmup + 1000 retained), acceptance 0.8, seed 4009.")
end
show(stdout,MIME"text/plain"(),mb_pf_diag; allrows=true,allcols=true)
println()
mb_pf_nuts.gate_pass || error("Gamma preflight convergence/divergence gate failed")
mb_pf_ll = TwoStageMarket.family_loglik(mb_pf_chain,mb_pf_data,:gamma)
println("Gamma preflight max absolute eta = ",mb_pf_ll.max_abs_eta)
println("R02_FAMILY_PREFLIGHT_DONE")
