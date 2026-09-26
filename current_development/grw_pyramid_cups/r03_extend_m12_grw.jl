# ==============================================================================
# r03 — Roll the live GRW hybrid (m12_joint_hybrid_synergy_grw, 3a9a4c7e) forward
# ==============================================================================
#
# grw_player_hybrid/r03_extend_2627.jl pins 43 folds and refuses anything else, so
# the 2026/27 roll-forward past fold 43 lives here. Same splitter (canonical 56/57,
# target 24/25 + 25/26 + 26/27), fresh ScottishLower DataStore from SQL, extend_fit
# samples only the folds the run lacks and writes back to the same UUID.
#
#   julia --project -t 16 current_development/grw_pyramid_cups/r03_extend_m12_grw.jl
# ==============================================================================

using ThreadPinning
using LinearAlgebra
pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

using BayesianFootball, DataFrames, Dates, Printf, UUIDs
include(joinpath(@__DIR__, "..", "grw_player_hybrid", "l01_loader.jl"))

const R03_RUN = UUID("3a9a4c7e-378b-45d0-a2d2-c8b69b46786b")
const R03_CFG = GPHConfig()
db = gph_database(R03_CFG.experiment)
ds = Data.load_datastore_cached(Data.ScottishLower(); force = true)
println("DataStore: ", nrow(ds.matches), " matches, latest ", maximum(ds.matches.match_date))
sp = gph_splitter(["24/25", "25/26", "26/27"])
plan = Training.preview_extension(db, string(R03_RUN), ds; splitter = sp)
println("preview: ", plan.new_count, " new fold(s)")
if plan.new_count == 0
    println("R03_DONE nothing to extend")
else
    t0 = time()
    ext = extend_fit(db, string(R03_RUN), ds; splitter = sp, execution = gph_execution(R03_CFG))
    re = load_fit(db, string(R03_RUN))
    d = re.diagnostics
    @printf("extended to %d folds in %.1f min | R̂ %.4f | ESS %.0f | div %d | passed %s\n",
            length(re.folds), (time() - t0) / 60, d.max_rhat, d.min_ess_bulk, d.n_divergent, d.passed)
    println("R03_DONE folds=", length(re.folds))
end
