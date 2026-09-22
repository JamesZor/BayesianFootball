# Verify user-approved one-in-four persistence on existing smoke draws. No MCMC.
using ThreadPinning,LinearAlgebra
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"l10_momentum_grw_loader.jl"))
const M = MomentumGRW
const C = M.MomentumGRWConfig()
certificate = M.Serialization.deserialize(joinpath(C.save_root,"smoke_gate.jls"))
certificate.passed && certificate.source==M.source_fingerprint() || error("Smoke gate not current")
ds = M.gph_load_data()
splitter = M.gph_splitter(C.target_seasons)
db = M.PostgresStorage(C.smoke_experiment)
for (name,model) in M.models()
    fit = M.load_fit(db,M.UUID(certificate.runs[name]))
    inputs,_ = M.selected_inputs(ds,splitter,model,C;smoke=true)
    stored = M.gph_thin_for_persistence(fit,inputs,4)
    @assert stored.diagnostics===fit.diagnostics
    @assert stored.latents.λ_home==fit.latents.λ_home[:,1:4:end]
    @assert stored.latents.λ_away==fit.latents.λ_away[:,1:4:end]
    M.score_grid_audit(stored)
    bytes = length(M.GPH_INF._db_artifact_blob(stored))
    println("STORAGE PASS ",name," retained=",M.n_draws(fit.latents),
        " persisted=",M.n_draws(stored.latents)," smoke_compressed_bytes=",bytes)
end
