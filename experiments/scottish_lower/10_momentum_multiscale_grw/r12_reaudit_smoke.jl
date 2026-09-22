# Correct the v1 smoke adapter's impossible strict '< 0' divergence threshold.
# NO SAMPLING. Re-audit exact immutable v1 chains; preserve original runs and record
# their UUIDs beside new audited artifacts. Zero divergences remains mandatory.
using ThreadPinning, LinearAlgebra, CSV, DataFrames, Dates
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"l10_momentum_grw_loader.jl"))
const M = MomentumGRW
const C = M.MomentumGRWConfig()
const R = M.runtime_config(C;smoke=true)
const SOURCE = M.source_fingerprint()
const OUT = joinpath(C.save_root,"smoke",SOURCE)
const ORIGINAL_SOURCE = "6eeb24377db63becfca8d30245846b13cecff8f9c1fca3c0fe5b54a8bc0e41f2"
const ORIGINALS = Dict(
    "m01_poisson_time_decay"=>M.UUID("f12080da-0f12-4da6-9f6d-43a00647a350"),
    "m02_poisson_grw_1st_order"=>M.UUID("0a6d1038-3d1e-402b-b681-368f6ba84a7b"),
    "m03_poisson_momentum_grw"=>M.UUID("ecc30d2f-a60c-4cf2-a0b5-9ea6057f1b5f"))
mkpath(OUT)
db = M.PostgresStorage(C.smoke_experiment)
ds = M.gph_load_data()
odds = M.gph_betfair_closing_odds(ds)
splitter = M.gph_splitter(C.target_seasons)
rows = NamedTuple[]
posteriors = NamedTuple[]
original_fits = Dict(name=>M.load_fit(db,id) for (name,id) in ORIGINALS)
panel,refusals = M.tradeable_panel(original_fits,odds,ds)
CSV.write(joinpath(OUT,"portfolio_refusals.csv"),refusals)
CSV.write(joinpath(OUT,"portfolio_panel.csv"),DataFrame(match_id=panel))
for (name,model) in M.models()
    original = original_fits[name]
    occursin(ORIGINAL_SOURCE,original.config.description) || error("Wrong original source")
    config = M.fit_recipe(C,name,model,splitter,R;smoke=true)
    string(config.model)==string(original.config.model) || error("Model changed; cannot reuse chains")
    string(config.sampler)==string(original.config.sampler) || error("Sampler changed; cannot reuse chains")
    original.diagnostics.n_divergent==0 || error("Nonzero divergences: cannot promote")
    diagnostics = M.GPH_INF.audit_convergence(original.folds;
        thresholds=M.gph_thresholds(R),max_depth=R.max_depth)
    fit = M.Fit(config,original.folds,original.latents,diagnostics,original.metadata,original.save_path)
    M.convergence_pass(fit,3) || error("$name fails corrected convergence audit")
    M.gph_latent_audit(fit)
    grid = M.score_grid_audit(fit)
    M.register_recipe!(db,name,config)
    existing = M.gph_completed_run(db,config)
    run_id = existing===nothing ? M.gph_save_and_verify(db,fit) : existing
    restored = M.load_fit(db,run_id)
    for (a,b) in zip(original.folds,restored.folds)
        parent(a.chain.value)==parent(b.chain.value) || error("Re-audit changed posterior draws")
    end
    portfolio_id,_ = M.portfolio_roundtrip(db,run_id,restored,ds,odds;panel)
    row = M.gph_convergence_row(name,restored,R;run_id)
    push!(rows,(;row...,grid...,gate_pass=true,portfolio_id=string(portfolio_id),
        source=SOURCE,original_run_id=string(ORIGINALS[name])))
    append!(posteriors,M.momentum_posterior(restored,C.smoke_folds))
    CSV.write(joinpath(OUT,"smoke_gates.csv"),DataFrame(rows))
    println("REAUDIT PASS ",last(rows))
end
CSV.write(joinpath(OUT,"momentum_posterior.csv"),DataFrame(posteriors))
certificate = (;source=SOURCE,passed=true,generated=now(),
    runs=Dict(r.model=>r.run_id for r in rows),report=joinpath(OUT,"smoke_gates.csv"),
    original_runs=ORIGINALS,reason="Correct strict zero-divergence threshold adapter; no resampling")
M.Serialization.serialize(joinpath(C.save_root,"smoke_gate.jls"),certificate)
println("SMOKE_VERDICT PASS — unchanged chains, corrected audit, all persistence/portfolio round-trips")
