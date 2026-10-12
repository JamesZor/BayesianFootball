# One deterministic rerun per offending fold; preserve original run and sampler settings.
using BayesianFootball, CSV, DataFrames, Random, LinearAlgebra, ThreadPinning
pinthreads(:cores); BLAS.set_num_threads(1)
const R3C=Module(:S3RerunCandidates)
Base.include(R3C,joinpath(@__DIR__,"..","candidates.jl"))
arm=only(ARGS)
runrows=CSV.read(joinpath(@__DIR__,"runs.csv"),DataFrame)
original_id=only(filter(r->r.label==arm,runrows).run_id)
candidate=only(filter(c->c.name==arm,R3C.CANDIDATES))
db=Training.PostgresStorage(R3C.EXPERIMENT)
original=Training.load_fit(db,original_id)
ds=Data.load_datastore_cached(Data.ScottishPyramid();max_age_hours=10000)
inputs=Harness._fold_inputs(candidate,ds;stage=:grid)
cfg=Harness.fit_config(candidate;stage=:grid,experiment=R3C.EXPERIMENT)
diag=CSV.read(joinpath(@__DIR__,"fold_diagnostics_$(arm).csv"),DataFrame)
bad=filter(r->coalesce(r.r_hat_max>1.05,false)||r.divergences/4000>0.001,diag).fold_idx
isempty(bad) && error("No offending folds")
chains=[f.chain for f in original.folds]
rerun_records=NamedTuple[]
for fold in bad
    seed=2026101200+fold
    Random.seed!(seed)
    result=Training.Inference.run_folds(cfg.model,cfg.sampler,cfg.execution,inputs.feature_sets[[fold]])
    length(result)==1 && result[1]!==nothing || error("Rerun failed fold$fold")
    chains[fold]=result[1]
    audit=Training.audit_fold(Training.FoldFit(fold,result[1],inputs.feature_sets[fold][2]);max_depth=cfg.sampler.max_depth)
    push!(rerun_records,(;arm,original_run_id=original_id,fold,seed,max_rhat=audit.max_rhat,min_ess_bulk=audit.min_ess_bulk,min_ess_tail=audit.min_ess_tail,divergences=audit.n_divergent))
    CSV.write(joinpath(@__DIR__,"reruns_$(arm).csv"),DataFrame(rerun_records))
    println("S3_SEEDED_RERUN fold=$fold seed=$seed Rhat=$(audit.max_rhat) div=$(audit.n_divergent)")
end
replay_cfg=Training.FitConfig(name=cfg.name,model=cfg.model,splitter=cfg.splitter,sampler=Training.ReplaySampler(chains),execution=Training.SequentialExecution(),tags=cfg.tags,description=cfg.description,save_dir=cfg.save_dir)
replay=Training.fit_model(replay_cfg;feature_sets=inputs.feature_sets,oos_fixtures=inputs.oos,quiet=false)
final_cfg=Training.FitConfig(name=cfg.name,model=cfg.model,splitter=cfg.splitter,sampler=cfg.sampler,execution=cfg.execution,tags=vcat(cfg.tags,["seeded_rerun_of:$original_id","seeded_rerun_folds:$(join(bad,','))"]),description=cfg.description,save_dir=cfg.save_dir)
final=Training.Fit(final_cfg,replay.folds,replay.latents,replay.diagnostics,replay.metadata,replay.save_path)
Harness._latent_audit(final;require_variance=true)
Harness._target_coverage(candidate,final,inputs;expected_folds=60,expected_target=1070)
new_id=Training.save_fit(final,db;on_duplicate=:error)
Harness._fit_parity(final,Training.load_fit(db,new_id))
CSV.write(joinpath(@__DIR__,"rerun_run_$(arm).csv"),DataFrame(arm=[arm],original_run_id=[original_id],run_id=[string(new_id)],folds=[join(bad,';')]))
println("S3_SEEDED_RERUN_DONE arm=$arm original=$original_id final=$new_id")
