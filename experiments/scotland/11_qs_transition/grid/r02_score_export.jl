# W2 score_runs.jl recipe, with committed CSV evidence (B=10000 default).
using BayesianFootball, CSV, DataFrames
include("/root/BF_runs/qs_transition/scripts/score_runs.jl")
s3path=joinpath(@__DIR__,"runs.csv")
s3groups=load_run_groups(s3path)
s3refs=unique(reduce(vcat,(g.refs for g in s3groups)))
include_run_loaders(s3refs)
s3ds=Data.load_datastore_cached(Data.ScottishPyramid();max_age_hours=10000)
s3result=Base.invokelatest(score_csv_groups,s3groups,s3ds;target_seasons=["23/24","24/25","25/26"],expected_fixtures=1070)
isempty(s3result.failures) || error(repr(s3result.failures))
s3store=Training.PostgresStorage("harness")
Harness.write_scores!(s3store,s3result.scores)
for ref in s3refs
    CSV.write(joinpath(@__DIR__,"scores_$(ref.label).csv"),filter(r->string(r.run_id)==string(ref.run_id),s3result.scores))
end
s3c=Training.Inference._db_connect(s3store)
try
    for ref in s3refs
        uuid=string(ref.run_id)
        folds=Training.Inference._db_rows(s3c,"SELECT * FROM fold_results WHERE run_id='$uuid' ORDER BY fold_idx;")
        expected_folds=ref.label=="qs_weekly" ? 115 : 60
        nrow(folds)==expected_folds || error("Expected$expected_folds folds for $(ref.label)")
        CSV.write(joinpath(@__DIR__,"fold_diagnostics_$(ref.label).csv"),folds)
        checks=Harness.read_checks(s3store;run_id=ref.run_id)
        CSV.write(joinpath(@__DIR__,"checks_$(ref.label).csv"),checks)
        bad=filter(r->coalesce(r.r_hat_max>1.05,false) || r.divergences/4000>0.001,folds)
        println("S3_EXPORT $(ref.label) run=$uuid folds=$(nrow(folds)) rerun_folds=$(join(bad.fold_idx,';'))")
    end
finally
    close(s3c)
end
println("S3_W2_SCORE_EXPORT_DONE rows=$(nrow(s3result.scores))")
