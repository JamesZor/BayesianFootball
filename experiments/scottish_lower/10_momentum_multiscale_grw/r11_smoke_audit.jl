# Post-sampling audit only. Does not change the smoke certificate or fit anything.
# Small-cohort decompression numbers are descriptive, not benchmark conclusions.
using ThreadPinning, LinearAlgebra, CSV, DataFrames
pinthreads(:cores)
BLAS.set_num_threads(1)
include(joinpath(@__DIR__,"l10_momentum_grw_loader.jl"))
include(joinpath(@__DIR__,"l12_evaluation.jl"))
const M = MomentumGRW
const E = MomentumEvaluation
const C = M.MomentumGRWConfig()
const OUT = joinpath(C.save_root,"smoke",M.source_fingerprint())
summary = CSV.read(joinpath(OUT,"smoke_gates.csv"),DataFrame)
nrow(summary)==3 || error("Smoke is incomplete; expected three recorded arms")
db = M.PostgresStorage(C.smoke_experiment)
ds = M.gph_load_data()
odds = M.gph_betfair_closing_odds(ds)
folds = NamedTuple[]
shape = NamedTuple[]
for row in eachrow(summary)
    fit = M.load_fit(db,M.UUID(row.run_id))
    isempty(fit.diagnostics.abstained) || error("Unmeasured convergence gates: $(fit.diagnostics.abstained)")
    for d in fit.diagnostics.folds
        push!(folds,(;model=row.model,fold=C.smoke_folds[d.fold],
            rhat=d.max_rhat,rhat_parameter=string(d.worst_rhat_param),
            ess_bulk=d.min_ess_bulk,bulk_parameter=string(d.worst_ess_bulk_param),
            ess_tail=d.min_ess_tail,tail_parameter=string(d.worst_ess_tail_param),
            divergences=d.n_divergent,bfmi=d.min_bfmi,treedepth_rate=d.treedepth_rate))
    end
    rates = E.market_reference(odds,fit.latents.match_ids)
    decompression,_,favourites = E.decompression(fit,odds,ds,rates)
    push!(shape,(;model=row.model,converged=row.gate_pass,decompression...))
    CSV.write(joinpath(OUT,row.model*"_smoke_favourites.csv"),favourites)
end
CSV.write(joinpath(OUT,"fold_convergence.csv"),DataFrame(folds))
CSV.write(joinpath(OUT,"smoke_decompression.csv"),DataFrame(shape))
println(M.gph_markdown_table(DataFrame(folds)))
println(M.gph_markdown_table(DataFrame(shape)))

# Persistence size facts, not a guess about a future full-grid artifact.
conn = M.GPH_INF._db_connect(db)
try
    sizes = M.GPH_INF._db_rows(conn,"""
        SELECT r.name,r.run_id,octet_length(a.fit_blob) AS compressed_bytes
        FROM runs r JOIN fit_artifacts a USING(run_id)
        WHERE r.experiment_name=\$1 ORDER BY r.id;
        """,(C.smoke_experiment,))
    CSV.write(joinpath(OUT,"smoke_artifact_sizes.csv"),sizes)
    show(stdout,MIME"text/plain"(),sizes;allcols=true)
finally
    close(conn)
end
