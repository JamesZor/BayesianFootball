# Repair diagnostic manifests only; previously scored outputs are not recomputed.
include(joinpath(@__DIR__,"l01_manifests.jl"))
s1audit("cutoffs") do conn
    runs=CSV.read(joinpath(S1OUT,"resolved_runs.csv"),DataFrame)
    labels=Dict("0dd74a52"=>"td_lower_base","6c0aa3ea"=>"td_lower_a2full_carry_jump","ce7ea22f"=>"grw_base","c4a0fa94"=>"grw_step_a2_carry_jump","973baafa"=>"m12_td_ref","81fdf817"=>"qs_marketobs")
    for r in eachrow(runs)
        label=labels[first(r.run_id,8)]
        fit=Training.load_fit(Training.PostgresStorage(r.experiment_name),UUID(r.run_id))
        dbfolds=CSV.read(joinpath(S1OUT,"dbfolds_"*label*".csv"),DataFrame)
        lat=CSV.read(joinpath(S1OUT,"rates_"*label*".csv"),DataFrame)
        s1manifests(fit,label,dbfolds,lat)
        println("CUTOFF_MANIFEST ",label," folds=",length(fit.folds))
        fit=nothing;GC.gc()
    end
end
