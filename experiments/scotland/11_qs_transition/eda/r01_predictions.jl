# Existing runs only: harness read-side scoring, no persistence.
if !isdefined(Main, :s1audit)
    include(joinpath(@__DIR__, "r00_inventory.jl"))
end
using UUIDs, Statistics
include(joinpath(@__DIR__,"l01_manifests.jl"))
const PG = BayesianFootball.Models.PreGame
# Same read-side legacy compatibility as scripts/score_runs.jl.
function Serialization.deserialize(s::Serialization.AbstractSerializer, T::Type{<:PG.JointGammaPoissonObservation})
    T isa DataType && return invoke(Serialization.deserialize, Tuple{Serialization.AbstractSerializer,DataType}, s, T)
    fields = Any[]
    for _ in 1:3
        tag = Int32(read(s.io, UInt8)::UInt8)
        push!(fields, Serialization.handle_deserialize(s, tag))
    end
    return PG.JointGammaPoissonObservation(fields[1], fields[2], fields[3], PG.SharedKappa())
end
s1audit("predictions") do conn
    prefixes = ["0dd74a52","6c0aa3ea","ce7ea22f","c4a0fa94","973baafa","81fdf817"]
    labels = ["td_lower_base","td_lower_a2full_carry_jump","grw_base","grw_step_a2_carry_jump","m12_td_ref","qs_marketobs"]
    runs = QSBacktestPhase0.read_query(conn,"SELECT DISTINCT r.run_id::text,r.experiment_name,r.status,r.name FROM runs r JOIN harness_scores h ON h.run_id=r.run_id WHERE " * join(["r.run_id::text LIKE '$p%'" for p in prefixes]," OR "))
    CSV.write(joinpath(S1OUT,"resolved_runs.csv"),runs)
    nrow(runs)==6 || error("Could not resolve all existing fit UUIDs through harness tables; STOP")
    tiers = Harness.club_season_tiers(s1ds)
    for (prefix,label) in zip(prefixes,labels)
        r = only(eachrow(filter(r->startswith(r.run_id,prefix),runs)))
        println("SCORING ",label," ",r.run_id); flush(stdout)
        fit = Training.load_fit(Training.PostgresStorage(r.experiment_name),UUID(r.run_id))
        seasons = label=="qs_marketobs" ? ["24/25","25/26"] : ["23/24","24/25","25/26"]
        ref = Harness.RunRef(label,r.experiment_name,r.run_id,:control)
        bundle = Harness._score_one(ref,fit,s1ds,tiers;target_seasons=seasons,bootstrap_B=2000,expected_count=label=="qs_marketobs" ? 710 : 1070)
        CSV.write(joinpath(S1OUT,"scores_"*label*".csv"),bundle.scores)
        CSV.write(joinpath(S1OUT,"observations_"*label*".csv"),bundle.observations)
        dbfolds = QSBacktestPhase0.read_query(conn,"SELECT fold_idx,first_match_date,last_match_date,n_matches FROM fold_results WHERE run_id=\$1::uuid ORDER BY fold_idx",(r.run_id,))
        CSV.write(joinpath(S1OUT,"dbfolds_"*label*".csv"),dbfolds)
        lat = QSBacktestPhase0.read_query(conn,"SELECT ml.match_id,fr.fold_idx,ml.mean_lambda_h,ml.mean_lambda_a FROM match_latents ml JOIN fold_results fr ON fr.fold_id=ml.fold_id WHERE fr.run_id=\$1::uuid",(r.run_id,))
        CSV.write(joinpath(S1OUT,"rates_"*label*".csv"),lat)
        s1manifests(fit,label,dbfolds,lat)
        println("SCORED ",label," panel=",length(bundle.panel)," observations=",nrow(bundle.observations)," folds=",length(fit.folds)); flush(stdout)
        fit=nothing; bundle=nothing; GC.gc()
    end
end
