# Bias-by-game-bin curves, using the v1.2 observations and cohort definitions.
using BayesianFootball, CSV, DataFrames
include("/root/BF_runs/qs_transition/scripts/score_runs.jl")
refs=load_run_refs(joinpath(@__DIR__,"runs.csv"))
include_run_loaders(refs)
ds=Data.load_datastore_cached(Data.ScottishPyramid();max_age_hours=10000)
tiers=Harness.club_season_tiers(ds)
function s3_bias_bins(refs,ds,tiers)
    games=Dict{Tuple{Int,String},Int}()
    counts=Dict{Tuple{String,String},Int}()
    matches=filter(r->Int(r.tournament_id) in (54,55,56,57),ds.matches)
    sort!(matches,[:match_date,:match_id])
    for r in eachrow(matches), club in (String(r.home_team),String(r.away_team))
        key=(club,String(r.season))
        get(tiers,key,0)==Int(r.tournament_id) || continue
        count=get(counts,key,0)+1; counts[key]=count
        games[(Int(r.match_id),club)]=count
    end
    output=NamedTuple[]; timing=NamedTuple[]
    for ref in refs
        fit=Training.load_fit(Training.PostgresStorage(ref.experiment),ref.run_id)
        bundle=Harness._score_one(ref,fit,ds,tiers;target_seasons=["23/24","24/25","25/26"],bootstrap_B=10000,expected_count=1070)
        CSV.write(joinpath(@__DIR__,"observations_$(ref.label).csv"),bundle.observations)
        push!(timing,(;arm=ref.label,run_id=string(ref.run_id),fit_elapsed_seconds=fit.metadata.elapsed_seconds,git_commit=fit.metadata.git_commit))
        for direction in (:relegated_into_L1,:promoted_into_L1,:entered_spfl)
            cohort=Harness.transition_cohort(ds,bundle.panel,tiers;first_n=20,direction)
            values=Harness._transition_bias_observations(bundle.observations,ds,cohort.clubs)
            values.game=[games[(r.match_id,r.club)] for r in eachrow(values)]
            for (label,lo,hi) in (("1",1,1),("2-5",2,5),("6-10",6,10),("11-20",11,20))
                selected=filter(r->lo<=r.game<=hi,values)
                boot=Harness._clustered_bias_bootstrap(selected;B=10000)
                push!(output,(;arm=ref.label,direction=string(direction),game_bin=label,bias_pp=boot.value,lo=boot.lo,hi=boot.hi,n_obs=nrow(selected),n_fixtures=length(unique(selected.match_id)),n_club_seasons=nrow(unique(selected[:,[:club,:season]]))))
            end
        end
        println("S3_BIAS_BINS $(ref.label) DONE")
        fit=nothing; GC.gc()
    end
    CSV.write(joinpath(@__DIR__,"bias_game_bins.csv"),DataFrame(output))
    CSV.write(joinpath(@__DIR__,"timing.csv"),DataFrame(timing))
end
Base.invokelatest(s3_bias_bins,refs,ds,tiers)
