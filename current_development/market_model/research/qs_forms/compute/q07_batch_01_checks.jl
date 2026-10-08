# Included inside QSFormsBatch01. Artifact gates, not new fitted-model experiments.
function repair_null_statuses()
    for (request,name) in (("R04","dynamics_screen.csv"),("R05","volatility_screen.csv"))
        df=CSV.read(joinpath(OUT,name),DF.DataFrame;stringtype=String)
        for r in eachrow(df)
            ismissing(r.null_rank)||(r.null_status="SIMULATED: conditional plug-in99")
        end
        output(request,name,df)
        if request=="R05"
            covariance=DF.filter(r->r.league in SENTINELS&&r.stratum=="all"&&r.statistic=="mean_offdiagonal_centered_step_covariance",df)
            line="mean_offdiagonal_centered_step_covariance: "*join(["$(r.league)/$(r.axis) $(round(r.value;sigdigits=4)) (rank=$(r.null_rank))" for r in eachrow(covariance)],"; ")*"."
            path=joinpath(OUT,"SUMMARY.md")
            write(path,replace(read(path,String),r"(?m)^mean_offdiagonal_centered_step_covariance:.*$"=>line))
        end
    end
    verification("metadata","PASS corrected populated null envelopes to SIMULATED; original fallback label incorrectly said non-sentinel even when all99 replicates existed. Numerical values unchanged.")
    flush_manifest!()
end
function final_checks()
    df=CSV.read(joinpath(OUT,"geometry_summary.csv"),DF.DataFrame;stringtype=String)
    raw=DF.DataFrame(CSV.File(joinpath(OUT,"geometry_draws.csv");select=[:league,:rung,:protocol,:chain,:draw],stringtype=String))
    @assert DF.nrow(raw)==792000&&DF.nrow(unique(raw))==792000
    groups=DF.groupby(raw,[:league,:rung,:protocol]);@assert length(groups)==66
    @assert all(DF.nrow(g)==12000&&maximum(g.chain)==4&&maximum(g.draw)==3000 for g in groups)
    @assert all(abs.(df.median[(df.rung.=="R2").&(df.quantity.=="r")].-1).<=1e-12)
    @assert all(df.q05[(df.rung.=="C0").&(df.quantity.=="c")].==0)
    for (name,tol) in (("reproduction_checks.csv",1e-6),("forecast_reproduction_checks.csv",1e-8))
        df=CSV.read(joinpath(OUT,name),DF.DataFrame;stringtype=String)
        @assert all(df.pass)&&maximum(df.error)<=tol
    end
    forecasts=CSV.read(joinpath(OUT,"forecast_joint.csv"),DF.DataFrame;stringtype=String)
    @assert DF.nrow(forecasts)==DF.nrow(unique(forecasts,[:league,:rung,:fixture_id]))
    @assert all(forecasts.n_draws.==128)
    @assert all(forecasts.var_D.*forecasts.var_M.-forecasts.cov_DM.^2 .>0)
    @assert all(isfinite,forecasts.logp_joint)
    paired=DF.innerjoin(DF.filter(:rung=>==("C0"),forecasts),DF.filter(:rung=>==("R6"),forecasts);on=[:league,:fixture_id],makeunique=true)
    @assert DF.nrow(paired)==3053
    for league in SENTINELS
        p,_=panel(league);f=fit(league,"C0","10a";request="final_checks")
        sim=shared_null(league)
        @assert length(sim.paths)==99&&sim.cutoff==sort(unique(p.obs_week))[8]
        @assert sim.theta==MID.median_theta(f)
    end
    nulls=CSV.read(joinpath(OUT,"null_statistics.csv"),DF.DataFrame;stringtype=String)
    keyed=Dict{Tuple,Any}()
    for g in DF.groupby(nulls,[:league,:request,:axis,:stratum,:statistic])
        @assert length(unique(g.replicate))==99&&DF.nrow(g)==99
        keyed[(first(g.league),first(g.request),first(g.axis),string(first(g.stratum)),first(g.statistic))]=g.value
    end
    compared=0
    for (request,name) in (("R04","dynamics_screen.csv"),("R05","volatility_screen.csv"))
        data=CSV.read(joinpath(OUT,name),DF.DataFrame;stringtype=String)
        for r in eachrow(data)
            ismissing(r.null_rank)&&continue
            @assert startswith(r.null_status,"SIMULATED")&&r.league in SENTINELS
            request=="R04"&&(@assert r.rung=="C0")
            stratum=request=="R04" ? "lag_$(r.lag)" : string(r.stratum)
            vals=keyed[(r.league,request,r.axis,stratum,r.statistic)]
            env=null_envelope(vals,r.value)
            @assert maximum(abs.([env.null_q05-r.null_q05,env.null_median-r.null_median,env.null_q95-r.null_q95,env.null_rank-r.null_rank]))<1e-12
            compared+=1
        end
    end
    goals=CSV.read(joinpath(OUT,"goal_ablation_fixtures.csv"),DF.DataFrame;stringtype=String)
    @assert DF.nrow(goals)==DF.nrow(unique(goals,[:league,:fixture_id,:method,:variant]))
    @assert all(isfinite,goals.logp_joint)&&all(isfinite,goals.logp_total)&&all(isfinite,goals.logp_allocation)
    @assert maximum(abs.(goals.logp_joint.-goals.logp_total.-goals.logp_allocation))<1e-8
    plugin=DF.filter(:method=>==("plugin_median_theta"),goals)
    allocation=DF.innerjoin(DF.filter(:variant=>==("full"),plugin),DF.filter(:variant=>==("no_style"),plugin);on=[:league,:fixture_id],makeunique=true)
    @assert maximum(abs.(allocation.logp_allocation.-allocation.logp_allocation_1))<1e-8
    @assert !any((goals.variant.=="frozen_style").&(goals.method.!="plugin_median_theta"))
    @assert all(goals.n_latent_samples[goals.method.=="mixture_128x2"].==256)
    @assert all(goals.n_latent_samples[goals.method.=="mixture_128x4"].==512)
    reproduction=CSV.read(joinpath(OUT,"goal_reproduction_checks.csv"),DF.DataFrame;stringtype=String)
    @assert all(reproduction.pass)&&maximum(reproduction.error)<=1e-8
    structure=CSV.read(joinpath(OUT,"league_structure.csv"),DF.DataFrame;stringtype=String)
    @assert all(structure.q05[structure.quantity.=="SD_log_r"].>0)
    movers=CSV.read(joinpath(OUT,"mover_summary.csv"),DF.DataFrame;stringtype=String)
    @assert all(startswith.(movers.status,"NOT_AVAILABLE"))
    manifest=CSV.read(joinpath(OUT,"manifest.csv"),DF.DataFrame;stringtype=String)
    verified=0
    for r in eachrow(manifest)
        endswith(r.kind,"output")||continue
        @assert isfile(r.path)&&filehash(r.path)==r.sha256
        header=names(DF.DataFrame(CSV.File(r.path;limit=1,ntasks=1)))
        @assert Set(header)==Set(split(r.columns,';'))
        @assert length(CSV.File(r.path;select=[1],ntasks=1))==r.n_rows
        if r.kind=="committed_output";@assert filesize(r.path)<=5_000_000;end
        verified+=1
    end
    raw_index=CSV.read(joinpath(OUT,"large_move_raw_source_index.csv"),DF.DataFrame;stringtype=String)
    for g in DF.groupby(raw_index,:snapshot_path)
        ds=Serialization.deserialize(first(g.snapshot_path))
        for r in eachrow(g)
            r.n_rows==0&&continue
            ix=parse.(Int,split(r.source_rows,';'))
            @assert length(ix)==r.n_rows&&length(unique(ix))==r.n_rows
            table=getproperty(ds,Symbol(r.table))
            @assert all(table.match_id[ix].==r.fixture_id)
        end
    end
    text=read(joinpath(OUT,"SUMMARY.md"),String)
    for request in ("R01","R02","R03","R04","R05","R06","R07","R08")
        section=split(split(text,"## $request —\n";limit=2)[2],"\n## ";limit=2)[1]
        @assert length(filter(!isempty,split(strip(section),'\n')))<=10
    end
    verification("final_checks","PASS R2/C0 identities,12000-draw retention, point/forecast round trips;3053 paired honest forecasts with PD covariance;99 replicates/sentinel and $compared null envelopes/ranks exactly reproduced from saved raw null rows; goal finite/unique keys and joint=total+allocation ≤1e−8; plug-in full/no-style allocation equality ≤1e−8; no fabricated frozen mixture or movers; $verified output hashes/row-counts/column-lists, exact selected-fixture raw-cache row links, committed-size bounds; all8 summary sections≤10 lines. Julia only on mcmc-beast, owned %304,16 threads/BLAS1. No MCMC/new fits/SQL/src edits.")
    flush_manifest!()
end
