# Final artifact and source-preservation checks, without recomputation.
function final_checks()
    manifest=CSV.read(joinpath(OUT,"manifest.csv"),DF.DataFrame;stringtype=String)
    latest=Dict{String,Int}()
    for i in 1:DF.nrow(manifest)
        r=manifest[i,:]
        ismissing(r.path)||isempty(r.path)|| (latest[r.path]=i)
    end
    countout=0;countin=0
    for (path,i) in latest
        r=manifest[i,:];@assert isfile(path)
        @assert sha(path)==r.sha256 "final artifact/source hash mismatch: $path"
        if r.kind in ("committed_output","beast_only_output")
            f=CSV.read(path,DF.DataFrame;stringtype=String);@assert DF.nrow(f)==r.n_rows
            countout+=1
        elseif r.kind=="input"
            countin+=1
        end
    end
    id=CSV.read(joinpath(OUT,"projection_identity.csv"),DF.DataFrame)
    checked=DF.filter(r->r.identity!="R05_subset_deviation_from_complete_identity",id)
    @assert all(checked.max_abs_error.<=1e-10)
    @assert count(==("C0_design_common_loading_zero"),id.identity)==11
    hp=CSV.read(joinpath(OUT,"horizon_reproduction.csv"),DF.DataFrame)
    @assert all(==( "PASS"),hp.status)
    sp=CSV.read(joinpath(OUT,"score_reproduction.csv"),DF.DataFrame)
    @assert maximum(sp.abs_error)<=1e-10
    sr=CSV.read(joinpath(OUT,"score_robustness.csv"),DF.DataFrame;stringtype=String)
    @assert all(sr.B[sr.bootstrap.=="circular"].==1999)
    @assert Set(sr.L)==Set([4,8,12])
    @assert all(isfinite,sr.point)&&all(isfinite,sr.boot_mean)
    for g in DF.groupby(DF.filter(:source=>==("R07"),sr),[:league,:weighting,:method,:comparison,:bootstrap,:L])
        @assert DF.nrow(g)==3
        for col in (:point,:boot_mean)
            get(c)=only(g[g.channel.==c,col])
            @assert abs(get("joint")-get("total")-get("allocation"))<=1e-10
        end
    end
    mem=CSV.read(joinpath(OUT,"scottish_tier_membership.csv"),DF.DataFrame;stringtype=String)
    @assert all(mem.n_draws[mem.window.=="full_season"].==128)
    windows=loadcsv("season_windows.csv");cutoff=nulls("sco_premiership").cutoff
    for r in eachrow(DF.filter(:window=>==("null_suffix"),mem))
        w=only(eachrow(DF.filter(x->x.league=="sco_premiership"&&x.season==r.season,windows)))
        @assert r.n_draws==(w.first_week>cutoff ? 128 : 0)
        @assert ismissing(r.prob_upper_group)==(r.n_draws==0)
    end
    tiers=CSV.read(joinpath(OUT,"scottish_tier_robustness.csv"),DF.DataFrame;stringtype=String)
    for r in eachrow(DF.filter(:season_or_pair=>==("EXISTING_COMPLETE_WINDOWS"),tiers))
        a=only(eachrow(DF.filter(x->x.rung==r.rung&&x.window==r.window&&x.quantity==r.quantity&&x.season_or_pair=="ALL_WINDOWS",tiers)))
        for col in (:n_shared,:q05,:median,:q95,:null_q05,:null_median,:null_q95,:null_rank)
            @assert isequal(r[col],a[col])
        end
    end
    audit=CSV.read(joinpath(OUT,"tier_window_audit.csv"),DF.DataFrame)
    @assert all(audit.n_raw_ingested_matches.==198)
    @assert all(audit.n_fixtures+audit.n_accepted_panel_exclusions.==198)
    text=read(joinpath(OUT,"SUMMARY.md"),String)
    for request in ("R09","R10","R11","R12")
        section=match(Regex("(?ms)^## $request\\n(.*?)(?=^## |\\z)"),text)
        @assert section!==nothing&&length(filter(!isempty,split(strip(section.captures[1]),'\n')))<=10
    end
    verify("final_checks","PASS$countout output hashes/row counts; $countin immutable input hashes;11 common-mode zero-loading/prior checks; energy/product/covariance tolerance≤1e−10; old horizon reproduction; original score point/interval reproduction≤1e−10;1999 circular B per L4/8/12; paired goal point/bootstrap-mean additivity≤1e−10;128 full-season tier means, identical later-season suffix means reused and first-season suffix posterior explicitly unavailable; complete pre-split mask identical to main, raw198 matches/season versus accepted-panel exclusions explicit; all4 summaries≤10 lines. No MCMC/new draws/SQL/src/package changes. Julia only on beast, pane%304,threads16/BLAS1.")
end
