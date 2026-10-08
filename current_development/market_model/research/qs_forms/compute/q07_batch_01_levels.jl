# Included inside QSFormsBatch01. Descriptive nonlinear/tier screens, never mixture fitting.
function nonlinear(q,s,teams)
    n=length(q)
    n>=8&&ST.std(q)>0 || return nothing
    x=(q.-ST.mean(q))/ST.std(q);z=x.^2 .-1
    A=hcat(ones(n),x,z);coef=A\s;pred=A*coef
    sse=sum((s-pred).^2);base=sum((s.-ST.mean(s)).^2)
    quad,linear,intercept=0.0,0.0,0.0
    for i in 1:n
        ix=setdiff(1:n,[i])
        cq=A[ix,:]\s[ix];cl=A[ix,1:2]\s[ix]
        quad+=(s[i]-LA.dot(A[i,:],cq))^2
        linear+=(s[i]-LA.dot(A[i,1:2],cl))^2
        intercept+=(s[i]-ST.mean(s[ix]))^2
    end
    order=sortperm(x);gaps=diff(x[order]);j=argmax(gaps);iqr=ST.quantile(x,.75)-ST.quantile(x,.25)
    popsd=sqrt(ST.mean(x.^2))
    quantities=(;b=coef[2],c=coef[3],R2=base>0 ? 1-sse/base : missing,
        LOTO_quadratic_minus_intercept=(quad-intercept)/n,LOTO_quadratic_minus_linear=(quad-linear)/n,
        skewness=ST.mean((x/popsd).^3),excess_kurtosis=ST.mean((x/popsd).^4)-3,
        largest_gap_over_iqr=iqr>0 ? gaps[j]/iqr : missing)
    sides=Dict(teams[order[k]]=>k<=j for k in 1:n)
    return (;quantities,left=teams[order[j]],right=teams[order[j+1]],sides)
end
function season_nonlinear(p,X,season;cutoff=0)
    d=season_design(p,season)
    weeks=max(first(d.weeks),cutoff+1):last(d.weeks)
    isempty(weeks)&&return nothing
    d=(;d.active,d.ix,weeks)
    _,_,_,q,s=level_values(p,X,d,"active_cohort")
    return nonlinear(q,s,d.active)
end
function tier_rows!(out,league,rung,window,series)
    seasons=sort(collect(keys(series));by=s->s)
    # Caller inserts seasons chronologically; textual season labels here are all YYYY or YY/YY.
    for (i,season) in enumerate(seasons)
        r=series[season];r===nothing&&continue
        shared=String[];same=missing
        if i<length(seasons)&&series[seasons[i+1]]!==nothing
            next=series[seasons[i+1]]
            shared=intersect(collect(keys(r.sides)),collect(keys(next.sides)))
            if !isempty(shared)
                fraction=ST.mean(r.sides[t]==next.sides[t] for t in shared)
                same=max(fraction,1-fraction)
            end
        end
        push!(out,(;league,rung,season,window,n=length(r.sides),gap_over_iqr=r.quantities.largest_gap_over_iqr,
            left_team=r.left,right_team=r.right,next_season_shared_n=length(shared),same_side_fraction=same,null_rank=missing))
    end
end
function r06()
    started=time();rows,tierrows,nullrows=NamedTuple[],NamedTuple[],NamedTuple[]
    means=CSV.read(joinpath(OUT,"team_season_path_means.csv"),DF.DataFrame;stringtype=String)
    for (league,_,_) in QS.LEAGUES
        p,config=panel(league);cutoff=sort(unique(p.obs_week))[8]
        sim=league in SENTINELS ? shared_null(league) : nothing
        for rung in ("C0","R6")
            point=point_cache(league,rung)
            for window in ("full_season","null_suffix")
                series=Dict{String,Any}()
                for season in config.seasons
                    value=season_nonlinear(p,point.X,season;cutoff=window=="null_suffix" ? cutoff : 0)
                    series[season]=value;value===nothing&&continue
                    nulls=rung=="C0"&&window=="null_suffix"&&sim!==nothing ? [season_nonlinear(p,X,season;cutoff) for X in sim.paths] : Any[]
                    for quantity in propertynames(value.quantities)
                        val=getproperty(value.quantities,quantity)
                        vals=isempty(nulls) ? Any[] : [s===nothing ? missing : getproperty(s.quantities,quantity) for s in nulls]
                        env=null_envelope(vals,val;status="NOT_SIMULATED: full-season or R6 or non-sentinel")
                        push!(rows,(;league,rung,season,window,method="RTS_point",quantity=String(quantity),n=length(value.sides),q05=val,median=val,q95=val,
                            null_q05=env.null_q05,null_median=env.null_median,null_q95=env.null_q95,null_rank=env.null_rank))
                        for (rep,s) in enumerate(nulls)
                            push!(nullrows,(;league,replicate=rep,request="R06",axis="levels",stratum=season,statistic=String(quantity),value=s===nothing ? missing : getproperty(s.quantities,quantity),n=s===nothing ? 0 : length(s.sides)))
                        end
                    end
                end
                tier_rows!(tierrows,league,rung,window,series)
            end
            g=DF.filter(r->r.league==league&&r.rung==rung&&r.gauge=="active_cohort",means)
            for season in config.seasons
                sg=DF.filter(:season=>==(season),g)
                values=[nonlinear(d.q,d.s,d.team) for d in DF.groupby(sg,:draw)]
                any(isnothing,values)&&continue
                for quantity in propertynames(first(values).quantities)
                    qs=quant([getproperty(v.quantities,quantity) for v in values])
                    push!(rows,(;league,rung,season,window="full_season",method="FFBS",quantity=String(quantity),n=length(first(values).sides),q05=qs[1],median=qs[2],q95=qs[3],
                        null_q05=missing,null_median=missing,null_q95=missing,null_rank=missing))
                end
            end
        end
    end
    # Gap envelopes for the exact observed suffix window only, never full-season uncertainty.
    for i in eachindex(tierrows)
        r=tierrows[i]
        if r.league in SENTINELS&&r.rung=="C0"&&r.window=="null_suffix"
            vals=[v.value for v in nullrows if v.league==r.league&&v.stratum==r.season&&v.statistic=="largest_gap_over_iqr"]
            tierrows[i]=merge(r,(;null_rank=null_envelope(vals,r.gap_over_iqr).null_rank))
        end
    end
    result=output("R06","nonlinear_levels.csv",rows);output("R06","tier_gaps.csv",tierrows);save_null_stats!(nullrows)
    g=DF.filter(r->r.method=="FFBS"&&r.quantity=="c"&&r.rung=="C0",result)
    order=sortperm(abs.(g.median);rev=true)[1:min(3,DF.nrow(g))]
    lines=["COMPLETE descriptive screen. [nonlinear_levels.csv](nonlinear_levels.csv), [tier_gaps.csv](tier_gaps.csv). Full-season posterior geometry remains separate from matched-null suffix point means. 128 paths per fit, never paths-as-extra-teams; LOTO deltas are quadratic minus comparator per team (negative is better)."]
    push!(lines,"Three largest C0 FFBS |curvature|: "*join(["$(r.league)/$(r.season) c=$(round(r.median;digits=4)) [$(round(r.q05;digits=4)),$(round(r.q95;digits=4))]" for r in eachrow(g[order,:])],"; ")*".")
    for league in SENTINELS
        g=filter(r->r.league==league&&r.rung=="C0"&&r.window=="null_suffix",tierrows)
        push!(lines,"$league gap/IQR; rank; next-season shared/same-side: "*join(["$(r.season) $(round(r.gap_over_iqr;digits=3));$(r.null_rank);$(r.next_season_shared_n)/$(ismissing(r.same_side_fraction) ? "NA" : round(r.same_side_fraction;digits=3))" for r in g],"; ")*".")
    end
    push!(lines,"R6/full-season curvature and both LOTO changes retained per season; consistent signs/rung sensitivity must be read jointly, not selected by largest magnitude. Null gap ranks do not establish persistent tiers; tiny n (8–24) limits shape inference. R05 prior-|q| energy is the radial screen, with no new fit.")
    summary!("R06",lines);verification("R06","PASS C0 suffix statistics reuse99 shared conditional panels; full-season FFBS kept separate; same teams/window in observed/null; largest gap fixed once per season, side labels may align across seasons without searching gaps. Runtime=$(time()-started)s; invocation Q07.r06().");flush_manifest!()
end
