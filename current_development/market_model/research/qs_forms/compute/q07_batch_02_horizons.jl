# R10 masks depend exclusively on fixed fixture design, not observed/null state values.
const HORIZONS=(1,2,4,8,13)
function horizon_masks(p,config,cutoff)
    designs=Dict(s=>A.season_design(p,s) for s in config.seasons)
    masks=Dict{Tuple{String,String,Int},Vector{Tuple{Int,Int}}}()
    for season in config.seasons
        d=designs[season];e=A.eligibility(p,d,season);T=length(d.weeks)
        common=[(i,t) for i in eachindex(d.active) for t in 1:max(0,T-13) if d.weeks[t]>cutoff&&all(e[i,t+h] for h in (0,1,2,4,8,13))]
        for h in HORIZONS
            endpoint=[(i,t) for i in eachindex(d.active) for t in 1:max(0,T-h) if d.weeks[t]>cutoff&&e[i,t]&&e[i,t+h]]
            masks[(season,"endpoint",h)]=endpoint
            masks[(season,"continuous",h)]=[(i,t) for (i,t) in endpoint if all(e[i,t:t+h])]
            masks[(season,"matched_start",h)]=[(i,t) for (i,t) in endpoint if e[i,t+1]]
            masks[(season,"common_horizon",h)]=common
        end
    end
    designs,masks
end
function horizon_arrays(p,X,designs)
    out=Dict{Tuple{String,String,String},Matrix{Float64}}()
    for (season,d) in designs
        raw=rawaxes(p,X,d)
        for axis in ("q","s")
            z=getproperty(raw,Symbol(axis));b=ST.mean(z;dims=1)
            out[(season,axis,"relative")]=z.-b
            out[(season,axis,"raw")]=z
            out[(season,axis,"common")]=repeat(b,length(d.active),1)
        end
    end
    out
end
function horizon_value(arrays,designs,masks,seasons,axis,projection,selection,h)
    num,den,px,dy=Float64[],Float64[],Float64[],Float64[]
    qsq,ssq=Float64[],Float64[];teams=Set{String}();ns=0;eligible=0
    for season in seasons
        d=designs[season];z=arrays[(season,axis,projection)]
        rows=masks[(season,selection,h)];eligible+=length(masks[(season,"endpoint",h)]);isempty(rows)||(ns+=1)
        # Old/endpoint baselines use their h=1 cohort; matched/common use the identical numerator rows.
        baseline=selection in ("matched_start","common_horizon") ? rows : masks[(season,selection,1)]
        for (i,t) in baseline;push!(den,(z[i,t+1]-z[i,t])^2);end
        for (i,t) in rows
            push!(num,(z[i,t+h]-z[i,t])^2);push!(teams,d.active[i])
            q,s=arrays[(season,"q",projection)],arrays[(season,"s",projection)]
            push!(qsq,(q[i,t+h]-q[i,t])^2);push!(ssq,(s[i,t+h]-s[i,t])^2)
            h==1&&(push!(px,z[i,t]);push!(dy,z[i,t+1]-z[i,t]))
        end
    end
    gh,g1=meanof(num),meanof(den)
    shape=ismissing(gh)||ismissing(g1)||g1==0 ? missing : gh/(h*g1)
    ratio=isempty(qsq)||sum(qsq)==0 ? missing : sqrt(sum(ssq)/sum(qsq))
    slope=length(px)>2&&ST.var(px)>0 ? ST.cov(px,dy)/ST.var(px) : missing
    (;Gh=gh,G1_same_rows=g1,shape_ratio=shape,r_s_over_q_h=ratio,OLS_delta_on_prior=slope,
        n_pairs=length(num),n_teams=length(teams),n_seasons=ns,retained_fraction=eligible>0 ? length(num)/eligible : missing)
end
function r10()
    rows,repro=NamedTuple[],NamedTuple[];old=loadcsv("dynamics_screen.csv";request="R10")
    for league in SENTINELS
        p,config=panel(league);sim=nulls(league);designs,masks=horizon_masks(p,config,sim.cutoff)
        nullarrays=[horizon_arrays(p,X,designs) for X in sim.paths]
        @assert length(nullarrays)==99
        for rung in ("C0","R6")
            arrays=horizon_arrays(p,point(league,rung).X,designs)
            for pool in ["POOL";config.seasons], axis in ("q","s"),projection in ("relative","common","raw"), selection in ("continuous","endpoint","matched_start","common_horizon"),h in HORIZONS
                seasons=pool=="POOL" ? config.seasons : [pool]
                v=horizon_value(arrays,designs,masks,seasons,axis,projection,selection,h)
                nv=rung=="C0" ? [horizon_value(x,designs,masks,seasons,axis,projection,selection,h) for x in nullarrays] : Any[]
                if selection in ("matched_start","common_horizon")
                    # Counts in baseline and numerator are identically the stored mask; exact calendar Q time=h.
                    @assert v.n_pairs==sum(length(masks[(s,selection,h)]) for s in seasons)
                end
                if h==1&&!ismissing(v.shape_ratio);@assert abs(v.shape_ratio-1)<=1e-10;end
                quantities=h==1&&selection=="endpoint" ? ("Gh","shape_ratio","r_s_over_q_h","OLS_delta_on_prior") : ("Gh","shape_ratio","r_s_over_q_h")
                for quantity in quantities
                    value=getproperty(v,Symbol(quantity));env=envelope([getproperty(x,Symbol(quantity)) for x in nv],value)
                    h==1&&quantity=="shape_ratio"&&(env=merge(env,(;null_rank=missing)))
                    status=ismissing(value) ? "NOT_AVAILABLE: empty rows/zero denominator" : selection=="common_horizon"&&(v.n_pairs<30||v.n_teams<6) ? "LOW_SUPPORT: <30 pairs or <6 teams; no ranking" : "OK"
                    rung=="R6"&&(status*="; observed-only rung sensitivity, no null")
                    h==1&&quantity=="shape_ratio"&&(status*="; identity check, no extreme rank")
                    push!(rows,(;league,rung,season_or_pool=pool,axis,projection,selection,h,v.n_pairs,v.n_teams,v.n_seasons,v.retained_fraction,v.Gh,v.G1_same_rows,v.shape_ratio,v.r_s_over_q_h,env...,quantity,value,status))
                end
                if pool=="POOL"&&projection=="relative"&&selection=="continuous"
                    for (quantity,new,statistic) in (("n_pairs",Float64(v.n_pairs),"G"),("Gh",v.Gh,"G"),("shape_ratio",v.shape_ratio,"G_over_hG1"))
                        r=only(eachrow(DF.filter(r->r.league==league&&r.rung==rung&&r.axis==axis&&r.lag==h&&r.statistic==statistic,old)))
                        expected=quantity=="n_pairs" ? Float64(r.n) : r.value
                        error=ismissing(expected)||ismissing(new) ? missing : abs(expected-new)
                        @assert (ismissing(expected)&&ismissing(new))||(!ismissing(error)&&error<=1e-10)
                        push!(repro,(;league,rung,axis,h,quantity,old=expected,new,abs_error=error,status="PASS"))
                    end
                end
            end
        end
        println("R10 $league completed design-matched99 nulls");flush(stdout)
    end
    output("R10","horizon_screen.csv",rows;bulk=true);output("R10","horizon_reproduction.csv",repro)
    r10_summary(rows)
    verify("R10","PASS old continuous count/G/normalized shape absolute errors≤1e−10; masks constructed once from actual schedule and reused identically for observed/R6/all99 nulls; matched baseline=numerator rows; h1 identity≤1e−10. Low-support common horizon labeled, no Q-time replacement or gap interpolation.")
end
function r10_summary(rows)
    lines=["COMPLETE. horizon_screen.csv beast-only (manifest): primary relative q/s screens plus common/raw sensitivity, season and pool; energy and normalized shape reported separately. Old continuous counts/G/ratios reproduced ≤1e−10, including R6. All masks fixed by design;99 reused nulls/sentinel; calendar Q time=h."]
    for league in SENTINELS
        g=filter(r->r.league==league&&r.rung=="C0"&&r.season_or_pool=="POOL"&&r.axis=="s"&&r.projection=="relative"&&r.selection=="matched_start"&&r.quantity=="shape_ratio"&&r.h>1,rows)
        push!(lines,"$league matched style shape/rank/pairs: "*join(["h$(r.h) $(r.shape_ratio)/$(r.null_rank)/$(r.n_pairs)" for r in g],"; ")*".")
    end
    for league in ("eng_premier","sco_league_two")
        g=filter(r->r.league==league&&r.rung=="C0"&&r.season_or_pool=="POOL"&&r.projection=="relative"&&r.selection=="endpoint"&&r.quantity=="OLS_delta_on_prior",rows)
        sensitivity=filter(r->r.league==league&&r.rung=="R6"&&r.season_or_pool=="POOL"&&r.axis=="s"&&r.projection=="relative"&&r.selection=="matched_start"&&r.quantity=="shape_ratio"&&r.h>1,rows)
        push!(lines,"$league endpoint OLS axis/slope/rank "*join(["$(r.axis) $(r.value)/$(r.null_rank)" for r in g],"; ")*"; R6 style shapes "*join(["h$(r.h)=$(r.shape_ratio)" for r in sensitivity],"; ")*".")
    end
    push!(lines,"Common-horizon LOW_SUPPORT cells (quantity rows)=$(count(r->r.selection=="common_horizon"&&startswith(r.status,"LOW_SUPPORT"),rows)); raw/common modes kept separate. Season-wise slopes/shapes/counts permit inspection of pooled cancellations. h1 normalized rank suppressed; no half-life or prospective-fit superiority claim.")
    qrows=filter(r->r.league in ("eng_premier","sco_league_two")&&r.rung=="C0"&&r.season_or_pool=="POOL"&&r.axis=="q"&&r.projection=="relative"&&r.selection=="matched_start"&&r.quantity=="shape_ratio"&&r.h>1,rows)
    push!(lines,"Matched quality shape/rank: "*join(["$(r.league)/h$(r.h) $(r.shape_ratio)/$(r.null_rank)" for r in qrows],"; ")*".")
    grows=filter(r->r.rung=="C0"&&r.season_or_pool=="POOL"&&r.axis=="s"&&r.projection=="relative"&&r.selection=="matched_start"&&r.quantity=="Gh"&&r.h==13,rows)
    push!(lines,"Matched style h13 raw squared-log-rate energy/null90%/rank: "*join(["$(r.league) $(r.Gh) [$(r.null_q05),$(r.null_q95)]/$(r.null_rank)" for r in grows],"; ")*"; raw energy ranks need not match normalized-shape ranks.")
    summary!("R10",lines)
end
