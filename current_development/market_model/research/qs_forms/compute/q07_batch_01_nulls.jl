# Included inside QSFormsBatch01. Matched conditional no-refit null for R04-R06.
const NULL_PATHS = Dict{String,Any}()
function point_cache(league,rung)
    path=joinpath(OUT,"cache","$(league)_$(rung)_point.jls")
    return Serialization.deserialize(path)
end
function replace_y(p,y)
    return MID.MarketPanel(p.matches,p.refusals,p.teams,p.n_weeks,p.week_start,p.obs_week,
        p.obs_home,p.obs_att,p.obs_def,y,p.obs_match,p.obs_season,p.week_ptr,p.obs_X,p.feature_names)
end
"Condition on first eight observed weeks, draw their joint filtered endpoint, simulate original design thereafter."
function shared_null(league)
    haskey(NULL_PATHS,league) && return NULL_PATHS[league]
    path=joinpath(OUT,"cache","$(league)_conditional_nulls.jls")
    if isfile(path)
        NULL_PATHS[league]=Serialization.deserialize(path)
        @assert length(NULL_PATHS[league].paths)==99
        return NULL_PATHS[league]
    end
    p,config=panel(league)
    li=only(findall(e->e[1]==league,QS.LEAGUES))
    f=fit(league,"C0","10a");theta=MID.median_theta(f)
    filtered=filter_store(f,p,theta,"C0")
    cutoff=sort(unique(p.obs_week))[8];N=MID.n_teams(p)
    H=PM.fullbook_design(f.arm,p,theta)
    paths=Vector{Matrix{Float64}}(undef,99)
    ts=time()
    Threads.@threads for j in 1:99
        rng=Random.Xoshiro(seed(li,3,1,500+j))
        x=zeros(size(filtered.m_filt,1))
        MID.draw_psd!(x,rng,filtered.m_filt[:,cutoff],filtered.P_filt[:,:,cutoff])
        y=copy(p.obs_y)
        for t in cutoff+1:p.n_weeks
            dq=exp(theta[2])*Random.randn(rng,N);ds=exp(theta[3])*Random.randn(rng,N)
            x[3:2+N].+=dq+ds;x[3+N:2+2N].+=ds-dq
            for k in p.week_ptr[t]:(p.week_ptr[t+1]-1)
                y[k]=LA.dot(view(H,k,:),x)+exp(theta[1])*Random.randn(rng)
            end
        end
        @assert y[p.obs_week.<=cutoff]==p.obs_y[p.obs_week.<=cutoff]
        simulated=replace_y(p,y)
        paths[j]=point_path(f,simulated,theta,"C0")
    end
    result=(;paths,cutoff,theta)
    Serialization.serialize(path,result)
    NULL_PATHS[league]=result
    record!("R04-R06","beast_only_cache",basename(path),path;n=99,
        detail="conditional prefix observed weeks1:8; cutoff calendar slot=$cutoff; seed=$(seed(li,3,1,500))+replicate; joint filtered state, actual C0 Q and R; whole suffix RTS; seconds=$(time()-ts)")
    println("NULL $league 99 shared replicates $(time()-ts)s");flush(stdout)
    return result
end
function centered_axes(p,X,design)
    N=MID.n_teams(p)
    a,b=X[3:2+N,design.weeks],X[3+N:2+2N,design.weeks]
    a,b=a[design.ix,:],b[design.ix,:]
    a.-=ST.mean(a;dims=1);b.-=ST.mean(b;dims=1)
    return (q=(a-b)/2,s=(a+b)/2)
end
"Conservative gap policy: a team must appear in every calendar slot of a compared interval."
function eligibility(p,design,season)
    eligible=falses(length(design.ix),length(design.weeks))
    matches=DF.filter(:season=>==(season),p.matches)
    for (i,team) in enumerate(design.active)
        w=matches.week[(matches.home_team.==team).|(matches.away_team.==team)]
        for t in w
            eligible[i,t-first(design.weeks)+1]=true
        end
    end
    return eligible
end
function motion_statistics(league,p,X,theta,cutoff)
    dyn,vol,teamrows,moves,gaps,excluded=NamedTuple[],NamedTuple[],NamedTuple[],NamedTuple[],NamedTuple[],NamedTuple[]
    sq,ss=exp.(theta[2:3])
    for axis in ("q","s")
        energies=NamedTuple[];lagx,lagy=Float64[],Float64[];driftx,drifty=Float64[],Float64[]
        variograms=Dict(h=>Float64[] for h in (1,2,4,8,13))
        weekly=NamedTuple[];offdiagonal_covariances=Float64[]
        for season in unique(p.obs_season)
            d=season_design(p,season);elig=eligibility(p,d,season)
            z=getproperty(centered_axes(p,X,d),Symbol(axis));q=centered_axes(p,X,d).q
            eligible_steps=0;excluded_steps=0
            for i in eachindex(d.active),t in 2:length(d.weeks)
                w=d.weeks[t]
                w-1>cutoff || continue
                if elig[i,t]&&elig[i,t-1]
                    delta=z[i,t]-z[i,t-1]
                    u=(w-first(d.weeks))/(last(d.weeks)-first(d.weeks))
                    phase=u<.2 ? "early" : u>.8 ? "late" : "middle"
                    push!(energies,(;season,team=d.active[i],week=w,delta,energy=delta^2/(axis=="q" ? sq^2 : ss^2),
                        prior_abs_q=abs(q[i,t-1]),prior_z=z[i,t-1],phase,half=u<.5 ? "first" : "second"))
                    push!(driftx,z[i,t-1]);push!(drifty,delta);eligible_steps+=1
                    if t>=3&&elig[i,t-2]&&d.weeks[t-1]>cutoff
                        push!(lagx,delta);push!(lagy,z[i,t-1]-z[i,t-2])
                    end
                else
                    excluded_steps+=1
                end
            end
            push!(excluded,(;league,season,axis,eligible_steps,excluded_steps,gauge="fixed_active_season_cohort",
                policy="requires appearances in every calendar slot; excludes bye/missing-book/gap intervals conservatively"))
            for h in keys(variograms),i in eachindex(d.active),t in 1:length(d.weeks)-h
                d.weeks[t]>cutoff && all(elig[i,t:t+h]) || continue
                push!(variograms[h],(z[i,t+h]-z[i,t])^2)
            end
            for i in 1:length(d.active), j in i+1:length(d.active)
                valid=[t for t in 2:length(d.weeks) if d.weeks[t-1]>cutoff&&elig[i,t]&&elig[i,t-1]&&elig[j,t]&&elig[j,t-1]]
                length(valid)>=3 || continue
                dx=[z[i,t]-z[i,t-1] for t in valid];dy=[z[j,t]-z[j,t-1] for t in valid]
                push!(offdiagonal_covariances,ST.cov(dx,dy))
            end
            for (i,team) in enumerate(d.active)
                appeared=findall(elig[i,:])
                for j in 2:length(appeared)
                    a,b=appeared[j-1],appeared[j]
                    d.weeks[a]>cutoff&&b-a>1 || continue
                    push!(gaps,(;league,season,team,axis,previous_week=d.weeks[a],next_week=d.weeks[b],gap_weeks=b-a,net_change=z[i,b]-z[i,a],label="appearance gap, not a transfer-window label"))
                end
            end
        end
        g1=isempty(variograms[1]) ? missing : ST.mean(variograms[1])
        for h in sort(collect(keys(variograms)))
            v=variograms[h]
            push!(dyn,(;axis,statistic="G",lag=h,n=length(v),value=isempty(v) ? missing : ST.mean(v)))
            push!(dyn,(;axis,statistic="G_over_hG1",lag=h,n=length(v),value=isempty(v)||ismissing(g1)||g1==0 ? missing : ST.mean(v)/(h*g1)))
            push!(dyn,(;axis,statistic="process_time_ratio",lag=h,n=length(v),value=Float64(h)))
        end
        push!(dyn,(;axis,statistic="increment_lag1_correlation",lag=1,n=length(lagx),value=safe_cor(lagx,lagy)))
        push!(dyn,(;axis,statistic="squared_increment_lag1_correlation",lag=1,n=length(lagx),value=safe_cor(lagx.^2,lagy.^2)))
        slope=length(driftx)>2&&ST.var(driftx)>0 ? ST.cov(driftx,drifty)/ST.var(driftx) : missing
        push!(dyn,(;axis,statistic="OLS_delta_on_prior_level",lag=1,n=length(driftx),value=slope))
        energy=DF.DataFrame(energies)
        isempty(energies) && continue
        for g in DF.groupby(energy,[:season,:team])
            DF.nrow(g)>=12 || continue
            firsthalf=g.energy[g.half.=="first"];secondhalf=g.energy[g.half.=="second"]
            push!(teamrows,(;league,season=first(g.season),team=first(g.team),axis,n=DF.nrow(g),
                energy_first=isempty(firsthalf) ? missing : ST.mean(firsthalf),energy_second=isempty(secondhalf) ? missing : ST.mean(secondhalf),energy_all=ST.mean(g.energy)))
        end
        teamaxis=filter(r->r.axis==axis,teamrows)
        means=[r.energy_all for r in teamaxis]
        stable=filter(r->!ismissing(r.energy_first)&&!ismissing(r.energy_second),teamaxis)
        push!(vol,(;axis,stratum="all",statistic="cross_team_energy_CV",n=length(means),value=length(means)>1&&ST.mean(means)>0 ? ST.std(means)/ST.mean(means) : missing))
        push!(vol,(;axis,stratum="all",statistic="team_first_second_energy_spearman",n=length(stable),value=spearman([r.energy_first for r in stable],[r.energy_second for r in stable])))
        for season in unique(energy.season)
            sg=DF.filter(:season=>==(season),energy)
            mid=sg.energy[sg.phase.=="middle"]
            for phase in ("early","middle","late")
                vals=sg.energy[sg.phase.==phase]
                push!(vol,(;axis,stratum="$(season):$phase",statistic="phase_energy_ratio_to_middle",n=length(vals),value=isempty(vals)||isempty(mid)||ST.mean(mid)==0 ? missing : ST.mean(vals)/ST.mean(mid)))
            end
            tr=filter(r->r.axis==axis&&r.season==season,teamrows)
            av=[r.energy_all for r in tr];sr=filter(r->!ismissing(r.energy_first)&&!ismissing(r.energy_second),tr)
            push!(vol,(;axis,stratum=season,statistic="cross_team_energy_CV",n=length(av),value=length(av)>1&&ST.mean(av)>0 ? ST.std(av)/ST.mean(av) : missing))
            push!(vol,(;axis,stratum=season,statistic="team_first_second_energy_spearman",n=length(sr),value=spearman([r.energy_first for r in sr],[r.energy_second for r in sr])))
        end
        middle=energy.energy[energy.phase.=="middle"]
        for phase in ("early","middle","late")
            vals=energy.energy[energy.phase.==phase]
            push!(vol,(;axis,stratum=phase,statistic="phase_energy_ratio_to_middle",n=length(vals),value=isempty(vals)||isempty(middle)||ST.mean(middle)==0 ? missing : ST.mean(vals)/ST.mean(middle)))
        end
        magnitudes=energy.delta.^2
        push!(vol,(;axis,stratum="all",statistic="max_over_median_squared_step",n=length(magnitudes),value=ST.median(magnitudes)>0 ? maximum(magnitudes)/ST.median(magnitudes) : missing))
        sorted=sort(magnitudes;rev=true)
        push!(vol,(;axis,stratum="all",statistic="top5pct_energy_share",n=length(sorted),value=sum(sorted)>0 ? sum(sorted[1:ceil(Int,.05length(sorted))])/sum(sorted) : missing))
        for g in DF.groupby(energy,[:season,:week])
            delta=g.delta;n=length(delta)
            push!(weekly,(;season=first(g.season),week=first(g.week),mean_energy=ST.mean(g.energy),offdiag=n>1 ? (sum(delta)^2-sum(delta.^2))/(n*(n-1)) : missing))
        end
        wf=DF.DataFrame(weekly)
        xx,yy=Float64[],Float64[]
        for g in DF.groupby(wf,:season),r in eachrow(g)
            previous=g.mean_energy[g.week.==r.week-1]
            isempty(previous)|| (push!(xx,r.mean_energy);push!(yy,only(previous)))
        end
        push!(vol,(;axis,stratum="all",statistic="weekly_mean_energy_lag1_correlation",n=length(xx),value=safe_cor(xx,yy)))
        push!(vol,(;axis,stratum="all",statistic="mean_offdiagonal_centered_step_product",n=length(collect(skipmissing(wf.offdiag))),value=mean_available(wf.offdiag)))
        push!(vol,(;axis,stratum="all",statistic="mean_offdiagonal_centered_step_covariance",n=length(offdiagonal_covariances),value=mean_available(offdiagonal_covariances)))
        for season in unique(energy.season)
            g=DF.filter(:season=>==(season),energy)
            cuts=ST.quantile(g.prior_abs_q,[1/3,2/3])
            for bin in 1:3
                vals=g.energy[[v<=cuts[1] ? 1 : v<=cuts[2] ? 2 : 3 for v in g.prior_abs_q].==bin]
                push!(vol,(;axis,stratum="$(season):prior_abs_q_tertile_$bin",statistic="mean_energy",n=length(vals),value=isempty(vals) ? missing : ST.mean(vals)))
            end
        end
        if league in SENTINELS
            idx=sortperm(abs.(energy.delta);rev=true)[1:min(10,DF.nrow(energy))]
            for row in eachrow(energy[idx,:])
                fixtures=DF.filter(r->r.week in row.week-1:row.week && (r.home_team==row.team||r.away_team==row.team),p.matches)
                push!(moves,(;league,axis,team=row.team,week=row.week,delta=row.delta,scheduled_sd=axis=="q" ? sq : ss,
                    previous_gap_weeks=1,fixture_ids=join(fixtures.match_id,";"),available_book_flags="rates: kl,n_selections,start_spread,optim_converged; raw flags joined in large_move_books.csv"))
            end
        end
    end
    return (;dyn,vol,teamrows,moves,gaps,excluded)
end
mean_available(v)=isempty(collect(skipmissing(v))) ? missing : ST.mean(collect(skipmissing(v)))
function ranks(v)
    ix=sortperm(v);out=zeros(length(v));j=1
    while j<=length(v)
        k=j;while k<length(v)&&v[ix[k+1]]==v[ix[j]];k+=1;end
        out[ix[j:k]].=(j+k)/2;j=k+1
    end
    return out
end
spearman(x,y)=safe_cor(ranks(x),ranks(y))
function null_envelope(values,observed;status="SIMULATED")
    v=collect(skipmissing(values))
    if isempty(v)||ismissing(observed)
        return (;null_q05=missing,null_median=missing,null_q95=missing,null_rank=missing,null_status=ismissing(observed) ? "NOT_AVAILABLE: insufficient eligible pairs" : status)
    end
    @assert length(v)==99
    q=quant(v)
    return (;null_q05=q[1],null_median=q[2],null_q95=q[3],null_rank=(1+count(<=(observed),v))/100,null_status=status)
end
function save_null_stats!(rows)
    path=joinpath(OUT,"null_statistics.csv")
    old=isfile(path) ? CSV.read(path,DF.DataFrame;stringtype=String) : DF.DataFrame()
    frame=vcat(old,DF.DataFrame(rows);cols=:union)
    lastrow=Dict{Tuple,Int}()
    for i in 1:DF.nrow(frame)
        r=frame[i,:];lastrow[(r.league,r.replicate,r.request,r.axis,r.stratum,r.statistic)]=i
    end
    output("R04-R06","null_statistics.csv",frame[sort(collect(values(lastrow))),:])
end
function r04()
    started=time();rows,nullrows,exclusions=NamedTuple[],NamedTuple[],NamedTuple[]
    for (league,_,_) in QS.LEAGUES
        p,config=panel(league)
        cutoff=sort(unique(p.obs_week))[8]
        sim=league in SENTINELS ? shared_null(league) : nothing
        theta=point_cache(league,"C0").theta
        nulls=sim===nothing ? Any[] : [motion_statistics(league,p,X,theta,cutoff) for X in sim.paths]
        for rung in (league in SENTINELS ? ("C0","R6") : ("C0",))
            point=point_cache(league,rung)
            # R6 has unequal alpha/beta scales; use its rotated scales for energy only.
            th=rung=="C0" ? point.theta : vcat(point.theta[1],log.([sqrt(geometry_for_theta(fit(league,rung,"10a"),rung,point.theta).Vq),sqrt(geometry_for_theta(fit(league,rung,"10a"),rung,point.theta).Vs)]))
            stats=motion_statistics(league,p,point.X,th,cutoff)
            append!(exclusions,stats.excluded)
            for (j,r) in enumerate(stats.dyn)
                vals=rung=="C0"&&sim!==nothing ? [s.dyn[j].value for s in nulls] : Any[]
                env=null_envelope(vals,r.value;status=rung=="R6" ? "NOT_SIMULATED: R6 sensitivity" : "NOT_SIMULATED: non-sentinel")
                push!(rows,(;league,rung,r...,env...))
                if rung=="C0"&&sim!==nothing
                    for (rep,s) in enumerate(nulls)
                        v=s.dyn[j]
                        push!(nullrows,(;league,replicate=rep,request="R04",axis=r.axis,stratum="lag_$(r.lag)",statistic=r.statistic,value=v.value,n=v.n))
                    end
                end
            end
        end
    end
    result=output("R04","dynamics_screen.csv",rows)
    output("R04","step_exclusions.csv",exclusions);save_null_stats!(nullrows)
    lines=["COMPLETE. [dynamics_screen.csv](dynamics_screen.csv), [step_exclusions.csv](step_exclusions.csv). Fixed active-season cohort centering; conservative eligibility requires a fixture appearance in every compared calendar slot, so missing books/byes are excluded as gaps. All G ratios use actual weekly Q (process-time ratio=h)."]
    for league in SENTINELS
        g=DF.filter(r->r.league==league&&r.rung=="C0"&&r.statistic in ("OLS_delta_on_prior_level","increment_lag1_correlation","squared_increment_lag1_correlation"),result)
        push!(lines,"$league "*join(["$(r.axis) $(r.statistic)=$(round(r.value;digits=3)) (rank=$(r.null_rank))" for r in eachrow(g)],"; ")*".")
    end
    push!(lines,"99 shared conditional C0 null panels/sentinel, suffix only after first eight observed weeks. Null ranks are plug-in predictive ranks, not p-values; non-sentinels and R6 sensitivity have no envelopes. No half-life is estimated; isolated RTS slopes are not dynamic-law evidence.")
    summary!("R04",lines);verification("R04","PASS exactly99 shared conditional nulls for each sentinel, unchanged prefix, original missingness/design/Q/R; all comparisons suffix-only. Runtime=$(time()-started)s; invocation Q07.r04().")
    flush_manifest!()
end
function r05()
    started=time();rows,nullrows,teams,moves,gaps=NamedTuple[],NamedTuple[],NamedTuple[],NamedTuple[],NamedTuple[]
    for (league,_,_) in QS.LEAGUES
        p,config=panel(league);point=point_cache(league,"C0");cutoff=sort(unique(p.obs_week))[8]
        obs=motion_statistics(league,p,point.X,point.theta,cutoff)
        append!(teams,obs.teamrows);append!(moves,obs.moves);append!(gaps,obs.gaps)
        sim=league in SENTINELS ? shared_null(league) : nothing
        nulls=sim===nothing ? Any[] : [motion_statistics(league,p,X,point.theta,cutoff) for X in sim.paths]
        for r in obs.vol
            matches(s)=only(filter(v->v.axis==r.axis&&v.stratum==r.stratum&&v.statistic==r.statistic,s.vol))
            vals=sim===nothing ? Any[] : [matches(s).value for s in nulls]
            env=null_envelope(vals,r.value;status="NOT_SIMULATED: non-sentinel")
            push!(rows,(;league,r...,env...))
            if sim!==nothing
                for (rep,s) in enumerate(nulls)
                    v=matches(s);push!(nullrows,(;league,replicate=rep,request="R05",axis=r.axis,stratum=r.stratum,statistic=r.statistic,value=v.value,n=v.n))
                end
            end
        end
    end
    result=output("R05","volatility_screen.csv",rows)
    output("R05","team_energy.csv",teams);mf=output("R05","large_moves.csv",moves);output("R05","appearance_gaps.csv",gaps)
    books=NamedTuple[]
    for league in SENTINELS
        rates=CSV.read(joinpath(Q,"rates_$league.csv"),DF.DataFrame;stringtype=String)
        ids=unique(vcat([parse.(Int,split(r.fixture_ids,";")) for r in eachrow(DF.filter(:league=>==(league),mf))]...))
        for r in eachrow(DF.filter(:match_id=>in(ids),rates))
            push!(books,(;league,fixture_id=r.match_id,kl=r.kl,n_selections=r.n_selections,start_spread=r.start_spread,optim_converged=r.optim_converged,reason=coalesce(r.reason,"accepted")))
        end
    end
    output("R05","large_move_books.csv",books);save_null_stats!(nullrows)
    lines=["COMPLETE. [volatility_screen.csv](volatility_screen.csv), [team_energy.csv](team_energy.csv), [large_moves.csv](large_moves.csv), [large_move_books.csv](large_move_books.csv), [appearance_gaps.csv](appearance_gaps.csv). All observed suffixes, same99 conditional nulls on sentinels; gaps never labelled transfer windows."]
    for stat in ("team_first_second_energy_spearman","cross_team_energy_CV","top5pct_energy_share","weekly_mean_energy_lag1_correlation","mean_offdiagonal_centered_step_covariance")
        g=DF.filter(r->r.league in SENTINELS&&r.statistic==stat&&r.stratum=="all",result)
        push!(lines,"$stat: "*join(["$(r.league)/$(r.axis) $(round(r.value;digits=3)) (rank=$(r.null_rank))" for r in eachrow(g) if !ismissing(r.value)],"; ")*".")
    end
    push!(lines,"Calendar ratios and prior-|q| tertiles are retained per axis/season in volatility_screen; nulls condition on estimated theta and selected books, so none separates inversion noise/selection from football shocks. Available book diagnostics are KL/selection count/optimizer-start spread, not quote age or exchange depth.")
    summary!("R05",lines);verification("R05","PASS reuses R04 exact99 paths; actual sigma_q/s² per calendar step; fixed-cohort centering; steps/gaps separated. Runtime=$(time()-started)s; invocation Q07.r05().");flush_manifest!()
end
