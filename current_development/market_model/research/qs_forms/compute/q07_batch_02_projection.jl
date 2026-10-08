# R09: raw median RTS energy, with exactly matched conditional-null designs.
function projection_stats(p,X,d,season,cutoff,selection,axis)
    z=getproperty(rawaxes(p,X,d),Symbol(axis));elig=A.eligibility(p,d,season)
    u,c,r=Float64[],Float64[],Float64[];identity=0.0
    for t in 2:length(d.weeks)
        d.weeks[t-1]>cutoff || continue
        ix=selection=="all_calendar" ? collect(eachindex(d.active)) : findall(elig[:,t-1].&elig[:,t])
        isempty(ix)&&continue
        v=z[ix,t]-z[ix,t-1];b=ST.mean(v)
        uu,cc,rr=ST.mean(v.^2),b^2,ST.mean((v.-b).^2)
        identity=max(identity,abs(uu-cc-rr));push!(u,uu);push!(c,cc);push!(r,rr)
    end
    stats=Dict("uncentered_energy"=>meanof(u),"common_energy"=>meanof(c),"relative_energy"=>meanof(r),
        "common_fraction"=>isempty(u)||sum(u)==0 ? missing : sum(c)/sum(u))
    (;stats,n_weeks=length(u),error=identity)
end
function r09()
    rows,identities,components=NamedTuple[],NamedTuple[],NamedTuple[]
    for league in SENTINELS
        p,config=panel(league);obs=point(league,"C0");sim=nulls(league)
        for season in config.seasons,axis in ("q","s"),selection in ("all_calendar","continuous_appearance")
            d=A.season_design(p,season)
            v=projection_stats(p,obs.X,d,season,sim.cutoff,selection,axis)
            nv=[projection_stats(p,X,d,season,sim.cutoff,selection,axis) for X in sim.paths]
            @assert maximum([v.error;getproperty.(nv,:error)])<=1e-10
            for (statistic,value) in sort(collect(v.stats);by=first)
                push!(rows,(;league,season,axis,selection,statistic,n_weeks=v.n_weeks,n_teams=length(d.active),value,
                    envelope([s.stats[statistic] for s in nv],value)...,status=ismissing(value) ? "NOT_AVAILABLE: empty suffix or zero denominator" : "OK; RTS mean energy, not latent process variance"))
            end
        end
        for axis in ("q","s")
            err,coverr,subseterr=0.0,0.0,0.0
            for season in config.seasons
                d=A.season_design(p,season);z=getproperty(A.centered_axes(p,obs.X,d),Symbol(axis));n=length(d.active)
                ts=[t for t in 2:length(d.weeks) if d.weeks[t-1]>sim.cutoff]
                isempty(ts)&&continue
                V=z[:,ts]-z[:,ts.-1]
                for v in eachcol(V)
                    direct=sum(v[i]*v[j] for i in 1:n for j in 1:n if i!=j)/(n*(n-1))
                    err=max(err,abs(direct+sum(v.^2)/(n*(n-1))))
                end
                if length(ts)>1
                    C=ST.cov(permutedims(V))
                    coverr=max(coverr,abs((sum(C)-LA.tr(C))/(n*(n-1))+LA.tr(C)/(n*(n-1))))
                end
                elig=A.eligibility(p,d,season)
                for t in ts
                    ix=findall(elig[:,t-1].&elig[:,t]);length(ix)>1||continue
                    v=z[ix,t]-z[ix,t-1];m=length(v)
                    subseterr=max(subseterr,abs((sum(v)^2-sum(v.^2))/(m*(m-1))+sum(v.^2)/(m*(m-1))))
                end
            end
            @assert max(err,coverr)<=1e-10
            for (identity,max_abs_error,estimator,status) in (
                ("offdiag_product_equals_negative_energy",err,"complete fixed-roster vector each week","PASS; redundant centered energy, not evidence of common shocks"),
                ("offdiag_temporal_cov_equals_negative_trace",coverr,"common weekly rows; sample covariance with temporal demeaning","PASS; sum-zero covariance identity"),
                ("R05_subset_deviation_from_complete_identity",subseterr,"R05 pair/appearance selection","DESCRIPTIVE; selected subset is not sum-zero; pairwise covariance uses different rows and temporal demeaning"))
                push!(identities,(;league,axis,identity,max_abs_error,estimator,status))
            end
        end
    end
    # Manager's scoped amendment: no scoring an unobserved common mode.
    # Verify C0 design annihilates both team-common modes and filtering preserves prior means.
    for (league,_,_) in QS.LEAGUES
        p,config=panel(league);f=fit(league,"C0","10b");theta=MID.median_theta(f)
        N=MID.n_teams(p);H=A.PM.fullbook_design(f.arm,p,theta)
        vs=zeros(size(H,2));vs[3:2+2N].=1
        vq=copy(vs);vq[3+N:2+2N].=-1
        loading=max(maximum(abs,H*vs),maximum(abs,H*vq))
        filtered=A.filter_store(f,p,theta,"C0")
        prior,_=MID.initial_state(MID.GRW1(),N)
        prior_s=ST.mean((prior[3:2+N]+prior[3+N:2+2N])/2)
        smean=vec(ST.mean((filtered.m_filt[3:2+N,:]+filtered.m_filt[3+N:2+2N,:])/2;dims=1))
        err=maximum(abs,smean.-prior_s)
        @assert loading<=1e-10 && err<=1e-10 "common-mode check failed: STOP and ask manager"
        for (identity,max_abs_error) in (("C0_design_common_loading_zero",loading),("C0_filtered_common_mean_equals_prior",err))
            push!(identities,(;league,axis="s",identity,max_abs_error,estimator="full-roster C0 10b; all weeks",status="PASS; prior-only common mode"))
        end
        for (quantity,value) in (("prior_common_mean",prior_s),("raw_training_end_b0",missing),("roster_n",missing))
            push!(components,(;league,quantity,value,n=N,cohort_source="full structural roster for loading/prior check only",status="NOT_IDENTIFIABLE: honest partition canceled by manager; C0 common style has zero rate loading"))
        end
    end
    output("R09","axis_projection.csv",rows)
    output("R09","projection_identity.csv",identities)
    output("R09","style_partition_components.csv",components)
    empty=DF.DataFrame(league=String[],season=String[],week=Int[],fixture_id=Int[],variant=String[],D=Float64[],M=Float64[],b0=Float64[],b_t=Float64[],relative_sum=Float64[],logp_joint=Float64[],logp_total=Float64[],logp_allocation=Float64[],method=String[])
    output("R09","style_partition_fixtures.csv",empty;bulk=true)
    lines=["PARTIAL (manager amendment). Raw RTS suffix energies and matched99-null ranks: axis_projection.csv; identities: projection_identity.csv. Fixed active roster, actual calendar transitions; old continuous-appearance selection shown separately."]
    for league in SENTINELS
        g=filter(r->r.league==league&&r.axis=="s"&&r.selection=="all_calendar"&&r.statistic=="common_fraction",rows)
        push!(lines,"$league style common-energy fraction/rank: "*join(["$(r.season) $(r.value)/$(r.null_rank)" for r in g],"; ")*". Active-cohort mean motion is relative to the full structural roster, not a directly observed league intensity; no 1/n process-variance claim.")
    end
    push!(lines,"Complete centered off-diagonal products = −sum(v²)/(n(n−1)); aligned temporal covariances = −trace(C)/(n(n−1)), max error $(maximum(r.max_abs_error for r in identities if startswith(r.identity,"offdiag"))). R05 used appearance subsets and pair-specific temporal covariance rows: not common-shock evidence.")
    push!(lines,"ALL11 C0 10b: design common-mode max loading $(maximum(r.max_abs_error for r in identities if r.identity=="C0_design_common_loading_zero")); filtered style common mean minus prior max $(maximum(r.max_abs_error for r in identities if r.identity=="C0_filtered_common_mean_equals_prior")). Check passes: the full-structural-roster common mode is prior-only. The smaller season-cohort mean can move relative to teams outside that cohort; its retrospective energy is not the engine's unobserved common mode.")
    push!(lines,"Honest partition, b0, component score variances and full−partition scores NOT_IDENTIFIABLE under C0; no literal uncentered formula computed, no gauge shift or dynamic mu silently introduced. Gauge/allocation/leakage assertions for these canceled scores are NOT_RUN, not passes.")
    push!(lines,"A league scoring state needs dynamic mu_t in a future fit (out of scope). Existing related evidence per manager: earlier B2 weekly-shock ICC ≈0.10 retrospective, ≈0 honest; not recomputed here. These energy diagnostics do not estimate OU or goal benefit.")
    summary!("R09",lines)
    verify("R09","PASS energy decomposition observed/all99 nulls absolute tolerance1e−10 (finite squared log-rates); full-centered product/covariance identities ≤1e−10. ALL11 common loading/prior-mean checks ≤1e−10. Canceled honest partition NOT_IDENTIFIABLE; no forecast scoring or future-cohort selection performed; no new draws.")
end
