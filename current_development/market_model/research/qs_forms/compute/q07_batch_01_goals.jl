# Included inside QSFormsBatch01. Goal deletions reuse C0's unchanged honest forward states.
function state_rates(p,state,fixture;variant="full",frozen_s=nothing)
    N=MID.n_teams(p)
    a=state[3:2+N].-ST.mean(state[3:2+N]);b=state[3+N:2+2N].-ST.mean(state[3+N:2+2N])
    q,s=(a-b)/2,(a+b)/2
    variant=="frozen_style"&&(s=frozen_s)
    h,away=p.obs_att[2fixture-1],p.obs_att[2fixture]
    return QS.goal_rates(state[1],state[2],q[h],q[away],s[h],s[away];zero_quality=variant=="no_quality",zero_style=variant=="no_style")
end
function goal_channels(h,a,rates)
    joint=QS.goal_logscore(h,a,rates)
    total=DS.logpdf(DS.Poisson(sum(rates)),h+a)
    return (;joint,total,allocation=joint-total)
end
function r07()
    started=time();rows,checks=NamedTuple[],NamedTuple[]
    goals=CSV.read(joinpath(Q,"goal_fixtures.csv"),DF.DataFrame;stringtype=String)
    @assert DF.nrow(goals)==3053
    record!("R07","input_goals","goal_fixtures.csv",joinpath(Q,"goal_fixtures.csv");n=3053,columns=join(names(goals),";"),detail="published complete honest goal intersection; no outcome refetch")
    for (li,(league,_,_)) in enumerate(QS.LEAGUES)
        p,config=panel(league);f=fit(league,"C0","10b");theta=MID.median_theta(f)
        medianpred=forward(f,p,theta,"C0";keep_states=true,min_state_week=minimum(p.matches.week[in.(p.matches.season,Ref(config.honest_test))]))
        trainend=maximum(p.obs_week[in.(p.obs_season,Ref(config.honest_train))])
        filtered=filter_store(f,p,theta,"C0");frozen=filtered.m_filt[:,trainend]
        N=MID.n_teams(p)
        frozen_s=(frozen[3:2+N]+frozen[3+N:2+2N])/2;frozen_s.-=ST.mean(frozen_s)
        # Full roster exists at initialization, including teams never directly seen in training.
        # Their frozen values use only their training-conditioned entry prior, never any test book.
        cache=Serialization.deserialize(joinpath(OUT,"cache","$(league)_C0_10b_forecasts.jls"))
        @assert cache.indices==indices(f)
        gg=DF.filter(:league=>==(league),goals)
        fixturemap=Dict(p.matches.match_id[i]=>i for i in 1:DF.nrow(p.matches))
        ix=[fixturemap[id] for id in gg.match_id]
        @assert Set(ix)==Set(findall(in.(p.matches.season,Ref(config.honest_test))))
        for (gi,i) in enumerate(ix)
            fixture=p.matches[i,:];target=gg[gi,:];state=medianpred.states[fixture.week].m
            for variant in ("full","no_style","no_quality","frozen_style")
                scores=goal_channels(target.goals_home,target.goals_away,state_rates(p,state,i;variant,frozen_s))
                if variant!="frozen_style"
                    expected=target[variant=="full" ? :full_logpd : variant=="no_style" ? :no_style_logpd : :no_quality_logpd]
                    err=abs(scores.joint-expected)
                    push!(checks,(;league,fixture_id=fixture.match_id,variant,error=err,pass=err<=1e-8))
                end
                @assert abs(scores.joint-scores.total-scores.allocation)<=1e-10
                push!(rows,(;league,season=fixture.season,week=fixture.week,fixture_id=fixture.match_id,variant,method="plugin_median_theta",logp_joint=scores.joint,logp_total=scores.total,logp_allocation=scores.allocation,n_latent_samples=0))
            end
        end
        @assert all(r.pass for r in checks)
        # Current-state joint samples are reused across all fixtures and deletions of a forecast week.
        samples=Vector{Any}(undef,length(cache.predictions))
        Threads.@threads for j in eachindex(cache.predictions)
            draws=Dict{Int,Matrix{Float64}}()
            for (week,pre) in cache.predictions[j].states
                @assert pre.last_observed<week
                rng=Random.Xoshiro(SEED+100_000_000li+3_000_000+200_000+1000j+week)
                X=zeros(length(pre.m),4)
                for rep in 1:4;MID.draw_psd!(view(X,:,rep),rng,pre.m,pre.V);end
                draws[week]=X
            end
            samples[j]=draws
        end
        for (gi,i) in enumerate(ix)
            fixture=p.matches[i,:];target=gg[gi,:]
            for variant in ("full","no_style","no_quality")
                scores=[goal_channels(target.goals_home,target.goals_away,state_rates(p,samples[j][fixture.week][:,rep],i;variant)) for j in eachindex(samples),rep in 1:4]
                for ns in (2,4)
                    joint=logmeanexp(vec(getproperty.(scores[:,1:ns],:joint)))
                    total=logmeanexp(vec(getproperty.(scores[:,1:ns],:total)))
                    allocation=joint-total
                    @assert abs(joint-total-allocation)<=1e-10
                    push!(rows,(;league,season=fixture.season,week=fixture.week,fixture_id=fixture.match_id,variant,method="mixture_128x$ns",logp_joint=joint,logp_total=total,logp_allocation=allocation,n_latent_samples=length(samples)*ns))
                end
            end
        end
        record!("R07","state_samples",league,"";n=length(samples)*4,detail="4 joint current-state draws per theta/week; seed=610801+100000000*league_index+3000000+200000+1000*theta_index+calendar_week; all variants/fixtures share samples; frozen-style integrated NOT_AVAILABLE: no joint frozen/current conditional cache")
        println("R07 $league n=$(DF.nrow(gg))");flush(stdout)
    end
    result=output("R07","goal_ablation_fixtures.csv",rows);output("R07","goal_reproduction_checks.csv",checks)
    summaries=NamedTuple[];keys=[:league,:season,:week,:fixture_id,:method]
    full=DF.filter(:variant=>==("full"),result)
    for alternative in ("no_style","no_quality","frozen_style")
        alt=DF.filter(:variant=>==(alternative),result)
        paired=DF.innerjoin(full,alt;on=keys,makeunique=true)
        @assert DF.nrow(paired)==DF.nrow(alt)
        for method in unique(paired.method),channel in ("joint","total","allocation"),block in (4,8,12)
            gm=DF.filter(:method=>==(method),paired);col=Symbol("logp_"*channel);other=Symbol("logp_"*channel*"_1")
            perleague=NamedTuple[]
            for (li,(league,_,_)) in enumerate(QS.LEAGUES)
                g=DF.filter(:league=>==(league),gm);delta=g[!,col]-g[!,other]
                boots=[ST.mean(delta[ix]) for ix in bootstrap_indices(g;block,rng=Random.Xoshiro(seed(li,3,2,700+block)))];ci=quant(boots)
                push!(perleague,(;n=length(delta),mean=ST.mean(delta),boots))
                mcchange=missing
                if startswith(method,"mixture")
                    g2=DF.filter(r->r.league==league&&r.method=="mixture_128x2",paired)
                    g4=DF.filter(r->r.league==league&&r.method=="mixture_128x4",paired)
                    mcchange=ST.mean(g4[!,col]-g4[!,other])-ST.mean(g2[!,col]-g2[!,other])
                end
                push!(summaries,(;league,weighting="fixture",method,comparison="full_minus_$alternative",channel,n=length(delta),mean_delta=ST.mean(delta),boot_q05=ci[1],boot_q95=ci[3],block_weeks=block,mc_delta_change=mcchange,status=!ismissing(mcchange)&&abs(mcchange)>.001 ? "MC_UNSTABLE" : "OK"))
            end
            for weighting in ("fixture","equal_league")
                weights=weighting=="fixture" ? [g.n for g in perleague] : ones(length(perleague));weights=weights/sum(weights)
                boot=sum(weights[i]*perleague[i].boots for i in eachindex(perleague));ci=quant(boot)
                mcchange=missing
                if startswith(method,"mixture")
                    changes=[only([s.mc_delta_change for s in summaries if s.league==league&&s.method==method&&s.comparison=="full_minus_$alternative"&&s.channel==channel&&s.block_weeks==block]) for (league,_,_) in QS.LEAGUES]
                    mcchange=sum(weights.*changes)
                end
                push!(summaries,(;league="ALL",weighting,method,comparison="full_minus_$alternative",channel,n=sum(g.n for g in perleague),mean_delta=sum(weights[i]*perleague[i].mean for i in eachindex(perleague)),boot_q05=ci[1],boot_q95=ci[3],block_weeks=block,mc_delta_change=mcchange,status=!ismissing(mcchange)&&abs(mcchange)>.001 ? "MC_UNSTABLE" : "OK"))
            end
        end
    end
    summary=output("R07","goal_ablation_summary.csv",summaries)
    lines=["PARTIAL: all published3053 fixtures reproduced (≤1e−8); [goal_reproduction_checks.csv](goal_reproduction_checks.csv), [goal_ablation_fixtures.csv](goal_ablation_fixtures.csv), [goal_ablation_summary.csv](goal_ablation_summary.csv). Integrated frozen-style NOT_AVAILABLE: no joint training-end/current conditional cache; plug-in frozen-style is available, never independently spliced."]
    for comparison in ("full_minus_no_style","full_minus_frozen_style")
        method=comparison=="full_minus_frozen_style" ? "plugin_median_theta" : "mixture_128x4"
        for weighting in ("fixture","equal_league")
            g=DF.filter(r->r.league=="ALL"&&r.weighting==weighting&&r.method==method&&r.comparison==comparison&&r.block_weeks==8,summary)
            push!(lines,"$comparison/$method/$weighting "*join(["$(r.channel)=$(round(r.mean_delta;digits=5)) [$(round(r.boot_q05;digits=5)),$(round(r.boot_q95;digits=5))],MCchange=$(ismissing(r.mc_delta_change) ? "NA" : round(r.mc_delta_change;digits=5)),$(r.status)" for r in eachrow(g)],"; ")*".")
        end
    end
    for league in ("sco_league_two","sco_premiership","fin_veikkausliiga")
        g=DF.filter(r->r.league==league&&r.method=="mixture_128x4"&&r.comparison=="full_minus_no_style"&&r.block_weeks==8,summary)
        push!(lines,"$league full−no-style "*join(["$(r.channel) $(round(r.mean_delta;digits=5)) [$(round(r.boot_q05;digits=5)),$(round(r.boot_q95;digits=5))] $(r.status)" for r in eachrow(g)],"; ")*".")
    end
    push!(lines,"Mixture allocation=joint−total exactly (not equally weighted Binomial mixing). MC_UNSTABLE means first2 versus all4 samples change delta by>0.001; no extra samples. Ablations retain C0 filter/q/mu/gamma: deletion scores do not compare refitted models.")
    summary!("R07",lines);verification("R07","PASS published fixture intersection3053 and every plug-in joint score ≤1e−8; joint=total+allocation ≤1e−10; same whole-week joint current-state samples across variants, same128 indices as R03; first2/all4 MC check retained. Runtime=$(time()-started)s; invocation Q07.r07().");flush_manifest!()
end
