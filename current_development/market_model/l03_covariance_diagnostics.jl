# Included inside CovarianceMarket; deterministic evaluation and diagnostics only.

"Exact Gaussian pre-week axes; nonlinear log-total uses the Phase B seeded MC/KDE convention."
function covariance_predictions(a,p,theta; seed=3906,mc_draws=4000)
    f = covariance_filter(a,p,theta; predict=true)
    rng = Random.Xoshiro(seed+999999)
    rows = NamedTuple[]
    for m in 1:MID.n_fixtures(p)
        j = 2m-1
        h,away = f.pair_mean[m,:]
        vh,va,cross = f.pair_cov[m,1,1],f.pair_cov[m,2,2],f.pair_cov[m,1,2]
        actual = TB.axis_values(p.obs_y[j],p.obs_y[j+1])
        means = (h,away,h-away,(h+away)/2)
        vars = (vh,va,vh+va-2cross,(vh+va+2cross)/4)
        for k in 1:4
            dist = DS.Normal(means[k],sqrt(vars[k]))
            cdf = DS.cdf(dist,actual[k])
            push!(rows,(; match_id=p.obs_match[j],week=p.obs_week[j],axis=TB.AXES[k],
                observed=actual[k],predicted=means[k],variance=vars[k],logpd=DS.logpdf(dist,actual[k]),
                cover90=0.05<=cdf<=0.95,method="conditional Gaussian mixture"))
        end
        draws = zeros(mc_draws)
        for d in eachindex(draws)
            rand(rng) # retain Phase B's one-particle mixture RNG convention
            z1,z2 = randn(rng,2)
            yh = h+sqrt(vh)*z1
            ya = away+cross/sqrt(vh)*z1+sqrt(max(va-cross^2/vh,0))*z2
            draws[d] = TB.axis_values(yh,ya)[5]
        end
        bw = max(1.06ST.std(draws)*length(draws)^(-0.2),1e-6)
        density = TB.logsumexp(DS.logpdf.(DS.Normal.(draws,bw),actual[5]))-log(length(draws))
        push!(rows,(; match_id=p.obs_match[j],week=p.obs_week[j],axis=TB.AXES[5],
            observed=actual[5],predicted=ST.mean(draws),variance=ST.var(draws),logpd=density,
            cover90=ST.quantile(draws,0.05)<=actual[5]<=ST.quantile(draws,0.95),method="predictive MC / KDE density"))
    end
    return (; rows=DF.DataFrame(rows),loglik=f.loglik)
end

"Full Gaussian RTS moments, including static prior directions."
function smoothing_moments(arm,p,sch,f)
    means = copy(f.m_filt)
    covs = copy(f.P_filt)
    N = MID.n_teams(p)
    for t in (p.n_weeks-1):-1:1
        Pt = f.P_filt[:,:,t]
        Ppred = LA.Symmetric(Pt+MID.process_cov(arm,N,sch,t+1))
        C = LA.cholesky(Ppred; check=false)
        J = LA.issuccess(C) ? (C\Pt')' : Pt*LA.pinv(Matrix(Ppred))
        means[:,t] .= f.m_filt[:,t]+J*(means[:,t+1]-f.m_filt[:,t])
        cov = Pt+J*(covs[:,:,t+1]-Ppred)*J'
        covs[:,:,t] .= (cov+cov')/2
    end
    return means,covs
end

"Linear RTS means; log-total is the MC mean of its smoothed latent-rate distribution."
function covariance_smoothed(fit,p; seed=3923,mc_draws=4000)
    a,theta = fit.arm,MID.median_theta(fit)
    if a isa TB.Rung
        f = TB.rung_filter(a,p,theta; store=true)
        arm,sch = a,TB.rung_schedule(a,p,theta)
    else
        f = covariance_filter(a,p,theta; store=true)
        arm,sch = conditional_arm(a,theta),covariance_schedule(a,p,theta)
    end
    means,covs = smoothing_moments(arm,p,sch,f)
    rng = Random.Xoshiro(seed)
    hh,ha = zeros(size(means,1)),zeros(size(means,1))
    rows = NamedTuple[]
    for m in 1:MID.n_fixtures(p)
        j,t = 2m-1,p.obs_week[2m-1]
        MID.obs_row!(hh,p,j,0)
        MID.obs_row!(ha,p,j+1,0)
        h,away = LA.dot(hh,means[:,t]),LA.dot(ha,means[:,t])
        vh,va,cross = LA.dot(hh,covs[:,:,t]*hh),LA.dot(ha,covs[:,:,t]*ha),LA.dot(hh,covs[:,:,t]*ha)
        sh = sqrt(max(vh,1e-300))
        draws = [TB.axis_values(h+sh*z[1],away+cross/sh*z[1]+sqrt(max(va-cross^2/max(vh,1e-300),0))*z[2])[5]
            for z in (randn(rng,2) for _ in 1:mc_draws)]
        observed = TB.axis_values(p.obs_y[j],p.obs_y[j+1])
        predicted = (h,away,h-away,(h+away)/2,ST.mean(draws))
        for k in 1:5
            push!(rows,(; rung=MID.arm_name(a),match_id=p.obs_match[j],axis=TB.AXES[k],
                observed=observed[k],predicted=predicted[k],residual=observed[k]-predicted[k],
                method=k == 5 ? "RTS log-total MC at median theta" : "RTS at median theta"))
        end
    end
    return DF.DataFrame(rows),f.loglik
end

"Consecutive team-signed supremacy errors WITHIN a season, sorted by date then match ID."
function persistence_pairs(events::DF.AbstractDataFrame)
    pairs = NamedTuple[]
    for g in DF.groupby(events,[:team,:season])
        ordered = DF.sort(DF.DataFrame(g),[:date,:match_id])
        for j in 2:DF.nrow(ordered)
            push!(pairs,(; team=String(ordered.team[j]),previous=ordered.residual[j-1],current=ordered.residual[j]))
        end
    end
    return DF.DataFrame(pairs)
end

"Pooled Pearson lag-1 correlation; percentile CI resamples whole TEAM clusters across seasons."
function persistence_statistic(events; reps=2000,seed=3924)
    pairs = persistence_pairs(events)
    teams = sort(unique(pairs.team))
    by_team = Dict(t=>findall(==(t),pairs.team) for t in teams)
    rng = Random.Xoshiro(seed)
    boot = zeros(reps)
    for b in 1:reps
        indices = reduce(vcat,[by_team[t] for t in rand(rng,teams,length(teams))])
        boot[b] = ST.cor(pairs.previous[indices],pairs.current[indices])
    end
    return (; n_pairs=DF.nrow(pairs),n_teams=length(teams),autocorrelation=ST.cor(pairs.previous,pairs.current),
        ci_low=ST.quantile(boot,0.025),ci_high=ST.quantile(boot,0.975),reps,seed)
end

function team_signed_events(panel,fixtures)
    meta = Dict(r.match_id=>r for r in eachrow(panel.matches))
    events = NamedTuple[]
    for r in eachrow(DF.filter(:axis=>==("supremacy"),fixtures))
        m = meta[r.match_id]
        for (team,sign) in ((m.home_team,1),(m.away_team,-1))
            push!(events,(; team,season=m.season,date=m.match_date,match_id=r.match_id,residual=sign*(r.observed-r.predicted)))
        end
    end
    return DF.DataFrame(events)
end

function residual_diagnostics(panel,fixtures,out)
    correlations,persistence = NamedTuple[],NamedTuple[]
    for g in DF.groupby(fixtures,[:rung,:protocol])
        h = DF.filter(:axis=>==("log_lambda_h"),g)
        a = DF.filter(:axis=>==("log_lambda_a"),g)
        h.match_id == a.match_id || error("side residuals are not paired")
        push!(correlations,(; rung=first(g.rung),protocol=first(g.protocol),n=DF.nrow(h),
            correlation=ST.cor(h.observed-h.predicted,a.observed-a.predicted)))
        statistic = persistence_statistic(team_signed_events(panel,g))
        push!(persistence,(; rung=first(g.rung),protocol=first(g.protocol),statistic...))
    end
    CSV.write(joinpath(out,"residual_correlation.csv"),DF.DataFrame(correlations))
    CSV.write(joinpath(out,"persistence.csv"),DF.DataFrame(persistence))
    # The requested decision score is the sum of d and level marginal log densities,
    # not a joint Gaussian density (the predictive state can correlate those axes).
    selected = DF.filter(r -> r.protocol == "10b" && r.axis in ("supremacy","level"),fixtures)
    scores = DF.combine(DF.groupby(selected,[:rung,:match_id]),:logpd=>sum=>:logpd)
    r2 = DF.select(DF.filter(:rung=>==("R2"),scores),:match_id,:logpd=>:r2_logpd)
    paired = NamedTuple[]
    for rung in ("R5","R6","R7")
        g = DF.innerjoin(DF.filter(:rung=>==(rung),scores),r2; on=:match_id,order=:left)
        DF.nrow(g) == DF.nrow(r2) || error("paired comparison lost fixtures")
        delta = g.logpd-g.r2_logpd
        push!(paired,(; rung,protocol="10b",n=length(delta),mean_delta=ST.mean(delta),
            se=ST.std(delta)/sqrt(length(delta)),sum_delta=sum(delta),
            score="sum of supremacy and level marginal logpd"))
    end
    CSV.write(joinpath(out,"paired_vs_r2.csv"),DF.DataFrame(paired))
    return DF.DataFrame(persistence)
end

function noise_equivalence(fits,out)
    rows = NamedTuple[]
    for protocol in ("10a","10b"), level in (2,5,6,7)
        f = fits[(level,protocol)]
        if level in (5,7)
            vd,vl = vec(f.draws[:,1,:]).^2,vec(f.draws[:,2,:]).^2
            s2 = (vd+4vl)/4
            rho = (4vl-vd)./(4vl+vd)
        else
            s2 = vec(f.draws[:,1,:]).^2
            rho = zeros(length(s2))
        end
        for (name,values) in (("side_variance",s2),("rho_epsilon",rho))
            q = ST.quantile(values,[0.05,0.5,0.95])
            push!(rows,(; rung="R$level",protocol,parameter=name,q05=q[1],median=q[2],q95=q[3]))
        end
    end
    result = DF.DataFrame(rows)
    CSV.write(joinpath(out,"noise_equivalence.csv"),result)
    return result
end

function evaluate_covariance(panel,config,fits,out)
    smooth,metrics,fixtures = DF.DataFrame[],DF.DataFrame[],DF.DataFrame[]
    likelihoods = Dict{Int,Float64}()
    for level in (2,5,6,7)
        sr,ll = covariance_smoothed(fits[(level,"10a")],panel)
        push!(smooth,sr)
        likelihoods[level] = ll
        for protocol in ("10a","10b")
            fit = fits[(level,protocol)]
            theta = MID.median_theta(fit)
            pred = level == 2 ? TB.preweek_predictions(TB.Rung(2),panel,theta) :
                covariance_predictions(CovarianceRung(level),panel,theta)
            push!(metrics,TB.prediction_summary(panel,pred,protocol,"R$level"; config))
            pred.rows.rung .= "R$level"
            pred.rows.protocol .= protocol
            allowed = Set(panel.matches.match_id[in.(panel.matches.season,Ref(config.honest_test))])
            push!(fixtures,protocol == "10a" ? pred.rows : DF.filter(r -> r.match_id in allowed,pred.rows))
        end
    end
    sm,met,raw = vcat(smooth...),vcat(metrics...),vcat(fixtures...)
    CSV.write(joinpath(out,"smoothed_fit_b2.csv"),sm)
    CSV.write(joinpath(out,"onestep_metrics_b2.csv"),met)
    CSV.write(joinpath(out,"onestep_fixture_b2.csv"),raw)
    ladder = NamedTuple[]
    for level in (2,5,6,7)
        fit = fits[(level,"10a")]
        row = Dict{Symbol,Any}(:rung=>"R$level",:collapsed_loglik=>likelihoods[level],
            :conditional_smoothed_loglik=>likelihoods[level],:loglik_method=>"exact Kalman at median theta")
        for name in ("sigma_obs","sigma_sup","sigma_lev","sigma_att","sigma_def","rho_ab")
            j = findfirst(==(name),fit.names)
            q = j === nothing ? fill(NaN,3) : ST.quantile(vec(fit.draws[:,j,:]),[0.05,0.5,0.95])
            for (suffix,value) in zip(("q05","median","q95"),q)
                row[Symbol(name,"_",suffix)] = value
            end
        end
        for axis in TB.AXES
            s = DF.filter(r -> r.rung == "R$level" && r.axis == axis,sm)
            row[Symbol("smoothed_r2_",axis)] = 1-sum(s.residual.^2)/sum((s.observed.-ST.mean(s.observed)).^2)
            row[Symbol("smoothed_rmse_",axis)] = sqrt(ST.mean(s.residual.^2))
            for protocol in ("10a","10b")
                g = DF.filter(r -> r.rung == "R$level" && r.protocol == protocol && r.axis == axis && r.subset == "all",met)
                row[Symbol("onestep_rmse_",protocol,"_",axis)] = only(g.rmse)
            end
        end
        keys_sorted = sort!(collect(keys(row)); by=String)
        push!(ladder,NamedTuple{Tuple(keys_sorted)}(Tuple(row[k] for k in keys_sorted)))
    end
    CSV.write(joinpath(out,"ladder_summary_b2.csv"),DF.DataFrame(ladder))
    persistence = residual_diagnostics(panel,raw,out)
    noise_equivalence(fits,out)
    return (; metrics=met,persistence)
end

function covariance_figures(P,evaluation,fig)
    p = P.plot(layout=(1,2),size=(1050,420),bottom_margin=8P.mm,left_margin=8P.mm)
    for (k,axis) in enumerate(("supremacy","level")), protocol in ("10a","10b")
        g = DF.filter(r -> r.axis == axis && r.protocol == protocol && r.subset == "all",evaluation.metrics)
        P.scatter!(p[k],g.rung,100g.cover90; label=protocol,xlabel="rung",ylabel="90% coverage (%)",title=axis)
        P.hline!(p[k],[90]; label="",color=:black,linestyle=:dash)
    end
    P.savefig(p,joinpath(fig,"B2_coverage.png"))
    p = P.plot(layout=(1,2),size=(1050,420),bottom_margin=8P.mm,left_margin=8P.mm)
    for (k,protocol) in enumerate(("10a","10b"))
        g = DF.filter(:protocol=>==(protocol),evaluation.persistence)
        P.scatter!(p[k],g.rung,g.autocorrelation; yerror=(g.autocorrelation-g.ci_low,g.ci_high-g.autocorrelation),
            label="team-cluster 95% CI",xlabel="rung",ylabel="team-signed lag-1 correlation",title=protocol)
        P.hline!(p[k],[0]; label="",color=:black,linestyle=:dash)
    end
    P.savefig(p,joinpath(fig,"B2_persistence.png"))
    return nothing
end
