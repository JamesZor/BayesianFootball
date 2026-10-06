# Included in CopulaGridMarket; no execution or fits on include.

"Active step endpoints: first through last accepted match of a team in each season, t>=2."
function active_step_keys(p)
    keys = Tuple{Int,Int,String}[]
    for season in unique(p.obs_season), i in 1:MID.n_teams(p)
        weeks = p.obs_week[(p.obs_season .== season) .& ((p.obs_att .== i) .| (p.obs_def .== i))]
        isempty(weeks) && continue
        for t in max(2,minimum(weeks)):maximum(weeks)
            push!(keys,(i,t,season))
        end
    end
    return keys
end

"Observable centred alpha/beta paths; remove unidentifiable common-team gauge at every week."
function path_steps(X,p,keys)
    N = MID.n_teams(p)
    attack = X[3:2+N,:]
    concessions = X[3+N:2+2N,:]
    attack = attack .- ST.mean(attack; dims=1)
    concessions = concessions .- ST.mean(concessions; dims=1)
    da = [attack[i,t]-attack[i,t-1] for (i,t,_) in keys]
    db = [concessions[i,t]-concessions[i,t-1] for (i,t,_) in keys]
    return (; da,db,dq=(da-db)/2,ds=(da+db)/2)
end

"Kendall tau-a for continuous steps, via inversion counting (O(n log n))."
function kendall_tau(x,y)
    n = length(x)
    order = sortperm(x)
    ranks = invperm(sortperm(y))
    tree = zeros(Int,n)
    inversions = 0
    for (i,k) in enumerate(order)
        r = ranks[k]
        lower = 0
        j = r
        while j > 0
            lower += tree[j]
            j -= j & -j
        end
        inversions += i-1-lower
        j = r
        while j <= n
            tree[j] += 1
            j += j & -j
        end
    end
    return 1-4inversions/(n*(n-1))
end
excess_kurtosis(x) = ST.mean((x .- ST.mean(x)).^4)/ST.mean((x .- ST.mean(x)).^2)^2-3

"Empirical one-sided 95% thresholds per draw, split improvement/collapse directions."
function step_statistics(steps,keys,theta)
    da,db,dq,ds = steps.da,steps.db,steps.dq,steps.ds
    rows = Dict{String,Float64}()
    for (name,v) in (("alpha",da),("beta",db),("quality",dq),("style",ds))
        rows["kurtosis_$name"] = excess_kurtosis(v)
    end
    rows["kendall_alpha_beta"] = kendall_tau(da,db)
    rows["kendall_alpha_beta_minus_gaussian"] = rows["kendall_alpha_beta"]-2/π*asin(tanh(theta[4]))
    sa,sb,rho = exp(theta[2]),exp(theta[3]),tanh(theta[4])
    vq = (sa^2+sb^2-2rho*sa*sb)/4
    vs = (sa^2+sb^2+2rho*sa*sb)/4
    qs_rho = (sa^2-sb^2)/(4sqrt(vq*vs))
    rows["kendall_quality_style"] = kendall_tau(dq,ds)
    rows["kendall_quality_style_minus_gaussian"] = rows["kendall_quality_style"]-2/π*asin(clamp(qs_rho,-1,1))
    ahi,alo = ST.quantile(da,0.95),ST.quantile(da,0.05)
    bhi,blo = ST.quantile(db,0.95),ST.quantile(db,0.05)
    rows["joint_improvement_95"] = ST.mean((da .> ahi) .& (db .< blo))
    rows["joint_collapse_95"] = ST.mean((da .< alo) .& (db .> bhi))
    rows["joint_direction_asymmetry"] = rows["joint_improvement_95"]-rows["joint_collapse_95"]
    qhi,qlo = ST.quantile(dq,0.95),ST.quantile(dq,0.05)
    shi,slo = ST.quantile(ds,0.95),ST.quantile(ds,0.05)
    rows["joint_quality_style_upper_95"] = ST.mean((dq .> qhi) .& (ds .> shi))
    rows["joint_quality_style_lower_95"] = ST.mean((dq .< qlo) .& (ds .< slo))
    lookup = Dict(key=>k for (k,key) in enumerate(keys))
    previous,current = Int[],Int[]
    for (k,(i,t,s)) in enumerate(keys)
        before = get(lookup,(i,t-1,s),0)
        before == 0 && continue
        push!(previous,before)
        push!(current,k)
    end
    for (name,v) in (("quality",dq),("style",ds),("alpha",da),("beta",db))
        rows["lag1_squared_$name"] = ST.cor(v[previous].^2,v[current].^2)
    end
    return rows
end

function panel_with_observations(p,y)
    return MID.MarketPanel(p.matches,p.refusals,p.teams,p.n_weeks,p.week_start,
        p.obs_week,p.obs_home,p.obs_att,p.obs_def,y,p.obs_match,p.obs_season,
        p.week_ptr,p.obs_X,p.feature_names)
end

"Replicate same fixtures/weeks, conditioning on drawn initial states; all new steps/noise Gaussian."
function simulate_panel(p,theta,rng; initial=nothing,heavy_quality=false)
    arm = CM.CovarianceRung(6)
    conditional = CM.conditional_arm(arm,theta)
    N,T = MID.n_teams(p),p.n_weeks
    m0,P0 = MID.initial_state(conditional,N)
    X = zeros(length(m0),T)
    if initial === nothing
        MID.draw_psd!(view(X,:,1),rng,m0,P0)
    else
        X[:,1] .= initial
    end
    sa,sb,rho = exp(theta[2]),exp(theta[3]),tanh(theta[4])
    Q = [sa^2 rho*sa*sb; rho*sa*sb sb^2]
    A = [0.5 -0.5; 0.5 0.5]
    L = LA.cholesky(LA.Symmetric(heavy_quality ? A*Q*A' : Q)).L
    for t in 2:T
        X[:,t] .= X[:,t-1]
        for i in 1:N
            z = randn(rng,2)
            heavy_quality && (z[1] = rand(rng,DS.TDist(3))/sqrt(3))
            innovation = L*z
            heavy_quality && (innovation = [innovation[1]+innovation[2],innovation[2]-innovation[1]])
            X[2+i,t] += innovation[1]
            X[2+N+i,t] += innovation[2]
        end
    end
    y = MID.fitted_logrates(p,X)+exp(theta[1])*randn(rng,MID.n_obs(p))
    return panel_with_observations(p,y),X
end

"Observed FFBS and independently regenerated/refiltered replicate FFBS, at EACH theta draw."
function tail_ppc(p,thetas; draws=200,seed=3942)
    draws >= 200 || error("PPC requires at least 200 posterior draws")
    keys = active_step_keys(p)
    isempty(keys) && error("no active team steps")
    arm = CM.CovarianceRung(6)
    selected = round.(Int,range(1,size(thetas,1); length=draws))
    tables = Vector{Vector{NamedTuple}}(undef,draws)
    quality = zeros(length(keys),draws)
    Threads.@threads for d in 1:draws
        rng = Random.Xoshiro(seed+1000d)
        theta = vec(thetas[selected[d],:])
        sch = CM.covariance_schedule(arm,p,theta)
        conditional = CM.conditional_arm(arm,theta)
        f = CM.covariance_filter(arm,p,theta; store=true)
        X = MID.ffbs(conditional,p,sch,f,rng)
        steps = path_steps(X,p,keys)
        quality[:,d] .= steps.dq
        observed = step_statistics(steps,keys,theta)
        rep,_ = simulate_panel(p,theta,rng; initial=X[:,1])
        rf = CM.covariance_filter(arm,rep,theta; store=true)
        RX = MID.ffbs(conditional,rep,sch,rf,rng)
        replicated = step_statistics(path_steps(RX,rep,keys),keys,theta)
        tables[d] = [(; draw=d,theta_index=selected[d],statistic=name,
            observed=observed[name],replicated=replicated[name],n_steps=length(keys))
            for name in sort(collect(Base.keys(observed)))]
    end
    return DF.DataFrame(reduce(vcat,tables)),quality,keys
end

function tail_summary(raw)
    rows = NamedTuple[]
    for g in DF.groupby(raw,[:protocol,:statistic])
        pvalue = ST.mean(g.replicated .>= g.observed)
        qo,qr = ST.quantile(g.observed,[0.05,0.5,0.95]),ST.quantile(g.replicated,[0.05,0.5,0.95])
        push!(rows,(; protocol=first(g.protocol),statistic=first(g.statistic),draws=DF.nrow(g),
            n_steps=first(g.n_steps),ppp=pvalue,extreme=pvalue < 0.01 || pvalue > 0.99,
            obs_q05=qo[1],obs_q50=qo[2],obs_q95=qo[3],rep_q05=qr[1],rep_q50=qr[2],rep_q95=qr[3]))
    end
    return DF.DataFrame(rows)
end

"Rank posterior median signed quality steps, retaining both protocol-specific tables."
function largest_steps(p,quality,keys,protocol; n=15)
    medians = [ST.median(quality[k,:]) for k in axes(quality,1)]
    order = sortperm(abs.(medians); rev=true)[1:min(n,length(keys))]
    rows = NamedTuple[]
    for (rank,k) in enumerate(order)
        i,t,season = keys[k]
        q = ST.quantile(quality[k,:],[0.05,0.95])
        push!(rows,(; protocol,rank,team=p.teams[i],season,week=t,date=p.week_start[t],
            delta_q=medians[k],abs_delta_q=abs(medians[k]),q05=q[1],q95=q[2]))
    end
    return DF.DataFrame(rows)
end

"Same R6 priors, 4 chains x (2000 warmup + 3000 retained), thin 1, seeded as B2."
function refit_r6(p,config,out,label; seed=3922)
    fits = Dict{String,MID.ArmFit}()
    diagnostics = DF.DataFrame[]
    sha = strip(read(`git rev-parse HEAD`,String))
    for protocol in ("10a","10b")
        pp = protocol == "10a" ? p : MID.restrict_panel(p,in.(p.obs_season,Ref(config.honest_train)))
        runseed = seed+600+(protocol == "10b" ? 1 : 0)
        fit = CM.fit_covariance(CM.CovarianceRung(6),pp; seed=runseed)
        fits[protocol] = fit
        Serialization.serialize(joinpath(out,"R6_$(label)_$protocol.jls"),fit)
        diag = MID.convergence_table(fit)
        diag.protocol .= protocol
        diag.grid .= label
        diag.sha .= sha
        diag.seed .= runseed
        diag.gate_pass = (diag.rhat .<= 1.05) .& (diag.ess_bulk .>= 200) .& (diag.ess_tail .>= 200)
        push!(diagnostics,diag)
        CSV.write(joinpath(out,"convergence_$label.csv"),vcat(diagnostics...))
        open(joinpath(out,"RUN_PROVENANCE.md"),"a") do io
            println(io,"- R6 $label $protocol, SHA `$sha`, seed $runseed, $(fit.seconds) seconds; 4 x (2000+3000), thin 1.")
        end
        println("R6 $label $protocol: gates $(count(diag.gate_pass))/$(DF.nrow(diag)); $(fit.seconds) seconds")
        flush(stdout)
        all(diag.gate_pass) || error("R6 $label $protocol fails convergence; no promotion")
    end
    return fits
end

"B2 five-axis smoothed R2/RMSE and preweek RMSE/logpd/coverage, both theta protocols."
function r6_measures(p,config,fits,label)
    rows = NamedTuple[]
    for protocol in ("10a","10b")
        fit = fits[protocol]
        theta = MID.median_theta(fit)
        smooth,ll = CM.covariance_smoothed(fit,p)
        pred = CM.covariance_predictions(CM.CovarianceRung(6),p,theta)
        summary = TB.prediction_summary(p,pred,protocol,"R6"; config)
        for metric in eachrow(DF.filter(:subset=>==("all"),summary))
            s = DF.filter(:axis=>==(metric.axis),smooth)
            push!(rows,(; grid=label,protocol,axis=metric.axis,n=metric.n,
                smoothed_r2=1-sum(s.residual.^2)/sum((s.observed .- ST.mean(s.observed)).^2),
                smoothed_rmse=sqrt(ST.mean(s.residual.^2)),collapsed_loglik=ll,
                onestep_rmse=metric.rmse,onestep_logpd=metric.mean_logpd,cover90=metric.cover90,
                sigma_obs=exp(theta[1]),sigma_att=exp(theta[2]),sigma_def=exp(theta[3]),rho_ab=tanh(theta[4])))
        end
    end
    return DF.DataFrame(rows)
end
