# Included inside TwoStageMarket; definitions only.
const AXES = ("log_lambda_h", "log_lambda_a", "supremacy", "level", "log_total")
axis_values(h, a) = (h, a, h-a, (h+a)/2, max(h,a)+log1p(exp(-abs(h-a))))
logsumexp(x) = (m=maximum(x); m+log(sum(exp.(x.-m))))

"""
Exact Gaussian pre-week filter (R0–R3, one particle), or R4 RBPF.
Proposal: independent PRIOR Gamma(ν/2,scale 2/ν) weights, one per current
fixture, shared by both sides. New weights each week; never posterior weights
from future data. Each particle carries its own exact conditional Kalman mean
and covariance (including static HA). Resample at ESS < Np/2, after the entire
week; predictions precede ALL updates in that week. `fixed_omega` is a toy gate.
Linear-axis densities and CDFs are exact conditional mixture averages. Log-total
uses seeded predictive MC and a Gaussian KDE for density, separately labelled.
For ν≤2 the true predictive variance is infinite, not a finite MC estimate.
"""
function preweek_predictions(a::Rung, panel, θ; particles=20000, seed=3906,
                             fixed_omega=false, mc_draws=4000)
    p = a.level >= 3 ? home_panel(panel) : panel
    arm = layout(a,p,θ)
    sch = rung_schedule(a,p,θ)
    N = MID.n_teams(p)
    np = a.level == 4 ? particles : 1
    m0, P0 = MID.initial_state(arm,N)
    ms = [copy(m0) for _ in 1:np]
    Ps = [copy(P0) for _ in 1:np]
    rngs = [Random.Xoshiro(seed+17q) for q in 1:np]
    rng = Random.Xoshiro(seed+999999)
    logw = fill(-log(np),np)
    ll = 0.0
    metrics = NamedTuple[]
    weeks = NamedTuple[]
    obs_means = zeros(MID.n_obs(p))
    obs_vars = zeros(MID.n_obs(p))
    resampling_events = 0
    for t in 1:p.n_weeks
        js = collect(p.week_ptr[t]:(p.week_ptr[t+1]-1))
        nf = length(js) ÷ 2
        means = zeros(np,nf,2)
        covs = zeros(np,nf,3) # home variance, away variance, cross covariance
        increments = zeros(np)
        weights = exp.(logw)
        cumulative_weights = cumsum(weights)
        cumulative_weights[end] = 1.0
        # Static chunks avoid task migration races and make RNG independent of scheduling.
        Threads.@threads for q in 1:np
            m = ms[q]
            P = Ps[q]
            h = zeros(length(m))
            ha = zeros(length(m))
            kh = zeros(length(m))
            ka = zeros(length(m))
            if t >= 2
                MID.time_update!(m,P,sch,t,N,false,zeros(0,0))
            end
            omega = a.level == 4 && !fixed_omega ? rand(rngs[q],DS.Gamma(exp(θ[5])/2,2/exp(θ[5])),nf) : ones(nf)
            for f in 1:nf
                j = js[2*f-1]
                MID.obs_row!(h,p,j,MID.n_features(arm))
                MID.obs_row!(ha,p,j+1,MID.n_features(arm))
                LA.mul!(kh,P,h)
                LA.mul!(ka,P,ha)
                means[q,f,1] = LA.dot(h,m)
                means[q,f,2] = LA.dot(ha,m)
                covs[q,f,1] = LA.dot(h,kh)+exp(2θ[1])/omega[f]
                covs[q,f,2] = LA.dot(ha,ka)+exp(2θ[1])/omega[f]
                covs[q,f,3] = LA.dot(h,ka)
            end
            for (k,j) in enumerate(js)
                MID.obs_row!(h,p,j,MID.n_features(arm))
                LA.mul!(kh,P,h)
                S = LA.dot(h,kh)+exp(2θ[1])/omega[cld(k,2)]
                residual = p.obs_y[j]-LA.dot(h,m)
                increments[q] += -0.5*(log(2π*S)+residual^2/S)
                LA.axpy!(residual/S,kh,m)
                LA.BLAS.ger!(-1/S,kh,kh,P)
            end
            P .= (P .+ P') ./ 2
        end
        # Predictive summaries use old weights, before this week's likelihood.
        for f in 1:nf
            j = js[2*f-1]
            h = means[:,f,1]
            away = means[:,f,2]
            vh = covs[:,f,1]
            va = covs[:,f,2]
            cross = covs[:,f,3]
            actual = axis_values(p.obs_y[j],p.obs_y[j+1])
            axis_mu = (h,away,h.-away,(h.+away)./2)
            axis_var = (vh,va,vh.+va.-2cross,(vh.+va.+2cross)./4)
            for k in 1:4
                mu = axis_mu[k]
                variance = axis_var[k]
                pred = sum(weights.*mu)
                vpred = sum(weights.*(variance.+mu.^2))-pred^2
                ld = @. logw-0.5*(log(2π*variance)+(actual[k]-mu)^2/variance)
                cdf = sum(weights .* DS.cdf.(DS.Normal.(mu,sqrt.(variance)),actual[k]))
                push!(metrics,(; match_id=p.obs_match[j],week=t,axis=AXES[k],
                    observed=actual[k],predicted=pred,variance=a.level == 4 && exp(θ[5]) <= 2 && !fixed_omega ? Inf : vpred,
                    logpd=logsumexp(ld),cover90=0.05<=cdf<=0.95,method="conditional Gaussian mixture"))
                if k <= 2
                    obs_means[j+k-1] = pred
                    obs_vars[j+k-1] = vpred
                end
            end
            # MC is necessary only for the nonlinear log-sum-exp axis.
            draws = zeros(mc_draws)
            for d in eachindex(draws)
                q = searchsortedfirst(cumulative_weights,rand(rng))
                z1,z2 = randn(rng,2)
                sh = sqrt(vh[q])
                yh = h[q]+sh*z1
                ya = away[q]+cross[q]/sh*z1+sqrt(max(va[q]-cross[q]^2/vh[q],0))*z2
                draws[d] = axis_values(yh,ya)[5]
            end
            bw = max(1.06ST.std(draws)*length(draws)^(-0.2),1e-6)
            density = logsumexp(DS.logpdf.(DS.Normal.(draws,bw),actual[5]))-log(length(draws))
            push!(metrics,(; match_id=p.obs_match[j],week=t,axis=AXES[5],observed=actual[5],
                predicted=ST.mean(draws),variance=ST.var(draws),logpd=density,
                cover90=ST.quantile(draws,0.05)<=actual[5]<=ST.quantile(draws,0.95),method="predictive MC / KDE density"))
        end
        lw = logw.+increments
        inc = logsumexp(lw)
        ll += inc
        logw = lw.-inc
        w = exp.(logw)
        ess = 1/sum(w.^2)
        resampled = ess < np/2
        if resampled
            idx = MID.systematic_resample(w,rng)
            ms = [copy(ms[q]) for q in idx]
            Ps = [copy(Ps[q]) for q in idx]
            logw .= -log(np)
            resampling_events += 1
        end
        fixtures = join(unique(p.obs_match[js]),";")
        push!(weeks,(; week=t,week_start=p.week_start[t],n_fixtures=nf,fixtures,ess,
                     resampled,resampling_events,particles=np,proposal="prior Gamma fixture-shared",seed))
    end
    return (; rows=DF.DataFrame(metrics),weeks=DF.DataFrame(weeks),loglik=ll,
              pred_mean=obs_means,pred_var=obs_vars,resampling_events)
end

"RBPF ν→∞ gate: ω identically 1, no dependence on a finite ν approximation."
function rbpf_gate()
    p = MID.toy_panel(Random.Xoshiro(3907))
    θ3 = log.([0.07,0.05,0.03,0.1])
    θ4 = vcat(θ3,log(1e12))
    r3 = rung_filter(Rung(3),home_panel(p),θ3; predict=true)
    r4 = preweek_predictions(Rung(4),p,θ4; particles=32,fixed_omega=true,seed=3907,mc_draws=100)
    rows = DF.DataFrame(gate=["B RBPF fixed-omega loglik","B RBPF fixed-omega pre-week mean","B RBPF fixed-omega pre-week variance"],
        value=[abs(r3.loglik-r4.loglik),maximum(abs.(r3.pred_mean.-r4.pred_mean)),maximum(abs.(r3.pred_var.-r4.pred_var))],
        tol=fill(1e-8,3))
    rows.pass = rows.value .<= rows.tol
    return rows
end

function prediction_summary(p, pred, protocol, rung; seed=3906, config=MM.scottish_lower_2425_2526())
    meta = Dict(r.match_id=>r for r in eachrow(p.matches))
    open_week = Dict(s=>minimum(p.obs_week[p.obs_season.==s]) for s in unique(p.obs_season))
    first_week = minimum(p.obs_week)
    rows = NamedTuple[]
    for axis in AXES, subset in ("all","warm","season-open","in-season")
        g = DF.filter(r -> r.axis == axis && (protocol == "10a" || meta[r.match_id].season in config.honest_test) &&
            (subset == "all" || (subset == "warm" && r.week >= first_week+3) ||
             (subset == "season-open" && r.week < open_week[meta[r.match_id].season]+3) ||
             (subset == "in-season" && r.week >= open_week[meta[r.match_id].season]+3)),pred.rows)
        DF.nrow(g) == 0 && continue
        residual = g.observed.-g.predicted
        push!(rows,(; rung,protocol,subset,axis,seed,n=DF.nrow(g),rmse=sqrt(ST.mean(residual.^2)),
            mae=ST.mean(abs.(residual)),mean_logpd=ST.mean(g.logpd),sum_logpd=sum(g.logpd),
            cover90=ST.mean(g.cover90),collapsed_loglik=pred.loglik))
    end
    return DF.DataFrame(rows)
end

"RTS means at median θ; R4 is conditional on posterior mean ω (explicit in method)."
function smoothed_rows(fit, panel)
    a = fit.arm
    p = a.level >= 3 ? home_panel(panel) : panel
    θ = MID.median_theta(fit)
    omega = a.level == 4 ? vec(fit.aux_mean) : Float64[]
    filter = rung_filter(a,p,θ; store=true,omega)
    X = MID.smoothed_mean(layout(a,p,θ),p,rung_schedule(a,p,θ),filter)
    fitted = MID.fitted_logrates(p,X; K=a.level >= 3 ? MID.n_teams(p) : 0)
    rows = NamedTuple[]
    for m in 1:MID.n_fixtures(p)
        observed = axis_values(p.obs_y[2m-1],p.obs_y[2m])
        predicted = axis_values(fitted[2m-1],fitted[2m])
        for k in 1:5
            push!(rows,(; rung=MID.arm_name(a),match_id=p.obs_match[2m-1],axis=AXES[k],
                observed=observed[k],predicted=predicted[k],residual=observed[k]-predicted[k],
                method=a.level == 4 ? "RTS at median theta / posterior mean omega" : "RTS at median theta"))
        end
    end
    return DF.DataFrame(rows),filter.loglik
end
