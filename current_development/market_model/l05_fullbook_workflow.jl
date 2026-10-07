module FullBookWorkflow

import CSV
import DataFrames
import Distributions
import LinearAlgebra
import Random
import Serialization
import SHA
import Statistics

const PM = parentmodule(@__MODULE__).PooledMarket
const MID = PM.MID
const TB = PM.CM.TB
const DF = DataFrames
const LA = LinearAlgebra
const DS = Distributions
const ST = Statistics

"""
Fixture theta=H*x+u smoothing under the frozen forward Gaussian book factors.
Given z=H*x+u+e, u~N(0,U), e~N(0,R), independent, K=U*(U+R)^-1.
Conditional on x,z, E[theta]=(I-K)*H*x+K*z and Var(theta)=U-K*U.
Integrate x against its ALL-book RTS posterior: add (I-K)*H*Vx*H'*(I-K)'.
For u use E[u]=K*(z-H*x), Var(u)=U-K*U+K*H*Vx*H'*K'.
This conditions on the fixture's own factor exactly once; adding an independent
fresh u after smoothing would be wrong. C1 factors/RTS remain approximate.
C0/H1/H2 theta paths are the structured latent rate, not the observed isolated
rate with independent observation noise re-added. No book is reinverted here.
"""
function fixture_moments(a,p,f,means,covs)
    rows = NamedTuple[]
    for fixture in 1:MID.n_fixtures(p)
        j = 2*fixture-1
        t = p.obs_week[j]
        H = f.H[j:j+1,:]
        structured = H*means[:,t]
        state_cov = H*covs[:,:,t]*H'
        if a.name == :C1
            factor = f.factors[fixture]
            factor.match_id == p.obs_match[j] || error("smoothing factor ID mismatch")
            K = (LA.Symmetric(f.noise+factor.R)\f.noise')'
            M = Matrix{Float64}(LA.I,2,2)-K
            conditional = f.noise-K*f.noise
            theta_mean = M*structured+K*factor.z
            theta_cov = conditional+M*state_cov*M'
            u_mean = K*(factor.z-structured)
            u_cov = conditional+K*state_cov*K'
        else
            theta_mean,theta_cov = structured,state_cov
            u_mean,u_cov = zeros(2),zeros(2,2)
        end
        push!(rows,(; match_id=p.obs_match[j],week=t,structured,theta_mean,
            theta_cov=Matrix(LA.Symmetric(theta_cov)),u_mean,
            u_cov=Matrix(LA.Symmetric(u_cov)),structured_cov=state_cov))
    end
    return rows
end

"""
B2-compatible forecasts at median hyperparameters, scored on unchanged isolated
Phase A targets. Fullbook_filter predicts before EVERY update in the week.
C1 pair_cov=H*P_preweek*H'+sigma_u^2*I: no own-book Hessian/noise or posterior
mean enters a forecast. We predict the underlying rate, not another noisy book.
Linear axes have exact conditional-Gaussian marginals; nonlinear log-total uses
B2's seeded MC/KDE convention, including its one-particle RNG draw.
"""
function predictions(a,p,theta; markets=nothing,seed=3963,mc_draws=4000)
    f = PM.fullbook_filter(a,p,theta; markets,predict=true)
    return prediction_rows(p,f; seed,mc_draws)
end

function prediction_rows(p,f; seed=3963,mc_draws=4000)
    rng = Random.Xoshiro(seed+999999)
    rows = NamedTuple[]
    for fixture in 1:MID.n_fixtures(p)
        j = 2*fixture-1
        h,away = f.pair_mean[fixture,:]
        vh,va,cross = f.pair_cov[fixture,1,1],f.pair_cov[fixture,2,2],f.pair_cov[fixture,1,2]
        actual = TB.axis_values(p.obs_y[j],p.obs_y[j+1])
        means = (h,away,h-away,(h+away)/2)
        variances = (vh,va,vh+va-2cross,(vh+va+2cross)/4)
        all(x -> isfinite(x) && x > 0,variances) || error("nonpositive forecast variance $(p.obs_match[j])")
        for axis in 1:4
            dist = DS.Normal(means[axis],sqrt(variances[axis]))
            cdf = DS.cdf(dist,actual[axis])
            push!(rows,(; match_id=p.obs_match[j],week=p.obs_week[j],axis=TB.AXES[axis],
                observed=actual[axis],predicted=means[axis],variance=variances[axis],
                logpd=DS.logpdf(dist,actual[axis]),cover90=0.05<=cdf<=0.95,
                method="conditional Gaussian mixture"))
        end
        draws = zeros(mc_draws)
        for d in eachindex(draws)
            rand(rng)
            z1,z2 = randn(rng,2)
            yh = h+sqrt(vh)*z1
            ya = away+cross/sqrt(vh)*z1+sqrt(max(va-cross^2/vh,0))*z2
            draws[d] = TB.axis_values(yh,ya)[5]
        end
        bandwidth = max(1.06ST.std(draws)*length(draws)^(-0.2),1e-6)
        density = TB.logsumexp(DS.logpdf.(DS.Normal.(draws,bandwidth),actual[5]))-log(length(draws))
        push!(rows,(; match_id=p.obs_match[j],week=p.obs_week[j],axis=TB.AXES[5],
            observed=actual[5],predicted=ST.mean(draws),variance=ST.var(draws),logpd=density,
            cover90=ST.quantile(draws,0.05)<=actual[5]<=ST.quantile(draws,0.95),
            method="predictive MC / KDE density"))
    end
    return (; rows=DF.DataFrame(rows),loglik=f.loglik)
end

"Fixture-paired sum of supremacy/level marginal logpd, with fixture SE (not serially robust)."
function paired_scores(fixtures; baseline="C0")
    selected = DF.filter(r -> r.axis in ("supremacy","level"),fixtures)
    all(DF.combine(DF.groupby(selected,[:rung,:protocol,:match_id]),DF.nrow=>:n).n .== 2) ||
        error("paired score needs exactly two axes per fixture")
    scores = DF.combine(DF.groupby(selected,[:rung,:protocol,:match_id]),:logpd=>sum=>:logpd)
    rows = NamedTuple[]
    for protocol in unique(scores.protocol)
        control = DF.select(DF.filter(r -> r.rung == baseline && r.protocol == protocol,scores),
            :match_id,:logpd=>:control_logpd)
        isempty(control) && error("missing paired control $baseline $protocol")
        for rung in unique(scores.rung)
            rung == baseline && continue
            candidate = DF.filter(r -> r.rung == rung && r.protocol == protocol,scores)
            isempty(candidate) && continue
            paired = DF.innerjoin(candidate,control; on=:match_id,order=:left)
            DF.nrow(paired) == DF.nrow(control) == DF.nrow(candidate) || error("paired fixtures differ")
            delta = paired.logpd-paired.control_logpd
            push!(rows,(; rung,protocol,baseline,n=length(delta),mean_delta=ST.mean(delta),
                se=ST.std(delta)/sqrt(length(delta)),sum_delta=sum(delta),
                score="sum of supremacy and level marginal logpd"))
        end
    end
    return DF.DataFrame(rows)
end

"Independent augmented-state batch posterior for frozen C1 fixture theta/u toy gates."
function augmented_batch(p,theta,f)
    a = PM.FullBookRung(:C1)
    m0,P0 = PM.fullbook_initial(a,MID.n_teams(p))
    Q = PM.fullbook_process(a,MID.n_teams(p),theta)
    n,T,nf = length(m0),p.n_weeks,MID.n_fixtures(p)
    latent_mean = vcat(repeat(m0,T),zeros(2*nf))
    Sigma = zeros(length(latent_mean),length(latent_mean))
    block(t) = ((t-1)*n+1):(t*n)
    for t in 1:T, v in 1:T
        Sigma[block(t),block(v)] .= P0+(min(t,v)-1)*Q
    end
    Sigma[n*T+1:end,n*T+1:end] .= exp(2theta[3])*Matrix{Float64}(LA.I,2*nf,2*nf)
    design = zeros(2*nf,length(latent_mean))
    R = zeros(2*nf,2*nf)
    z = zeros(2*nf)
    for fixture in 1:nf
        ix = (2*fixture-1):(2*fixture)
        factor = f.factors[fixture]
        design[ix,block(factor.week)] .= factor.H
        design[ix,n*T .+ ix] .= Matrix{Float64}(LA.I,2,2)
        R[ix,ix] .= factor.R
        z[ix] .= factor.z
    end
    C = LA.cholesky(LA.Symmetric(design*Sigma*design'+R))
    gain = (C\(design*Sigma))'
    mean = latent_mean+gain*(z-design*latent_mean)
    covariance = Sigma-gain*design*Sigma
    return (; mean,covariance,design,n,T)
end

include("l05_fullbook_reports.jl")

end # module
