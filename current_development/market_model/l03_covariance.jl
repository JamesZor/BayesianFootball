module CovarianceMarket

import CSV
import DataFrames
import Dates
import Distributions
import LinearAlgebra
import Random
import Serialization
import Statistics

if !isdefined(parentmodule(@__MODULE__), :TwoStageMarket)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "l02_two_stage.jl"))
end
const TB = parentmodule(@__MODULE__).TwoStageMarket
const MM = TB.MM
const MID = TB.MID
const DF = DataFrames
const LA = LinearAlgebra
const ST = Statistics
const DS = Distributions

"R5–R7: R2 states, fixture-axis noise and/or same-team innovation correlation."
struct CovarianceRung <: MID.AbstractArm
    level::Int
    rho::Float64 # conditional layout; fitted correlation lives in theta
    function CovarianceRung(level::Int, rho::Float64=0.0)
        level in 5:7 || error("covariance rung must be R5–R7")
        abs(rho) < 1 || error("innovation correlation must lie strictly inside (-1,1)")
        new(level,rho)
    end
end
rotated(a::CovarianceRung) = a.level in (5,7)
correlated(a::CovarianceRung) = a.level in (6,7)
MID.arm_name(a::CovarianceRung) = "R$(a.level)"
MID.param_names(a::CovarianceRung) = vcat(rotated(a) ? ["sigma_sup","sigma_lev"] : ["sigma_obs"],
    ["sigma_att","sigma_def"], correlated(a) ? ["rho_ab"] : String[])
MID.init_centre(a::CovarianceRung) = vcat(rotated(a) ? log.([0.10,0.03,0.03,0.03]) : log.([0.08,0.03,0.03]),
    correlated(a) ? [0.0] : Float64[])

"Positive scales on log, Uniform(-1,1) correlation on atanh, including its Jacobian."
function MID.log_prior(a::CovarianceRung, theta)
    nnoise = rotated(a) ? 2 : 1
    lp = sum(MID.halfnormal_lp(exp(theta[j]),0.20)+theta[j] for j in 1:nnoise)
    lp += sum(MID.halfnormal_lp(exp(theta[j]),0.10)+theta[j] for j in (nnoise+1):(nnoise+2))
    if correlated(a)
        z = abs(theta[end])
        lp += -log(2)+2*(log(2)-z-log1p(exp(-2z)))
    end
    return lp
end
physical_draws(a, U) = cat([correlated(a) && j == size(U,2) ? tanh.(U[:,j:j,:]) : exp.(U[:,j:j,:])
    for j in axes(U,2)]...; dims=2)
conditional_arm(a, theta) = CovarianceRung(a.level,correlated(a) ? tanh(theta[end]) : 0.0)

function covariance_schedule(a, p, theta)
    k = rotated(a) ? 3 : 2
    return MID.schedule(MID.GRW1(),[theta[1],theta[k],theta[k+1]],MID.n_teams(p),p.n_weeks)
end

"Q is independent across teams; static mu/gamma and independent initial states are unchanged."
function MID.process_cov(a::CovarianceRung, N, sch, t)
    Q = zeros(2+2N,2+2N)
    for i in 1:N
        va, vb = sch.D[i,t],sch.D[N+i,t]
        Q[2+i,2+i] = va
        Q[2+N+i,2+N+i] = vb
        Q[2+i,2+N+i] = Q[2+N+i,2+i] = a.rho*sqrt(va*vb)
    end
    return Q
end

"Protect exact static directions in the FFBS draw against numerical backward noise."
function MID.backward_step!(xt,xnext,a::CovarianceRung,N,sch,filt,F,t,rng)
    mean,cov = MID.backward_moments(xnext,a,N,sch,filt,F,t)
    MID.draw_psd!(xt,rng,mean,cov)
    xt[1:2] .= xnext[1:2]
    return nothing
end

"Unit-determinant (home,away) -> (supremacy,level) transform."
const ROTATION = [1.0 -1.0; 0.5 0.5]
function prepared_observations(a,p)
    H = zeros(MID.n_obs(p),2+2MID.n_teams(p))
    h = zeros(size(H,2))
    for j in axes(H,1)
        MID.obs_row!(h,p,j,0)
        H[j,:] .= h
    end
    y = copy(p.obs_y)
    if rotated(a)
        for j in 1:2:length(y)
            H[j:j+1,:] .= ROTATION*H[j:j+1,:]
            y[j:j+1] .= ROTATION*y[j:j+1]
        end
    end
    return (; H,y)
end

"Exact scalar Kalman updates; all predictions precede every update of their week."
function covariance_filter(a,p,theta; store=false,predict=false,prepared=prepared_observations(a,p))
    arm = conditional_arm(a,theta)
    sch = covariance_schedule(a,p,theta)
    N = MID.n_teams(p)
    m,P = MID.initial_state(arm,N)
    n,T = length(m),p.n_weeks
    k = zeros(n)
    noise = rotated(a) ? exp.(2theta[1:2]) : fill(exp(2theta[1]),2)
    ll = 0.0
    Ms = store ? zeros(n,T) : zeros(0,0)
    Ps = store ? zeros(n,n,T) : zeros(0,0,0)
    pm = predict ? zeros(MID.n_fixtures(p),2) : zeros(0,0)
    pc = predict ? zeros(MID.n_fixtures(p),2,2) : zeros(0,0,0)
    hh,ha = zeros(n),zeros(n)
    for t in 1:T
        if t >= 2
            MID.time_update!(m,P,sch,t,N,false,zeros(0,0))
            for i in 1:N
                cross = arm.rho*sqrt(sch.D[i,t]*sch.D[N+i,t])
                P[2+i,2+N+i] += cross
                P[2+N+i,2+i] += cross
            end
        end
        js = p.week_ptr[t]:(p.week_ptr[t+1]-1)
        if predict
            R = side_noise(a,theta)
            for j in first(js):2:last(js)
                MID.obs_row!(hh,p,j,0)
                MID.obs_row!(ha,p,j+1,0)
                f = cld(j,2)
                pm[f,1],pm[f,2] = LA.dot(hh,m),LA.dot(ha,m)
                pc[f,1,1] = LA.dot(hh,P*hh)+R[1,1]
                pc[f,2,2] = LA.dot(ha,P*ha)+R[2,2]
                pc[f,1,2] = pc[f,2,1] = LA.dot(hh,P*ha)+R[1,2]
            end
        end
        for j in js
            h = view(prepared.H,j,:)
            LA.mul!(k,P,h)
            S = LA.dot(h,k)+noise[mod1(j,2)]
            if !(S > 0 && isfinite(S))
                return (; loglik=-Inf,m_filt=Ms,P_filt=Ps,pair_mean=pm,pair_cov=pc)
            end
            e = prepared.y[j]-LA.dot(h,m)
            ll += -0.5*(log(2π*S)+e^2/S)
            LA.axpy!(e/S,k,m)
            LA.BLAS.ger!(-1/S,k,k,P)
        end
        for j in 1:n, i in 1:(j-1)
            P[i,j] = P[j,i] = (P[i,j]+P[j,i])/2
        end
        if store
            Ms[:,t] .= m
            Ps[:,:,t] .= P
        end
    end
    return (; loglik=ll,m_filt=Ms,P_filt=Ps,pair_mean=pm,pair_cov=pc)
end

"Original-side observation covariance, derived from axis variances (not from a filter)."
function side_noise(a,theta)
    if !rotated(a)
        return Matrix(LA.Diagonal(fill(exp(2theta[1]),2)))
    end
    vd,vl = exp.(2theta[1:2])
    s2 = vd/4+vl
    c = vl-vd/4
    return [s2 c; c s2]
end

"Independent batch Gaussian in ORIGINAL side coordinates, including full process covariance."
function batch_gaussian(a,p,theta)
    N,T = MID.n_teams(p),p.n_weeks
    m0,P0 = MID.initial_state(MID.GRW1(),N)
    n = length(m0)
    k = rotated(a) ? 3 : 2
    sa,sb = exp.(theta[k:k+1])
    rho = correlated(a) ? tanh(theta[end]) : 0.0
    Q = zeros(n,n)
    for i in 1:N
        Q[2+i,2+i],Q[2+N+i,2+N+i] = sa^2,sb^2
        Q[2+i,2+N+i] = Q[2+N+i,2+i] = rho*sa*sb
    end
    M = repeat(m0,T)
    Sigma = zeros(n*T,n*T)
    blk(t) = ((t-1)*n+1):(t*n)
    for t in 1:T, u in 1:T
        Sigma[blk(t),blk(u)] .= P0+(min(t,u)-1)*Q
    end
    H = zeros(MID.n_obs(p),n*T)
    h = zeros(n)
    R = zeros(MID.n_obs(p),MID.n_obs(p))
    for j in 1:MID.n_obs(p)
        MID.obs_row!(h,p,j,0)
        H[j,blk(p.obs_week[j])] .= h
    end
    for j in 1:2:MID.n_obs(p)
        R[j:j+1,j:j+1] .= side_noise(a,theta)
    end
    C = LA.cholesky(LA.Symmetric(H*Sigma*H'+R))
    residual = p.obs_y-H*M
    ll = -0.5*(length(residual)*log(2π)+LA.logdet(C)+LA.dot(residual,C\residual))
    K = Sigma*H'/C
    return (; loglik=ll,post_mean=M+K*residual,post_cov=Sigma-K*H*Sigma)
end

function engine_gates_b2()
    rows = vcat(TB.engine_gates(),TB.rbpf_gate())
    p = MID.toy_panel(Random.Xoshiro(3920))
    theta2 = log.([0.07,0.05,0.03])
    ref = TB.rung_filter(TB.Rung(2),p,theta2).loglik
    theta5 = log.([sqrt(2)*0.07,0.07/sqrt(2),0.05,0.03])
    push!(rows,("B2 R5 isotropic reduction",abs(covariance_filter(CovarianceRung(5),p,theta5).loglik-ref),1e-9,false))
    push!(rows,("B2 R6 rho-zero reduction",abs(covariance_filter(CovarianceRung(6),p,vcat(theta2,0.0)).loglik-ref),1e-9,false))
    for (a,theta) in ((CovarianceRung(5),log.([0.12,0.025,0.05,0.03])),
                      (CovarianceRung(6),vcat(theta2,atanh(-0.6))),
                      (CovarianceRung(7),vcat(log.([0.12,0.025,0.05,0.03]),atanh(-0.6))))
        f = covariance_filter(a,p,theta; store=true)
        arm = conditional_arm(a,theta)
        sch = covariance_schedule(a,p,theta)
        batch = batch_gaussian(a,p,theta)
        smooth = MID.smoothed_mean(arm,p,sch,f)
        push!(rows,("B2 loglik $(MID.arm_name(a))",abs(f.loglik-batch.loglik),1e-9,false))
        push!(rows,("B2 RTS $(MID.arm_name(a))",maximum(abs.(vec(smooth)-batch.post_mean)),1e-8,false))
        if a.level == 6
            rng = Random.Xoshiro(3921)
            draws = reduce(hcat,[vec(MID.ffbs(arm,p,sch,f,rng)) for _ in 1:4000])
            se = sqrt.(max.(LA.diag(batch.post_cov),1e-300)/4000)
            z = maximum(abs.(vec(ST.mean(draws; dims=2))-batch.post_mean)./max.(se,1e-12))
            err = maximum(abs.(ST.cov(draws; dims=2)-batch.post_cov))/maximum(abs.(batch.post_cov))
            push!(rows,("B2 R6 FFBS mean max z (4000)",z,4.5,false))
            push!(rows,("B2 R6 FFBS covariance relative error",err,0.1,false))
        end
    end
    rows.pass = rows.value .<= rows.tol
    return rows
end

"Coordinate slice on the exact collapsed likelihood; adaptation ends at warmup."
function covariance_chain(a,p,chain,seed,warmup,samples)
    rng = Random.Xoshiro(seed+1000chain)
    theta = MID.init_theta(a,rng)
    prepared = prepared_observations(a,p)
    target = z -> maximum(abs,z) > MID.THETA_BOUND ? -Inf :
        covariance_filter(a,p,z; prepared).loglik+MID.log_prior(a,z)
    U = zeros(samples,length(theta))
    hist = zeros(warmup,length(theta))
    widths = ones(length(theta))
    for it in 1:(warmup+samples)
        theta,_ = MID.slice_sweep(target,theta,target(theta),widths,rng)
        if it <= warmup
            hist[it,:] .= theta
            if it >= 50 && it % 25 == 0
                widths .= clamp.(3 .* vec(ST.std(hist[max(1,it-199):it,:]; dims=1)),0.02,3.0)
            end
        else
            U[it-warmup,:] .= theta
        end
    end
    return U
end

function fit_covariance(a,p; seed=3922,warmup=2000,samples=3000,chains=4)
    start = time()
    tasks = [Threads.@spawn covariance_chain(a,p,c,seed,warmup,samples) for c in 1:chains]
    U = cat(fetch.(tasks)...; dims=3)
    return MID.ArmFit(a,MID.param_names(a),physical_draws(a,U),U,
        Matrix{Float64}[],Vector{Float64}[],zeros(0,0),time()-start)
end

"Refit R2 control and R5–R7 in both protocols; refuse inference after any failed gate."
function train_covariance(panel,config,out; seed=3922)
    fits = Dict{Tuple{Int,String},MID.ArmFit}()
    diagnostics = DF.DataFrame[]
    sha = strip(read(`git rev-parse HEAD`,String))
    for protocol in ("10a","10b"), level in (2,5,6,7)
        p = protocol == "10a" ? panel : MID.restrict_panel(panel,in.(panel.obs_season,Ref(config.honest_train)))
        runseed = seed+100level+(protocol == "10b" ? 1 : 0)
        fit = level == 2 ? TB.fit_rung(TB.Rung(2),p; seed=runseed) : fit_covariance(CovarianceRung(level),p; seed=runseed)
        fits[(level,protocol)] = fit
        Serialization.serialize(joinpath(out,"R$(level)_$(protocol).jls"),fit)
        diag = MID.convergence_table(fit)
        diag.protocol .= protocol
        diag.sha .= sha
        diag.seed .= runseed
        diag.gate_pass = (diag.rhat .<= 1.05) .& (diag.ess_bulk .>= 200) .& (diag.ess_tail .>= 200)
        push!(diagnostics,diag)
        CSV.write(joinpath(out,"convergence_b2.csv"),vcat(diagnostics...))
        open(joinpath(out,"RUN_PROVENANCE.md"),"a") do io
            println(io,"- $(Dates.now(Dates.UTC)): R$level $protocol; SHA `$sha`; seed $runseed; $(fit.seconds) seconds; 4 × (2000+3000), thin 1.")
        end
        println("R$level $protocol: $(fit.seconds) seconds; gates $(count(diag.gate_pass))/$(DF.nrow(diag))")
        flush(stdout)
        all(diag.gate_pass) || error("R$level $protocol failed convergence; no promotion")
    end
    return fits
end

include("l03_covariance_diagnostics.jl")
end # module
