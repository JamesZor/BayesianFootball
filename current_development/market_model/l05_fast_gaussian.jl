module FastGaussianBook

import DataFrames
import LinearAlgebra
import Random
const PM = parentmodule(@__MODULE__).PooledMarket
const LA = LinearAlgebra

"""
Exact scalar Gaussian Kalman likelihood for C0/H1/H2 only. Independent equal
side noise makes sequential scalar observations exactly equivalent to the
existing 2-D updates. This changes arithmetic/allocation, NOT the model,
priors, sampler, adaptation, budgets or seeds. C1 is explicitly refused.
Fixed C0/H1 designs are prepared once per fit; H2's design remains conditional
on sampled kappa. Q contains ordinary weekly q/s increments through gaps.
The state covariance update reuses B2's BLAS rank-one form and symmetrises at
week boundaries. Stored moments exist for independent parity gates only.
"""
function scalar_filter(a,p,theta; store=false,base=PM.fullbook_design(a,p,theta),quality=nothing)
    a.name in (:C0,:H1,:H2) || error("Gaussian fast path forbids C1")
    m,P = PM.fullbook_initial(a,PM.MID.n_teams(p))
    H = a.name == :H2 && quality !== nothing ? base+theta[4]*quality : base
    N,T,n = PM.MID.n_teams(p),p.n_weeks,length(m)
    vq,vs = exp.(2theta[2:3])
    variance = exp(2theta[1])
    Ms = store ? zeros(n,T) : zeros(0,0)
    Ps = store ? zeros(n,n,T) : zeros(0,0,0)
    k = zeros(n)
    ll = 0.0
    for t in 1:T
        if t >= 2
            for team in 1:N
                i,j = 2+team,2+N+team
                P[i,i] += vq+vs
                P[j,j] += vq+vs
                P[i,j] += vs-vq
                P[j,i] += vs-vq
            end
        end
        for j in p.week_ptr[t]:(p.week_ptr[t+1]-1)
            h = view(H,j,:)
            LA.mul!(k,P,h)
            S = LA.dot(h,k)+variance
            S > 0 && isfinite(S) || error("nonpositive Gaussian innovation variance")
            e = p.obs_y[j]-LA.dot(h,m)
            ll += -0.5*(log(2pi*S)+e^2/S)
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
    return (; loglik=ll,m_filt=Ms,P_filt=Ps,Q=PM.fullbook_process(a,N,theta))
end

"Prepared exact Gaussian callback for the existing native slice sampler."
function evaluator(a,p)
    a.name in (:C0,:H1,:H2) || error("Gaussian callback forbids C1")
    theta = PM.MID.init_centre(a)
    base = PM.fullbook_design(a,p,theta)
    quality = a.name == :H2 ? PM.fullbook_design(a,p,vcat(theta[1:3],1.0))-base : nothing
    return function (arm,panel,z; markets=nothing,likelihoods=nothing,store=false)
        arm.name == a.name && panel === p || error("prepared Gaussian panel/rung mismatch")
        markets === nothing && likelihoods === nothing || error("Gaussian callback forbids book likelihoods")
        return scalar_filter(arm,panel,z; store,base,quality)
    end
end

"Scalar/dense likelihood AND full filtered moments parity at unchanged engine tolerances."
function parity_gates(p; scope)
    rows = DataFrames.DataFrame(gate=String[],value=Float64[],tol=Float64[],pass=Bool[])
    for rung in (:C0,:H1,:H2)
        a = PM.FullBookRung(rung)
        callback = evaluator(a,p)
        for (point,scales) in enumerate(([0.07,0.04,0.015],[0.12,0.01,0.06],[0.03,0.09,0.02]))
            theta = rung == :H2 ? vcat(log.(scales),(-0.4,0.0,0.3)[point]) : log.(scales)
            fast = callback(a,p,theta; store=true)
            ref = PM.fullbook_filter(a,p,theta; store=true)
            for (name,value,tol) in (("loglik",abs(fast.loglik-ref.loglik),1e-9),
                ("filtered mean",maximum(abs,fast.m_filt-ref.m_filt),1e-8),
                ("filtered covariance",maximum(abs,fast.P_filt-ref.P_filt),1e-8))
                push!(rows,("$scope $rung scalar/dense $point $name",value,tol,value<=tol))
            end
        end
    end
    return rows
end

end # module
