# Included inside PooledMarket. Full-book-only Phase C revision 4 engines.

"C0/C1 quality-style GRW; H1 split static HA; H2 quality-linked home loading."
struct FullBookRung <: MID.AbstractArm
    name::Symbol
    function FullBookRung(name::Symbol)
        name in (:C0,:C1,:H1,:H2) || error("unknown full-book rung $name")
        new(name)
    end
end
MID.arm_name(a::FullBookRung) = String(a.name)
MID.param_names(a::FullBookRung) = a.name == :C1 ? ["sigma_q","sigma_s","sigma_u","n"] :
    vcat(["sigma_obs","sigma_q","sigma_s"],a.name == :H2 ? ["kappa"] : String[])
MID.init_centre(a::FullBookRung) = a.name == :C1 ? log.([0.03,0.01,0.06,1000.0]) :
    vcat(log.([0.07,0.03,0.01]),a.name == :H2 ? [0.0] : Float64[])

"Normalised HalfNormal log-density; inherited MID helper omits scale-dependent constants."
fullbook_halfnormal_lp(x,sd) = log(2)+DS.logpdf(DS.Normal(0,sd),x)

"Log-coordinate priors include Jacobians; n has NO inherited linked-coordinate bound."
function MID.log_prior(a::FullBookRung,theta)
    length(theta) == length(MID.param_names(a)) || error("wrong parameter layout")
    all(isfinite,theta) || return -Inf
    if a.name == :C1
        lp = sum(fullbook_halfnormal_lp(exp(theta[j]),0.10)+theta[j] for j in 1:2)
        lp += fullbook_halfnormal_lp(exp(theta[3]),0.20)+theta[3]
        # LogNormal(n) plus log-Jacobian is exactly Normal(log n).
        return lp+DS.logpdf(DS.Normal(log(1000.0),1.5),theta[4])
    end
    lp = fullbook_halfnormal_lp(exp(theta[1]),0.20)+theta[1]
    lp += sum(fullbook_halfnormal_lp(exp(theta[j]),0.10)+theta[j] for j in 2:3)
    return a.name == :H2 ? lp+DS.logpdf(DS.Normal(0,0.5),theta[4]) : lp
end

"Inherited independent initial alpha/beta prior, NOT independent initial q/s at SD .5."
function fullbook_initial(a,N; gamma_def_sd=0.25)
    m0,P0 = MID.initial_state(MID.GRW1(),N)
    a.name == :H1 || return m0,P0
    # Manager clarification: gamma_att N(.15,.25²), gamma_def N(0,.25²), independent.
    m = vcat(m0,0.0)
    P = zeros(length(m),length(m))
    P[1:length(m0),1:length(m0)] .= P0
    P[end,end] = gamma_def_sd^2
    return m,P
end

"Ordinary weekly q/s innovations in raw alpha/beta coordinates; static terms have zero Q."
function fullbook_process(a,N,theta)
    iq = a.name == :C1 ? 1 : 2
    vq,vs = exp.(2theta[iq:iq+1])
    n = 2+2N+(a.name == :H1 ? 1 : 0)
    Q = zeros(n,n)
    for team in 1:N
        i,j = 2+team,2+N+team
        Q[i,i] = Q[j,j] = vq+vs
        Q[i,j] = Q[j,i] = vs-vq
    end
    return Q
end

"Original-side design with the inherited zero-sum projection; fixture order is panel order."
function fullbook_design(a,p,theta)
    N = MID.n_teams(p)
    n = 2+2N+(a.name == :H1 ? 1 : 0)
    H = zeros(MID.n_obs(p),n)
    h = zeros(n)
    for j in axes(H,1)
        MID.obs_row!(h,p,j,0)
        if a.name == :H1
            h[end] = -(1-p.obs_home[j])
        elseif a.name == :H2 && p.obs_home[j] == 1
            # q_home = (C alpha_home - C beta_home)/2, not q_away.
            kappa = theta[4]
            for team in 1:N
                centred = (team == p.obs_att[j] ? 1.0 : 0.0)-1/N
                h[2+team] += kappa*centred/2
                h[2+N+team] -= kappa*centred/2
            end
        end
        H[j,:] .= h
    end
    return H
end

"Join only the panel's full books; targets/IDs/order are never regenerated from a fit."
function fullbook_markets(ds,p,config)
    book,_ = MM.gated_close(ds,config)
    books = Dict(Int(first(g.match_id))=>DF.DataFrame(g) for g in DF.groupby(book,:match_id))
    ids = p.obs_match[1:2:end]
    ids == Int.(p.matches.match_id) || error("panel fixture/target order differs")
    length(unique(ids)) == length(ids) || error("duplicate full-book fixture")
    for id in ids
        haskey(config.excluded_matches,id) && error("excluded fixture $id in panel")
        haskey(books,id) && book_type(books[id]) == "full" || error("non-full book $id")
    end
    return [PF.market_vectors(books[id]) for id in ids]
end

"""
Propagate theta=H*x+u posterior moments b,V to x. With a=H*m,
S=H*P*H'+U and K=P*H'/S, Gaussian conditioning gives
m_new=m+K*(b-a), P_new=P+K*(V-S)*K'. U is included ONCE.
No inverse of the full state covariance is required; centred/static directions
may be singular. Symmetrisation only removes arithmetic asymmetry, not curvature.
"""
function conditional_state_update(m,P,H,U,b,V)
    a = H*m
    S = LA.Symmetric(H*P*H'+U)
    K = (S\(H*P))'
    mean = m+K*(b-a)
    covariance = P+K*(V-S)*K'
    return mean,Matrix(LA.Symmetric((covariance+covariance')/2))
end

"Exact Gaussian observation update, used for C0/H1/H2 and independent toy checks."
function gaussian_state_update(m,P,H,z,R)
    S = LA.Symmetric(H*P*H'+R)
    e = z-H*m
    marginal = DS.logpdf(DS.MvNormal(H*m,S),z)
    K = (S\(H*P))'
    mean = m+K*e
    covariance = P-K*H*P
    return (; mean,covariance=Matrix(LA.Symmetric((covariance+covariance')/2)),marginal)
end

"""
Frozen Gaussian factor matching the returned Laplace marginal and moments.
R=J_clip^-1, z=R*information, and L_G(theta)=exp(constant)*N(z;theta,R).
The constant is chosen so its expectation under the CURRENT N(a,S) equals
update.marginal. It includes the original exact-density peak/normalisers.
Integrating u changes R to R+U for a state observation of H*x, without another
likelihood contribution. Freezing these factors gives a linear-Gaussian batch
reference, NOT an exact batch representation of the nonlinear Dirichlet model.
"""
function frozen_book_factor(update,a,S)
    J = LA.Symmetric(inv(update.covariance)-inv(S))
    R = inv(J)
    z = R*update.information
    constant = update.marginal-DS.logpdf(DS.MvNormal(a,LA.Symmetric(S+R)),z)
    return (; z,R=Matrix(R),constant)
end

"""
Full-book filter. C0/H1/H2 are exact Gaussian. C1 is a sequential approximate
collapsed Laplace filter with the revision-3 mean and unchanged revision-2
marginal/clipping. All pre-week forecasts precede EVERY update of that week.
Within-week updates follow the existing (week,date,match_id) order.
Store the local factors for approximate RTS, frozen-factor gates and u smoothing.
`likelihoods` is only a deterministic toy-Gaussian test injection.
"""
function fullbook_filter(a,p,theta; markets=nothing,store=false,predict=false,
                         gamma_def_sd=0.25,likelihoods=nothing,mode_audit=nothing,mode_solver=joint_mode)
    m,P = fullbook_initial(a,MID.n_teams(p); gamma_def_sd)
    H = fullbook_design(a,p,theta)
    Q = fullbook_process(a,MID.n_teams(p),theta)
    n,T = length(m),p.n_weeks
    Ms = store ? zeros(n,T) : zeros(0,0)
    Ps = store ? zeros(n,n,T) : zeros(0,0,0)
    pm = predict ? zeros(MID.n_fixtures(p),2) : zeros(0,0)
    pc = predict ? zeros(MID.n_fixtures(p),2,2) : zeros(0,0,0)
    factors = NamedTuple[]
    noise = exp(2theta[a.name == :C1 ? 3 : 1])*Matrix(LA.I,2,2)
    if a.name == :C1
        (markets !== nothing || likelihoods !== nothing) || error("C1 needs full markets")
        countbooks = likelihoods === nothing ? length(markets) : length(likelihoods)
        countbooks == MID.n_fixtures(p) || error("book/fixture count differs")
    end
    ll = 0.0
    for t in 1:T
        t >= 2 && (P += Q)
        js = p.week_ptr[t]:(p.week_ptr[t+1]-1)
        if predict
            for j in first(js):2:last(js)
                f = cld(j,2)
                B = H[j:j+1,:]
                pm[f,:] .= B*m
                pc[f,:,:] .= B*P*B'+noise
            end
        end
        for j in first(js):2:last(js)
            f = cld(j,2)
            B = H[j:j+1,:]
            if a.name == :C1
                pred = B*m
                S = Matrix(LA.Symmetric(B*P*B'+noise))
                if likelihoods === nothing
                    raw = x -> PF.book_logdensity(x,markets[f],exp(theta[4]))
                    adf = x -> differentiable_logdensity(x,markets[f],exp(theta[4]))
                    audit = mode_audit === nothing ? nothing :
                        (optimum,raw,a,S) -> mode_audit(p.obs_match[j],optimum,raw,a,S,adf)
                    update = laplace_update(raw,pred,S; third_likelihood=adf,
                        derivative=fullbook_derivative_rule(markets[f]),mode_audit=audit,mode_solver)
                else
                    raw = likelihoods[f]
                    update = laplace_update(raw,pred,S; derivative=ad_derivatives)
                end
                ll += update.marginal # already integrates the book x state prediction
                if store
                    factor = frozen_book_factor(update,pred,S)
                    push!(factors,(; match_id=p.obs_match[j],week=t,H=B,
                        z=factor.z,R=factor.R,constant=factor.constant,
                        a=pred,S,theta_mean=update.mean,theta_cov=update.covariance,
                        raw_covariance=update.raw_covariance,J=update.J))
                end
                m,P = conditional_state_update(m,P,B,noise,update.mean,update.covariance)
            else
                update = gaussian_state_update(m,P,B,p.obs_y[j:j+1],noise)
                ll += update.marginal
                m,P = update.mean,update.covariance
            end
        end
        if store
            Ms[:,t] .= m
            Ps[:,:,t] .= P
        end
    end
    return (; loglik=ll,m_filt=Ms,P_filt=Ps,pair_mean=pm,pair_cov=pc,factors,H,Q,noise)
end

"RTS on the forward Gaussian moments; approximate for C1, exact for two-stage rungs."
function fullbook_smoothing(f)
    means,covs = copy(f.m_filt),copy(f.P_filt)
    for t in (size(means,2)-1):-1:1
        Pt = f.P_filt[:,:,t]
        prediction = LA.Symmetric(Pt+f.Q)
        C = LA.cholesky(prediction; check=false)
        gain = LA.issuccess(C) ? (C\Pt')' : Pt*LA.pinv(Matrix(prediction))
        means[:,t] .= f.m_filt[:,t]+gain*(means[:,t+1]-f.m_filt[:,t])
        V = Pt+gain*(covs[:,:,t+1]-prediction)*gain'
        covs[:,:,t] .= (V+V')/2
    end
    return means,covs
end

"Independent joint Gaussian over all weeks, original-side observations or frozen C1 factors."
function fullbook_batch(a,p,theta; factors=nothing,gamma_def_sd=0.25)
    m0,P0 = fullbook_initial(a,MID.n_teams(p); gamma_def_sd)
    Q = fullbook_process(a,MID.n_teams(p),theta)
    n,T = length(m0),p.n_weeks
    M = repeat(m0,T)
    Sigma = zeros(n*T,n*T)
    block(t) = ((t-1)*n+1):(t*n)
    for t in 1:T, u in 1:T
        Sigma[block(t),block(u)] .= P0+(min(t,u)-1)*Q
    end
    H = zeros(MID.n_obs(p),n*T)
    R = zeros(MID.n_obs(p),MID.n_obs(p))
    z = copy(p.obs_y)
    design = fullbook_design(a,p,theta)
    constants = 0.0
    for j in 1:2:MID.n_obs(p)
        f,t = cld(j,2),p.obs_week[j]
        H[j:j+1,block(t)] .= design[j:j+1,:]
        if a.name == :C1
            factors === nothing && error("batch C1 requires frozen factors")
            factor = factors[f]
            factor.match_id == p.obs_match[j] || error("frozen factor order differs")
            z[j:j+1] .= factor.z
            R[j:j+1,j:j+1] .= factor.R+exp(2theta[3])*Matrix(LA.I,2,2)
            constants += factor.constant
        else
            R[j:j+1,j:j+1] .= exp(2theta[1])*Matrix(LA.I,2,2)
        end
    end
    C = LA.cholesky(LA.Symmetric(H*Sigma*H'+R))
    e = z-H*M
    ll = constants-0.5*(length(e)*log(2pi)+LA.logdet(C)+LA.dot(e,C\e))
    K = (C\(H*Sigma))'
    return (; loglik=ll,post_mean=M+K*e,post_cov=Sigma-K*H*Sigma)
end

"Deterministic C0/HA reductions and independent Gaussian batch/RTS gates, plus inherited gates."
function fullbook_engine_gates()
    rows = CM.engine_gates_b2()
    p = MID.toy_panel(Random.Xoshiro(3951))
    theta = log.([0.07,0.04,0.015])
    c0 = fullbook_filter(FullBookRung(:C0),p,theta; store=true)
    r6 = CM.covariance_filter(CM.CovarianceRung(6),p,c0_parameters(exp.(theta)...))
    push!(rows,("C0 matched R6 loglik",abs(c0.loglik-r6.loglik),1e-9,false))
    h1 = fullbook_filter(FullBookRung(:H1),p,theta; gamma_def_sd=0.0)
    h2 = fullbook_filter(FullBookRung(:H2),p,vcat(theta,0.0))
    push!(rows,("H1 gamma_def pinned zero reduction",abs(h1.loglik-c0.loglik),1e-9,false))
    push!(rows,("H2 kappa zero reduction",abs(h2.loglik-c0.loglik),1e-9,false))
    for (a,z) in ((FullBookRung(:C0),theta),(FullBookRung(:H1),theta),
                  (FullBookRung(:H2),vcat(theta,0.3)))
        f = fullbook_filter(a,p,z; store=true)
        batch = fullbook_batch(a,p,z)
        means,covs = fullbook_smoothing(f)
        push!(rows,("$(a.name) batch loglik",abs(f.loglik-batch.loglik),1e-9,false))
        push!(rows,("$(a.name) batch RTS mean",maximum(abs.(vec(means)-batch.post_mean)),1e-8,false))
        n = size(means,1)
        err = maximum(maximum(abs.(covs[:,:,t]-batch.post_cov[(t-1)*n+1:t*n,(t-1)*n+1:t*n]))
            for t in 1:p.n_weeks)
        push!(rows,("$(a.name) batch RTS covariance",err,1e-8,false))
    end
    # Gaussian books test C1 collapse without nonlinear approximation error.
    gaussian = [x -> DS.logpdf(DS.MvNormal(p.obs_y[j:j+1],0.03^2*Matrix(LA.I,2,2)),x)
        for j in 1:2:MID.n_obs(p)]
    z = log.([0.04,0.015,0.05,1000.0])
    f = fullbook_filter(FullBookRung(:C1),p,z; store=true,likelihoods=gaussian)
    batch = fullbook_batch(FullBookRung(:C1),p,z; factors=f.factors)
    means,covs = fullbook_smoothing(f)
    push!(rows,("C1 toy Gaussian frozen batch loglik",abs(f.loglik-batch.loglik),1e-9,false))
    push!(rows,("C1 toy Gaussian frozen batch RTS",maximum(abs.(vec(means)-batch.post_mean)),1e-8,false))
    n = size(means,1)
    cov_error = maximum(maximum(abs.(covs[:,:,t]-batch.post_cov[(t-1)*n+1:t*n,(t-1)*n+1:t*n]))
        for t in 1:p.n_weeks)
    push!(rows,("C1 toy Gaussian frozen batch RTS covariance",cov_error,1e-8,false))
    # Gate 2: rotate a correlated book error, then integrate isotropic fixture u.
    m0,P0 = fullbook_initial(FullBookRung(:C0),MID.n_teams(p))
    H = c0.H[1:2,:]
    z0 = p.obs_y[1:2]
    R = [0.07 0.01; 0.01 0.03]
    U = 0.05^2*Matrix(LA.I,2,2)
    full = gaussian_state_update(m0,P0,H,z0,R+U)
    eig = LA.eigen(LA.Symmetric(R))
    design,values = eig.vectors'*H,eig.vectors'*z0
    m,P = copy(m0),copy(P0)
    ll = 0.0
    for j in 1:2
        scalar = gaussian_state_update(m,P,reshape(design[j,:],1,:),[values[j]],
            reshape([eig.values[j]+U[1,1]],1,1))
        m,P = scalar.mean,scalar.covariance
        ll += scalar.marginal
    end
    push!(rows,("C Gate2 scalar eigen mean",maximum(abs.(m-full.mean)),1e-10,false))
    push!(rows,("C Gate2 scalar eigen covariance",maximum(abs.(P-full.covariance)),1e-10,false))
    push!(rows,("C Gate2 scalar eigen loglik",abs(ll-full.marginal),1e-10,false))
    direct = fullbook_filter(FullBookRung(:C0),p,
        log.([sqrt(0.03^2+0.05^2),0.04,0.015]); store=true)
    push!(rows,("C1 Gaussian exact observation reduction",abs(f.loglik-direct.loglik),1e-9,false))
    rows.pass = rows.value .<= rows.tol
    return rows
end
