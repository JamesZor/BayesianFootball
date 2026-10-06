module PooledMarket

import CSV
import DataFrames
import Dates
import Distributions
import LinearAlgebra
import Optim
import Statistics

if !isdefined(parentmodule(@__MODULE__), :LaplaceBookPreflight)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "l05_laplace_preflight.jl"))
end
if !isdefined(parentmodule(@__MODULE__), :CovarianceMarket)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "l03_covariance.jl"))
end
const PF = parentmodule(@__MODULE__).LaplaceBookPreflight
const CM = parentmodule(@__MODULE__).CovarianceMarket
const MM = PF.MM
const MID = CM.MID
const DF = DataFrames
const LA = LinearAlgebra
const ST = Statistics
const DS = Distributions

"Central derivatives of the exact density on the unchanged production grid."
function derivatives(f, x; step=2e-4)
    k = length(x)
    gradient = zeros(k)
    hessian = zeros(k, k)
    centre = f(x)
    for i in 1:k
        ei = zeros(k)
        ei[i] = step
        plus, minus = f(x+ei), f(x-ei)
        gradient[i] = (plus-minus)/(2step)
        hessian[i,i] = (plus-2centre+minus)/step^2
        for j in 1:(i-1)
            ej = zeros(k)
            ej[j] = step
            hessian[i,j] = hessian[j,i] = (f(x+ei+ej)-f(x+ei-ej)-
                f(x-ei+ej)+f(x-ei-ej))/(4step^2)
        end
    end
    return gradient, hessian
end

"Newton ascent with backtracking from the prediction, not from Phase A's inversion."
function joint_mode(f, start; tolerance=2e-6, iterations=100)
    x = copy(start)
    for iteration in 1:iterations
        gradient, hessian = derivatives(f, x)
        if maximum(abs, gradient) <= tolerance
            return (; mode=x, gradient, precision=-hessian, iteration)
        end
        eigen = LA.eigen(LA.Symmetric(-hessian))
        precision = eigen.vectors*LA.Diagonal(max.(eigen.values, 1e-6))*eigen.vectors'
        direction = precision\gradient
        if LA.norm(direction) <= 2e-7
            return (; mode=x, gradient, precision=-hessian, iteration)
        end
        direction ./= max(1.0, LA.norm(direction))
        value = f(x)
        scale = 1.0
        while scale >= 2.0^-30
            candidate = x+scale*direction
            next = f(candidate)
            if isfinite(next) && next >= value + 1e-4scale*LA.dot(gradient,direction)
                x = candidate
                break
            end
            scale /= 2
        end
        if scale < 2.0^-30
            # At floating-point density resolution, the Newton displacement is
            # a stronger stopping diagnostic than an unresolvable Armijo gain.
            LA.norm(direction) <= 2e-7 || error("joint-mode line search stalled: $gradient")
            return (; mode=x, gradient, precision=-hessian, iteration)
        end
    end
    error("joint mode did not converge in $iterations iterations")
end

"Log-sum-exp with a loud failure when the integration has no finite support."
function logsumexp(terms)
    largest = maximum(terms)
    isfinite(largest) || error("quadrature has no finite support")
    return largest+log(sum(exp.(terms.-largest)))
end

"""
Revision 2 joint-mode Laplace integral and information-form book update.
Let f(x)=log L(x)+log N(x;a,S), x*=argmax f, and B=-f''(x*).
Taylor expansion gives exp(f(x)) ≈ exp(f(x*)) exp(-δ'Bδ/2).
Thus log integral ≈ f(x*) + k/2 log(2π) - 1/2 log det B.
All Dirichlet normalisers remain in f and depend on n. No isolated-KL
centre, cross-entropy Hessian or omitted peak term enters this calculation.

For the approximate posterior, the local likelihood has score g and precision J.
Its information vector is J*x*+g; adding prior information S^-1*a gives
posterior covariance (S^-1+J)^-1 and corresponding mean. At a converged
joint mode without clipping the mean is x*. Eigenvalues of J below 1e-8
are clipped as prescribed; raw J is retained for diagnosis. The marginal's
B is the actual negative Hessian of the bracket, not the clipped likelihood.
This distinction matters when a curved constant-total contour gives negative
observed curvature in its tangent direction. Both covariances are returned.

For 1X2-only books x=d=theta_h-theta_a, L means the level-integrated likelihood
with flat dℓ, ℓ=(theta_h+theta_a)/2. That likelihood is independent of the
state's ℓ by definition. k=1 and S=c'S_side*c, c=(1,-1).
This is an approximate update, not an exact nonlinear Kalman filter. No second
relinearisation is used: a second Newton expansion at the same joint mode
would not restore higher-order mass or moments. RTS would be approximate.
"""
function laplace_update(loglikelihood, a, S)
    prior = DS.MvNormal(a, LA.Symmetric(S))
    target = x -> loglikelihood(x)+DS.logpdf(prior,x)
    optimum = joint_mode(target,a)
    x = optimum.mode
    gradient, hessian = derivatives(loglikelihood,x)
    J = -hessian
    B = LA.Symmetric(J+inv(S))
    LA.isposdef(B) || error("joint mode has nonpositive bracket precision")
    marginal = target(x)+length(a)/2*log(2pi)-LA.logdet(B)/2
    eigen = LA.eigen(LA.Symmetric(J))
    clipped = eigen.vectors*LA.Diagonal(max.(eigen.values,1e-8))*eigen.vectors'
    covariance = inv(LA.Symmetric(inv(S)+clipped))
    mean = covariance*(S\a+clipped*x+gradient)
    return (; marginal,mean,covariance,mode=x,raw_covariance=inv(B),J,
        mode_gradient=maximum(abs,optimum.gradient),iteration=optimum.iteration)
end

"""
Exact-density quadrature and posterior moments, in mode-centred Gaussian
importance coordinates. The Gaussian proposal is a change of variables only;
EVERY node uses the original exact integrand/proposal ratio. Increase order
independently to check mass and moment resolution, including non-Gaussian tails.
Supports 1-D and 2-D; uses the unclipped bracket Hessian for the proposal.
"""
function quadrature_moments(loglikelihood,a,S; order=32)
    prior = DS.MvNormal(a,LA.Symmetric(S))
    target = x -> loglikelihood(x)+DS.logpdf(prior,x)
    optimum = joint_mode(target,a)
    covariance = inv(LA.Symmetric(optimum.precision))
    proposal = DS.MvNormal(optimum.mode,LA.Symmetric(covariance))
    factor = LA.cholesky(LA.Symmetric(covariance)).L
    nodes,weights = PF.normal_quadrature(order)
    k = length(a)
    coordinates = k == 1 ? [[x] for x in nodes] : [[x,y] for y in nodes for x in nodes]
    logweights = k == 1 ? log.(weights) : [log(x)+log(y) for y in weights for x in weights]
    points = [optimum.mode+factor*x for x in coordinates]
    terms = [w+target(x)-DS.logpdf(proposal,x) for (w,x) in zip(logweights,points)]
    marginal = logsumexp(terms)
    probabilities = exp.(terms.-marginal)
    mean = sum(p*x for (p,x) in zip(probabilities,points))
    cov = sum(p*(x-mean)*(x-mean)' for (p,x) in zip(probabilities,points))
    return (; marginal,mean,covariance=cov)
end

"""
Level integral for a 1X2-only book: integrate the EXACT Dirichlet density over
flat Lebesgue dℓ. Locate its level mode, then Gaussian importance quadrature.
The omitted tails are checked at explicit endpoints; no Gaussian state-level
prior is introduced. The mode/proposal changes coordinates, not the integrand.
The grid is evaluated only at representable rates; underflow excursions carry
zero density, never an invented probability or repaired book.
"""
function level_integral(d,markets,n; order=32, bounds=(-8.0,2.5))
    f = ell -> begin
        theta = [ell+d/2,ell-d/2]
        value = PF.book_logdensity(theta,markets,n)
        isfinite(value) ? value : -Inf
    end
    optimum = Optim.optimize(ell->-f(ell),bounds[1],bounds[2],Optim.Brent(); abs_tol=1e-9)
    mode = Optim.minimizer(optimum)
    step = 2e-4
    precision = -(f(mode+step)-2f(mode)+f(mode-step))/step^2
    precision > 0 || error("nonpositive level-integral mode curvature")
    sd = 1/sqrt(precision)
    proposal = DS.Normal(mode,sd)
    nodes,weights = PF.normal_quadrature(order)
    terms = [log(w)+f(mode+sd*x)-DS.logpdf(proposal,mode+sd*x) for (x,w) in zip(nodes,weights)]
    marginal = logsumexp(terms)
    return (; marginal,mode,sd,
        lower_relative_logdensity=f(bounds[1])-f(mode),
        upper_relative_logdensity=f(bounds[2])-f(mode))
end

"Type is determined by complete gated markets, not the rank of a Hessian."
function book_type(book)
    names = Set(book.market_name)
    if "1X2" in names
        return length(names) == 1 ? "1X2-only" : "full"
    end
    return "BTTS" in names ? ("OverUnder" in names ? "BTTS+OU" : "BTTS-only") : "OU-only"
end

"C0 covariance expressed in alpha/beta coordinates; initial states remain B2's."
function c0_parameters(sigma_obs,sigma_q,sigma_s)
    scale = sqrt(sigma_q^2+sigma_s^2)
    rho = (sigma_s^2-sigma_q^2)/(sigma_s^2+sigma_q^2)
    return vcat(log.([sigma_obs,scale,scale]),atanh(rho))
end

"""
Fixed-parameter C0 pre-week predictions for ALL fixtures, updating only on the
517 full-book isolated rates. Parameters are the published R6 10a rotation
medians, not a newly estimated C0 posterior. Used only as realistic centres for
books with no Phase A isolated rate. No observation of any fixture in a week
enters that week's prediction; no thin-book placeholder enters an update.
"""
function c0_gate_predictions(ds,config,rates)
    frame = DF.DataFrame(match_id=Int.(rates.match_id),accepted=trues(DF.nrow(rates)),
        lambda_mkt_h=ones(DF.nrow(rates)),lambda_mkt_a=ones(DF.nrow(rates)),
        sse=zeros(DF.nrow(rates)),n_targets=rates.n_selections,reason=fill("",DF.nrow(rates)))
    p = MID.build_market_panel(ds,frame; config)
    rotation = CSV.read(joinpath(@__DIR__,"results","B2","rotation_quality_style.csv"),DF.DataFrame)
    diag = CSV.read(joinpath(@__DIR__,"results","B2","convergence_b2.csv"),DF.DataFrame)
    getrot(name) = first(DF.filter(r->r.rung == "R6" && r.protocol == "10a" && r.parameter == name,rotation).median)
    obs = first(DF.filter(r->r.arm == "R6" && r.protocol == "10a" && r.parameters == "sigma_obs",diag)[!,Symbol("50.0%")])
    theta = c0_parameters(obs,getrot("sigma_quality"),getrot("sigma_style"))
    arm = CM.conditional_arm(CM.CovarianceRung(6),theta)
    schedule = CM.covariance_schedule(arm,p,theta)
    N = MID.n_teams(p)
    m,P = MID.initial_state(arm,N)
    metadata = Dict(Int(r.match_id)=>r for r in eachrow(rates))
    predictions = Dict{Int,Vector{Float64}}()
    h = zeros(length(m))
    for t in 1:p.n_weeks
        if t >= 2
            P += MID.process_cov(arm,N,schedule,t)
        end
        js = p.week_ptr[t]:(p.week_ptr[t+1]-1)
        for j in first(js):2:last(js)
            H = zeros(2,length(m))
            for side in 0:1
                MID.obs_row!(h,p,j+side,0)
                H[side+1,:] .= h
            end
            predictions[p.obs_match[j]] = H*m
        end
        for j in js
            r = metadata[p.obs_match[j]]
            r.accepted && r.n_selections >= config.min_selections_ladder || continue
            MID.obs_row!(h,p,j,0)
            k = P*h
            variance = LA.dot(h,k)+obs^2
            y = log(isodd(j) ? r.lambda_h : r.lambda_a)
            m += (y-LA.dot(h,m))/variance*k
            P -= k*k'/variance
        end
        P = Matrix(LA.Symmetric(P))
    end
    return predictions
end

"""
Revised Gate 1: first min(30,available) per type in match-ID order, all settings.
Realistic centres are Phase A isolated rates where finite, fixed-parameter C0
pre-week predictions otherwise. Offset adds +0.10 to each side. Moments use
max_i |mean_L-mean_Q|/SD_Q and max_i |SD_L/SD_Q-1| <= .05 per fixture.
Raw (unclipped) covariance errors are also retained to isolate clipping effects.
Fail fast after the first book type with a failed fixed gate, as the brief requires.
"""
function laplace_gate(ds,config,out)
    rates = CSV.read(joinpath(@__DIR__,"results","A","rates.csv"),DF.DataFrame)
    rates = DF.sort(DF.filter(r->r.season in config.seasons && r.tournament in config.tournaments,rates),:match_id)
    book,_ = MM.gated_close(ds,config)
    books = Dict(Int(first(g.match_id))=>DF.DataFrame(g) for g in DF.groupby(book,:match_id))
    predictions = c0_gate_predictions(ds,config,rates)
    rows = NamedTuple[]
    kinds = ("full","1X2-only","OU-only","BTTS+OU","BTTS-only")
    availability = Dict(kind => [r for r in eachrow(rates) if
        !haskey(config.excluded_matches,r.match_id) && haskey(books,r.match_id) &&
        book_type(books[r.match_id]) == kind] for kind in kinds)
    inventory = DF.DataFrame(book_type=collect(kinds),
        available=[length(availability[k]) for k in kinds],tested=zeros(Int,length(kinds)))
    for (type_index,kind) in enumerate(kinds)
        available = availability[kind]
        selected = available[1:min(30,length(available))]
        inventory.tested[type_index] = length(selected)
        for r in selected
            markets = PF.market_vectors(books[r.match_id])
            centre = r.accepted ? log.([r.lambda_h,r.lambda_a]) : predictions[r.match_id]
            for n in (250.0,1000.0,4000.0), spread in (0.05,0.20), offset in (0.0,0.10)
                a = centre .+ offset
                lower,upper,level_delta = -Inf,-Inf,0.0
                if kind == "1X2-only"
                    d = a[1]-a[2]
                    integral32 = level_integral(d,markets,n; order=32)
                    integral64 = level_integral(d,markets,n; order=64)
                    lower,upper = integral64.lower_relative_logdensity,integral64.upper_relative_logdensity
                    level_delta = abs(integral32.marginal-integral64.marginal)
                    f32 = x -> level_integral(x[1],markets,n; order=32).marginal
                    f64 = x -> level_integral(x[1],markets,n; order=64).marginal
                    a = [d]
                    S = reshape([2spread^2],1,1)
                    approximate = laplace_update(f64,a,S)
                    exact32 = quadrature_moments(f32,a,S; order=32)
                    exact64 = quadrature_moments(f64,a,S; order=64)
                else
                    f = x -> PF.book_logdensity(x,markets,n)
                    S = spread^2*Matrix(LA.I,2,2)
                    approximate = laplace_update(f,a,S)
                    exact32 = quadrature_moments(f,a,S; order=32)
                    exact64 = quadrature_moments(f,a,S; order=64)
                end
                sd = sqrt.(LA.diag(exact64.covariance))
                mean_error = maximum(abs.(approximate.mean-exact64.mean)./sd)
                sd_error = maximum(abs.(sqrt.(LA.diag(approximate.covariance))./sd.-1))
                raw_sd_error = maximum(abs.(sqrt.(LA.diag(approximate.raw_covariance))./sd.-1))
                push!(rows,(; book_type=kind,match_id=r.match_id,n,spread,offset,
                    centre_source=r.accepted ? "Phase A isolated" : "C0 fixed-R6-rotation preweek",
                    exact32=exact32.marginal,exact64=exact64.marginal,
                    laplace=approximate.marginal,error=approximate.marginal-exact64.marginal,
                    quadrature_delta=abs(exact64.marginal-exact32.marginal),
                    moment_quadrature_delta=maximum(abs.(exact64.mean-exact32.mean)./sd),
                    sd_quadrature_delta=maximum(abs.(sqrt.(LA.diag(exact32.covariance))./sd.-1)),
                    mean_error,sd_error,raw_sd_error,moment_pass=max(mean_error,sd_error)<=0.05,
                    prediction_1=a[1],prediction_2=length(a)==2 ? a[2] : NaN,
                    mode_1=approximate.mode[1],mode_2=length(a)==2 ? approximate.mode[2] : NaN,
                    laplace_mean_1=approximate.mean[1],
                    laplace_mean_2=length(a)==2 ? approximate.mean[2] : NaN,
                    exact_mean_1=exact64.mean[1],exact_mean_2=length(a)==2 ? exact64.mean[2] : NaN,
                    exact_sd_1=sd[1],exact_sd_2=length(a)==2 ? sd[2] : NaN,
                    mean_mode_delta=maximum(abs.(approximate.mean-approximate.mode)),
                    eigen_min=minimum(LA.eigvals(LA.Symmetric(approximate.J))),
                    mode_gradient=approximate.mode_gradient,level_delta,
                    lower_relative_logdensity=lower,upper_relative_logdensity=upper))
            end
        end
        fixture = DF.DataFrame(rows)
        summary = gate_summary(fixture)
        CSV.write(joinpath(out,"laplace_gate_fixture.csv"),fixture)
        CSV.write(joinpath(out,"laplace_gate.csv"),summary)
        CSV.write(joinpath(out,"book_inventory_c.csv"),inventory)
        println("Revised Laplace gate: $kind, $(length(selected))/$(length(available)) books")
        flush(stdout)
        all(summary.gate_pass) || break
    end
    fixture = DF.DataFrame(rows)
    summary = gate_summary(fixture)
    gates = NamedTuple[]
    for r in eachrow(summary)
        setting = "$(r.book_type) n=$(r.n) sd=$(r.spread) offset=$(r.offset)"
        for (measure,value,tol) in (("median marginal",r.median_abs_error,0.01),
            ("p95 marginal",r.p95_abs_error,0.05),
            ("max mean/SD",r.max_mean_error,0.05),("max SD relative",r.max_sd_error,0.05))
            push!(gates,(; gate="C Gate1 $measure $setting",value,tol,pass=value<=tol))
        end
    end
    CSV.write(joinpath(out,"engine_gates_c.csv"),DF.DataFrame(gates))
    return (; fixture,summary)
end

"Error distributions by type and fixed setting; no fixture subset chosen by error."
function gate_summary(fixture)
    summaries = NamedTuple[]
    for g in DF.groupby(fixture,[:book_type,:n,:spread,:offset])
        errors = abs.(g.error)
        median = ST.median(errors)
        p95 = ST.quantile(errors,0.95)
        push!(summaries,(; book_type=first(g.book_type),n=first(g.n),spread=first(g.spread),
            offset=first(g.offset),books=DF.nrow(g),median_abs_error=median,p95_abs_error=p95,
            max_quadrature_delta=maximum(g.quadrature_delta),
            max_moment_quadrature_delta=maximum(g.moment_quadrature_delta),
            max_sd_quadrature_delta=maximum(g.sd_quadrature_delta),
            max_mean_error=maximum(g.mean_error),max_sd_error=maximum(g.sd_error),
            max_raw_sd_error=maximum(g.raw_sd_error),
            mean_failures=count(g.mean_error .> 0.05),sd_failures=count(g.sd_error .> 0.05),
            max_mean_mode_delta=maximum(g.mean_mode_delta),moment_pass=all(g.moment_pass),
            marginal_pass=median<=0.01 && p95<=0.05,
            gate_pass=median<=0.01 && p95<=0.05 && all(g.moment_pass)))
    end
    return DF.DataFrame(summaries)
end

end # module
