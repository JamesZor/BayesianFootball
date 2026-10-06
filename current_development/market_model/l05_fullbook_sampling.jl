# Included inside PooledMarket. No execution or database access.
import Serialization

"Synthetic probabilities carried in log-space, including draws below Float64's positive range."
struct FullBookLogMarket
    selections::Vector{Symbol}
    p::Vector{Float64} # display/diagnostic projection only; may underflow to zero
    logp::Vector{Float64} # the likelihood consumes this, never log(p)
end
fullbook_market_logp(market) = log.(market.p)
fullbook_market_logp(market::FullBookLogMarket) = market.logp

"""
Exact Gamma(a)=Gamma(a+1)*U^(1/a) identity for a<1, evaluated in log-space.
Normalising independent Gamma draws gives Dirichlet(n*q) with no clipping,
rejection/resampling or changed concentration. Only storage/arithmetic changes.
"""
function fullbook_logdirichlet(rng,alpha)
    all(x -> x > 0 && isfinite(x),alpha) || error("invalid synthetic Dirichlet shape")
    logg = [a < 1 ? log(rand(rng,DS.Gamma(a+1,1)))+log1p(-rand(rng))/a :
        log(rand(rng,DS.Gamma(a,1))) for a in alpha]
    all(isfinite,logg) || error("synthetic log-Gamma is nonfinite")
    return logg.-logsumexp(logg)
end

"Production density algebra with synthetic log-probabilities; all normalisers retained."
function PF.book_logdensity(theta,markets::Vector{FullBookLogMarket},n)
    probabilities = MM.selection_probabilities(Vector{Float64}(theta))
    value = 0.0
    for market in markets
        alpha = n.*[probabilities[s] for s in market.selections]
        value += SpecialFunctions.loggamma(sum(alpha))-sum(SpecialFunctions.loggamma,alpha)
        value += sum((alpha.-1).*market.logp)
    end
    return value
end

# Synthetic logp can be enormous: finite differences then lose the last digits
# at the joint mode. Dual evaluation uses the already parity-tested grid algebra,
# while primal likelihood/marginal evaluation stays on the production grid.
PF.book_logdensity(theta::AbstractVector{<:ForwardDiff.Dual},
    markets::Vector{FullBookLogMarket},n) = differentiable_logdensity(theta,markets,n)
fullbook_derivative_rule(markets) = derivatives
fullbook_derivative_rule(markets::Vector{FullBookLogMarket}) = ad_derivatives

"Physical scales are exponentiated; H2 kappa is already on its physical scale."
function fullbook_physical_draws(a,U)
    return cat([a.name == :H2 && j == 4 ? U[:,j:j,:] : exp.(U[:,j:j,:])
        for j in axes(U,2)]...; dims=2)
end

"""
Numerical support only, NOT the inherited ±12 box. Positive coordinates must
produce representable nonzero variances; n must be positive/finite and n*log(n)
representable for loggamma. log n otherwise retains its unbounded Normal prior.
Nonfinite input is rejected before any filter. Numerical mode/PSD failures at
supported parameters propagate as errors, never as posterior rejections.
"""
function fullbook_supported(a,theta)
    length(theta) == length(MID.param_names(a)) || error("wrong full-book parameter layout")
    all(isfinite,theta) || return false
    variances = exp.(2theta[1:3])
    all(x -> isfinite(x) && x > 0,variances) || return false
    if a.name == :C1
        n = exp(theta[4])
        return isfinite(n) && n > 0 && isfinite(n*log(n))
    end
    return true
end

"Collapsed likelihood plus normalised rung prior; no R6 fitted draws or prior."
function fullbook_logtarget(a,p,theta; markets=nothing,likelihoods=nothing)
    fullbook_supported(a,theta) || return -Inf
    lp = MID.log_prior(a,theta)
    isfinite(lp) || return -Inf
    ll = fullbook_filter(a,p,theta; markets,likelihoods).loglik
    isfinite(ll) || error("$(a.name) supported filter returned nonfinite likelihood at $theta")
    return lp+ll
end

"""
Dedicated coordinate slice chain. Initial jitter uses a chain-local Xoshiro;
widths adapt on the last <=200 warmup draws, every 25 iterations after 50,
and freeze exactly at warmup. All retained iterations are saved (thin=1).
The MID slice kernel is reused, not its bounded target or R6 fit/prior.
"""
function fullbook_chain(a,p,seed; markets=nothing,warmup=2000,samples=3000,
                        progress=nothing,cancellation=nothing)
    warmup >= 0 && samples > 0 || error("invalid chain budget")
    rng = Random.Xoshiro(seed)
    theta = MID.init_centre(a)+0.35randn(rng,length(MID.param_names(a)))
    last_evaluation = copy(theta)
    iteration = 0
    completed = false
    function target(z)
        cancellation !== nothing && cancellation[] && error("peer full-book chain failed")
        last_evaluation .= z
        return fullbook_logtarget(a,p,z; markets)
    end
    U = zeros(samples,length(theta))
    history = zeros(warmup,length(theta))
    widths = ones(length(theta))
    try
        lf = target(theta)
        isfinite(lf) || error("nonfinite initial full-book target")
        for it in 1:(warmup+samples)
            iteration = it
            theta,lf = MID.slice_sweep(target,theta,lf,widths,rng)
            if it <= warmup
                history[it,:] .= theta
                if it >= 50 && it % 25 == 0
                    widths .= clamp.(3vec(ST.std(history[max(1,it-199):it,:]; dims=1)),0.02,3.0)
                end
            else
                U[it-warmup,:] .= theta
            end
            if progress !== nothing && (it % 250 == 0 || (it <= 250 && it % 25 == 0))
                progress(it,lf)
            end
        end
        completed = true
    finally
        # No catch/rejection: the original error propagates. Cancel peers and
        # retain the exact failing coordinate for a deterministic diagnosis.
        if !completed
            cancellation !== nothing && (cancellation[] = true)
            println(stderr,"C4_CHAIN_ABORT rung=$(a.name) seed=$seed iteration=$iteration theta=$last_evaluation")
            flush(stderr)
        end
    end
    return (; U,widths)
end

"Four independent chain tasks, indexed seed vector; scheduling does not affect RNG."
function fit_fullbook(a,p; markets=nothing,seeds,warmup=2000,samples=3000,progress=true)
    length(seeds) == 4 && length(unique(seeds)) == 4 || error("exactly four unique chain seeds required")
    start = time()
    cancellation = Threads.Atomic{Bool}(false)
    tasks = Task[]
    @sync for c in eachindex(seeds)
        callback = progress ? (it,lf) -> begin
            println("C4 $(a.name) chain=$c iteration=$it target=$lf")
            flush(stdout)
        end : nothing
        push!(tasks,Threads.@spawn fullbook_chain(a,p,seeds[c]; markets,warmup,samples,
            progress=callback,cancellation))
    end
    results = fetch.(tasks)
    U = cat([r.U for r in results]...; dims=3)
    fit = MID.ArmFit(a,MID.param_names(a),fullbook_physical_draws(a,U),U,
        Matrix{Float64}[],Vector{Float64}[],zeros(0,0),time()-start)
    return (; fit,widths=reduce(hcat,[r.widths for r in results]))
end

"Restrict fixture pairs AND metadata, join their books by ID, never positional reuse."
function restrict_fullbook(p,markets,seasons)
    length(markets) == MID.n_fixtures(p) || error("full-panel market count mismatch")
    p.obs_match[1:2:end] == Int.(p.matches.match_id) || error("full-panel metadata order mismatch")
    keep = in.(p.obs_season,Ref(seasons))
    keep[1:2:end] == keep[2:2:end] || error("restriction splits a fixture pair")
    r = MID.restrict_panel(p,keep)
    ids = r.obs_match[1:2:end]
    meta = Dict(Int(row.match_id)=>NamedTuple(row) for row in DF.eachrow(p.matches))
    matches = DF.DataFrame([meta[id] for id in ids])
    restricted = MID.MarketPanel(matches,r.refusals,r.teams,r.n_weeks,r.week_start,
        r.obs_week,r.obs_home,r.obs_att,r.obs_def,r.obs_y,r.obs_match,r.obs_season,
        r.week_ptr,r.obs_X,r.feature_names)
    byid = Dict(id=>market for (id,market) in zip(p.obs_match[1:2:end],markets))
    return restricted,[byid[id] for id in ids]
end

"Populate a fresh panel with synthetic log-rate observations, preserving fixture IDs."
function fullbook_panel_values(p,y)
    length(y) == MID.n_obs(p) || error("synthetic observation count mismatch")
    matches = copy(p.matches)
    matches.lambda_mkt_h = exp.(y[1:2:end])
    matches.lambda_mkt_a = exp.(y[2:2:end])
    return MID.MarketPanel(matches,copy(p.refusals),copy(p.teams),p.n_weeks,
        copy(p.week_start),copy(p.obs_week),copy(p.obs_home),copy(p.obs_att),
        copy(p.obs_def),copy(y),copy(p.obs_match),copy(p.obs_season),copy(p.week_ptr),
        copy(p.obs_X),copy(p.feature_names))
end

"Generate state/deviation population separately from book draws (also tests extreme priors)."
function synthetic_fullbook_latents(a,p,theta,rng)
    m0,P0 = fullbook_initial(a,MID.n_teams(p))
    x = rand(rng,DS.MvNormal(m0,LA.Symmetric(P0)))
    H = fullbook_design(a,p,theta)
    N = MID.n_teams(p)
    iq = a.name == :C1 ? 1 : 2
    sq,ss = exp.(theta[iq:iq+1])
    noise = exp(theta[a.name == :C1 ? 3 : 1])
    states = zeros(length(x),p.n_weeks)
    y = zeros(MID.n_obs(p))
    fixture_u = zeros(MID.n_fixtures(p),2)
    for t in 1:p.n_weeks
        if t >= 2
            dq,ds = sq*randn(rng,N),ss*randn(rng,N)
            x[3:2+N] .+= dq+ds
            x[3+N:2+2N] .+= ds-dq
        end
        states[:,t] .= x
        js = p.week_ptr[t]:(p.week_ptr[t+1]-1)
        for j in first(js):2:last(js)
            f = cld(j,2)
            u = noise*randn(rng,2)
            fixture_u[f,:] .= u
            y[j:j+1] .= H[j:j+1,:]*x+u
        end
    end
    return (; states,y,fixture_u)
end

"""
Generate a full-book-only panel from known q/s/u/n on the SAME production grid.
The supplied schedule fixes fixture/team/week population and market line mix;
only probabilities are generated. Raw independent alpha/beta initial states
come from the inherited prior. Independent raw q/s steps imply alpha=q+s,
beta=s-q; the design applies zero-sum centring. Static mu/gamma (and H1 def)
are generated from their stated priors. Each fixture theta=H*x+u is independent
conditional on its week. C1 draws each market p ~ Dirichlet(n*q(theta)),
including the two-way Beta equivalent, using log-normalised Gamma draws.
Log probabilities remain finite even when their display p underflows; no
clipping, inversion gates or retries. Its obs_y stores the latent theta as
recovery truth, not an isolated inversion/scoring target. Gaussian rungs draw
obs_y with sigma_obs and do not use the supplied markets in fitting.
"""
function synthetic_fullbook(a,p,templates,theta; seed)
    length(templates) == MID.n_fixtures(p) || error("synthetic market schedule mismatch")
    for markets in templates
        any(m -> Set(m.selections) == Set((:home,:draw,:away)),markets) &&
            length(markets) >= 2 || error("synthetic recovery forbids thin books")
    end
    fullbook_supported(a,theta) || error("unsupported generating parameters")
    rng = Random.Xoshiro(seed)
    states,y,fixture_u = synthetic_fullbook_latents(a,p,theta,rng)
    markets_out = Vector{Vector{FullBookLogMarket}}(undef,length(templates))
    for f in eachindex(templates)
        j = 2*f-1
        probabilities = MM.selection_probabilities(y[j:j+1])
        markets_out[f] = FullBookLogMarket[]
        for m in templates[f]
            q = [probabilities[s] for s in m.selections]
            logp = a.name == :C1 ? fullbook_logdirichlet(rng,exp(theta[4])*q) : log.(q)
            push!(markets_out[f],FullBookLogMarket(copy(m.selections),exp.(logp),logp))
        end
        all(m -> all(isfinite,m.logp),markets_out[f]) ||
            error("synthetic log market nonfinite: seed=$seed fixture=$(p.obs_match[j])")
    end
    return (; panel=fullbook_panel_values(p,y),markets=markets_out,states,
        fixture_u,truth=fullbook_physical_draws(a,reshape(theta,1,:,1))[1,:,1],seed)
end

"Persist all-parameter convergence BEFORE accepting recovery or promoting inference."
function fullbook_diagnostics(fit; protocol,seed)
    diag = MID.convergence_table(fit)
    diag.protocol .= protocol
    diag.seed .= seed
    diag.gate_pass = (diag.rhat .<= 1.05) .& (diag.ess_bulk .>= 200) .& (diag.ess_tail .>= 200)
    return diag
end

"Prescribed C1 full-book recovery, all four truths inside marginal 90% intervals."
function recover_fullbook(p,markets,out; generation_seed,chain_seeds,
                          truth=log.([0.03,0.01,0.06,1000.0]))
    a = FullBookRung(:C1)
    generated = synthetic_fullbook(a,p,markets,truth; seed=generation_seed)
    Serialization.serialize(joinpath(out,"synthetic_panel.jls"),generated)
    result = fit_fullbook(a,generated.panel; markets=generated.markets,seeds=chain_seeds)
    Serialization.serialize(joinpath(out,"C1_recovery.jls"),result)
    diag = fullbook_diagnostics(result.fit; protocol="synthetic",seed=generation_seed)
    CSV.write(joinpath(out,"recovery_convergence_c.csv"),diag)
    rows = DF.DataFrame(parameter=MID.param_names(a),truth=generated.truth,
        lo=[ST.quantile(vec(result.fit.draws[:,j,:]),0.05) for j in 1:4],
        median=[ST.median(vec(result.fit.draws[:,j,:])) for j in 1:4],
        hi=[ST.quantile(vec(result.fit.draws[:,j,:]),0.95) for j in 1:4])
    rows.recovered = (rows.lo .<= rows.truth) .& (rows.truth .<= rows.hi)
    CSV.write(joinpath(out,"synthetic_recovery_c.csv"),rows)
    all(diag.gate_pass) || error("synthetic C1 convergence failed; no promotion")
    all(rows.recovered) || error("synthetic C1 parameter recovery failed; no promotion")
    println("C05_FULLBOOK_RECOVERY_DONE")
    return result
end
