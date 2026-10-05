# Included inside TwoStageMarket; definitions only.

"""
Static per-team-season families with non-centred season-zero-sum α/β.
μ ~ Normal(log(1.35),0.5), γ ~ Normal(0.15,0.25) (TODO 023).
Human-approved 2026-10-05: σ_α,σ_β ~ HalfNormal(0.5);
log-normal/log-t σ ~ HalfNormal(0.20); Gamma shape ν ~ Gamma(2,scale 100);
log-t df ν ~ Gamma(2,scale 10) (TODO 023 Phase 2).
Log-normal/log-t η is a log location, NOT log of their arithmetic mean.
Gamma uses mean exp(η), shape ν, scale exp(η)/ν. All densities are on λ scale.
"""
function family_data(p)
    pairs = sort!(unique([(p.teams[p.obs_att[j]],p.obs_season[j]) for j in eachindex(p.obs_y)]))
    index = Dict(pair=>i for (i,pair) in enumerate(pairs))
    n = length(pairs)
    C = Matrix{Float64}(LA.I,n,n)
    for season in unique(p.obs_season)
        ids = findall(pair -> pair[2] == season,pairs)
        C[ids,ids] .-= 1/length(ids)
    end
    att = [index[(p.teams[p.obs_att[j]],p.obs_season[j])] for j in eachindex(p.obs_y)]
    def = [index[(p.teams[p.obs_def[j]],p.obs_season[j])] for j in eachindex(p.obs_y)]
    return (; y=exp.(p.obs_y),logy=copy(p.obs_y),home=copy(p.obs_home),att,def,C,n,
              fixtures=copy(p.obs_match),pairs)
end

Turing.@model function family_noise(::Val{:gamma})
    nu ~ DS.Gamma(2.0,100.0)
    return (; nu)
end
Turing.@model function family_noise(::Val{:lognormal})
    sigma ~ DS.truncated(DS.Normal(0.0,0.20),0.0,Inf)
    return (; sigma)
end
Turing.@model function family_noise(::Val{:logt})
    sigma ~ DS.truncated(DS.Normal(0.0,0.20),0.0,Inf)
    nu ~ DS.Gamma(2.0,10.0)
    return (; sigma,nu)
end
family_ll(::Val{:gamma}, y, logy, eta, noise) = DS.logpdf.(DS.Gamma.(noise.nu,exp.(eta)./noise.nu),y)
family_ll(::Val{:lognormal}, y, logy, eta, noise) =
    -0.5log(2π) .- log(noise.sigma) .- 0.5.*((logy.-eta)./noise.sigma).^2 .- logy
family_ll(::Val{:logt}, y, logy, eta, noise) =
    DS.logpdf.(DS.TDist(noise.nu),(logy.-eta)./noise.sigma) .- log(noise.sigma) .- logy

Turing.@model function static_family(family, data, mask)
    mu ~ DS.Normal(log(1.35),0.5)
    gamma ~ DS.Normal(0.15,0.25)
    sigma_alpha ~ DS.truncated(DS.Normal(0.0,0.5),0.0,Inf)
    sigma_beta ~ DS.truncated(DS.Normal(0.0,0.5),0.0,Inf)
    z_alpha ~ Turing.filldist(DS.Normal(),data.n)
    z_beta ~ Turing.filldist(DS.Normal(),data.n)
    noise ~ DynamicPPL.to_submodel(family_noise(family))
    alpha = sigma_alpha .* (data.C*z_alpha)
    beta = sigma_beta .* (data.C*z_beta)
    eta = mu .+ gamma.*data.home .+ alpha[data.att] .+ beta[data.def]
    ll = family_ll(family,data.y,data.logy,eta,noise)
    Turing.@addlogprob! sum(ll.*mask)
    return (; eta,noise)
end

"Compiled/fresh/ForwardDiff/central-finite-difference agreement at three dispersed points."
function family_gradient_gates(data)
    rows = NamedTuple[]
    for family in (:gamma,:lognormal,:logt)
        Random.seed!(3908)
        model = static_family(Val(family),data,ones(length(data.y)))
        vi = DynamicPPL.link!!(DynamicPPL.VarInfo(model),model)
        lf = DynamicPPL.LogDensityFunction(model,DynamicPPL.getlogjoint_internal,vi)
        theta = DynamicPPL.getparams(lf)
        f = x -> LogDensityProblems.logdensity(lf,x)
        tape = ReverseDiff.compile(ReverseDiff.GradientTape(f,theta))
        for (point,delta) in enumerate((0.0,0.1,-0.2))
            x = theta .+ delta.*sin.(eachindex(theta))
            compiled = similar(x)
            ReverseDiff.gradient!(compiled,tape,x)
            fresh = ReverseDiff.gradient(f,x)
            forward = ForwardDiff.gradient(f,x)
            fd = similar(x)
            for j in eachindex(x)
                step = 1e-5*max(1,abs(x[j]))
                xp = copy(x)
                xm = copy(x)
                xp[j] += step
                xm[j] -= step
                fd[j] = (f(xp)-f(xm))/(2step)
            end
            relative(g) = LA.norm(compiled-g)/max(LA.norm(compiled),LA.norm(g),1)
            for (method,g,tol) in (("fresh ReverseDiff",fresh,1e-8),("ForwardDiff",forward,1e-6),("finite difference",fd,1e-6))
                value = relative(g)
                push!(rows,(; family=String(family),point,method,value,tol,pass=value<=tol))
            end
        end
    end
    return DF.DataFrame(rows)
end

function family_diagnostics(chain, family, fold)
    ch = MCMCChains.get_sections(chain,:parameters)
    ss = DF.DataFrame(MCMCChains.summarystats(ch))
    ss.family .= String(family)
    ss.fold .= fold
    ss.gate_pass = (ss.rhat .<= 1.05) .& (ss.ess_bulk .>= 200) .& (ss.ess_tail .>= 200)
    names = String.(MCMCChains.names(chain))
    "numerical_error" in names || error("NUTS divergence diagnostic unavailable")
    divergences = sum(Array(chain[:,["numerical_error"],:]))
    divergence_rate = divergences/(size(chain,1)*size(chain,3))
    return ss,(; family=String(family),fold,divergences,divergence_rate,
                gate_pass=all(ss.gate_pass) && divergence_rate<=0.001)
end

function fit_family(data,family,mask; seed=3909,warmup=1000,samples=1000,chains=4)
    model = static_family(Val(family),data,Float64.(mask))
    algorithm = Turing.NUTS(warmup,0.8; adtype=Turing.AutoReverseDiff(compile=true))
    # Explicit RNG, stable chain seeds, native Turing MCMCThreads execution.
    started = time()
    chain = Turing.sample(Random.Xoshiro(seed),model,algorithm,Turing.MCMCThreads(),samples,chains;
                          progress=false)
    return chain,time()-started
end

"Pointwise λ-scale densities; extraction uses the exact model projection and parameter names."
function family_loglik(chain,data,family)
    arr = Array(MCMCChains.get_sections(chain,:parameters).value)
    arr = permutedims(arr,(2,1,3))
    draws = reshape(arr,size(arr,1),:)
    names = String.(MCMCChains.names(MCMCChains.get_sections(chain,:parameters)))
    index = Dict(name=>i for (i,name) in enumerate(names))
    value(name,d) = draws[index[name],d]
    ll = zeros(length(data.y),size(draws,2))
    locations = similar(ll)
    noises = NamedTuple[]
    for d in axes(ll,2)
        za = [value("z_alpha[$i]",d) for i in 1:data.n]
        zb = [value("z_beta[$i]",d) for i in 1:data.n]
        alpha = value("sigma_alpha",d).*(data.C*za)
        beta = value("sigma_beta",d).*(data.C*zb)
        eta = value("mu",d) .+ value("gamma",d).*data.home .+ alpha[data.att] .+ beta[data.def]
        noise = family == :gamma ? (; nu=value("noise.nu",d)) : family == :lognormal ?
            (; sigma=value("noise.sigma",d)) : (; sigma=value("noise.sigma",d),nu=value("noise.nu",d))
        ll[:,d] .= family_ll(Val(family),data.y,data.logy,eta,noise)
        locations[:,d] .= eta
        push!(noises,noise)
    end
    return (; ll,locations,noises)
end

function waic_table(output,family)
    ll = output.ll
    lppd = sum(logsumexp(vec(ll[i,:]))-log(size(ll,2)) for i in axes(ll,1))
    effective = sum(ST.var(ll; dims=2))
    return (; family=String(family),n_obs=size(ll,1),lppd,p_waic=effective,
              elpd_waic=lppd-effective,waic=-2(lppd-effective))
end

function family_comparison(p,out; seed=3909)
    data = family_data(p)
    folds = fixture_folds(p)
    obs_folds = repeat(folds; inner=2)
    CSV.write(joinpath(out,"family_folds.csv"),DF.DataFrame(match_id=p.matches.match_id,fold=folds))
    waic = NamedTuple[]
    cv = NamedTuple[]
    diagnostics = DF.DataFrame[]
    nuts = NamedTuple[]
    qq = NamedTuple[]
    for (fi,family) in enumerate((:gamma,:lognormal,:logt))
        for fold in 0:10
            mask = fold == 0 ? trues(length(data.y)) : obs_folds .!= fold
            chain,seconds = fit_family(data,family,mask; seed=seed+100fi+fold)
            Serialization.serialize(joinpath(out,"family_$(family)_$(fold).jls"),chain)
            diag,nut = family_diagnostics(chain,family,fold)
            push!(diagnostics,diag)
            push!(nuts,(; nut...,seconds,sha=strip(read(`git rev-parse HEAD`,String))))
            CSV.write(joinpath(out,"family_diagnostics.csv"),vcat(diagnostics...))
            CSV.write(joinpath(out,"nuts_diagnostics.csv"),DF.DataFrame(nuts))
            nut.gate_pass || error("$family fold $fold fails NUTS convergence/divergence gate")
            output = family_loglik(chain,data,family)
            if fold == 0
                push!(waic,waic_table(output,family))
                eta = vec(ST.median(output.locations; dims=2))
                if family == :gamma
                    shape = ST.median(n.nu for n in output.noises)
                    z = (data.y.-exp.(eta))./(exp.(eta)./sqrt(shape))
                    theoretical = (DS.quantile.(DS.Gamma(shape,1/shape),(collect(eachindex(z)).-0.5)./length(z)).-1).*sqrt(shape)
                else
                    scale = ST.median(n.sigma for n in output.noises)
                    z = (data.logy.-eta)./scale
                    dist = family == :lognormal ? DS.Normal() : DS.TDist(ST.median(n.nu for n in output.noises))
                    theoretical = DS.quantile.(dist,(collect(eachindex(z)).-0.5)./length(z))
                end
                for (i,resid) in enumerate(sort(z))
                    push!(qq,(; family=String(family),rank=i,theoretical=theoretical[i],standardised_residual=resid))
                end
            else
                for m in findall(==(fold),folds)
                    lh = logsumexp(output.ll[2m-1,:])-log(size(output.ll,2))
                    la = logsumexp(output.ll[2m,:])-log(size(output.ll,2))
                    joint = logsumexp(output.ll[2m-1,:].+output.ll[2m,:])-log(size(output.ll,2))
                    push!(cv,(; family=String(family),fold,match_id=p.matches.match_id[m],
                              logpd_home=lh,logpd_away=la,logpd_sum=lh+la,joint_logpd=joint))
                end
            end
        end
    end
    CSV.write(joinpath(out,"waic.csv"),DF.DataFrame(waic))
    CV = DF.sort!(DF.DataFrame(cv),[:family,:match_id])
    CSV.write(joinpath(out,"family_cv_fixture.csv"),CV)
    summary = DF.combine(DF.groupby(CV,:family),:logpd_sum=>sum=>:sum_logpd,:logpd_sum=>(x->ST.mean(x)/2)=>:mean_logpd_per_observation)
    CSV.write(joinpath(out,"family_cv.csv"),summary)
    paired = NamedTuple[]
    for (a,b) in (("gamma","lognormal"),("gamma","logt"),("lognormal","logt"))
        ga = DF.filter(:family=>==(a),CV)
        gb = DF.filter(:family=>==(b),CV)
        ga.match_id == gb.match_id || error("CV fixture pairing lost")
        delta = ga.logpd_sum.-gb.logpd_sum
        push!(paired,(; family_a=a,family_b=b,n_fixtures=length(delta),difference_sum=sum(delta),
                      se_sum=sqrt(length(delta))*ST.std(delta),difference_mean_per_observation=ST.mean(delta)/2,
                      se_mean_per_observation=ST.std(delta)/(2sqrt(length(delta)))))
    end
    CSV.write(joinpath(out,"family_cv_paired.csv"),DF.DataFrame(paired))
    CSV.write(joinpath(out,"family_qq.csv"),DF.DataFrame(qq))
    return summary
end
