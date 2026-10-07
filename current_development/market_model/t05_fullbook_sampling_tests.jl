# Included within the accepted t05 testset. Deterministic generator/target checks;
# no posterior sampling on the laptop. Recovery evidence is produced on beast.
@testset "Full-book sampler contract and synthetic population" begin
    p = PC05.MID.toy_panel(Xoshiro(3960))
    p.matches.season = fill("toy",PC05.MID.n_fixtures(p))
    p.matches.week = p.obs_week[1:2:end]
    q = PC05.MM.selection_probabilities(log.([1.5,1.0]))
    template = [(selections=[:home,:draw,:away],p=[q[:home],q[:draw],q[:away]]),
                (selections=[:over_25,:under_25],p=[q[:over_25],q[:under_25]]),
                (selections=[:btts_yes,:btts_no],p=[q[:btts_yes],q[:btts_no]])]
    markets = [deepcopy(template) for _ in 1:PC05.MID.n_fixtures(p)]
    a = PC05.FullBookRung(:C1)
    theta = log.([0.03,0.01,0.06,1000.0])
    generated = PC05.synthetic_fullbook(a,p,markets,theta; seed=3961)
    again = PC05.synthetic_fullbook(a,p,markets,theta; seed=3961)
    @test generated.panel.obs_y == again.panel.obs_y
    @test [[(m.selections,m.logp) for m in b] for b in generated.markets] ==
        [[(m.selections,m.logp) for m in b] for b in again.markets]
    @test generated.states == again.states
    @test generated.truth ≈ [0.03,0.01,0.06,1000.0]
    @test generated.panel.obs_match == p.obs_match
    @test p.obs_y != generated.panel.obs_y
    @test !hasproperty(p.matches,:lambda_mkt_h)
    @test generated.panel.week_ptr == p.week_ptr
    @test generated.panel.matches.match_id == p.matches.match_id
    @test all(length(book) == 3 for book in generated.markets)
    @test all(abs(sum(m.p)-1) <= 1e-14 && all(isfinite,m.logp)
        for book in generated.markets for m in book)
    tiny = PC05.fullbook_logdirichlet(Xoshiro(3966),[1e-8,1000.0])
    @test all(isfinite,tiny)
    @test exp(tiny[1]) == 0.0 # no probability clipping: log likelihood stays finite
    tinybook = [PC05.FullBookLogMarket([:over_25,:under_25],exp.(tiny),tiny)]
    @test isfinite(PC05.PF.book_logdensity(log.([1.5,1.0]),tinybook,1000.0))
    @test abs(PC05.PF.book_logdensity(log.([1.5,1.0]),tinybook,1000.0)-
        PC05.differentiable_logdensity(log.([1.5,1.0]),tinybook,1000.0)) <= 1e-3
    # Moderate concentrations: Dirichlet mean/variance under exact log-Gamma identity.
    alpha = [0.5,0.75,1.25]
    draws = reduce(hcat,[exp.(PC05.fullbook_logdirichlet(Xoshiro(seed),alpha))
        for seed in 6001:16000])
    @test maximum(abs.(vec(PC05.ST.mean(draws; dims=2))-alpha/sum(alpha))) <= 0.01
    @test maximum(abs.(diag(PC05.ST.cov(draws; dims=2))-
        (alpha.*(sum(alpha).-alpha))/(sum(alpha)^2*(sum(alpha)+1)))) <= 0.002
    H = PC05.fullbook_design(a,p,theta)
    @test maximum(maximum(abs.(generated.panel.obs_y[j:j+1]-
        H[j:j+1,:]*generated.states[:,p.obs_week[j]]-generated.fixture_u[cld(j,2),:]))
        for j in 1:2:PC05.MID.n_obs(p)) <= 1e-15
    @test all(generated.states[1:2,t] == generated.states[1:2,1] for t in 1:p.n_weeks)
    for f in 1:length(markets)
        x = generated.panel.obs_y[2*f-1:2*f]
        prob = PC05.MM.selection_probabilities(x)
        reference = sum(logpdf(Dirichlet(1000*[prob[s] for s in m.selections]),m.p)
            for m in generated.markets[f])
        @test abs(PC05.PF.book_logdensity(x,generated.markets[f],1000)-reference) <= 1e-9
        raw = z -> PC05.PF.book_logdensity(z,generated.markets[f],1000.0)
        _,Had = PC05.ad_derivatives(raw,x)
        _,Hfd = PC05.derivatives(raw,x)
        @test norm(Had-Hfd)/norm(Had) <= 1e-6
    end
    @test PC05.fullbook_derivative_rule(markets[1]) === PC05.derivatives
    @test PC05.fullbook_derivative_rule(generated.markets[1]) === PC05.ad_derivatives
    @test isfinite(PC05.fullbook_filter(a,generated.panel,theta; markets=generated.markets).loglik)
    @test_throws ErrorException PC05.synthetic_fullbook(a,p,
        [[first(template)] for _ in markets],theta; seed=3961)
    @test PC05.fullbook_supported(a,[theta[1:3];13.0]) # no inherited ±12 on log n
    @test isfinite(PC05.MID.log_prior(a,[theta[1:3];13.0]))
    @test !PC05.fullbook_supported(a,[theta[1:3];1000.0])
    @test !PC05.fullbook_supported(a,[-1000.0;theta[2:4]])
    @test PC05.fullbook_logtarget(a,p,[-1000.0;theta[2:4]]; markets) == -Inf
    c0 = PC05.FullBookRung(:C0)
    t0 = log.([0.07,0.03,0.01])
    @test PC05.fullbook_logtarget(c0,p,t0) ==
        PC05.fullbook_filter(c0,p,t0).loglik+PC05.MID.log_prior(c0,t0)
    # Errors inside a supported likelihood are NOT treated as rejected proposals.
    @test_throws ErrorException PC05.fullbook_logtarget(a,p,theta;
        likelihoods=[x -> error("planted numerical failure") for _ in markets])
    h2 = PC05.FullBookRung(:H2)
    U = reshape([t0;-0.4],1,4,1)
    @test PC05.fullbook_physical_draws(h2,U)[1,:,1] ≈ [0.07,0.03,0.01,-0.4]
    # ID-safe restriction on non-contiguous fixture positions.
    pp = generated.panel
    pp.obs_season[5:8] .= "heldout"
    pp.matches.season[3:4] .= "heldout"
    restricted,books = PC05.restrict_fullbook(pp,generated.markets,["toy"])
    ids = restricted.obs_match[1:2:end]
    @test restricted.matches.match_id == ids
    @test books == generated.markets[[findfirst(==(id),pp.obs_match[1:2:end]) for id in ids]]
    @test PC05.MID.n_obs(restricted) == 2length(books)
    @test all(==("toy"),restricted.obs_season)
    # State/deviation population covariance needs no market draws. Broad-prior
    # Dirichlet replicates can underflow p to zero; logp storage preserves them.
    # This independent state-only ensemble imposes no interior-p requirement.
    steps_q,steps_s = Float64[],Float64[]
    noise = Float64[]
    for seed in 4001:4200
        g = PC05.synthetic_fullbook_latents(a,p,theta,Xoshiro(seed))
        N = PC05.MID.n_teams(p)
        da = diff(g.states[3:2+N,:]; dims=2)
        db = diff(g.states[3+N:2+2N,:]; dims=2)
        append!(steps_q,vec((da-db)/2))
        append!(steps_s,vec((da+db)/2))
        append!(noise,vec(g.fixture_u))
    end
    @test abs(PC05.ST.std(steps_q)/0.03-1) <= 0.05
    @test abs(PC05.ST.std(steps_s)/0.01-1) <= 0.05
    @test abs(PC05.ST.cor(steps_q,steps_s)) <= 0.05
    @test abs(PC05.ST.std(noise)/0.06-1) <= 0.05
end
