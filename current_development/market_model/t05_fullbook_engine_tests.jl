# Included in t05 acceptance testset; deterministic, no sampling.
@testset "Full-book prior and parameter layout" begin
    for name in (:C0,:C1,:H1,:H2)
        a = PC05.FullBookRung(name)
        theta = PC05.MID.init_centre(a)
        @test length(theta) == length(PC05.MID.param_names(a))
        @test isfinite(PC05.MID.log_prior(a,theta))
    end
    a = PC05.FullBookRung(:C1)
    theta = log.([0.03,0.01,0.06,1000.0])
    expected = sum(logpdf(truncated(Normal(0,s),0,Inf),exp(x))+x
        for (s,x) in zip([0.10,0.10,0.20],theta[1:3]))
    expected += logpdf(LogNormal(log(1000.0),1.5),exp(theta[4]))+theta[4]
    @test abs(PC05.MID.log_prior(a,theta)-expected) <= 1e-12
    @test isfinite(PC05.MID.log_prior(a,vcat(theta[1:3],13.0))) # no ±12 truncation of n
end

@testset "Conditional moments, information form and scalar eigen projections" begin
    rng = Random.Xoshiro(3952)
    m = randn(rng,5)
    A = randn(rng,5,5)
    P = A*A'
    P[end,:] .= 0
    P[:,end] .= 0 # prove singular latent P is allowed
    H = randn(rng,2,5)
    U = 0.05^2*Matrix(I,2,2)
    z = randn(rng,2)
    R = [0.07 0.01; 0.01 0.03]
    a,S = H*m,H*P*H'+U
    V = inv(inv(S)+inv(R))
    b = V*(S\a+R\z)
    mean,covariance = PC05.conditional_state_update(m,P,H,U,b,V)
    direct = PC05.gaussian_state_update(m,P,H,z,R+U)
    @test norm(mean-direct.mean) <= 1e-10
    @test norm(covariance-direct.covariance) <= 1e-10
    @test norm(covariance-covariance') <= 1e-14
    @test minimum(eigvals(Symmetric(covariance))) >= -1e-12
    eig = eigen(Symmetric(R))
    rows = eig.vectors'*H
    values = eig.vectors'*z
    sm,sp = copy(m),copy(P)
    ll = 0.0
    for j in 1:2
        row = reshape(rows[j,:],1,:)
        scalar = PC05.gaussian_state_update(sm,sp,row,[values[j]],
            reshape([eig.values[j]+U[1,1]],1,1))
        sm,sp = scalar.mean,scalar.covariance
        ll += scalar.marginal
    end
    @test norm(sm-direct.mean) <= 1e-10
    @test norm(sp-direct.covariance) <= 1e-10
    @test abs(ll-direct.marginal) <= 1e-10
end

@testset "C0/H1/H2 and Gaussian C1 independent batch gates" begin
    gates = PC05.fullbook_engine_gates()
    show(stdout,MIME"text/plain"(),gates; allrows=true,allcols=true)
    println()
    for r in eachrow(gates)
        @test r.pass
    end
end

@testset "Full-book population, frozen nonlinear factors and pre-week filtration" begin
    config = PC05.MM.scottish_lower_2425_2526()
    ds = BayesianFootball.Data.load_datastore_cached(config.segment; max_age_hours=10^6)
    p = PC05.CM.TB.phase_b_panel(ds; config).panel
    markets = PC05.fullbook_markets(ds,p,config)
    @test length(markets) == 517
    @test PC05.MID.n_obs(p) == 1034
    # Three real full books, retain chronology/state indexing but keep batch small.
    mask = falses(length(p.obs_y))
    mask[1:6] .= true
    toy = PC05.MID.restrict_panel(p,mask)
    toy_markets = markets[1:3]
    a = PC05.FullBookRung(:C1)
    theta = log.([0.03,0.01,0.06,1000.0])
    f = PC05.fullbook_filter(a,toy,theta; markets=toy_markets,store=true,predict=true)
    batch = PC05.fullbook_batch(a,toy,theta; factors=f.factors)
    means,covs = PC05.fullbook_smoothing(f)
    @test abs(f.loglik-batch.loglik) <= 1e-9
    @test maximum(abs.(vec(means)-batch.post_mean)) <= 1e-8
    @test minimum(minimum(eigvals(Symmetric(covs[:,:,t]))) for t in axes(covs,3)) >= -1e-10
    @test all(norm(covs[:,:,t]-covs[:,:,t]') <= 1e-12 for t in axes(covs,3))
    m0,P0 = PC05.fullbook_initial(a,PC05.MID.n_teams(toy))
    for fixture in 1:3
        if toy.obs_week[2fixture-1] == 1
            H = f.H[2fixture-1:2fixture,:]
            @test norm(f.pair_mean[fixture,:]-H*m0) <= 1e-14
            @test norm(f.pair_cov[fixture,:,:]-(H*P0*H'+f.noise)) <= 1e-14
        end
    end
end
