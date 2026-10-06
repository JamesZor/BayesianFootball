# Exact frozen seed4964 warmup8-coordinate replay; test-only legacy termination.
# The original forward prefix is retained until its FIRST stalled book. This
# diagnostic solver never enters sampling and never promotes a stalled mode.
function c05_legacy_mode(f,start; derivative,iterations=100)
    x = copy(start)
    for iteration in 1:iterations
        g,h = derivative(f,x)
        maximum(abs,g) <= 2e-6 && return (; mode=x,gradient=g,precision=-h,iteration,termination=:legacy)
        e = eigen(Symmetric(-h))
        direction = (e.vectors*Diagonal(max.(e.values,1e-6))*e.vectors')\g
        norm(direction) <= 2e-7 && return (; mode=x,gradient=g,precision=-h,iteration,termination=:legacy)
        direction ./= max(1.0,norm(direction))
        value = f(x)
        scale = 1.0
        while scale >= 2.0^-30
            candidate = x+scale*direction
            next = f(candidate)
            if isfinite(next) && next >= value+1e-4scale*dot(g,direction)
                x = candidate
                break
            end
            scale /= 2
        end
        if scale < 2.0^-30
            return (; mode=x,gradient=g,precision=-h,iteration,
                termination=norm(direction) <= 2e-7 ? :legacy : :legacy_stalled)
        end
    end
    error("legacy replay exhausted iterations")
end

# Tighter independent path: analytic prior, exact AD grid derivatives, no Armijo
# comparisons of nearly equal Float64 densities. Start at accepted mode and take
# damped Newton steps until displacement <=1e-13. This is verification only.
function c05_tighter_mode(adf,a,S,start)
    invS = inv(S)
    x = copy(start)
    step = fill(Inf,length(x))
    for iteration in 1:100
        g,h = PC05.ad_derivatives(adf,x)
        step = Symmetric(invS-h)\(g-invS*(x-a))
        if maximum(abs,step) <= 1e-13
            return x
        end
        x += 0.5step
    end
    error("tighter diagnostic Newton failed: step=$step")
end

@testset "Revision 6 exact legacy-stalled synthetic book" begin
    config = PC05.MM.scottish_lower_2425_2526()
    ds = BayesianFootball.Data.load_datastore_cached(config.segment; max_age_hours=10^6)
    panel = PC05.CM.TB.phase_b_panel(ds; config).panel
    templates = PC05.fullbook_markets(ds,panel,config)
    generated = PC05.synthetic_fullbook(PC05.FullBookRung(:C1),panel,templates,
        log.([0.03,0.01,0.06,1000.0]); seed=3962)
    theta = [-3.549527585137839,-4.460929121755582,-2.821549571263347,7.9567722491577495]
    captured = Ref{Any}(nothing)
    function capture(id,optimum,raw,a,S,adf)
        if optimum.termination == :legacy_stalled
            captured[] = (; id,optimum,raw,a=copy(a),S=copy(S),adf)
            error("C05_LEGACY_STALL_CAPTURED")
        end
    end
    failure = try
        PC05.fullbook_filter(PC05.FullBookRung(:C1),generated.panel,theta;
            markets=generated.markets,mode_audit=capture,mode_solver=c05_legacy_mode)
        nothing
    catch e
        e
    end
    @test failure isa ErrorException
    @test failure !== nothing && failure.msg == "C05_LEGACY_STALL_CAPTURED"
    @test captured[] !== nothing
    if captured[] !== nothing
        c = captured[]
        @test maximum(abs.(c.optimum.gradient-[-1.5699131339808048e-5,3.249019587192592e-5])) <= 1e-10
        accepted = PC05.laplace_update(c.raw,c.a,c.S;
            third_likelihood=c.adf,derivative=PC05.ad_derivatives)
        tighter = c05_tighter_mode(c.adf,c.a,c.S,accepted.mode)
        mode_delta = maximum(abs.(accepted.mode-tighter))
        prior = MvNormal(c.a,Symmetric(c.S))
        _,h = PC05.ad_derivatives(c.raw,tighter)
        marginal = c.raw(tighter)+logpdf(prior,tighter)+log(2pi)-logdet(Symmetric(inv(c.S)-h))/2
        marginal_delta = abs(accepted.marginal-marginal)
        tighter_g,_ = PC05.ad_derivatives(c.adf,tighter)
        tighter_residual = maximum(abs,tighter_g-c.S\(tighter-c.a))
        target = x -> c.raw(x)+logpdf(prior,x)
        optimum = PC05.joint_mode(target,c.a; derivative=PC05.ad_derivatives)
        @test mode_delta <= 1e-8
        @test marginal_delta <= 1e-9
        @test optimum.decrement <= 1e-9
        @test optimum.termination == :polished
        @test optimum.polish_steps <= 3
        @test tighter_residual <= 1e-10
        println("C05_EXACT_STALL fixture=$(c.id) legacy_gradient=$(c.optimum.gradient) mode_delta=$mode_delta marginal_delta=$marginal_delta termination=$(optimum.termination) decrement=$(optimum.decrement) tighter_residual=$tighter_residual")
        out = get(ENV,"C05_NEWTON_TEST_OUT",joinpath(@__DIR__,"results","C","v6_newton"))
        mkpath(out)
        CSV.write(joinpath(out,"newton_regression.csv"),DataFrame(match_id=[c.id],seed=[4964],
            generation_seed=[3962],mode_delta=[mode_delta],marginal_delta=[marginal_delta],
            decrement=[optimum.decrement],termination=[String(optimum.termination)],
            polish_steps=[optimum.polish_steps],
            legacy_gradient_1=[c.optimum.gradient[1]],legacy_gradient_2=[c.optimum.gradient[2]],
            tighter_residual=[tighter_residual],accepted_1=[accepted.mode[1]],accepted_2=[accepted.mode[2]],
            tighter_1=[tighter[1]],tighter_2=[tighter[2]],
            prediction_1=[c.a[1]],prediction_2=[c.a[2]],S11=[c.S[1,1]],S12=[c.S[1,2]],S22=[c.S[2,2]],
            n=[exp(theta[4])],density_delta=[target(accepted.mode)-target(tighter)],
            logdet_delta=[logdet(Symmetric(optimum.precision))-logdet(Symmetric(inv(c.S)-h))],
            primal_algebra_delta=[c.raw(tighter)-c.adf(tighter)],
            mode_gate_pass=[mode_delta <= 1e-8],marginal_gate_pass=[marginal_delta <= 1e-9]))
        f = findfirst(==(c.id),panel.obs_match[1:2:end])
        rows = [(; market=k,selection=String(s),logp=lp) for (k,m) in enumerate(generated.markets[f])
            for (s,lp) in zip(m.selections,m.logp)]
        CSV.write(joinpath(out,"newton_regression_book.csv"),DataFrame(rows))
    end
end
