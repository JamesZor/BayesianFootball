using Test, BayesianFootball, DataFrames, Random, Dates, Distributions
include("l01_market_model.jl")
const M = MarketModel
const F = BayesianFootball.Features

function exact_book(lh,la; subset=false, id=1)
    q=M.selection_probabilities(log.([lh,la]))
    rows=NamedTuple[]
    for (name,line,keys) in (("1X2",0.0,(:home,:draw,:away)),
                             ("BTTS",0.0,(:btts_yes,:btts_no)),
                             ("OverUnder",0.5,(:over_05,:under_05)),
                             ("OverUnder",1.5,(:over_15,:under_15)),
                             ("OverUnder",2.5,(:over_25,:under_25)),
                             ("OverUnder",3.5,(:over_35,:under_35)))
        subset && !(name=="1X2" || line==2.5) && continue
        for key in keys
            push!(rows,(;match_id=id,market_name=name,market_line=line,selection=key,
                         odds_close=1/q[key],prob_fair_close=q[key]))
        end
    end
    return DataFrame(rows)
end

@testset "Phase A market inversion" begin
    rng=MersenneTwister(39)
    @testset "grid parity" begin
        for _ in 1:50
            theta=log.(0.3 .+ 3.2 .* rand(rng,2))
            ph=pdf.(Poisson(exp(theta[1])),0:10)
            pa=pdf.(Poisson(exp(theta[2])),0:10)
            manual=ph*pa'
            manual ./= sum(manual)
            @test maximum(abs.(F.build_probability_matrix(F.DoublePoissonMarketFeature(),theta,10) .- manual)) <= 1e-14
        end
    end
    @testset "synthetic rates and KL identity" begin
        for lh in (0.3,0.8,1.5,2.6,3.5), la in (0.3,0.8,1.5,2.6,3.5)
            for subset in (false,true)
                b=exact_book(lh,la;subset)
                fit=M.solve_kl(b)
                @test fit.converged
                @test maximum(abs.(fit.theta .- log.([lh,la]))) <= 1e-6
                @test abs(M.kl_loss(log.([lh,la]),b)) <= 1e-14
                @test M.kl_loss(log.([lh+0.1,la+0.1]),b) > 0
            end
        end
    end
    @testset "completeness before normalisation" begin
        b=exact_book(1.5,1.0;subset=true)
        complete,ref=M.gate_book(b)
        @test nrow(ref)==0
        @test isapprox(sum(complete.prob_fair_close[complete.market_name.=="1X2"]),1;atol=1e-14)
        partial=b[b.selection .!= :draw,:]
        gated,ref=M.gate_book(partial)
        @test nrow(ref)==1 && ref.reason[1]=="incomplete market"
        @test !any(gated.market_name.=="1X2")
    end
    @testset "gate precedence" begin
        ds=(matches=DataFrame(match_id=[1,2,3,14035501],season=fill("24/25",4),
                tournament_id=fill(56,4), match_date=fill(Date(2024,8,3),4),
                home_team=fill("h",4),away_team=fill("a",4)),)
        book=exact_book(1.5,1.0;id=2,subset=true)
        # id 3: totals-only with >= 3 quoted selections
        totals=filter(r -> r.market_name=="OverUnder",exact_book(1.5,1.0;id=3))
        book=vcat(book,totals,exact_book(1.5,1.0;id=14035501))
        rates=M.invert_panel(ds,book)
        @test rates.reason == ["no Betfair book","","no complete 1X2 market","T014 swapped book"]
    end
    @testset "held-out comparison uses only the same fixtures" begin
        held=DataFrame(match_id=[1,2],scope=fill("heldout",2),tournament=fill(56,2),
                       n_markets=fill(3,2),line=fill("BTTS",2),selection=fill("btts_yes",2),
                       residual=[0.2,0.4])
        insample=DataFrame(match_id=[1,2,3],scope=fill("insample",3),tournament=fill(56,3),
                           n_markets=fill(3,3),line=fill("BTTS",3),selection=fill("btts_yes",3),
                           residual=[0.1,0.3,0.9])
        overall=copy(held)
        overall.tournament .= 0
        overall.n_markets .= 0
        summary=M.residual_summary(overall;reps=20)
        result=M.heldout_comparison(held,insample,summary;reps=20)
        @test nrow(result)==1
        @test result.n[1]==result.insample_n[1]==2
        @test isapprox(result.insample_mean[1],0.2;atol=1e-14)
        @test result.insample_ci_low[1] <= result.insample_mean[1] <= result.insample_ci_high[1]
        @test_throws ErrorException M.heldout_comparison(held,insample[2:3,:],summary;reps=20)
    end
    @testset "real panel bridge and T015" begin
        ds=BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower();max_age_hours=10^6)
        original,_,frame=M.MID.build_market_panel(ds)
        bridged=M.MID.build_market_panel(ds,frame)
        for col in (:obs_y,:obs_att,:obs_def,:obs_week,:week_ptr)
            @test getproperty(bridged,col)==getproperty(original,col)
        end
        b,_=M.gated_close(ds)
        r=M.invert_panel(ds,b)
        @test nrow(r)==710
        @test r.reason[findfirst(==(14035501),r.match_id)]=="T014 swapped book"
        @test all(r.reason[.!r.has_1x2 .& (r.n_selections .> 0)] .== "no complete 1X2 market")
        @test maximum(r.start_spread[isfinite.(r.start_spread)])<=1e-6
    end
end
println("T01_DONE")
