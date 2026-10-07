# Pure QS contracts; run by include in a fresh owned laptop tmux Julia REPL.
using BayesianFootball, Test
if !isdefined(@__MODULE__, :QualityStyleEDA)
    include(joinpath(@__DIR__, "l06_qs_eda.jl"))
end
const Q06 = QualityStyleEDA

@testset "QS local segments and league configuration" begin
    @test BayesianFootball.Data.tournament_ids(Q06.MarketModelEnglish()) == [1,2,3,84]
    @test length(Q06.LEAGUES) == 11
    inventory = Q06.DF.DataFrame(season=["22/23","23/24","24/25","25/26"])
    for (name,tournament,segment) in Q06.LEAGUES
        config = Q06.league_config(name,tournament,segment,inventory)
        @test config.tournaments == [tournament]
        expected = name == "sco_championship" ? inventory.season[1:3] : inventory.season
        @test config.seasons == expected
        @test config.honest_train == expected[1:end-1]
        @test config.honest_test == [last(expected)]
        @test config.price_window == (-20.0,0.0)
        @test config.min_selections_ladder == 5
    end
end

@testset "Per-draw rotation against closed-form covariance" begin
    # alpha=q+s and beta=s-q with independent q/s, here q=.04, s=.01.
    q, s = 0.04, 0.01
    side = sqrt(q^2+s^2)
    rho = (s^2-q^2)/(s^2+q^2)
    result = Q06.rotated_step(side,side,rho)
    @test result.sigma_q ≈ q atol=1e-15
    @test result.sigma_s ≈ s atol=1e-15
    @test result.r ≈ 0.25 atol=1e-15
    @test result.corr_qs ≈ 0 atol=1e-15
    unequal = Q06.rotated_step(0.06,0.03,-0.5)
    @test unequal.corr_qs ≈ (0.06^2-0.03^2) /
        sqrt((0.06^2+0.03^2)^2-4*0.5^2*0.06^2*0.03^2)
end

@testset "Goal variants reduce to the full predictor at zero contribution" begin
    mu,gamma,qh,qa,sh,sa = 0.3,0.15,0.12,-0.04,0.08,-0.02
    full = Q06.goal_rates(mu,gamma,qh,qa,sh,sa)
    @test all(isapprox.(full, (exp(mu+gamma+qh+sh+sa-qa), exp(mu+qa+sa+sh-qh)); atol=1e-14))
    @test Q06.goal_rates(mu,gamma,qh,qa,0.0,0.0) ==
        Q06.goal_rates(mu,gamma,qh,qa,0.0,0.0; zero_style=true)
    @test Q06.goal_rates(mu,gamma,0.0,0.0,sh,sa) ==
        Q06.goal_rates(mu,gamma,0.0,0.0,sh,sa; zero_quality=true)
    @test isfinite(Q06.goal_logscore(2,1,full))
end
println("T06_DONE")
