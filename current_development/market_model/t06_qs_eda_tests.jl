# Pure QS contracts; run by include in a fresh owned beast tmux Julia REPL.
using BayesianFootball, Test
if !isdefined(@__MODULE__, :QualityStyleEDA)
    include(joinpath(@__DIR__, "l06_qs_eda.jl"))
end
const Q06 = QualityStyleEDA
if !isdefined(@__MODULE__, :QualityStyleReport)
    include(joinpath(@__DIR__, "l06_qs_report.jl"))
end
const Q06R = QualityStyleReport

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
@testset "Predeclared verdict thresholds and protocol sensitivity" begin
    parameters = NamedTuple[]
    paired = NamedTuple[]
    for league in Q06R.LEAGUES
        for protocol in ("10a","10b")
            alignment_bad = league == "eng_championship" && protocol == "10a" ||
                league == "eng_premier" && protocol == "10b"
            style_bad = league == "fin_veikkausliiga" && protocol == "10a" ||
                league == "sco_league_two" && protocol == "10b"
            style_border = league == "sco_league_one" && protocol == "10a"
            push!(parameters,(; league,rung="R6",protocol,parameter="corr_delta_q_s",
                q05=alignment_bad ? 0.31 : -0.2,median=alignment_bad ? 0.4 : 0.1,
                q95=alignment_bad ? 0.5 : 0.2))
            push!(parameters,(; league,rung="C0",protocol,parameter="r",
                q05=style_bad ? 0.65 : style_border ? 0.55 : 0.2,
                median=style_bad ? 0.7 : style_border ? 0.6 : 0.3,
                q95=style_bad ? 0.8 : style_border ? 0.7 : 0.4))
        end
        push!(paired,(; league,candidate="C0",control="R6",n=200,
            mean_delta=league == "sco_championship" ? -0.1 : 0.0,se=0.01))
    end
    result = Q06R.verdicts(Q06.DF.DataFrame(parameters),Q06.DF.DataFrame(paired))
    lookup(name) = only(eachrow(Q06.DF.filter(:league=>==(name),result)))
    @test lookup("eng_championship").verdict == "unsuitable"
    @test lookup("fin_veikkausliiga").verdict == "unsuitable"
    @test lookup("sco_championship").verdict == "inconclusive"
    @test lookup("sco_league_one").verdict == "inconclusive"
    @test lookup("eng_premier").verdict == "suitable"
    @test lookup("eng_premier").flipped_criteria == "alignment"
    @test lookup("sco_league_two").flipped_criteria == "quality dominance"
end
println("T06_DONE")
