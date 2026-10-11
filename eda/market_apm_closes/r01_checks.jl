# Small synthetic and output-contract checks, run in the owned beast REPL.
using Test, Dates, DataFrames, CSV, LinearAlgebra
@testset "Phase A causal histories" begin
    f=DataFrame(match_id=1:7,season=fill("test",7),match_date=Date(2025,1,1).+Day.(0:6),
        home_team=fill("A",7),away_team=fill("B",7))
    s=DataFrame(match_id=Int[],team_side=String[],player_id=Int[])
    for i in 1:7, side in ("home","away")
        players=side=="home" ? (i<=5 ? collect(1:11) : vcat(1:10,12)) : collect(21:31)
        for p in players
            push!(s,(i,side,p))
        end
    end
    h=MarketAPMPhaseA.histories(f,s)
    home=h[h.team.=="A",:]
    @test home.ordinal==collect(1:7)
    @test home.absent[6]==1
    @test home.absent[7]==1
    @test home.prev_match[6]==5
    @test home.next_match[6]==7
    s2=copy(s)
    s2.player_id[s2.match_id.==7].+=100
    h2=MarketAPMPhaseA.histories(f,s2)
    @test h2.absent[h2.match_id.<7]==h.absent[h.match_id.<7]
    shared=copy(s)
    shared.player_id[(shared.match_id.==3).&(shared.player_id.==21)].=1
    bad=MarketAPMPhaseA.histories(f,shared)
    @test all(.!bad.valid[bad.match_id.==3])
    @test all(isnan,bad.absent[bad.match_id.==3])
    @test all(.!bad.history_complete[bad.match_id.==4])
    # Unchanging XIs are completely absorbed by team-season effects.
    j=DataFrame(match_id=1:5,season=fill("test",5),home=fill("A",5),away=fill("B",5))
    hd=Dict((r.match_id,r.side)=>r for r in eachrow(h))
    T,P,_=MarketAPMPhaseA.matrices(j,hd)
    @test MarketAPMPhaseA.sparse_rank(hcat(T,P))==MarketAPMPhaseA.sparse_rank(T)
    @test MarketAPMPhaseA.home_probability(1.5,1.0)>MarketAPMPhaseA.home_probability(1.0,1.5)
end
@testset "Phase A committed-output contract" begin
    out=joinpath(@__DIR__,"results/A")
    coverage=CSV.read(joinpath(out,"coverage.csv"),DataFrame)
    design=CSV.read(joinpath(out,"design.csv"),DataFrame)
    headroom=CSV.read(joinpath(out,"headroom.csv"),DataFrame)
    slopes=CSV.read(joinpath(out,"absent_slopes.csv"),DataFrame)
    joined=CSV.read(joinpath(out,"joined_matches.csv"),DataFrame)
    @test nrow(design)==8
    @test nrow(headroom)==8
    @test nrow(slopes)==8*2*5*2
    @test nrow(unique(joined,[:league,:match_id]))==nrow(joined)
    for d in eachrow(design)
        c=coverage[coverage.league.==d.league,:]
        j=joined[joined.league.==d.league,:]
        @test sum(c.joined_n)==d.joined_n==nrow(j)
        @test 0<=d.absorbed_starter_rank<=2d.joined_n-d.team_fe_rank
        @test d.players_ge20<=d.players_ge5<=d.players_ge1
        @test all(isfinite,j.residual_supremacy)
        @test all(isfinite,j.residual_total)
        @test all(j.absent_home.>=0) && all(j.absent_home.<=11)
        @test all(j.absent_away.>=0) && all(j.absent_away.<=11)
        @test all(.!j.eligible .| ((j.home_ordinal.>5).&(j.away_ordinal.>5)))
        @test isfile(joinpath(out,"figures","$(d.league).svg"))
    end
    @test isfile(joinpath(out,"figures","pooled.svg"))
    @test all(isfinite,slopes.slope)
    @test all(isfinite,slopes.se)
    @test all(slopes.ci_low.<=slopes.slope.<=slopes.ci_high)
    bins=CSV.read(joinpath(out,"figure_bins.csv"),DataFrame)
    audit=CSV.read(joinpath(out,"rank_audit.csv"),DataFrame)
    @test sum(bins.n[bins.league.=="pooled"])==count(joined.eligible)
    for d in eachrow(design)
        a=audit[audit.league.==d.league,:]
        @test only(a.absorbed_rank_tol_1e6)==d.absorbed_starter_rank==only(a.absorbed_rank_tol_1e10)
        s=slopes[(slopes.league.==d.league).&(slopes.response.=="supremacy").&(slopes.test.=="current").&(slopes.se_type.=="match_HC1"),:]
        @test sum(bins.n[bins.league.==d.league])==only(s.n)
    end
end
