# Numerical rank audit; no statistical fitting beyond the Phase A indicator matrices.
using DataFrames, CSV, LinearAlgebra, Test
root=normpath(joinpath(@__DIR__,"../.."))
out=joinpath(@__DIR__,"results/A")
stores=MarketAPMPhaseA.pinned_datastores(root)
joined_all=CSV.read(joinpath(out,"joined_matches.csv"),DataFrame)
audit=NamedTuple[]
@testset "Rank tolerance and dense cross-check" begin
    for (league,tournament,segment) in MarketAPMPhaseA.LEAGUES
        ds=stores[segment]
        joined=joined_all[joined_all.league.==league,:]
        seasons=Set(joined.season)
        fixtures=ds.matches[(ds.matches.tournament_id.==tournament).&[!ismissing(s)&&s in seasons for s in ds.matches.season],:]
        starters=ds.lineups[(ds.lineups.tournament_id.==tournament).& .!ds.lineups.is_substitute,:]
        hist=MarketAPMPhaseA.histories(fixtures,starters)
        hd=Dict((r.match_id,r.side)=>r for r in eachrow(hist))
        T,P,_=MarketAPMPhaseA.matrices(joined,hd)
        r6=MarketAPMPhaseA.sparse_rank(hcat(T,P);tol=1e-6)-MarketAPMPhaseA.sparse_rank(T;tol=1e-6)
        r10=MarketAPMPhaseA.sparse_rank(hcat(T,P);tol=1e-10)-MarketAPMPhaseA.sparse_rank(T;tol=1e-10)
        @test r6==r10
        small=first(joined,min(60,nrow(joined)))
        Ts,Ps,_=MarketAPMPhaseA.matrices(small,hd)
        sparse=MarketAPMPhaseA.sparse_rank(hcat(Ts,Ps))-MarketAPMPhaseA.sparse_rank(Ts)
        dense=rank(Matrix(hcat(Ts,Ps));atol=1e-8)-rank(Matrix(Ts);atol=1e-8)
        @test sparse==dense
        push!(audit,(league=league,absorbed_rank_tol_1e6=r6,absorbed_rank_tol_1e10=r10,
            dense_check_matches=nrow(small),subsample_sparse_absorbed_rank=sparse,subsample_dense_absorbed_rank=dense))
        println(league," rank audit ok")
    end
end
CSV.write(joinpath(out,"rank_audit.csv"),DataFrame(audit))
