# W2-cache closing-book coverage, not a new extraction or forecast.
s1audit("close_coverage") do conn
    odds=Harness._betfair_closing_odds(s1ds)
    fixtures=filter(r->Date(r.match_date)<Date(2026,7,1) && Int(r.tournament_id) in (54,55,56,57),s1ds.matches)
    priced=Set(Int.(filter(r->r.market_name=="1X2",odds).match_id))
    rows=NamedTuple[]
    for g in groupby(fixtures,[:season,:tournament_id])
        ids=Set(Int.(g.match_id))
        push!(rows,(;season=String(g.season[1]),tournament=Int(g.tournament_id[1]),n_fixtures=nrow(g),n_1x2_closes=length(intersect(ids,priced)),source="frozen W2 cache Betfair TWA(-20,0]"))
    end
    CSV.write(joinpath(S1OUT,"q6_w2_close_coverage.csv"),DataFrame(rows))
    println("CLOSE_COVERAGE_ROWS=",length(rows))
end
