# Exact saved market-feature mask, not an assumption that every training fixture has a quote.
s1audit("likelihood_ids") do conn
    fit=Training.load_fit(Training.PostgresStorage("scottish_lower_qs_wave2_2426"),UUID("81fdf817-f82d-4f66-9bfd-5f456c4fe29f"))
    f=fit.config.model.observation.feature
    dates=Dict(Int(r.match_id)=>Date(r.match_date) for r in eachrow(s1ds.matches))
    rows=[(;match_id=mid,full_book=f.full_book[i],date=string(get(dates,mid,Date(1900,1,1)))) for (i,mid) in enumerate(f.match_ids) if get(dates,mid,Date(2100,1,1))<Date(2026,7,1)]
    CSV.write(joinpath(S1OUT,"q12_qs_market_likelihood.csv"),DataFrame(rows))
    println("SAVED_MARKET_FEATURE rows=",length(rows)," covered=",count(r->r.full_book,rows))
end
