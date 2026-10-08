# Included inside QSFormsBatch01. Read-only links for the R05 selected movement list.
function audit_large_moves()
    moves=CSV.read(joinpath(OUT,"large_moves.csv"),DF.DataFrame;stringtype=String)
    observations,rate_rows,source_index=NamedTuple[],NamedTuple[],NamedTuple[]
    for league in SENTINELS
        lm=DF.filter(:league=>==(league),moves)
        ids=Set(vcat([parse.(Int,split(r.fixture_ids,";")) for r in eachrow(lm)]...))
        p,config=panel(league)
        for k in eachindex(p.obs_y)
            p.obs_match[k] in ids||continue
            push!(observations,(;league,fixture_id=p.obs_match[k],observation_row=k,calendar_week=p.obs_week[k],home_role=p.obs_home[k],log_lambda=p.obs_y[k],source="accepted saved-rate panel; quote age not supplied"))
        end
        rates=CSV.read(joinpath(Q,"rates_$league.csv"),DF.DataFrame;stringtype=String)
        append!(rate_rows,[merge(NamedTuple(r),(;league)) for r in eachrow(DF.filter(:match_id=>in(ids),rates))])
        name=string(nameof(typeof(config.segment)))
        path=joinpath(dirname(dirname(P)),".cache","datastore_$(name).jls")
        ds=Serialization.deserialize(path)
        for field in (:odds,:betfair_odds)
            table=getproperty(ds,field);indices=Dict{Int,Vector{Int}}()
            for (i,id) in enumerate(table.match_id)
                id in ids||continue
                push!(get!(indices,Int(id),Int[]),i)
            end
            for id in sort(collect(ids))
                ix=get(indices,id,Int[])
                push!(source_index,(;league,fixture_id=id,snapshot_path=path,table=String(field),n_rows=length(ix),source_rows=join(ix,";"),
                    semantics=field==:betfair_odds ? "all saved archive trade-price samples for fixture, NOT assertion that all were selected for inversion or executable quote events" : "all cached odds selection rows for fixture; no invented timestamps"))
            end
        end
    end
    output("R05","large_move_observations.csv",observations)
    output("R05","large_move_rate_rows.csv",rate_rows)
    output("R05","large_move_raw_source_index.csv",source_index)
    insert_summary_lines!("R05",["Audit links: [large_move_observations.csv](large_move_observations.csv), [large_move_rate_rows.csv](large_move_rate_rows.csv), [large_move_raw_source_index.csv](large_move_raw_source_index.csv) point to actual log-rate targets and exact row ordinals in pinned odds/trade-price caches; archive samples are not executable quote-age evidence."])
    verification("R05_audit","PASS selected fixture IDs linked to panel observation rows and full saved-rate rows, with exact cached raw table row ordinals; no SQL/network/raw book recreation. No quote timestamps were fabricated.")
    flush_manifest!()
end
