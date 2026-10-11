# SplitBoundary.target_match_ids are FITTED current-season rows, not held-out rows.
# Boundary.fold_id resets by season; persisted fold_idx is global. Match on metadata.
function s1manifests(fit, label, dbfolds, lat)
    boundaries=Data.create_id_boundaries(s1ds,fit.config.splitter)
    dates=Dict(Int(x.match_id)=>x.match_date for x in eachrow(s1ds.matches))
    rows=NamedTuple[]; history=NamedTuple[]; checks=NamedTuple[]
    for fold in fit.folds
        matches=filter(x->x[2]==fold.meta,boundaries)
        if isempty(matches)
            # Metadata structs need fieldwise comparison after deserialization.
            matches=filter(x->all(getfield(x[2],k)==getfield(fold.meta,k) for k in fieldnames(typeof(fold.meta))),boundaries)
        end
        length(matches)==1 || error("Cannot uniquely reconstruct fold metadata: $label fold $(fold.fold)")
        b,meta=only(matches)
        train_ids=vcat(b.history_match_ids,b.target_match_ids)
        cutoff=maximum(dates[mid] for mid in train_ids)
        heldout=Int.(filter(r->r.fold_idx==fold.fold,lat).match_id)
        for mid in heldout
            push!(rows,(;model=label,fold=fold.fold,match_id=mid,cutoff=string(cutoff),target_season=meta.target_season,time_step=meta.time_step,history_count=length(train_ids),valid=dates[mid]>cutoff))
        end
        for mid in train_ids
            push!(history,(;model=label,fold=fold.fold,match_id=mid,target_season=meta.target_season))
        end
        dbrow=only(eachrow(filter(r->r.fold_idx==fold.fold,dbfolds)))
        push!(checks,(;model=label,fold=fold.fold,n_heldout=length(heldout),n_training=length(train_ids),cutoff=string(cutoff),db_first_date=string(dbrow.first_match_date),db_last_date=string(dbrow.last_match_date),valid=all(dates[mid]>cutoff for mid in heldout),target_season=meta.target_season,time_step=meta.time_step))
    end
    CSV.write(joinpath(S1OUT,"folds_"*label*".csv"),DataFrame(rows))
    CSV.write(joinpath(S1OUT,"history_"*label*".csv"),DataFrame(history))
    CSV.write(joinpath(S1OUT,"q12_cutoff_check_"*label*".csv"),DataFrame(checks))
    all(r.valid for r in rows) || error("Prediction cutoff chronology invalid: $label")
end
