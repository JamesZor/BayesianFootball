# Frozen before/after default-off scope signatures; no fitting or DB writes.
using BayesianFootball, CSV, DataFrames, Serialization, SHA
const F6OUT = @__DIR__
ds = deserialize(joinpath(Base.pkgdir(BayesianFootball), ".cache", "datastore_ScottishPyramid.jls"))
rows = NamedTuple[]
for (panel, seasons) in (("W2", ["23/24","24/25","25/26"]), ("wave2", ["24/25","25/26"]))
    for (name, train, cups, monitor) in (
        ("lower", [56,57], :none, Int[]),
        ("spfl", [54,55,56,57], :none, [54,55]),
        ("spfl_cups", [54,55,56,57], :senior_spfl_ties, [54,55]),
        ("spfl_cups_no_monitor", [54,55,56,57], :senior_spfl_ties, Int[]))
        for clock in (:match_week, :match_biweek, :match_month)
            scope = Data.DataScope(; name, train_tournaments=train, cups,
                monitor_tournaments=monitor, target_seasons=seasons, dynamics_col=clock)
            scoped = Data.apply_scope(ds, scope)
            splitter = Data.ScopedWalkForwardCV(scope)
            for (boundary, meta) in Data.create_id_boundaries(scoped, splitter)
                held = Data.get_next_matches(scoped, meta, splitter)
                signature = repr((boundary.fold_id, boundary.target_step,
                    boundary.history_match_ids, boundary.target_match_ids, Int.(held.match_id),
                    meta.target_season, meta.time_step))
                push!(rows, (; panel, scope=name, clock=string(clock), fold=boundary.fold_id,
                    heldout=nrow(held), sha=bytes2hex(sha256(signature))))
            end
        end
    end
end
current = DataFrame(rows)
mode = get(ENV, "F6_SPLIT_MODE", "check")
path = joinpath(F6OUT, "legacy_split_baseline.csv")
if mode == "record"
    isfile(path) && error("baseline already exists")
    CSV.write(path, current)
    println("F6_SPLIT_BASELINE rows=$(nrow(current))")
else
    before = CSV.read(path, DataFrame)
    isequal(before, current) || error("default-off splits changed")
    CSV.write(joinpath(F6OUT, "legacy_split_identity.csv"), DataFrame([(
        ; pass=true, policies=4, panels=2, clocks=3, compared_folds=nrow(current))]))
    println("F6_SPLIT_IDENTITY_PASS rows=$(nrow(current))")
end
