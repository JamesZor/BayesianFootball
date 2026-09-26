using BayesianFootball
using DataFrames
using Dates

const D = BayesianFootball.Data
const CONFIG = D.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons = ["24/25", "25/26"],
    history_seasons = 2,
    dynamics_col = :match_biweek,
    warmup_period = 0,
)

function kickoff_map(matches::DataFrame)
    return Dict(Int(row.match_id) => DateTime(row.match_date) + Hour(row.match_hour)
                for row in eachrow(matches))
end

function fold_inventory()
    ds = D.load_datastore_cached(D.ScottishAll())
    boundaries = D.create_id_boundaries(ds, CONFIG)
    kickoff = kickoff_map(ds.matches)
    records = NamedTuple[]
    heldout_sets = Set{Int}[]

    for (boundary, meta) in boundaries
        heldout = D.get_next_matches(ds, (boundary, meta), CONFIG)
        heldout_ids = Set(Int.(heldout.match_id))
        isempty(heldout_ids) && continue
        fitted_ids = Set(vcat(boundary.history_match_ids, boundary.target_match_ids))
        overlap = intersect(fitted_ids, heldout_ids)
        cutoff_ok = isempty(fitted_ids) || maximum(kickoff[id] for id in fitted_ids) <
                                         minimum(kickoff[id] for id in heldout_ids)
        push!(records, (
            fold_id = boundary.fold_id,
            season = meta.target_season,
            train_step = boundary.target_step,
            heldout_matches = length(heldout_ids),
            train_heldout_overlap = length(overlap),
            strict_train_cutoff = cutoff_ok,
        ))
        push!(heldout_sets, heldout_ids)
    end

    unique_heldout_ids = Set{Int}()
    repeated_heldout_ids = Set{Int}()
    for ids in heldout_sets, id in ids
        id in unique_heldout_ids && push!(repeated_heldout_ids, id)
        push!(unique_heldout_ids, id)
    end
    report = DataFrame(records)
    per_season = combine(groupby(report, :season),
                         nrow => :predictive_boundaries,
                         :heldout_matches => sum => :heldout_match_observations)

    println("FOLD_INVENTORY julia_version=$(VERSION)")
    println("cache_only=true cache_path=$(joinpath(pkgdir(D), ".cache", "datastore_ScottishAll.jls"))")
    println("all_boundaries=$(length(boundaries)) predictive_boundaries=$(nrow(report)) unique_heldout_ids=$(length(unique_heldout_ids)) repeated_heldout_ids=$(length(repeated_heldout_ids))")
    println("train_heldout_overlap_total=$(sum(report.train_heldout_overlap)) strict_train_cutoff_all=$(all(report.strict_train_cutoff))")
    show(stdout, MIME("text/plain"), per_season); println()
    return (; report, per_season, boundaries, unique_heldout_ids, repeated_heldout_ids)
end

fold_inventory()
