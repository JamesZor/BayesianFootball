# Read-only Phase 2 provenance and fold-coverage preflight, on mcmc-beast only.
using BayesianFootball, Dates, DataFrames, SHA, LinearAlgebra, ThreadPinning
const PFD = BayesianFootball.Data
pinthreads(:cores)
BLAS.set_num_threads(1)
raw = PFD.load_datastore_cached(PFD.ScottishLower(); max_age_hours = 10^6)
cache = joinpath(Base.pkgdir(PFD), ".cache", "datastore_ScottishLower.jls")
sha = bytes2hex(open(SHA.sha256, cache))
sha == "c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4" || error("snapshot mismatch: $sha")
scope = PFD.DataScope(name = "lower", train_tournaments = [56, 57],
    cups = :none, target_tournaments = [56, 57], monitor_tournaments = Int[],
    clock_tournaments = [56, 57], target_seasons = ["24/25", "25/26"],
    history_seasons = 2, dynamics_col = :match_biweek)
ds = PFD.apply_scope(raw, scope)
splitter = PFD.ScopedWalkForwardCV(scope)
folds = PFD.create_id_boundaries(ds, splitter)
length(folds) == 40 || error("expected 40 folds, got $(length(folds))")
match = Dict(Int(r.match_id) => r for r in eachrow(ds.matches))
training_ids = Set{Int}()
heldout_ids = Set{Int}()
for (i, (boundary, meta)) in enumerate(folds)
    train = Set(Int.(vcat(boundary.history_match_ids, boundary.target_match_ids)))
    oos = Set(Int.(PFD.get_next_matches(ds, meta, splitter).match_id))
    isempty(intersect(train, oos)) || error("fold $i train overlaps target")
    union!(training_ids, train)
    union!(heldout_ids, oos)
    first_date = minimum(Date(match[id].match_date) for id in oos)
    first_week = first_date - Day(dayofweek(first_date) - 1)
    same_week = [id for id in train if Date(match[id].match_date) >= first_week]
    history = sort!(unique(String(match[id].season) for id in train))
    println("FOLD $i train=$(length(train)) oos=$(length(oos)) earliest=$(first_date) first_week=$(first_week) same_week_training=$(length(same_week)) seasons=$(join(history, ","))")
end
println("UNION train=$(length(training_ids)) heldout=$(length(heldout_ids)) seasons=$(sort!(unique(String(match[id].season) for id in training_ids))) SHA=$sha")
println("MARKET_PREFLIGHT_DONE")
