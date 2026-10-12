# Real frozen weekly panel, strict day filtration and state-clock audit. No sampling.
using BayesianFootball, CSV, DataFrames, Serialization, Dates
const F6Candidates = Module(:F6Candidates)
Base.include(F6Candidates, joinpath(@__DIR__, "..", "candidates.jl"))
struct F6ClockProbe <: BayesianFootball.AbstractFootballModel end
Features.required_features(::F6ClockProbe) = Features.AbstractFeatureConfig[Features.TimeIndicesFeature()]
ds = deserialize(joinpath(Base.pkgdir(BayesianFootball), ".cache", "datastore_ScottishPyramid.jls"))
control = only(filter(c->c.name=="qs_control", F6Candidates.CANDIDATES))
weekly = only(filter(c->c.name=="qs_weekly", F6Candidates.CANDIDATES))
Harness._structural_equal(control.model, weekly.model) || error("weekly model differs")
Harness._structural_equal(control.sampler, weekly.sampler) || error("weekly sampler differs")
for field in fieldnames(Data.DataScope)
    field == :fold_col && continue
    getfield(control.scope, field) == getfield(weekly.scope, field) || error("scope changed: $field")
end
panel = Data.apply_scope(ds, weekly.scope)
splitter = Data.ScopedWalkForwardCV(weekly.scope)
bounds = Data.create_id_boundaries(panel, splitter)
features = Features.create_features(bounds, panel, F6ClockProbe(), splitter)
by_id = Dict(Int(row.match_id)=>row.match_date for row in eachrow(panel.matches))
rows = NamedTuple[]
all_ids = Int[]
for ((boundary,meta), (fs,_)) in zip(bounds, features)
    held = Data.get_next_matches(panel, meta, splitter)
    isempty(held) && error("empty weekly fold")
    append!(all_ids, Int.(held.match_id))
    ids = vcat(boundary.history_match_ids, boundary.target_match_ids)
    first_day = minimum(Date.(held.match_date))
    train_last_day = maximum(Date(by_id[Int(id)]) for id in ids)
    train_last_day < first_day || error("weekly day filtration failed fold $(boundary.fold_id)")
    same = count(>(0), values(fs.data[:qs_oos_state_indices]))
    raw = Data._scope_step_map(panel.matches, weekly.scope, meta.target_season)
    effective = fs.data[:effective_target_steps]
    all(effective[Int(id)] == raw[Int(id)] for id in boundary.target_match_ids) || error("state clock changed")
    push!(rows, (; fold=boundary.fold_id, season=meta.target_season, fold_step=meta.time_step,
        heldout=nrow(held), first_day, train_last_day, filtration_pass=true,
        n_history=fs.data[:n_history_steps], n_target=fs.data[:n_target_steps],
        same_state_fixtures=same, next_state_fixtures=nrow(held)-same))
end
cp = Data.apply_scope(ds, control.scope)
cs = Data.ScopedWalkForwardCV(control.scope)
cb = Data.create_id_boundaries(cp,cs)
control_ids = vcat([Int.(Data.get_next_matches(cp,pair,cs).match_id) for pair in cb]...)
length(cb)==60 && length(control_ids)==1070 || error("control panel changed")
length(all_ids)==length(unique(all_ids))==1070 && Set(all_ids)==Set(control_ids) || error("weekly fixture panel differs")
CSV.write(joinpath(@__DIR__,"weekly_folds.csv"),DataFrame(rows))
CSV.write(joinpath(@__DIR__,"weekly_panel.csv"),DataFrame([(; folds=length(bounds), fixtures=length(all_ids),
    control_folds=length(cb), same_state_folds=count(r->r.same_state_fixtures>0,rows),
    same_state_fixtures=sum(r.same_state_fixtures for r in rows), filtration_pass=true,
    model_identical=true, sampler_identical=true, dynamics_col=string(weekly.scope.dynamics_col),
    fold_col=string(weekly.scope.fold_col))]))
println("F6_PANEL_PASS folds=$(length(bounds)) fixtures=$(length(all_ids)) same_state_folds=$(count(r->r.same_state_fixtures>0,rows))")
