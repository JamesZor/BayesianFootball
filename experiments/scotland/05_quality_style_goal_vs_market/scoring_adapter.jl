# Scores external posterior latents (the market arm) with the UNCHANGED harness scorecard.
# `Harness.score_fits` consumes a `Training.Fit`; the adapter swaps only the `latents` field
# of a template goal-run fit, so every metric, panel check and bootstrap is the harness's own.
module QSScoringAdapter

using BayesianFootball, DataFrames, UUIDs

const H = BayesianFootball.Harness
const T = BayesianFootball.Training
const M = BayesianFootball.Models

"""
A harness-scorable Fit carrying `latents`; config/folds/metadata come from a goal run.
Rows are put in the template's fixture order (the harness bootstrap's group order follows
row order), and the fixture sets must be identical.
"""
function wrap(template, latents::M.CountLatents)
    order_ids = M.latent_match_ids(getfield(template, :latents))
    Set(order_ids) == Set(latents.match_ids) || error(
        "external latents cover $(length(latents.match_ids)) fixtures, template $(length(order_ids))")
    row = Dict(id => i for (i, id) in enumerate(latents.match_ids))
    order = [row[Int(id)] for id in order_ids]
    aligned = M.CountLatents(latents.match_ids[order], latents.λ_home[order, :],
                             latents.λ_away[order, :])
    return T.Fit(getfield(template, :config), getfield(template, :folds), aligned,
                 getfield(template, :diagnostics), getfield(template, :metadata),
                 getfield(template, :save_path))
end

"Row-permuted copy: proves scoring keys on match_id, not on container order."
function permuted(latents::M.CountLatents, seed::Integer)
    order = sortperm([hash((id, seed)) for id in latents.match_ids])
    return M.CountLatents(latents.match_ids[order], latents.λ_home[order, :],
                          latents.λ_away[order, :])
end

"Score labelled latents against a control fit via `Harness.score_fits`."
function score_latents(pairs::AbstractVector{<:Pair}, template; ds, control_label,
                       bootstrap_B::Int = 10_000, expected_fixtures = 710)
    fits = Pair[]
    for (ref, latents) in pairs
        push!(fits, ref => (latents isa T.Fit ? latents : wrap(template, latents)))
    end
    tiers = H.club_season_tiers(ds)
    return H.score_fits(fits; ds, tiers, control = control_label,
                        target_seasons = ["24/25", "25/26"], expected_fixtures, bootstrap_B)
end

"Max |Δ| over every numeric score cell, matched on (subset, market, metric)."
function max_difference(a::AbstractDataFrame, b::AbstractDataFrame)
    keys = [:subset, :market, :metric]
    nrow(a) == nrow(b) || error("score row counts differ: $(nrow(a)) vs $(nrow(b))")
    joined = innerjoin(select(a, keys..., :value, :lo, :hi, :n_obs, :n_fixtures),
                       select(b, keys..., :value => :value_b, :lo => :lo_b, :hi => :hi_b,
                              :n_obs => :n_obs_b, :n_fixtures => :n_fixtures_b); on = keys)
    nrow(joined) == nrow(a) || error("score keys differ between paths")
    all(joined.n_obs .== joined.n_obs_b) && all(joined.n_fixtures .== joined.n_fixtures_b) ||
        error("observation or fixture counts differ between paths")
    gap(x, y) = (isnan(x) && isnan(y)) ? 0.0 : abs(x - y)
    return maximum(max(gap(r.value, r.value_b), gap(r.lo, r.lo_b), gap(r.hi, r.hi_b))
                   for r in eachrow(joined))
end

end # module
println("QS_SCORING_ADAPTER_LOADED")
