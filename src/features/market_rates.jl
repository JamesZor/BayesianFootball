# Stream-independent, injected match-level market log rates. Inversion belongs to the caller.

"""
    MarketRatesFeature(table)

Inject a table with `match_id`, `log_lambda_h`, `log_lambda_a`, and `full_book` columns.
Only rows flagged as a full book are observations. The caller owns book gating/inversion
and snapshot provenance; this feature performs no SQL, inversion, or prototype imports.
Duplicate IDs and nonfinite covered rates are errors. Missing/uncovered rows become finite
zero dummies with an exact binary mask. Extraction aligns by match ID, never table position.
"""
struct MarketRatesFeature <: AbstractFeatureConfig
    rates::Dict{Int,Tuple{Float64,Float64,Bool}}
    function MarketRatesFeature(table::AbstractDataFrame)
        required = (:match_id, :log_lambda_h, :log_lambda_a, :full_book)
        all(k -> k in propertynames(table), required) ||
            error("MarketRatesFeature requires columns $required")
        rates = Dict{Int,Tuple{Float64,Float64,Bool}}()
        for row in eachrow(table)
            id = Int(row.match_id)
            haskey(rates, id) && error("duplicate market-rate match ID $id")
            row.full_book isa Bool || error("full_book must be Bool for match $id")
            covered = row.full_book
            h, a = covered ? (Float64(row.log_lambda_h), Float64(row.log_lambda_a)) : (0.0, 0.0)
            isfinite(h) && isfinite(a) || error("nonfinite covered market rate for match $id")
            rates[id] = (h, a, covered)
        end
        new(rates)
    end
end

function add_feature!(F_data::Dict, config::MarketRatesFeature, ordered_ids,
                      team_map::Dict, ds::Data.DataStore)
    n = length(ordered_ids)
    h, a, mask = zeros(n), zeros(n), zeros(n)
    for (i, id) in enumerate(ordered_ids)
        lh, la, covered = get(config.rates, Int(id), (0.0, 0.0, false))
        h[i], a[i], mask[i] = lh, la, Float64(covered)
    end
    F_data[:flat_market_log_h] = h
    F_data[:flat_market_log_a] = a
    F_data[:flat_market_available] = mask
    return nothing
end
