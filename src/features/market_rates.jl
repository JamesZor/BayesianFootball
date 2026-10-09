# Stream-independent, injected match-level market log rates. Inversion belongs to the caller.
import SHA

"""
    MarketRatesFeature(table)

Inject a table with `match_id`, `log_lambda_h`, `log_lambda_a`, and `full_book` columns.
Only rows flagged as a full book are observations. The caller owns book gating/inversion
and snapshot provenance; this feature performs no SQL, inversion, or prototype imports.
Duplicate IDs and nonfinite covered rates are errors. Missing/uncovered rows become finite
zero dummies with an exact binary mask. Storage uses sorted plain vectors for structural
persistence parity. Extraction aligns by match ID, never table position.
"""
struct MarketRatesFeature <: AbstractFeatureConfig
    match_ids::Vector{Int}
    log_lambda_h::Vector{Float64}
    log_lambda_a::Vector{Float64}
    full_book::Vector{Bool}
    function MarketRatesFeature(table::AbstractDataFrame)
        required = (:match_id, :log_lambda_h, :log_lambda_a, :full_book)
        all(k -> k in propertynames(table), required) ||
            error("MarketRatesFeature requires columns $required")
        ids = collect(Int, table.match_id)
        order = sortperm(ids)
        ids = ids[order]
        n = length(ids)
        h, a = zeros(n), zeros(n)
        full_book = Vector{Bool}(undef, n)
        for (i, source) in enumerate(order)
            id = ids[i]
            i > 1 && id == ids[i - 1] && error("duplicate market-rate match ID $id")
            covered = table.full_book[source]
            covered isa Bool || error("full_book must be Bool for match $id")
            full_book[i] = covered
            h[i], a[i] = covered ? (Float64(table.log_lambda_h[source]), Float64(table.log_lambda_a[source])) : (0.0, 0.0)
            isfinite(h[i]) && isfinite(a[i]) || error("nonfinite covered market rate for match $id")
        end
        new(ids, h, a, full_book)
    end
end

"Canonical content identity, recomputed from the plain arrays; no cached mutable digest."
function market_rates_digest(config::MarketRatesFeature)
    io = IOBuffer()
    print(io, "MarketRatesFeature:sorted_arrays_v1\n")
    for i in eachindex(config.match_ids)
        print(io, config.match_ids[i], '\t', repr(config.log_lambda_h[i]), '\t',
            repr(config.log_lambda_a[i]), '\t', Int(config.full_book[i]), '\n')
    end
    return bytes2hex(SHA.sha256(take!(io)))
end

function add_feature!(F_data::Dict, config::MarketRatesFeature, ordered_ids,
                      team_map::Dict, ds::Data.DataStore)
    n = length(ordered_ids)
    h, a, mask = zeros(n), zeros(n), zeros(n)
    for (i, id) in enumerate(ordered_ids)
        index = searchsortedfirst(config.match_ids, Int(id))
        if index <= length(config.match_ids) && config.match_ids[index] == id
            h[i], a[i], mask[i] = config.log_lambda_h[index], config.log_lambda_a[index], Float64(config.full_book[index])
        end
    end
    F_data[:flat_market_log_h] = h
    F_data[:flat_market_log_a] = a
    F_data[:flat_market_available] = mask
    return nothing
end
