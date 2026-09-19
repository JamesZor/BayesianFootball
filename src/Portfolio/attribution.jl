# src/Portfolio/attribution.jl
#
# Bet-level confidence, capture-ratio, selectivity, and shared-bet sizing attribution.
# Kept separate from metrics.jl: these statistics consume ledgers, while path metrics consume
# the compounded bankroll trajectory.

const _BetKey = Tuple{Int, String, Symbol}
const _EDGE_COLUMNS = (:stake, :pnl, :payoff, :odds, :p_model, :p_market)
const _BET_KEY_COLUMNS = (:match_id, :family, :selection)

_empty_edge_summary() = EdgeSummary(
    n_bets = 0, n_wins = 0, win_rate = NaN, cap_weighted_win_rate = NaN,
    stake_sum = 0.0, pnl_sum = 0.0, roi = NaN, edge_mean = NaN,
    edge_win = NaN, edge_loss = NaN, capture_ratio = NaN, stake_mean = NaN,
    odds_mean = NaN, p_model_mean = NaN, p_market_mean = NaN)

_float_or_nan(values) = Float64[ismissing(x) ? NaN : Float64(x) for x in values]
_finite_sum(values) = sum(x for x in values if isfinite(x); init = 0.0)

function _require_bet_columns(bets::AbstractDataFrame, columns, caller::AbstractString)
    missing_columns = Symbol[c for c in columns if !hasproperty(bets, c)]
    isempty(missing_columns) || error(
        "$caller needs ledger columns $(join(string.(columns), ", ")); missing " *
        join(string.(missing_columns), ", "))
    return nothing
end

"""
    edge_summary(bets::AbstractDataFrame) -> EdgeSummary

Summarise confidence and realised returns over one bet ledger. Edge is
`p_model - p_market`; the three reported edge means are in probability-percentage points.
A win has `payoff > 0` and a loss has `payoff < 0`; pushes remain in win-rate denominators but
are excluded from both conditional edge means. This intentionally improves on the research
prototype's `.!won` shortcut, which counted pushes as losses, so push-bearing ledgers are not
numerically comparable to prototype Capture Ratios without recomputation. The production API
also returns `NaN` when mean losing-bet edge is non-positive, where the prototype could report an
uninterpretable negative ratio.

`capture_ratio = E[edge | win] / E[edge | loss]` is `NaN` for an empty ledger, when either
outcome class is absent, or when mean losing-bet edge is non-positive. ROI is
`100 * sum(pnl) / sum(stake)` and is `NaN` when total stake is non-positive. Union-merged legacy
ledgers may contain `missing` in columns added by a newer schema; aggregate stake/P&L remain
available, while every statistic whose inputs are incomplete is conservatively `NaN`.
"""
function edge_summary(bets::AbstractDataFrame)
    n = nrow(bets)
    n == 0 && return _empty_edge_summary()
    _require_bet_columns(bets, _EDGE_COLUMNS, "edge_summary")

    stake = _float_or_nan(bets.stake)
    pnl = _float_or_nan(bets.pnl)
    payoff = _float_or_nan(bets.payoff)
    odds = _float_or_nan(bets.odds)
    p_model = _float_or_nan(bets.p_model)
    p_market = _float_or_nan(bets.p_market)

    stake_complete = all(isfinite, stake)
    pnl_complete = all(isfinite, pnl)
    payoff_complete = all(isfinite, payoff)
    odds_complete = all(isfinite, odds)
    probabilities_complete = all(isfinite, p_model) && all(isfinite, p_market)

    won = payoff .> 0.0
    lost = payoff .< 0.0
    edge = p_model .- p_market
    stake_sum = _finite_sum(stake)
    pnl_sum = _finite_sum(pnl)
    conditional_edges_complete = payoff_complete && probabilities_complete
    edge_win_raw = conditional_edges_complete && any(won) ? mean(edge[won]) : NaN
    edge_loss_raw = conditional_edges_complete && any(lost) ? mean(edge[lost]) : NaN
    ratio = conditional_edges_complete && any(won) && any(lost) &&
            isfinite(edge_loss_raw) && edge_loss_raw > 0.0 ?
            edge_win_raw / edge_loss_raw : NaN

    return EdgeSummary(
        n_bets = n,
        # Known wins remain countable in a union-merged ledger, but an incomplete outcome
        # column makes the denominator unknowable and therefore makes win_rate undefined.
        n_wins = count(won),
        win_rate = payoff_complete ? count(won) / n : NaN,
        cap_weighted_win_rate = stake_complete && payoff_complete && stake_sum > 0.0 ?
                                sum(stake[won]) / stake_sum : NaN,
        stake_sum = stake_sum,
        pnl_sum = pnl_sum,
        roi = stake_complete && pnl_complete && stake_sum > 0.0 ?
              100.0 * pnl_sum / stake_sum : NaN,
        edge_mean = probabilities_complete ? 100.0 * mean(edge) : NaN,
        edge_win = 100.0 * edge_win_raw,
        edge_loss = 100.0 * edge_loss_raw,
        capture_ratio = ratio,
        stake_mean = stake_complete ? mean(stake) : NaN,
        odds_mean = odds_complete ? mean(odds) : NaN,
        p_model_mean = probabilities_complete ? mean(p_model) : NaN,
        p_market_mean = probabilities_complete ? mean(p_market) : NaN,
    )
end

capture_ratio(bets::AbstractDataFrame) = edge_summary(bets).capture_ratio
edge_summary(t::Trajectory) = edge_summary(t.bets)
edge_summary(r::PortfolioResult) = edge_summary(r.trajectory)
capture_ratio(t::Trajectory) = capture_ratio(t.bets)
capture_ratio(r::PortfolioResult) = capture_ratio(r.trajectory)

_has_edge_schema(bets::AbstractDataFrame) = all(c -> hasproperty(bets, c), _EDGE_COLUMNS)
_reporting_edge_summary(t::Trajectory) =
    _has_edge_schema(t.bets) ? edge_summary(t.bets) : _empty_edge_summary()
_reporting_edge_summary(r::PortfolioResult) = _reporting_edge_summary(r.trajectory)

"""
    attribution(t::Trajectory) -> DataFrame

Stake, P/L, ROI and hit rate per selection family. This is the legacy family view retained for
compatibility; use [`family_breakdown`](@ref) for edge and capture-ratio statistics.
"""
function attribution(t::Trajectory)
    isempty(t.bets) && return DataFrame()
    grouped = combine(groupby(t.bets, :family),
                      nrow => :n,
                      :stake => sum => :stake,
                      :pnl => sum => :pnl,
                      :odds => median => :med_odds,
                      :payoff => (x -> mean(x .> 0)) => :hit)
    grouped.roi = 100 .* grouped.pnl ./ grouped.stake
    return sort!(grouped, :pnl, rev = true)
end

function _bet_keys(bets::AbstractDataFrame, ledger_name::AbstractString)
    nrow(bets) == 0 && return _BetKey[]
    _require_bet_columns(bets, _BET_KEY_COLUMNS, "partition_bets ledger $ledger_name")
    return _BetKey[(Int(r.match_id), String(r.family), Symbol(r.selection)) for r in eachrow(bets)]
end

function _assert_unique_keys(keys::Vector{_BetKey}, ledger_name::AbstractString)
    allunique(keys) && return nothing
    seen = Set{_BetKey}()
    duplicate = first(k for k in keys if k in seen || (push!(seen, k); false))
    error("ledger $ledger_name has duplicate bet key $duplicate")
end

"""
    partition_bets(a, b) -> (both_a, both_b, only_a, only_b)

Partition two ledgers by `(match_id, family, selection)`. Duplicate keys are refused. Shared
frames are sorted by key and row-aligned; exclusive frames retain deterministic key order. Input
frames are not mutated and helper key columns are not added to the returned frames.
"""
function partition_bets(a::AbstractDataFrame, b::AbstractDataFrame)
    keys_a = _bet_keys(a, "A")
    keys_b = _bet_keys(b, "B")
    _assert_unique_keys(keys_a, "A")
    _assert_unique_keys(keys_b, "B")

    index_a = Dict(k => i for (i, k) in enumerate(keys_a))
    index_b = Dict(k => i for (i, k) in enumerate(keys_b))
    shared_keys = sort!(collect(intersect(Set(keys_a), Set(keys_b))))
    only_a_keys = sort!(collect(setdiff(Set(keys_a), Set(keys_b))))
    only_b_keys = sort!(collect(setdiff(Set(keys_b), Set(keys_a))))

    both_a = DataFrame(a[[index_a[k] for k in shared_keys], :])
    both_b = DataFrame(b[[index_b[k] for k in shared_keys], :])
    only_a = DataFrame(a[[index_a[k] for k in only_a_keys], :])
    only_b = DataFrame(b[[index_b[k] for k in only_b_keys], :])

    aligned_a = _bet_keys(both_a, "A shared")
    aligned_b = _bet_keys(both_b, "B shared")
    aligned_a == aligned_b || error("shared bet ledgers are not row-aligned")
    return both_a, both_b, only_a, only_b
end

"""
    shared_bet_sizing_attribution(both_a, both_b) -> Float64

Return the controlled sizing P&L contrast
`sum((stake_a - stake_b) * payoff)` over row-aligned shared bets. The method refuses different
keys, duplicate keys, or settlement payoffs; those would make the contrast something other than
pure sizing attribution.
"""
function shared_bet_sizing_attribution(both_a::AbstractDataFrame,
                                       both_b::AbstractDataFrame)
    nrow(both_a) == nrow(both_b) || error(
        "shared bet ledgers have different row counts: $(nrow(both_a)) and $(nrow(both_b))")
    nrow(both_a) == 0 && return 0.0
    _require_bet_columns(both_a, (:stake, :payoff), "shared_bet_sizing_attribution ledger A")
    _require_bet_columns(both_b, (:stake, :payoff), "shared_bet_sizing_attribution ledger B")

    keys_a = _bet_keys(both_a, "A shared")
    keys_b = _bet_keys(both_b, "B shared")
    _assert_unique_keys(keys_a, "A shared")
    _assert_unique_keys(keys_b, "B shared")
    keys_a == keys_b || error("shared bet ledgers are not row-aligned")

    payoff_a = Float64.(both_a.payoff)
    payoff_b = Float64.(both_b.payoff)
    all(isapprox.(payoff_a, payoff_b; atol = 1e-12, rtol = 0.0)) || error(
        "shared bets settle differently in the two ledgers")
    if hasproperty(both_a, :odds) && hasproperty(both_b, :odds)
        all(isapprox.(Float64.(both_a.odds), Float64.(both_b.odds);
                      atol = 1e-12, rtol = 0.0)) || error(
            "shared bets have different prices in the two ledgers")
    end
    return sum((Float64.(both_a.stake) .- Float64.(both_b.stake)) .* payoff_a)
end

"""
    compare_portfolios(res_a, res_b; name_a = "Model A", name_b = "Model B")
        -> ModelComparisonAttribution

Partition two result ledgers into shared and exclusive bets and compute the pure sizing P&L
contrast on the shared set. `summary_a` and `summary_b` describe each complete ledger; the shared
ROI fields describe only the controlled common set.
"""
function compare_portfolios(res_a::PortfolioResult, res_b::PortfolioResult;
                            name_a::AbstractString = "Model A",
                            name_b::AbstractString = "Model B")
    bets_a = res_a.trajectory.bets
    bets_b = res_b.trajectory.bets
    both_a, both_b, only_a, only_b = partition_bets(bets_a, bets_b)
    return ModelComparisonAttribution(
        name_a = String(name_a),
        name_b = String(name_b),
        shared_a = both_a,
        shared_b = both_b,
        exclusive_a = only_a,
        exclusive_b = only_b,
        sizing_delta_pnl = shared_bet_sizing_attribution(both_a, both_b),
        shared_roi_a = edge_summary(both_a).roi,
        shared_roi_b = edge_summary(both_b).roi,
        summary_a = edge_summary(bets_a),
        summary_b = edge_summary(bets_b),
    )
end

_edge_summary_namedtuple(s::EdgeSummary) =
    NamedTuple{fieldnames(EdgeSummary)}(getfield(s, f) for f in fieldnames(EdgeSummary))

function _empty_breakdown(label::Symbol)
    row = merge(NamedTuple{(label,)}(("",)), _edge_summary_namedtuple(_empty_edge_summary()))
    return DataFrame([row])[1:0, :]
end

_odds_bucket(odds::Real) = odds < 2.0 ? "< 2.0" : odds < 3.5 ? "2.0 - 3.5" : "≥ 3.5"

function _canonical_market_family(value)
    family = String(value)
    startswith(family, "1X2") && return "1X2"
    (startswith(family, "O/U") || startswith(family, "OverUnder")) && return "OverUnder"
    startswith(family, "BTTS") && return "BTTS"
    return family
end

_market_family(row::DataFrameRow) = _canonical_market_family(
    hasproperty(row, :group) ? row.group : row.family)

"""
    edge_breakdown(bets; by = :odds) -> DataFrame

Apply [`edge_summary`](@ref) by odds bucket (`< 2.0`, `2.0 - 3.5`, `≥ 3.5`) or canonical market
family (`1X2`, `OverUnder`, `BTTS`). Use `by = :family` (or its explicit alias
`:market_family`) for the latter. A non-empty ledger always emits all three canonical buckets so
breakdowns from two models are row-alignable; absent buckets carry an empty `EdgeSummary`. The
returned summary columns use exactly the same units and undefined-value rules as `EdgeSummary`.
"""
function edge_breakdown(bets::AbstractDataFrame; by::Symbol = :odds)
    by in (:odds, :family, :market_family) || error(
        "edge_breakdown by must be :odds, :family, or :market_family; got :$by")
    isempty(bets) && return _empty_breakdown(by == :odds ? :odds_bucket : :market_family)
    _require_bet_columns(bets, _EDGE_COLUMNS, "edge_breakdown")

    if by == :odds
        labels = _odds_bucket.(Float64.(bets.odds))
        order = ("< 2.0", "2.0 - 3.5", "≥ 3.5")
        rows = [merge((odds_bucket = label,),
                      _edge_summary_namedtuple(edge_summary(bets[labels .== label, :])))
                for label in order]
    else
        _require_bet_columns(bets, (:family,), "edge_breakdown")
        labels = [_market_family(r) for r in eachrow(bets)]
        preferred = ("1X2", "OverUnder", "BTTS")
        order = vcat(collect(preferred),
                     sort!(setdiff(unique(labels), collect(preferred))))
        rows = [merge((market_family = label,),
                      _edge_summary_namedtuple(edge_summary(bets[labels .== label, :])))
                for label in order]
    end
    return DataFrame(rows)
end

edge_breakdown(t::Trajectory; by::Symbol = :odds) = edge_breakdown(t.bets; by = by)
edge_breakdown(r::PortfolioResult; by::Symbol = :odds) = edge_breakdown(r.trajectory; by = by)
odds_breakdown(bets::AbstractDataFrame) = edge_breakdown(bets; by = :odds)
family_breakdown(bets::AbstractDataFrame) = edge_breakdown(bets; by = :family)
odds_breakdown(t::Trajectory) = odds_breakdown(t.bets)
odds_breakdown(r::PortfolioResult) = odds_breakdown(r.trajectory)
family_breakdown(t::Trajectory) = family_breakdown(t.bets)
family_breakdown(r::PortfolioResult) = family_breakdown(r.trajectory)
