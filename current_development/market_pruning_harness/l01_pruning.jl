module MarketPruningHarness

# Reusable machinery for Task 018. This file defines types and functions only; r01_sweep.jl owns
# database access, benchmark selection, execution, and reporting.

import BayesianFootball
import CSV
import DataFrames
import Dates
import Distributions
import Printf
import Statistics

const BF = BayesianFootball
const Data = BF.Data
const Markets = BF.Data.Markets
const Portfolio = BF.Portfolio
const Models = BF.Models

export CatalogSelection, CatalogMarket, MarketCatalog, PruningConfig, TierGrid,
       default_catalog, default_candidates, baseline_weights, candidate_weights,
       conviction_weights, active_market_keys, pruning_book_spec, pruning_policy,
       run_line_screening, run_t012_contrast, run_conviction_sweep,
       smile_coherence_gate, accretive_candidates, write_pruning_outputs,
       write_market_pruning_report, summary_frame

const SelectionKey = Tuple{String,Float64,Symbol}
const VALID_T012_MODES = (:excise_pruned, :retain_zero_trust)

"One directional selection in the catalog. `id` is the stable research-facing address."
struct CatalogSelection
    id::Symbol
    label::String
    market_key::Symbol
    group::String
    line::Float64
    selection::Symbol
end

"One complete quoted market. Excision is market-level because `BookSpec` admits whole markets."
struct CatalogMarket
    key::Symbol
    label::String
    market::Markets.AbstractMarket
    selections::Vector{CatalogSelection}
end

struct MarketCatalog
    markets::Vector{CatalogMarket}
    selection_index::Dict{Symbol,CatalogSelection}
    market_index::Dict{Symbol,CatalogMarket}
end

function MarketCatalog(markets::Vector{CatalogMarket})
    isempty(markets) && error("market catalog must contain at least one market")
    selection_index = Dict{Symbol,CatalogSelection}()
    market_index = Dict{Symbol,CatalogMarket}()
    for market in markets
        haskey(market_index, market.key) && error("duplicate market key $(market.key)")
        market_index[market.key] = market
        isempty(market.selections) && error("market $(market.key) has no selections")
        for selection in market.selections
            selection.market_key == market.key || error(
                "selection $(selection.id) points at $(selection.market_key), not $(market.key)")
            haskey(selection_index, selection.id) && error(
                "duplicate catalog selection id $(selection.id)")
            selection_index[selection.id] = selection
        end
    end
    return MarketCatalog(markets, selection_index, market_index)
end

"A named Phase-1 directional addition."
struct LineCandidate
    id::Symbol
    label::String
    selection_ids::Vector{Symbol}
end

Base.@kwdef struct PruningConfig
    t012_mode::Symbol = :excise_pruned
    candidate_trust::Float64 = 1.0 / 1.4
    initial_bankroll::Float64 = 1_000.0
    return_tolerance_pp::Float64 = 1.0e-9
    bootstrap::Bool = false
    bootstrap_draws::Int = 2_000
    seed::Int = 18
end

Base.@kwdef struct TierGrid
    tier1::Vector{Float64} = collect(0.30:0.05:0.45)
    tier2::Vector{Float64} = collect(0.15:0.05:0.25)
    tier3::Vector{Float64} = collect(0.00:0.05:0.10)
end

function _catalog_market(key::Symbol, label::AbstractString, market,
                         ids::NamedTuple)
    outcomes = Markets.outcomes(market)
    keys(ids) == keys(outcomes) || error(
        "$key catalog keys $(keys(ids)) do not match market outcome keys $(keys(outcomes))")
    group = Markets.market_group(market)
    line = Float64(Markets.market_line(market))
    selections = CatalogSelection[
        CatalogSelection(getproperty(ids, outcome_key),
                         String(getproperty(ids, outcome_key)), key, group, line,
                         getproperty(outcomes, outcome_key))
        for outcome_key in keys(outcomes)
    ]
    return CatalogMarket(key, String(label), market, selections)
end

"Canonical extensible menu for Task 018: 1X2, totals 0.5--4.5, and BTTS."
function default_catalog()
    markets = CatalogMarket[
        _catalog_market(:x1x2, "1X2", Data.Market1X2(),
                        (home = :home, draw = :draw, away = :away)),
        _catalog_market(:ou05, "O/U 0.5", Data.MarketOverUnder(0.5),
                        (over = :over_05, under = :under_05)),
        _catalog_market(:ou15, "O/U 1.5", Data.MarketOverUnder(1.5),
                        (over = :over_15, under = :under_15)),
        _catalog_market(:ou25, "O/U 2.5", Data.MarketOverUnder(2.5),
                        (over = :over_25, under = :under_25)),
        _catalog_market(:ou35, "O/U 3.5", Data.MarketOverUnder(3.5),
                        (over = :over_35, under = :under_35)),
        _catalog_market(:ou45, "O/U 4.5", Data.MarketOverUnder(4.5),
                        (over = :over_45, under = :under_45)),
        _catalog_market(:btts, "BTTS", Data.MarketBTTS(),
                        (yes = :btts_yes, no = :btts_no)),
    ]
    return MarketCatalog(markets)
end

"The nine predeclared one-direction additions. Over 1.5 is already in operational Option B."
default_candidates() = LineCandidate[
    LineCandidate(:under_05, "+U0.5", [:under_05]),
    LineCandidate(:under_15, "+U1.5", [:under_15]),
    LineCandidate(:under_35, "+U3.5", [:under_35]),
    LineCandidate(:under_45, "+U4.5", [:under_45]),
    LineCandidate(:over_25, "+O2.5", [:over_25]),
    LineCandidate(:over_35, "+O3.5", [:over_35]),
    LineCandidate(:over_45, "+O4.5", [:over_45]),
    LineCandidate(:btts_yes, "+BTTS_yes", [:btts_yes]),
    LineCandidate(:btts_no, "+BTTS_no", [:btts_no]),
]

function _zero_weights(catalog::MarketCatalog)
    return Dict{Symbol,Float64}(id => 0.0 for id in keys(catalog.selection_index))
end

"Exact operational Option B trust vector used by the S1 reproduction gate."
function baseline_weights(catalog::MarketCatalog)
    weights = _zero_weights(catalog)
    for (id, weight) in ((:home, 1.0), (:under_25, 1.0),
                         (:draw, 1.0 / 1.4), (:away, 1.0 / 1.4),
                         (:over_15, 1.0 / 1.4))
        haskey(weights, id) || error("default catalog is missing Option B selection $id")
        weights[id] = weight
    end
    return weights
end

function candidate_weights(catalog::MarketCatalog, candidate::LineCandidate;
                           trust::Real = 1.0 / 1.4)
    0.0 <= trust <= 1.0 || error("candidate trust must be in [0,1], got $trust")
    weights = baseline_weights(catalog)
    for id in candidate.selection_ids
        haskey(weights, id) || error("candidate $(candidate.id) names unknown selection $id")
        weights[id] = Float64(trust)
    end
    return weights
end

function _validate_weights(catalog::MarketCatalog, weights::AbstractDict{Symbol,<:Real})
    unknown = setdiff(Set(keys(weights)), Set(keys(catalog.selection_index)))
    isempty(unknown) || error("trust table names unknown selections: $(join(sort!(String.(collect(unknown))), ", "))")
    missing = setdiff(Set(keys(catalog.selection_index)), Set(keys(weights)))
    isempty(missing) || error("trust table omits selections: $(join(sort!(String.(collect(missing))), ", "))")
    for (id, weight) in weights
        0.0 <= weight <= 1.0 || error("trust for $id must be in [0,1], got $weight")
    end
    return nothing
end

function active_market_keys(catalog::MarketCatalog,
                            weights::AbstractDict{Symbol,<:Real})
    _validate_weights(catalog, weights)
    return Symbol[market.key for market in catalog.markets
                  if any(weights[selection.id] > 0.0 for selection in market.selections)]
end

function _market_keys(catalog::MarketCatalog, weights, mode::Symbol)
    mode in VALID_T012_MODES || error(
        "unknown T012 mode $mode; use :excise_pruned or :retain_zero_trust")
    return mode === :retain_zero_trust ? Symbol[m.key for m in catalog.markets] :
           active_market_keys(catalog, weights)
end

"Clone `base` while replacing only its market menu according to the explicit T012 mode."
function pruning_book_spec(base::Portfolio.BookSpec, catalog::MarketCatalog,
                           weights::AbstractDict{Symbol,<:Real};
                           mode::Symbol = :excise_pruned)
    keys = Set(_market_keys(catalog, weights, mode))
    markets = Data.MarketConfig(Markets.AbstractMarket[
        market.market for market in catalog.markets if market.key in keys
    ])
    isempty(markets.markets) && error("pruning produced an empty BookSpec")
    return Portfolio.BookSpec(
        markets = markets,
        price = base.price,
        allocator = base.allocator,
        shrink = base.shrink,
        exec = base.exec,
    )
end

"Clone `base` while replacing only its directional trust table."
function pruning_policy(base::Portfolio.PolicySpec, catalog::MarketCatalog,
                        weights::AbstractDict{Symbol,<:Real})
    _validate_weights(catalog, weights)
    table = Dict{SelectionKey,Float64}()
    for market in catalog.markets, selection in market.selections
        table[(selection.group, selection.line, selection.selection)] =
            Float64(weights[selection.id])
    end
    return Portfolio.PolicySpec(
        trust = Portfolio.TieredTrust(table; default = 0.0),
        risk = base.risk,
        cap = base.cap,
        filter = base.filter,
        grouping = base.grouping,
    )
end

_market_signature(catalog, weights, mode) =
    join(String.(sort!(_market_keys(catalog, weights, mode))), ";")

function _selection_families(catalog::MarketCatalog, ids)
    return Set(Portfolio.selection_family(selection.group, selection.line, selection.selection)
               for id in ids for selection in (catalog.selection_index[id],))
end

function _bet_summary(bets::DataFrames.AbstractDataFrame, families::AbstractSet{<:AbstractString})
    isempty(bets) && return (n_bets = 0, stake = 0.0, pnl = 0.0, roi_pct = NaN)
    keep = [String(family) in families for family in bets.family]
    frame = bets[keep, :]
    stake = sum(frame.stake; init = 0.0)
    pnl = sum(frame.pnl; init = 0.0)
    return (n_bets = DataFrames.nrow(frame), stake, pnl,
            roi_pct = stake > 0.0 ? 100.0 * pnl / stake : NaN)
end

function _result_row(model::AbstractString, policy::AbstractString, mode::Symbol, result)
    summary = result.summary
    return (
        model = String(model),
        policy = String(policy),
        t012_mode = String(mode),
        n_slates = summary.n_slates,
        n_bets = summary.n_bets,
        total_return_pct = summary.total_return_pct,
        roi_pct = summary.roi,
        sharpe_ann = summary.sharpe_ann,
        max_drawdown_pct = summary.mdd,
        total_turnover = summary.total_stake,
        mean_exposure = summary.mean_exposure,
        n_capped = summary.n_capped,
    )
end

function _build_cached!(cache::Dict{String,Any}, base_book, catalog, weights, mode,
                        fit, odds, fixtures)
    signature = _market_signature(catalog, weights, mode)
    return get!(cache, signature) do
        spec = pruning_book_spec(base_book, catalog, weights; mode)
        books, report = if fit isa Models.AbstractPosteriorLatents
            Portfolio.build_books_reported(
                spec, fit, odds, fixtures; converged = true, gated = true, quiet = true)
        else
            Portfolio.build_books_reported(
                spec, fit, odds, fixtures; require_converged = true, quiet = true)
        end
        (spec = spec, books = books, report = report)
    end
end

function _simulate(base_policy, catalog, weights, built, config::PruningConfig)
    policy = pruning_policy(base_policy, catalog, weights)
    return Portfolio.simulate_portfolio(
        policy,
        built.books,
        built.report;
        initial_bankroll = config.initial_bankroll,
        bootstrap = config.bootstrap,
        B = config.bootstrap_draws,
        seed = config.seed,
    )
end

function _classification(delta_return_pp::Real, added_roi_pct::Real,
                         core_stake_vs_p0::Real, n_added::Integer,
                         tolerance::Real)
    n_added == 0 && return "neutral"
    abs(delta_return_pp) <= tolerance && return "neutral"
    (!isfinite(added_roi_pct) || added_roi_pct <= 0.0) && return "toxic"
    delta_return_pp < -tolerance && return "cannibalizing"
    core_stake_vs_p0 < 1.0 - tolerance && return "accretive_with_cannibalization"
    return "accretive"
end

"Run P0 plus every one-direction Phase-1 addition for one model."
function run_line_screening(model::AbstractString, fit, odds, fixtures,
                            base_book::Portfolio.BookSpec,
                            base_policy::Portfolio.PolicySpec;
                            catalog::MarketCatalog = default_catalog(),
                            candidates::Vector{LineCandidate} = default_candidates(),
                            config::PruningConfig = PruningConfig())
    config.t012_mode in VALID_T012_MODES || error("invalid T012 mode $(config.t012_mode)")
    cache = Dict{String,Any}()
    p0_weights = baseline_weights(catalog)
    p0_built = _build_cached!(cache, base_book, catalog, p0_weights, config.t012_mode,
                              fit, odds, fixtures)
    p0 = _simulate(base_policy, catalog, p0_weights, p0_built, config)
    p0_ids = Set(Int.(p0_built.books[i].m_id for i in eachindex(p0_built.books)))
    # Capacity diagnostics use the work package's alpha core (1X2 + Under 2.5). Operational
    # Option B's Over 1.5 remains active for S1 reproduction but is a baseline diversifier, not
    # silently folded into `core_stake_vs_p0`.
    core_ids = Symbol[:home, :draw, :away, :under_25]
    core_families = _selection_families(catalog, core_ids)
    p0_core = _bet_summary(p0.trajectory.bets, core_families)

    rows = NamedTuple[]
    baseline = _result_row(model, "P0 Option B", config.t012_mode, p0)
    push!(rows, (; baseline..., candidate = "P0", candidate_label = "P0 Option B",
                   classification = "baseline", added_n_bets = 0, added_roi_pct = NaN,
                   core_stake_vs_p0 = 1.0, delta_core_roi_pp = 0.0,
                   delta_return_pp = 0.0, delta_sharpe = 0.0,
                   n_books = length(p0_built.books),
                   market_signature = _market_signature(catalog, p0_weights, config.t012_mode)))

    results = Dict{Symbol,Any}(:P0 => p0)
    builds = Dict{Symbol,Any}(:P0 => p0_built)
    for candidate in candidates
        weights = candidate_weights(catalog, candidate; trust = config.candidate_trust)
        built = _build_cached!(cache, base_book, catalog, weights, config.t012_mode,
                               fit, odds, fixtures)
        ids = Set(Int.(book.m_id for book in built.books))
        ids == p0_ids || error(
            "$model/$(candidate.label) changed the fixture panel: $(length(ids)) vs $(length(p0_ids))")
        result = _simulate(base_policy, catalog, weights, built, config)
        added_families = _selection_families(catalog, candidate.selection_ids)
        added = _bet_summary(result.trajectory.bets, added_families)
        core = _bet_summary(result.trajectory.bets, core_families)
        delta_return = result.summary.total_return_pct - p0.summary.total_return_pct
        ratio = p0_core.stake > 0.0 ? core.stake / p0_core.stake : NaN
        delta_core_roi = core.roi_pct - p0_core.roi_pct
        classification = _classification(delta_return, added.roi_pct, ratio,
                                           added.n_bets, config.return_tolerance_pp)
        headline = _result_row(model, candidate.label, config.t012_mode, result)
        push!(rows, (; headline..., candidate = String(candidate.id),
                       candidate_label = candidate.label, classification,
                       added_n_bets = added.n_bets, added_roi_pct = added.roi_pct,
                       core_stake_vs_p0 = ratio, delta_core_roi_pp = delta_core_roi,
                       delta_return_pp = delta_return,
                       delta_sharpe = result.summary.sharpe_ann - p0.summary.sharpe_ann,
                       n_books = length(built.books),
                       market_signature = _market_signature(catalog, weights, config.t012_mode)))
        results[candidate.id] = result
        builds[candidate.id] = built
    end
    return (rows = DataFrames.DataFrame(rows), baseline = p0, results, builds,
            cache, core_families)
end

"Gate S0: quantify the exact effect of retaining every all-zero market in P0's book geometry."
function run_t012_contrast(model::AbstractString, fit, odds, fixtures,
                           base_book::Portfolio.BookSpec,
                           base_policy::Portfolio.PolicySpec;
                           catalog::MarketCatalog = default_catalog(),
                           config::PruningConfig = PruningConfig())
    weights = baseline_weights(catalog)
    cache = Dict{String,Any}()
    excised = _build_cached!(cache, base_book, catalog, weights, :excise_pruned,
                             fit, odds, fixtures)
    retained = _build_cached!(cache, base_book, catalog, weights, :retain_zero_trust,
                              fit, odds, fixtures)
    result_excised = _simulate(base_policy, catalog, weights, excised, config)
    result_retained = _simulate(base_policy, catalog, weights, retained, config)
    same = isequal(result_excised.trajectory.bets, result_retained.trajectory.bets)
    row = (
        model = String(model),
        gate = "S0 T012 quantified",
        pass = true,
        shift_detected = !same,
        ledgers_identical = same,
        excised_markets = _market_signature(catalog, weights, :excise_pruned),
        retained_markets = _market_signature(catalog, weights, :retain_zero_trust),
        excised_n_books = length(excised.books),
        retained_n_books = length(retained.books),
        excised_n_bets = result_excised.summary.n_bets,
        retained_n_bets = result_retained.summary.n_bets,
        excised_return_pct = result_excised.summary.total_return_pct,
        retained_return_pct = result_retained.summary.total_return_pct,
        delta_return_pp = result_retained.summary.total_return_pct -
                          result_excised.summary.total_return_pct,
        excised_roi_pct = result_excised.summary.roi,
        retained_roi_pct = result_retained.summary.roi,
    )
    return (row = DataFrames.DataFrame([row]), result_excised, result_retained,
            books_excised = excised.books, books_retained = retained.books)
end

function accretive_candidates(screening::DataFrames.AbstractDataFrame)
    accepted = Set(["accretive", "accretive_with_cannibalization"])
    return Symbol.(screening.candidate[in.(screening.classification, Ref(accepted))])
end

function conviction_weights(catalog::MarketCatalog, survivors::AbstractVector{Symbol};
                            tau1::Real, tau2::Real, tau3::Real)
    for value in (tau1, tau2, tau3)
        0.0 <= value <= 1.0 || error("tier trust must be in [0,1], got $value")
    end
    weights = _zero_weights(catalog)
    for id in (:home, :under_25)
        weights[id] = Float64(tau1)
    end
    for id in (:draw, :away, :over_15)
        weights[id] = Float64(tau2)
    end
    survivor_set = Set(survivors)
    if :btts_no in survivor_set
        weights[:btts_no] = Float64(tau2)
        delete!(survivor_set, :btts_no)
    end
    for id in survivor_set
        haskey(weights, id) || error("survivor $id is absent from the catalog")
        weights[id] = Float64(tau3)
    end
    return weights
end

function _mark_pareto!(frame::DataFrames.DataFrame)
    frame.pareto = falses(DataFrames.nrow(frame))
    for i in 1:DataFrames.nrow(frame)
        dominated = false
        for j in 1:DataFrames.nrow(frame)
            i == j && continue
            growth_better = frame.total_return_pct[j] >= frame.total_return_pct[i]
            drawdown_better = frame.max_drawdown_pct[j] >= frame.max_drawdown_pct[i]
            strict = frame.total_return_pct[j] > frame.total_return_pct[i] ||
                     frame.max_drawdown_pct[j] > frame.max_drawdown_pct[i]
            if growth_better && drawdown_better && strict
                dominated = true
                break
            end
        end
        frame.pareto[i] = !dominated
    end
    return frame
end

"Phase 2: solve every conviction-tier cell and mark the Growth/Max-DD Pareto frontier."
function run_conviction_sweep(model::AbstractString, fit, odds, fixtures,
                              base_book::Portfolio.BookSpec,
                              base_policy::Portfolio.PolicySpec,
                              survivors::AbstractVector{Symbol};
                              catalog::MarketCatalog = default_catalog(),
                              grid::TierGrid = TierGrid(),
                              config::PruningConfig = PruningConfig())
    cache = Dict{String,Any}()
    rows = NamedTuple[]
    for tau1 in grid.tier1, tau2 in grid.tier2, tau3 in grid.tier3
        weights = conviction_weights(catalog, survivors; tau1, tau2, tau3)
        built = _build_cached!(cache, base_book, catalog, weights, config.t012_mode,
                               fit, odds, fixtures)
        result = _simulate(base_policy, catalog, weights, built, config)
        headline = _result_row(model, "tier_grid", config.t012_mode, result)
        push!(rows, (; headline..., tau1, tau2, tau3,
                       survivors = join(String.(sort!(collect(survivors))), ";"),
                       n_books = length(built.books),
                       market_signature = _market_signature(catalog, weights, config.t012_mode)))
    end
    frame = DataFrames.DataFrame(rows)
    _mark_pareto!(frame)
    return (rows = frame, cache)
end

"Gate S2: the score tensor consumed by Kelly reproduces every smile totals CDF."
function smile_coherence_gate(books, latents::Models.SmileLatents; tolerance::Real = 1.0e-9)
    row_of = Dict(Int(match_id) => i for (i, match_id) in enumerate(latents.match_ids))
    n_draws = size(latents.λ_home, 2)
    worst = 0.0
    worst_fixture = 0
    worst_strike = -1
    n_checked = 0
    for book in books
        row = get(row_of, book.m_id, 0)
        row == 0 && continue
        side = isqrt(length(book.p_grid))
        side * side == length(book.p_grid) || error(
            "book $(book.m_id) score grid is not square")
        grid = reshape(book.p_grid, side, side)
        for strike_index in eachindex(latents.strikes)
            K = strike_index - 1
            implied = 0.0
            for away in 0:(side - 1), home in 0:(side - 1)
                home + away <= K && (implied += grid[home + 1, away + 1])
            end
            reference = Statistics.mean(Distributions.cdf(
                Distributions.Poisson(latents.λ_tot[row, draw] *
                                      latents.φ[row, strike_index, draw]), K)
                for draw in 1:n_draws)
            gap = abs(implied - reference)
            if gap > worst
                worst = gap
                worst_fixture = book.m_id
                worst_strike = K
            end
            n_checked += 1
        end
    end
    return (gate = "S2 score-grid coherence", pass = worst <= tolerance,
            n_books = length(books), n_checks = n_checked, max_abs_gap = worst,
            tolerance = Float64(tolerance), worst_fixture, worst_strike)
end

smile_coherence_gate(books, ::Models.CountLatents; tolerance::Real = 1.0e-9) =
    (gate = "S2 score-grid coherence", pass = true, n_books = length(books), n_checks = 0,
     max_abs_gap = 0.0, tolerance = Float64(tolerance), worst_fixture = 0, worst_strike = -1)

function summary_frame(line_screening::DataFrames.DataFrame,
                       conviction::DataFrames.DataFrame,
                       t012::DataFrames.DataFrame)
    rows = NamedTuple[]
    for model in unique(line_screening.model)
        screen = line_screening[line_screening.model .== model, :]
        tiers = conviction[conviction.model .== model, :]
        p0 = screen[screen.classification .== "baseline", :]
        best = isempty(tiers) ? nothing : tiers[argmax(tiers.total_return_pct), :]
        push!(rows, (
            model = String(model),
            p0_return_pct = p0.total_return_pct[1],
            p0_roi_pct = p0.roi_pct[1],
            p0_sharpe_ann = p0.sharpe_ann[1],
            p0_max_drawdown_pct = p0.max_drawdown_pct[1],
            n_accretive = count(in(Set(["accretive", "accretive_with_cannibalization"])),
                                screen.classification),
            n_toxic = count(==("toxic"), screen.classification),
            n_cannibalizing = count(==("cannibalizing"), screen.classification),
            best_tier_return_pct = best === nothing ? NaN : best.total_return_pct,
            best_tier_sharpe_ann = best === nothing ? NaN : best.sharpe_ann,
            best_tier_max_drawdown_pct = best === nothing ? NaN : best.max_drawdown_pct,
            t012_delta_return_pp = only(t012.delta_return_pp[t012.model .== model]),
        ))
    end
    return DataFrames.DataFrame(rows)
end

function _markdown_table(io, frame::DataFrames.AbstractDataFrame)
    names_ = String.(DataFrames.names(frame))
    println(io, "| ", join(names_, " | "), " |")
    println(io, "|", join(fill("---", length(names_)), "|"), "|")
    for row in DataFrames.eachrow(frame)
        values = [replace(string(row[name]), "|" => "\\|") for name in DataFrames.names(frame)]
        println(io, "| ", join(values, " | "), " |")
    end
end

function write_market_pruning_report(path::AbstractString, summary, screening, tiers, t012, gates)
    open(path, "w") do io
        println(io, "# Market pruning report\n")
        println(io, "Generated `", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"), "`. ",
                "Phase 1 uses the exact operational Option B trust vector and ",
                "`:excise_pruned`; Phase 2 re-solves every tier cell from the posterior books.\n")
        println(io, "## Headline summary\n")
        _markdown_table(io, summary)
        println(io, "\n## Verification gates\n")
        _markdown_table(io, gates)
        println(io, "\n## T012 contrast\n")
        _markdown_table(io, t012)
        println(io, "\n## Cross-paradigm line decision\n")
        decision_rows = NamedTuple[]
        candidates = sort!(unique(String.(screening.candidate_label[
            screening.classification .!= "baseline"])))
        models = sort!(unique(String.(screening.model)))
        accepted = Set(["accretive", "accretive_with_cannibalization"])
        for candidate in candidates
            frame = screening[screening.candidate_label .== candidate, :]
            classes = Dict(String(row.model) => String(row.classification)
                           for row in DataFrames.eachrow(frame))
            deltas = Dict(String(row.model) => Float64(row.delta_return_pp)
                          for row in DataFrames.eachrow(frame))
            robust = all(get(classes, model, "missing") in accepted for model in models)
            push!(decision_rows, (
                candidate,
                robustly_accretive = robust,
                classifications = join([string(model, "=", get(classes, model, "missing"))
                                        for model in models], "; "),
                delta_return_pp = join([string(model, "=",
                                                round(get(deltas, model, NaN); digits = 2))
                                        for model in models], "; "),
            ))
        end
        _markdown_table(io, DataFrames.DataFrame(decision_rows))
        println(io, "\nA line is `robustly_accretive` only if every requested paradigm classifies it ",
                "as accretive. This deliberately prevents a gain isolated to one posterior ",
                "family from becoming the shared production basket.\n")
        println(io, "\n## Phase 1 line screening\n")
        cols = [:model, :candidate_label, :classification, :added_n_bets, :added_roi_pct,
                :core_stake_vs_p0, :delta_core_roi_pp, :delta_return_pp, :total_return_pct,
                :sharpe_ann, :max_drawdown_pct]
        _markdown_table(io, DataFrames.select(screening, cols))
        println(io, "\n## Phase 2 Pareto frontier\n")
        frontier = tiers[tiers.pareto, :]
        frontier = unique(frontier,
                          [:model, :total_return_pct, :max_drawdown_pct, :n_bets])
        tier_cols = [:model, :tau1, :tau2, :tau3, :survivors, :total_return_pct,
                     :sharpe_ann, :max_drawdown_pct, :n_bets]
        _markdown_table(io, DataFrames.select(frontier, tier_cols))
        println(io, "\n## Interpretation guardrails\n")
        println(io, "- T012 excision is deliberately **market-level**. A BookSpec admits a complete ",
                "market, so enabling one direction necessarily admits its zero-trust complement ",
                "to the Kelly geometry. The report does not call that selection-level excision.")
        println(io, "- `accretive_with_cannibalization` means net growth improved while core stake fell; ",
                "it is not equivalent to a free diversification gain.")
        println(io, "- The sweep is descriptive on the same held-out settlement period used to select ",
                "survivors. Promotion needs a later untouched period.")
    end
    return path
end

function write_pruning_outputs(output_dir::AbstractString; summary, line_screening,
                               conviction_tiers, t012_contrast, gates)
    mkpath(output_dir)
    outputs = [
        "sweep_summary.csv" => summary,
        "line_screening.csv" => line_screening,
        "conviction_tiers.csv" => conviction_tiers,
        "t012_contrast.csv" => t012_contrast,
        "gates.csv" => gates,
    ]
    for (filename, frame) in outputs
        CSV.write(joinpath(output_dir, filename), frame)
    end
    report = joinpath(output_dir, "MARKET_PRUNING_REPORT.md")
    write_market_pruning_report(report, summary, line_screening, conviction_tiers,
                                t012_contrast, gates)
    return (; outputs = first.(outputs), report)
end

end # module
