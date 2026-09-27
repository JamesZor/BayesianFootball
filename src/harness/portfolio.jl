# Finalist portfolio evaluation under the production pricing and staking path.
# The closing container deliberately pins the Task 013/r05 Option-B risk value
# (`lambda = 8`) so its historical acceptance target cannot drift when the live
# MatchDay default is re-pointed.

const PORTFOLIO_CONTAINERS = (:close_option_b, :t25_calibrated)
const PORTFOLIO_BOOTSTRAP_B = 4_000
const PORTFOLIO_SEED = 1

"The Task 013 Option-B system whose closing-line results are the Phase-4 parity target."
_portfolio_system() = MatchDay.option_b_system(; lambda = 8.0)

function _portfolio_odds(ds, container::Symbol)
    container === :close_option_b && return _betfair_closing_odds(ds)
    container === :t25_calibrated || throw(ArgumentError(
        "portfolio container must be :close_option_b or :t25_calibrated; got :$container"))
    book, refusals = Calibration.point_in_time_book(ds;
        config = Calibration.PointInTimeBookConfig(as_of_minutes = -25.0))
    nrow(book) > 0 || error("T-25 point-in-time book is empty ($(nrow(refusals)) refusals)")
    return book
end

function _portfolio_calibrator()
    return Calibration.GenerativeRateCalibrator(
        name = "scot_lower_t25_inv",
        law = Calibration.InverseGaussianLaw(w_base = 0.25, sigma = 0.35),
        book_as_of_minutes = -25.0,
    )
end

function _portfolio_source(fit::Training.Fit, panel::Vector{Int}, container::Symbol, odds)
    restricted = _restrict(fit, panel)
    container === :close_option_b && return restricted
    calibrated = Calibration.calibrate_fit(_portfolio_calibrator(), restricted, odds; quiet = true)
    return Calibration.calibrated_fit(calibrated)
end

_portfolio_build_books(spec, source::Training.Fit, odds, fixtures) =
    Portfolio.build_books_reported(
        spec, source, odds, fixtures; require_converged = false, quiet = true)
_portfolio_build_books(spec, source, odds, fixtures) =
    Portfolio.build_books_reported(spec, source, odds, fixtures; quiet = true)

"Return the shared buildable fixture panel and every unioned refusal."
function _portfolio_buildable_panel(spec, sources::AbstractDict, odds, fixtures,
                                    panel::Vector{Int})
    dropped = Dict{Int,String}()
    for (label, source) in sources
        _, report = _portfolio_build_books(spec, source, odds, fixtures)
        for (ids, why) in ((report.skipped_no_fixture, "no fixture row"),
                           (report.skipped_unplayed, "unplayed"),
                           (report.skipped_no_quotes, "no quotes"),
                           (report.skipped_no_selections, "no usable selections"))
            for match_id in ids
                dropped[Int(match_id)] = "$label: $why"
            end
        end
        for (match_id, message) in report.errored
            dropped[Int(match_id)] = "$label: error: $message"
        end
    end
    keep = sort!(collect(setdiff(Set(panel), keys(dropped))))
    frame = DataFrame(match_id = sort!(collect(keys(dropped))))
    frame.reason = [dropped[match_id] for match_id in frame.match_id]
    return keep, frame
end

"Build and simulate one source, refusing any skip from the already-common panel."
function _portfolio_simulate(spec, policy, source, odds, fixtures, panel::Vector{Int};
                             label::AbstractString,
                             B::Int = PORTFOLIO_BOOTSTRAP_B,
                             seed::Int = PORTFOLIO_SEED)
    restricted = source isa Training.Fit ? _restrict(source, panel) : source
    books, report = _portfolio_build_books(spec, restricted, odds, fixtures)
    Portfolio.n_skipped(report) == 0 || error(
        "$label skipped $(Portfolio.n_skipped(report)) common-panel fixtures")
    length(books) == length(panel) || error(
        "$label built $(length(books)) books for $(length(panel)) common-panel fixtures")
    return Portfolio.simulate_portfolio(
        policy, books, report;
        bootstrap = true, B, seed,
        metrics = BackTesting.AbstractWealthMetric[
            BackTesting.CalmarRatio(), BackTesting.SharpeRatio()])
end

"Convert one portfolio result to the stable wide finalist summary schema."
function _portfolio_summary_row(ref::RunRef, result; n_panel::Int)
    summary = result.summary
    interval = result.bootstrap_ci
    edge = Portfolio.edge_summary(result)
    return (;
        run_id = ref.run_id,
        model = ref.label,
        total_return_pct = summary.total_return_pct,
        cagr_pct = 100.0 * summary.cagr,
        growth_per_slate = summary.growth_per_slate,
        growth_lo = interval === nothing ? NaN : interval.growth_lo,
        growth_hi = interval === nothing ? NaN : interval.growth_hi,
        roi_pct = summary.roi,
        p_roi_positive = interval === nothing ? NaN : interval.p_roi_positive,
        sharpe_ann = summary.sharpe_ann,
        calmar = summary.calmar,
        max_drawdown_pct = summary.mdd,
        win_rate_pct = 100.0 * summary.win_rate,
        mean_edge_pp = edge.edge_mean,
        mean_exposure = summary.mean_exposure,
        capture_ratio = edge.capture_ratio,
        n_bets = summary.n_bets,
        n_slates = summary.n_slates,
        n_panel,
    )
end

const _PORTFOLIO_METRICS = (
    :total_return_pct, :cagr_pct, :growth_per_slate, :roi_pct, :p_roi_positive,
    :sharpe_ann, :calmar, :max_drawdown_pct, :win_rate_pct, :mean_edge_pp,
    :mean_exposure, :capture_ratio, :n_bets, :n_slates, :n_panel,
)

"Convert wide finalist summaries to versioned long rows for `harness_scores`."
function _portfolio_score_rows(summary::AbstractDataFrame, container::Symbol,
                               control_run_id)
    subset_name = "portfolio_$(container)"
    reference = container === :close_option_b ?
        "devigged_betfair_twa(-20,0]_option_b_lambda8" :
        "betfair_t-25_generative_inverse_option_b_lambda8"
    rows = NamedTuple[]
    for row in eachrow(summary), metric in _PORTFOLIO_METRICS
        lo = metric === :growth_per_slate ? row.growth_lo : NaN
        hi = metric === :growth_per_slate ? row.growth_hi : NaN
        push!(rows, (;
            run_id = row.run_id,
            model = String(row.model),
            stage = "finalist",
            scorecard_version = SCORECARD_VERSION,
            subset = subset_name,
            market = "book",
            metric = String(metric),
            value = Float64(row[metric]),
            lo,
            hi,
            n_obs = Int(row.n_bets),
            n_fixtures = Int(row.n_panel),
            reference,
            control_run_id,
        ))
    end
    return DataFrame(rows)
end

"An offline score sink used by finalist portfolio tests."
mutable struct InMemoryScoreStore
    scores::DataFrame
    InMemoryScoreStore() = new(DataFrame())
end
ensure_harness_schema!(::InMemoryScoreStore) = nothing
function _score_partition_key(row)
    control = ismissing(row.control_run_id) || row.control_run_id === nothing ? nothing :
              string(row.control_run_id)
    return (string(row.run_id), String(row.scorecard_version), control,
            String(row.stage), String(row.subset))
end

function write_scores!(store::InMemoryScoreStore, scores::AbstractDataFrame)
    incoming = DataFrame(scores)
    isempty(incoming) && return scores
    keys = Set(_score_partition_key(row) for row in eachrow(incoming))
    if !isempty(store.scores)
        keep = [_score_partition_key(row) ∉ keys for row in eachrow(store.scores)]
        store.scores = store.scores[keep, :]
    end
    store.scores = isempty(store.scores) ? incoming : vcat(store.scores, incoming; cols = :union)
    return scores
end

function _portfolio_requested_panel(panels, panel)
    requested = if panel === :common
        sort!(collect(reduce(intersect, (Set(ids) for ids in values(panels)))))
    elseif panel isa AbstractVector
        sort!(unique(Int.(panel)))
    else
        throw(ArgumentError("portfolio panel must be :common or a vector of match IDs"))
    end
    isempty(requested) && error("portfolio common panel is empty")
    return requested
end

"""
    portfolio_runs(refs; ds, container=:close_option_b, panel=:common,
                   db=PostgresStorage("harness"),
                   B=4000, seed=1) -> DataFrame

Load completed grid runs by immutable UUID, put every run through one common buildable
fixture panel, and return one wide summary row per run. By default the same results are written
to the harness database as long `stage = "finalist"`, `market = "book"` rows under
`portfolio_<container>` in `harness_scores`; pass `db = nothing` only for an explicit offline run.

`:close_option_b` is the exact Task 013/r05 closing contract, including its historical
`SlateDrawdown(8)` value. `:t25_calibrated` instead prices and calibrates against the complete
T−25 point-in-time book with the validated inverse-Gaussian rate law; staking remains the
same pinned Option-B contract so only the container and execution instant change.
"""
function portfolio_runs(refs::AbstractVector{RunRef};
                        ds,
                        container::Symbol = :close_option_b,
                        panel = :common,
                        db = Training.PostgresStorage("harness"),
                        B::Int = PORTFOLIO_BOOTSTRAP_B,
                        seed::Int = PORTFOLIO_SEED)
    container in PORTFOLIO_CONTAINERS || throw(ArgumentError(
        "portfolio container must be one of $(PORTFOLIO_CONTAINERS); got :$container"))
    isempty(refs) && error("portfolio_runs requires at least one RunRef")
    labels = getfield.(refs, :label)
    run_ids = getfield.(refs, :run_id)
    if !allunique(labels)
        duplicate = first(label for label in labels if count(==(label), labels) > 1)
        error("portfolio_runs requires unique labels; duplicate label $(repr(duplicate))")
    end
    if !allunique(run_ids)
        duplicate = first(run_id for run_id in run_ids if count(==(run_id), run_ids) > 1)
        error("portfolio_runs requires unique run IDs; duplicate run_id $duplicate")
    end
    control = _control_ref(refs, nothing)

    odds = _portfolio_odds(ds, container)
    system = _portfolio_system()
    spec = getproperty(system, :book)
    policy = getproperty(system, :policy)

    fits = Dict{UUID,Training.Fit}()
    panels = Dict{UUID,Vector{Int}}()
    for ref in refs
        fit = Training.load_fit(Training.PostgresStorage(ref.experiment), ref.run_id)
        fits[ref.run_id] = fit
        panels[ref.run_id] = _season_panel(ds, fit, ["24/25", "25/26"])
    end

    requested_panel = _portfolio_requested_panel(panels, panel)

    sources = Dict{String,Training.Fit}()
    for ref in refs
        available = Set(panels[ref.run_id])
        all(in(available), requested_panel) || error(
            "$(ref.label) is missing $(count(!in(available), requested_panel)) requested panel fixtures")
        sources[ref.label] = _portfolio_source(
            fits[ref.run_id], requested_panel, container, odds)
    end

    common_panel, dropped = _portfolio_buildable_panel(
        spec, sources, odds, ds, requested_panel)
    isempty(common_panel) && error("portfolio common buildable panel is empty")

    rows = NamedTuple[]
    for ref in refs
        result = _portfolio_simulate(
            spec, policy, sources[ref.label], odds, ds, common_panel;
            label = ref.label, B, seed)
        push!(rows, _portfolio_summary_row(ref, result; n_panel = length(common_panel)))
    end
    summary = DataFrame(rows)
    metadata!(summary, "dropped_fixtures", dropped; style = :note)

    if db !== nothing
        ensure_harness_schema!(db)
        write_scores!(db, _portfolio_score_rows(summary, container, control.run_id))
    end
    return summary
end
