# src/MatchDay/calibration.jl
#
# The optional generative-rate calibration seam between posterior extraction and Portfolio.
# MatchDay still extracts the legacy row-of-draw-vectors frame used by the live engines; this
# file lifts that frame into a typed CountLatents container, applies Calibration's production
# transform, and writes only the calibrated rate draws back into a copy of the frame.

export matchday_calibration_book, calibrate_matchday_latents,
       option_b_calibrator, option_b_scottish_lower_policy, option_b_book_spec,
       option_b_system

"""
    option_b_calibrator() -> Calibration.GenerativeRateCalibrator

The Scottish Lower T-25 Option B rate calibrator validated by experiment suite 07.
"""
option_b_calibrator() = Calibration.GenerativeRateCalibrator(
    name = "scot_lower_t25_inv",
    law = Calibration.InverseGaussianLaw(w_base = 0.25, sigma = 0.35),
    dispersion = Calibration.PoolDispersion(),
    anchor = :pool_mean,
    fallback = :identity,
    book_as_of_minutes = -25.0,
)

"""
    option_b_scottish_lower_policy() -> Portfolio.PolicySpec

The Option B basket and trust vector: Home and Under 2.5 at full trust; Draw, Away and
Over 1.5 at `1/1.4`; every other canonical selection gated. The slate-wide risk and cap are
`SlateDrawdown(8.0)` and `FixedCap(0.25)`.
"""
option_b_scottish_lower_policy() = Portfolio.PolicySpec(
    trust = Portfolio.TieredTrust(Dict(
        ("1x2", 0.0, :home)         => 1.0,
        ("over_under", 2.5, :under) => 1.0,
        ("1x2", 0.0, :draw)         => 1.0 / 1.4,
        ("1x2", 0.0, :away)         => 1.0 / 1.4,
        ("over_under", 1.5, :over)  => 1.0 / 1.4,
    ); default = 0.0),
    risk = Portfolio.SlateDrawdown(8.0),
    cap = Portfolio.FixedCap(0.25),
    grouping = Portfolio.DailySlate(),
)

"""
    option_b_book_spec() -> Portfolio.BookSpec

The Option B execution book: canonical MatchDay markets, de-arbitraged prices, 30% fractional
Kelly, 2% per-bet commission and a 99% per-match budget. Portfolio's minimum is a bankroll
fraction (`0.001` as validated in suite 07); MatchDay applies the £1 exchange minimum later via
`FloorOrDrop`.
"""
option_b_book_spec() = Portfolio.BookSpec(
    markets = canonical_markets(),
    price = Portfolio.DeArb(),
    allocator = Portfolio.KellyLogUtility(),
    shrink = Portfolio.FractionalKelly(0.30),
    exec = Portfolio.ExecutionConfig(
        commission = Portfolio.PerBetCommission(0.02),
        budget = 0.99,
        min_selection_stake = 0.001,
    ),
)

"The complete Scottish Lower Option B pricing and staking system."
option_b_system() = Portfolio.PortfolioSystem(option_b_book_spec(),
                                               option_b_scottish_lower_policy())

function _matchday_as_of_minutes(cards::Vector{<:FixtureCard}, as_of::DateTime)
    isempty(cards) && error("matchday calibration: no fixture card was supplied.")
    offsets = Float64[Dates.value(as_of - c.fixture.kickoff) / 60_000 for c in cards]
    first_offset = first(offsets)
    all(x -> isapprox(x, first_offset; atol = 1e-9), offsets) || error(
        "matchday calibration: the slate has multiple kick-off offsets at $as_of: " *
        join(sort!(unique(offsets)), ", ") * ". One calibrated book must describe one instant.")
    return first_offset
end

"""
    matchday_calibration_book(odds, cards, as_of) -> DataFrame

Build the complete, de-vigged book used by the generative calibrator from the exact scalar
quotes MatchDay handed to Portfolio. Invalid prices and incomplete markets are removed before
normalisation, so a one-sided total can never become a fabricated fair probability of one.

The returned frame carries `as_of_minutes`, measured from the fixtures rather than assigned from
the calibrator. `calibrate_matchday_latents` then asserts that it equals the instant at which the
calibrator was fitted.
"""
function matchday_calibration_book(odds::AbstractDataFrame,
                                   cards::Vector{<:FixtureCard}, as_of::DateTime)
    required = (:match_id, :market_name, :market_line, :selection, :odds_close)
    for col in required
        hasproperty(odds, col) || error(
            "matchday_calibration_book: odds has no :$col; it has $(propertynames(odds)).")
    end

    ids = Set{Int}(c.fixture.m_id for c in cards)
    offset = _matchday_as_of_minutes(cards, as_of)
    valid = DataFrame(odds)
    filter!(r -> Int(r.match_id) in ids && !ismissing(r.odds_close) &&
                 isfinite(Float64(r.odds_close)) && Float64(r.odds_close) > 1.0, valid)

    chunks = DataFrame[]
    for group in groupby(valid, [:match_id, :market_name, :market_line])
        market_name = String(first(group.market_name))
        market_line = Float64(first(group.market_line))
        expected = Calibration.expected_selection_count(market_name, market_line)
        expected > 0 || continue
        length(unique(Symbol.(group.selection))) == expected || continue
        nrow(group) == expected || continue

        out = DataFrame(group)
        implied = 1.0 ./ Float64.(out.odds_close)
        overround = sum(implied)
        (isfinite(overround) && overround > 0.0) || continue
        out.prob_implied_close = implied
        out.prob_fair_close = implied ./ overround
        out.overround = fill(overround, nrow(out))
        out.as_of_minutes = fill(offset, nrow(out))
        push!(chunks, out)
    end

    if isempty(chunks)
        empty = DataFrame(valid)
        empty.prob_implied_close = Float64[]
        empty.prob_fair_close = Float64[]
        empty.overround = Float64[]
        empty.as_of_minutes = Float64[]
        return empty
    end
    book = vcat(chunks...)
    sort!(book, [:match_id, :market_name, :market_line, :selection])
    return book
end

function _matchday_rate_columns(latents::AbstractDataFrame)
    if hasproperty(latents, :λ_h) && hasproperty(latents, :λ_a)
        return (:λ_h, :λ_a)
    elseif hasproperty(latents, :lambda_home) && hasproperty(latents, :lambda_away)
        return (:lambda_home, :lambda_away)
    end
    error("calibrate_matchday_latents: expected rate columns :λ_h/:λ_a or " *
          ":lambda_home/:lambda_away; got $(propertynames(latents)).")
end

function _matchday_draw_matrix(latents::AbstractDataFrame, col::Symbol)
    values = latents[!, col]
    isempty(values) && return Matrix{Float64}(undef, 0, 0)
    n_draws = length(first(values))
    n_draws > 0 || error("calibrate_matchday_latents: :$col has zero posterior draws.")
    matrix = Matrix{Float64}(undef, nrow(latents), n_draws)
    for i in 1:nrow(latents)
        draws = values[i]
        length(draws) == n_draws || error(
            "calibrate_matchday_latents: :$col row $i has $(length(draws)) draws; " *
            "row 1 has $n_draws.")
        @inbounds for draw in 1:n_draws
            matrix[i, draw] = Float64(draws[draw])
        end
    end
    return matrix
end

function _matchday_observation_params(latents::AbstractDataFrame)
    if hasproperty(latents, :r_h) && hasproperty(latents, :r_a)
        return (; r_h = _matchday_draw_matrix(latents, :r_h),
                r_a = _matchday_draw_matrix(latents, :r_a))
    elseif hasproperty(latents, :r)
        shared = _matchday_draw_matrix(latents, :r)
        return (; r_h = shared, r_a = copy(shared))
    end
    return nothing
end

_matchday_draw_rows(matrix::AbstractMatrix) =
    [collect(view(matrix, i, :)) for i in axes(matrix, 1)]

"""
    calibrate_matchday_latents(calibrator, latents, odds, cards, as_of)
        -> (; latents, diagnostics, rates, book, coverage)

Apply a generative-rate calibrator to a MatchDay legacy latent frame. The input frame is not
mutated. Observation parameters and all model-specific columns are preserved; only the home and
away intensity draw columns are replaced.

The book instant is checked against `calibrator.book_as_of_minutes`. A T-25 recipe therefore
cannot be applied to a T-15 or closing book by relabelling the frame.
"""
function calibrate_matchday_latents(calibrator::Calibration.AbstractGenerativeRateCalibrator,
                                    latents::AbstractDataFrame,
                                    odds::AbstractDataFrame,
                                    cards::Vector{<:FixtureCard}, as_of::DateTime)
    isempty(latents) && return (
        latents = DataFrame(latents), diagnostics = DataFrame(),
        rates = Dict{Int,Calibration.MarketRateFit}(), book = DataFrame(),
        coverage = (; n_fixtures = 0, n_accepted = 0, n_refused = 0, n_absent = 0,
                    n_quoted = 0, coverage = NaN, coverage_quoted = NaN),
    )

    home_col, away_col = _matchday_rate_columns(latents)
    count_latents = Models.CountLatents(
        Int.(latents.match_id),
        _matchday_draw_matrix(latents, home_col),
        _matchday_draw_matrix(latents, away_col),
        _matchday_observation_params(latents),
    )
    book = matchday_calibration_book(odds, cards, as_of)
    actual_offset = _matchday_as_of_minutes(cards, as_of)
    isapprox(actual_offset, calibrator.book_as_of_minutes; atol = 1e-9) || error(
        "calibrate_matchday_latents: expected T$(calibrator.book_as_of_minutes), " *
        "got T$actual_offset.")
    isempty(book) || Calibration.assert_book_as_of(book, calibrator.book_as_of_minutes)
    ids = Models.latent_match_ids(count_latents)
    rates = Calibration.invert_market_rates(calibrator, book; match_ids = ids)
    calibrated, diagnostics = Calibration.calibrate_latents(calibrator, count_latents, rates)

    out = DataFrame(latents)
    out[!, home_col] = _matchday_draw_rows(calibrated.λ_home)
    out[!, away_col] = _matchday_draw_rows(calibrated.λ_away)
    coverage = Calibration.inversion_coverage(rates, ids)
    return (; latents = out, diagnostics, rates, book, coverage)
end
