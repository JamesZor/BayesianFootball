using Test
using BayesianFootball
using BayesianFootball.Models
using BayesianFootball.Predictions
using BayesianFootball.Data
using DataFrames
using Dates
using Distributions

const SCOREGRID_PORTFOLIO = BayesianFootball.Portfolio
const SCOREGRID_EVALUATION = BayesianFootball.Evaluation

function scoregrid_baseline(λh::AbstractMatrix, λa::AbstractMatrix; max_goals::Int = 12)
    S = Array{Float64,3}(undef, max_goals, max_goals, size(λh, 2))
    @inbounds for k in axes(S, 3), a in 0:(max_goals - 1), h in 0:(max_goals - 1)
        S[h + 1, a + 1, k] = pdf(Poisson(λh[1, k]), h) * pdf(Poisson(λa[1, k]), a)
    end
    return S
end

function scoregrid_cdf(S::Array{Float64,3}, K::Int, k::Int)
    p = 0.0
    @inbounds for a in axes(S, 2), h in axes(S, 1)
        (h - 1) + (a - 1) <= K && (p += S[h, a, k])
    end
    return p
end

@testset "Score-grid hierarchy and smile anti-diagonal reweighting" begin
    ids = [9101]
    λh = [1.40 0.85 1.75]
    λa = [0.90 1.10 0.70]
    λtot = λh .+ λa
    strikes = collect(0.5:1.0:4.5)
    nd = size(λh, 2)
    nK = length(strikes)
    φ = fill(1.15, 1, nK, nd)
    smile = SmileLatents(ids, λh, λa, nothing, λtot, φ, strikes)

    ws = GridWorkspace(12)
    raw = alloc_score_grid(smile)
    buffers = alloc_smile_buffers(smile)
    holder = SmileScoreGrid(raw, buffers.λ_tot, buffers.φ, copy(strikes))
    standard = StandardScoreGrid(similar(raw))

    @test holder isa AbstractScoreGrid
    @test standard isa AbstractScoreGrid
    @test supertype(StandardScoreGrid) === AbstractScoreGrid
    @test supertype(SmileScoreGrid) === AbstractScoreGrid
    @test compute_score_grid!(holder, ws, smile, 1) === holder

    # Every non-identity draw is a genuine joint probability distribution, and each
    # learned totals CDF is reproduced directly by summing its anti-diagonals.
    for k in 1:nd
        @test abs(sum(holder.grid[:, :, k]) - 1.0) < 1e-14
        for K in 0:(nK - 1)
            target = cdf(Poisson(λtot[1, k] * φ[1, K + 1, k]), K)
            @test abs(scoregrid_cdf(holder.grid, K, k) - target) <= 1e-9
        end
    end

    # O/U now has no analytical side route: holder and raw-tensor dispatch are exact.
    ou = MarketOverUnder(2.5)
    holder_book = alloc_market_book(ou, nd)
    tensor_book = alloc_market_book(ou, nd)
    price_market!(holder_book, holder, ou)
    price_market!(tensor_book, holder.grid, ou)
    @test holder_book == tensor_book
    for k in 1:nd
        target = cdf(Poisson(λtot[1, k] * φ[1, 3, k]), 2)
        @test abs(holder_book[2][k] - target) <= 1e-9
    end

    # The hot fill/reweight/price path owns all scratch before the fixture loop.
    compute_score_grid!(holder, ws, smile, 1)
    price_market!(holder_book, holder, ou)
    @test @allocated(compute_score_grid!(holder, ws, smile, 1)) == 0
    @test @allocated(price_market!(holder_book, holder, ou)) == 0

    evaluation_probs = SCOREGRID_EVALUATION.market_probabilities(
        smile, BayesianFootball.Data.AbstractMarket[ou]; keep_draws = true, threaded = false)
    evaluation_workspace = SCOREGRID_EVALUATION.alloc_evaluation_workspace(smile, [ou])
    SCOREGRID_EVALUATION.price_match_markets!(evaluation_probs, evaluation_workspace, smile, 1)
    @test @allocated(SCOREGRID_EVALUATION.price_match_markets!(
        evaluation_probs, evaluation_workspace, smile, 1)) == 0
    @test vec(evaluation_probs.draws[:, 1, 2]) == holder_book[2]

    # φ ≡ 1 is an explicit shortcut: not merely equivalent, but bit-identical to the
    # original truncated count grid. The forced path differs by no more than the mass
    # that truncation had omitted and restores total mass to one.
    identity_φ = ones(1, nK, nd)
    identity = SmileLatents(ids, λh, λa, nothing, λtot, identity_φ, strikes)
    identity_grid = alloc_score_grid(identity)
    count_grid = alloc_score_grid(CountLatents(ids, λh, λa))
    compute_score_grid!(identity_grid, ws, identity, 1)
    count = CountLatents(ids, λh, λa)
    compute_score_grid!(count_grid, ws, count, 1)
    @test identity_grid == count_grid

    standard_holder = StandardScoreGrid(similar(count_grid))
    @test compute_score_grid!(standard_holder, ws, count, 1) === standard_holder
    standard_book = alloc_market_book(ou, nd)
    raw_book = alloc_market_book(ou, nd)
    price_market!(standard_book, standard_holder, ou)
    price_market!(raw_book, standard_holder.grid, ou)
    @test standard_book == raw_book

    forced = copy(count_grid)
    forced_φ = ones(nK, nd)
    reweight_grid_antidiagonals!(forced, vec(λtot), forced_φ, ws;
                                 identity_shortcut = false)
    for k in 1:nd
        truncation_mass = 1.0 - sum(count_grid[:, :, k])
        @test abs(sum(forced[:, :, k]) - 1.0) < 1e-14
        @test maximum(abs.(forced[:, :, k] .- count_grid[:, :, k])) <=
              truncation_mass + 1e-14
    end

    # A per-strike intensity curve is admitted only when its CDF values form a CDF.
    bad = copy(count_grid)
    bad_φ = ones(nK, nd)
    bad_φ[1, :] .= 0.10
    bad_φ[2, :] .= 10.0
    @test_throws ErrorException reweight_grid_antidiagonals!(bad, vec(λtot), bad_φ, ws)
end

@testset "Portfolio consumes the reweighted smile tensor" begin
    ids = [9201, 9202]
    λh = [1.35 1.50 1.20 1.45; 0.85 1.00 0.95 1.10]
    λa = [0.80 0.95 1.05 0.90; 1.20 1.30 1.10 1.25]
    λtot = λh .+ λa
    strikes = collect(0.5:1.0:4.5)
    φ = fill(1.12, length(ids), length(strikes), size(λh, 2))
    smile = SmileLatents(ids, λh, λa, nothing, λtot, φ, strikes)

    spec = SCOREGRID_PORTFOLIO.BookSpec(
        markets = MarketConfig(BayesianFootball.Data.AbstractMarket[
            Market1X2(), MarketOverUnder(2.5), MarketBTTS()]),
        shrink = SCOREGRID_PORTFOLIO.NoShrinkage())
    workspace = SCOREGRID_PORTFOLIO.BookWorkspace(spec, smile)
    @test workspace.grid isa SmileScoreGrid

    SCOREGRID_PORTFOLIO.price_fixture!(workspace, smile, 1)
    @test @allocated(SCOREGRID_PORTFOLIO.price_fixture!(workspace, smile, 2)) == 0
    for k in 1:size(workspace.S, 3)
        @test abs(sum(workspace.S[:, :, k]) - 1.0) < 1e-14
    end

    rows = NamedTuple[]
    for id in ids
        for (group, line, selections, probabilities) in (
            ("1X2", 0.0, [:home, :draw, :away], [0.42, 0.28, 0.30]),
            ("OverUnder", 2.5, [:over_25, :under_25], [0.52, 0.48]),
            ("BTTS", 0.0, [:btts_yes, :btts_no], [0.51, 0.49]))
            for (selection, probability) in zip(selections, probabilities)
                push!(rows, (; match_id = id, market_name = group, market_line = line,
                              selection, odds_close = 1.05 / probability))
            end
        end
    end
    odds = DataFrame(rows)
    fixtures = Dict(id => (date = Date(2026, 1, day), score = (1, 0))
                    for (day, id) in enumerate(ids))
    books = SCOREGRID_PORTFOLIO.build_books(spec, smile, odds, fixtures)
    @test length(books) == length(ids)
    for (i, book) in enumerate(books)
        grid = reshape(book.p_grid, 12, 12)
        implied_under = scoregrid_cdf(reshape(grid, 12, 12, 1), 2, 1)
        target = sum(cdf(Poisson(λtot[i, k] * φ[i, 3, k]), 2)
                     for k in axes(λtot, 2)) / size(λtot, 2)
        @test abs(implied_under - target) <= 1e-9
        under = only(s for s in book.sels if s.selection === :under_25)
        @test abs(under.p_model - target) <= 1e-9
    end

    # Identity smile and its CountLatents twin reach the same Portfolio ledger.
    identity = SmileLatents(ids, λh, λa, nothing, λtot,
                            ones(size(φ)), strikes)
    count = CountLatents(ids, λh, λa)
    @test SCOREGRID_PORTFOLIO.build_books(spec, identity, odds, fixtures) ==
          SCOREGRID_PORTFOLIO.build_books(spec, count, odds, fixtures)
end
