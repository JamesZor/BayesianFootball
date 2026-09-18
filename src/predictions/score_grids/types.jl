"The score-grid truncation used by the legacy prediction kernels."
const TPL_MAX_GOALS = 12

"""Reusable scratch for typed posterior score-grid kernels."""
struct GridWorkspace
    p_h::Vector{Float64}
    p_a::Vector{Float64}
    grid_mass::Vector{Float64}
    ratio::Vector{Float64}
    max_goals::Int

    function GridWorkspace(max_goals::Integer = TPL_MAX_GOALS)
        max_goals > 0 || error("GridWorkspace: max_goals must be positive, got $max_goals.")
        n = Int(max_goals)
        n_diagonals = 2 * n - 1
        return new(zeros(Float64, n), zeros(Float64, n),
                   zeros(Float64, n_diagonals), zeros(Float64, n_diagonals), n)
    end
end

"Allocate one `(home goals × away goals × draws)` destination grid."
alloc_score_grid(l::AbstractPosteriorLatents, max_goals::Integer = TPL_MAX_GOALS) =
    Array{Float64, 3}(undef, Int(max_goals), Int(max_goals), n_draws(l))

@inline function _tpl_check_target(S::Array{Float64,3}, ws::GridWorkspace,
                                   l::AbstractPosteriorLatents, i::Int)
    1 <= i <= n_matches(l) ||
        error("fixture index $i is out of range 1:$(n_matches(l)).")
    size(S) == (ws.max_goals, ws.max_goals, n_draws(l)) || error(
        "destination grid is $(size(S)); expected " *
        "$((ws.max_goals, ws.max_goals, n_draws(l))). Use `alloc_score_grid`.")
    return nothing
end

"A smile's per-strike CDF values do not define a valid totals distribution."
struct NonMonotoneSmileError <: Exception
    draw::Int
    strike::Int
    smile_cdf::Float64
    previous_cdf::Float64
    λ_tot::Float64
    φ::Float64
end

function Base.showerror(io::IO, e::NonMonotoneSmileError)
    print(io, "anti-diagonal reweighting: the smile curve is not a CDF on draw ", e.draw,
          " — P(total ≤ ", e.strike, ") = ", e.smile_cdf,
          " < P(total ≤ ", e.strike - 1, ") = ", e.previous_cdf,
          " at λ_tot = ", e.λ_tot, " and φ[", e.strike + 1, "] = ", e.φ,
          ". The container is refused rather than clipped.")
end

"Common interface for score tensors consumed by market pricing."
abstract type AbstractScoreGrid end

"One fixture's ordinary joint score tensor."
struct StandardScoreGrid <: AbstractScoreGrid
    grid::Array{Float64,3}
end

"One fixture's anti-diagonal-reweighted joint score tensor and its source smile curve."
struct SmileScoreGrid <: AbstractScoreGrid
    grid::Array{Float64,3}
    λ_tot::Vector{Float64}
    φ::Matrix{Float64}
    strikes::Vector{Float64}
end

"Allocate the per-fixture smile buffers for reuse across fixtures."
alloc_smile_buffers(l::SmileLatents) = (;
    λ_tot = Vector{Float64}(undef, n_draws(l)),
    φ = Matrix{Float64}(undef, n_strikes(l), n_draws(l)),
)

market_keys(m::Market1X2) = (outcomes(m).home, outcomes(m).draw, outcomes(m).away)
market_keys(m::MarketBTTS) = (outcomes(m).yes, outcomes(m).no)
market_keys(m::MarketOverUnder) = (outcomes(m).over, outcomes(m).under)
market_arity(m) = length(market_keys(m))

"Allocate a concretely typed tuple of market-outcome vectors."
alloc_market_book(market, n_draws::Integer) =
    ntuple(_ -> Vector{Float64}(undef, Int(n_draws)), market_arity(market))
