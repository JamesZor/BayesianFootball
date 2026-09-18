# src/predictions/score_computation/smile_poisson.jl
#
# Prediction path for the LOCAL-INTENSITY SMILE double-Poisson engine
# (DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel).
#
# The smile is a PRICING object: its per-strike intensity Λ^model(K)=λ_tot·φ(K) defines
# P(N≤K)=cdf(Poisson(Λ^model(K)),K). The baseline (λ_h,λ_a) grid is reweighted along
# anti-diagonals to reproduce that CDF, so O/U, 1X2, BTTS, correct score and portfolio sizing
# all consume one coherent joint tensor.

using Distributions
using ..Models.PreGame: DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel,
                        DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel
using ..Data: AbstractMarket

# All engines that price O/U through the smile (λ_tot·φ(K)) — extend this Union when a new
# smile engine graduates; do NOT let a smile engine fall through to a plain grid route.
const AbstractSmilePoissonEngines = Union{
    DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel,
    DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel,
}

# Carry the reweighted joint grid plus the source per-strike intensities for diagnostics and
# compatibility with research code that inspects `Λ` directly.
struct SmileScoreMatrix <: AbstractScoreMatrix
    grid::ScoreMatrix                # [max_goals × max_goals × n_samples], reweighted in place
    Λ::Matrix{Float64}               # [nK × n_samples] per-strike total intensity (K = row-1)
end

score_matrix_data(sm::SmileScoreMatrix) = sm.grid.data


# 1. Adapter: DataFrame Row -> NamedTuple
extract_params(::AbstractSmilePoissonEngines, row) =
    (λ_h = row.λ_h, λ_a = row.λ_a, λ_tot = row.λ_tot, φ = row.φ)

# Grid kernel (independent double-Poisson), identical math to score_computation/poisson.jl.
function _smile_poisson_grid(λ_h, λ_a; max_goals::Int=12)
    n = length(λ_h)
    S = zeros(Float64, max_goals, max_goals, n)
    p_h = zeros(Float64, max_goals); p_a = zeros(Float64, max_goals)
    goals = 0:(max_goals-1)
    @inbounds for k in 1:n
        @. p_h = pdf(Poisson(λ_h[k]), goals)
        @. p_a = pdf(Poisson(λ_a[k]), goals)
        for j in 1:max_goals
            pj = p_a[j]
            for i in 1:max_goals
                S[i, j, k] = p_h[i] * pj
            end
        end
    end
    return ScoreMatrix(S)
end

# 2. Kernel: Params -> SmileScoreMatrix
function compute_score_matrix(::AbstractSmilePoissonEngines, params; max_goals::Int=12)
    grid = _smile_poisson_grid(params.λ_h, params.λ_a; max_goals)
    # params.φ is [n_samples × nK]; the typed reweighting kernel consumes [nK × n_samples].
    φ = Matrix{Float64}(transpose(params.φ))
    λ_tot = Vector{Float64}(params.λ_tot)
    Λ = φ .* transpose(λ_tot)
    reweight_grid_antidiagonals!(grid.data, λ_tot, φ, GridWorkspace(max_goals))
    return SmileScoreMatrix(grid, Λ)
end

# Every derivative now reads the same reweighted joint tensor. In particular O/U is no longer
# an analytical side route that can disagree with the scenario distribution used by Kelly.
compute_market_probs(S::SmileScoreMatrix, m::AbstractMarket) = compute_market_probs(S.grid, m)
