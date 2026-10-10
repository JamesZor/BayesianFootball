module QSBacktestScenario

import BayesianFootball as BF
import Random
import Dates
import Test
import CSV
import DataFrames as DF
const PF = BF.Portfolio

"Prototype configuration, validated before graduation to the src allocator."
struct PrototypeScenarioKelly <: PF.AbstractAllocator
    n_scenarios::Int
    seed::Int
    function PrototypeScenarioKelly(; n_scenarios::Integer = 1000, seed::Integer = 20261010)
        1 <= n_scenarios <= 1000 || throw(ArgumentError("scenario count must be in 1:1000"))
        seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
        new(Int(n_scenarios), Int(seed))
    end
end

"Explicit joint-scenario weights: plain per-match mean vectors are refused."
struct PrototypeScenarioWeights <: AbstractVector{Float64}
    values::Vector{Float64}
end
Base.size(p::PrototypeScenarioWeights) = size(p.values)
Base.getindex(p::PrototypeScenarioWeights, i::Int) = p.values[i]
Base.IndexStyle(::Type{PrototypeScenarioWeights}) = Base.IndexLinear()

function PF.allocate(alloc::PrototypeScenarioKelly, p::PrototypeScenarioWeights,
                     R::AbstractMatrix{Float64}, exec::PF.ExecutionConfig)
    length(p) == size(R, 1) || throw(DimensionMismatch("scenario weights and payoff rows differ"))
    1 <= length(p) <= alloc.n_scenarios || throw(ArgumentError("too many or no scenarios"))
    all(isfinite, p) && all(>=(0.0), p) && abs(sum(p) - 1.0) <= 1e-10 ||
        throw(ArgumentError("scenario weights must be a finite probability vector"))
    all(isfinite, R) && all(>=(-1.0), R) || throw(ArgumentError("invalid net payoff matrix"))
    0 < exec.budget <= 1 && 0 < exec.max_selection_stake <= 1 ||
        throw(ArgumentError("scenario budget and per-bet cap must be in (0,1]"))
    return PF.allocate(PF.KellyLogUtility(), p, R, exec)
end
PF.allocate(::PrototypeScenarioKelly, ::AbstractVector, ::AbstractMatrix, ::PF.ExecutionConfig) =
    throw(ArgumentError("joint scenario weights required; stock per-match builders are not scenario Kelly"))

scenario_allocator(native::Bool; kw...) = native ? PF.ScenarioKelly(; kw...) :
                                                 PrototypeScenarioKelly(; kw...)
scenario_weights(native::Bool, p) = native ? PF.ScenarioWeights(p) :
                                            PrototypeScenarioWeights(collect(Float64, p))

"Balanced shuffled posterior indices, shared by every fixture; never decorrelated by match."
function joint_draw_indices(rng, n_draws::Int, count::Int)
    out = Int[]
    while length(out) < count
        permutation = Random.randperm(rng, n_draws)
        append!(out, permutation[1:min(n_draws, count - length(out))])
    end
    return out
end

"Inverse-CDF sample of a full joint home/away score grid, conditional on one draw."
function score_row(grid, draw::Int, u::Float64)
    q = vec(grid[:, :, draw])
    mass = sum(q)
    mass > 0 && isfinite(mass) && all(isfinite, q) && all(>=(0.0), q) ||
        throw(ArgumentError("draw-level score grid must have finite nonnegative positive mass"))
    target, accumulated = u * mass, 0.0
    for row in eachindex(q)
        accumulated += q[row]
        accumulated > target && return row
    end
    return length(q)                 # floating-point endpoint only
end

"""
    scenario_problem(allocator, slate, grids, draw_groups; native = false)

`grids[i]` is home × away × draw for books[i], in the SAME posterior column order.
`draw_groups[i]` identifies the fitted posterior/fold and chain-iteration layout.
The caller must obtain these witnesses from saved-fit provenance, not invent them.
Mixed groups and unequal draw counts fail loudly: matching column counts alone
are not evidence of joint posterior draws. There is no mixture-only C0 fallback.

Each scenario shares one posterior column across the slate, but samples scorelines
independently conditional on it. All same-match legs use one shared scoreline row.
Within each draw, grid truncation is explicitly normalised. Replicate shuffled
posterior passes when fewer draws than requested scenarios are available.
"""
function scenario_problem(allocator, slate::PF.Slate, grids::AbstractVector,
                          draw_groups::AbstractVector{<:AbstractString}; native::Bool = false)
    L = length(slate.books)
    length(grids) == length(draw_groups) == L || throw(DimensionMismatch("one grid/group per book required"))
    if L == 0
        return (; weights = scenario_weights(native, [1.0]), R = zeros(1, 0),
                columns = UnitRange{Int}[], draw_indices = Int[], score_rows = zeros(Int, 1, 0),
                draw_group = "")
    end
    all(s -> !isempty(s), draw_groups) && length(unique(draw_groups)) == 1 ||
        throw(ArgumentError("daily slate crosses posterior/fold layouts; do not pair unrelated draw indices"))
    nd = size(grids[1], 3)
    nd > 0 || throw(ArgumentError("draw-level posterior grids required"))
    all(g -> ndims(g) == 3 && size(g, 3) == nd, grids) ||
        throw(DimensionMismatch("posterior draw axes must be aligned"))
    length(unique(b.m_id for b in slate.books)) == L || throw(ArgumentError("duplicate fixture in slate"))
    all(b -> b.date == slate.window, slate.books) || throw(ArgumentError("one calendar day required"))
    day_seed = allocator.seed + Dates.value(slate.window)
    draws = joint_draw_indices(Random.MersenneTwister(day_seed), nd, allocator.n_scenarios)
    columns = UnitRange{Int}[]
    offset = 0
    for (b, grid) in zip(slate.books, grids)
        size(b.R, 1) == size(grid, 1) * size(grid, 2) ||
            throw(DimensionMismatch("score states and payoff rows differ"))
        size(b.R, 2) == length(b.sels) || throw(DimensionMismatch("payoff legs and selections differ"))
        push!(columns, (offset + 1):(offset + length(b.sels)))
        offset += length(b.sels)
    end
    R = zeros(length(draws), offset)
    rows = zeros(Int, length(draws), L)
    for (i, b) in enumerate(slate.books)
        rng = Random.MersenneTwister(day_seed + b.m_id)
        for s in eachindex(draws)
            row = score_row(grids[i], draws[s], Random.rand(rng))
            rows[s, i] = row
            R[s, columns[i]] .= b.R[row, :]
        end
    end
    return (; weights = scenario_weights(native, fill(1.0 / length(draws), length(draws))),
            R, columns, draw_indices = draws, score_rows = rows, draw_group = String(first(draw_groups)))
end

"""
    scenario_books(allocator, slate, grids, draw_groups, exec; native = false)

Opt-in daily solve, not a BookSpec plug-in. Preserve every field including the
production per-match shrink factors; replace only the raw allocation in NEW books.
Then use ordinary stake_slate/simulate_portfolio. Existing risk still uses its
marginal/sequential law, not the joint scenarios; that overlay is deliberately unchanged.
"""
function scenario_books(allocator, slate::PF.Slate, grids, draw_groups, exec;
                        native::Bool = false)
    problem = scenario_problem(allocator, slate, grids, draw_groups; native = native)
    solution = PF.allocate(allocator, problem.weights, problem.R, exec)
    solution.converged || error("joint scenario allocation did not converge; stop before staking")
    all(isfinite, solution.a) && all(>=(0.0), solution.a) &&
        sum(solution.a) <= min(1.0, exec.budget) + 1e-10 &&
        all(<=(exec.max_selection_stake + 1e-10), solution.a) || error("scenario constraints failed")
    books = PF.MatchBook[
        PF.MatchBook(b.m_id, b.date, b.sels, b.p_grid, b.R, b.settle,
                     solution.a[problem.columns[i]], b.k_shrink, solution.kkt, solution.converged)
        for (i, b) in enumerate(slate.books)]
    return (; slate = PF.Slate(slate.window, books), problem, solution)
end

"Synthetic home-bet books with coherent goal score states; no fits, cache or SQL."
function toy_slate(; shared::Bool = true)
    exec = PF.ExecutionConfig(commission = PF.NoCommission(), min_selection_stake = 0.0)
    books, grids = PF.MatchBook[], Array{Float64,3}[]
    date = Dates.Date(2025, 1, 4)
    for id in (1, 2)
        sel = PF.Selection("1X2_home", "1X2", 0.0, :home, 2.0, 2.0, 0.6, 0.5)
        grid = zeros(2, 1, 10)
        if shared
            grid[2, 1, 1:6] .= 1.0
            grid[1, 1, 7:10] .= 1.0
        else
            grid[1, 1, :] .= 0.4
            grid[2, 1, :] .= 0.6
        end
        p = [0.4, 0.6]
        R = PF.payoff_matrix([sel], 2, 1, exec.commission)
        raw = PF.allocate(PF.KellyLogUtility(), p, R, exec)
        push!(books, PF.MatchBook(id, date, [sel], p, R, [1.0], raw.a, 0.30, raw.kkt, raw.converged))
        push!(grids, grid)
    end
    return PF.Slate(date, books), grids, exec
end

include("l01_scenario_kelly_tests.jl")
end
