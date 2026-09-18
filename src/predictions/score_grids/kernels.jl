# ==============================================================================
# 06 — TYPED POSTERIOR LATENTS : SCORE GRIDS AND MARKET PRICING
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# ------------------------------------------------------------------------------
# THE TWO RULES EVERY KERNEL IN THIS FILE FOLLOWS
# ------------------------------------------------------------------------------
#
# RULE 1 — BIT PARITY, NOT ALGEBRAIC PARITY.
#
#   Floating-point addition is not associative and multiplication does not
#   distribute exactly. "Same formula" is therefore not the same as "same number":
#   `(a*b) + (c*d) + e` and `a*b + (c*d + e)` are different Float64 in general.
#
#   So every kernel below reproduces the legacy kernel's OPERAND ORDER, not merely
#   its formula, and each one names the file and lines it is mirroring. Where this
#   file departs from the legacy shape at all (the negative-binomial grid, §3) the
#   departure is proved value-identical before it is taken, and the proof is in the
#   comment next to it.
#
#   The threshold in `l04_parity.jl` is |Δ| < 1e-12 because that is what the briefing
#   asks for. What the kernels actually deliver is 0 ULP, and the parity report
#   reports the ULP distance rather than the absolute one, because "agrees to 1e-12"
#   would hide a systematic last-bit drift that "agrees to 0 ULP" does not.
#
# RULE 2 — THE HOT PATH ALLOCATES NOTHING.
#
#   Every kernel comes in two forms:
#
#       compute_score_grid!(S, ws, latents, i)   0 bytes; caller owns S and ws
#       compute_score_grid(latents, i)           allocates S and ws, calls the above
#
#   The bang forms are the API a backtest should use: one destination array and one
#   workspace, reused across every fixture in the fold. The allocating forms exist
#   for the REPL and for one-off inspection.
#
#   Concretely, the bang forms hold to:
#     * no `view` of a latent matrix — a `SubArray` is a heap object unless escape
#       analysis elides it, and a zero-allocation claim that depends on an
#       optimisation firing is not a claim worth making. Scalar `A[i, k]` instead.
#     * no distribution vector, no `Ref`, no comprehension, no closure.
#     * scalar accumulators inside the draw loop, written to the output vector once.
#
#   Verified by `@allocated`, not by inspection — `l04_parity.jl` §7.
#
# ==============================================================================

const TPL_MD = MyDistributions

# ==============================================================================
# 2. COUNT FAMILY — POISSON
# ==============================================================================
#
# Mirrors src/predictions/score_computation/poisson.jl:29-62.

"""
Fill `p[1:n]` with `pdf(Poisson(λ), 0:n-1)`.

The legacy line is `@. p_h = pdf(d_h, goals)` with `goals::UnitRange{Int}`, which
lowers to exactly this loop over the same `Poisson` object with the same `Int`
arguments. Written out rather than broadcast so there is no fused-broadcast object
to construct.
"""
@inline function _tpl_poisson_pmf!(p::Vector{Float64}, λ::Float64, n::Int)
    d = Poisson(λ)
    @inbounds for g in 1:n
        p[g] = pdf(d, g - 1)
    end
    return nothing
end

"Outer product into one draw slice. Column-major: `j` (away, columns) outer, `i` inner."
@inline function _tpl_outer!(S::Array{Float64,3}, p_h::Vector{Float64},
                             p_a::Vector{Float64}, n::Int, k::Int)
    @inbounds for j in 1:n
        pj = p_a[j]
        for i in 1:n
            S[i, j, k] = p_h[i] * pj
        end
    end
    return nothing
end

"""
    compute_score_grid!(S, ws, l::CountLatents{T,Nothing}, i) -> S

Independent double-Poisson grid for fixture row `i`. 0 bytes.
"""
function compute_score_grid!(S::Array{Float64,3}, ws::GridWorkspace,
                             l::CountLatents{Float64, Nothing}, i::Int)
    _tpl_check_target(S, ws, l, i)
    n = ws.max_goals
    @inbounds for k in 1:n_draws(l)
        _tpl_poisson_pmf!(ws.p_h, l.λ_home[i, k], n)
        _tpl_poisson_pmf!(ws.p_a, l.λ_away[i, k], n)
        _tpl_outer!(S, ws.p_h, ws.p_a, n, k)
    end
    return S
end


# ==============================================================================
# 3. COUNT FAMILY — NEGATIVE BINOMIAL
# ==============================================================================
#
# Mirrors src/predictions/score_computation/negativebinomial.jl:53-88, which builds
# a `DoubleNegativeBinomial` per draw and evaluates it on a pre-built `12×12` grid of
# `[h, a]` vectors:
#
#     S[i, j, k] = pdf(DoubleNegativeBinomial(λ_h, λ_a, r_h, r_a), [i-1, j-1])
#
# THE DEPARTURE, AND WHY IT IS EXACT. That is `M²` = 144 joint evaluations per draw,
# each of which internally computes TWO independent marginal log-pdfs and adds them
# (double_negative_binomial.jl:37-38):
#
#     logpdf(d, x) = _nbinom_logpdf_robust(r_h, μ_h, x[1]) + _nbinom_logpdf_robust(r_a, μ_a, x[2])
#     pdf(d, x)    = exp(logpdf(d, x))
#
# The `2M` distinct marginal values are therefore recomputed `M` times each. This
# kernel computes each once into `ws.p_h` / `ws.p_a` and forms
# `exp(lp_h[i] + lp_a[j])`.
#
# The result is BIT-IDENTICAL, not merely close, because it is the same three
# floating-point operations on the same three operands in the same order: the same
# `_nbinom_logpdf_robust` call with the same arguments, the same `+`, the same `exp`.
# What is skipped is only the redundant recomputation. `r01_demo.jl` §8 checks this
# against the live legacy kernel at 0 ULP rather than trusting the argument.
#
# `exp(a) * exp(b)` would ALSO be algebraically correct here and is NOT used, because
# it is a different pair of operations and does not round to the same Float64.
#
# The `max(·, 1e-9)` guards reproduce `DoubleNegativeBinomial`'s inner constructor
# (double_negative_binomial.jl:17). They never bind on a validated container — `l01`
# already refuses non-positive rates — but they are the legacy code's, so they stay.

"Fill `lp[1:n]` with the NegBin log-pmf at 0:n-1, matching DoubleNegativeBinomial's clamps."
@inline function _tpl_negbin_logpmf!(lp::Vector{Float64}, r::Float64, μ::Float64, n::Int)
    rc = max(r, 1e-9)
    μc = max(μ, 1e-9)
    @inbounds for g in 1:n
        lp[g] = TPL_MD._nbinom_logpdf_robust(rc, μc, g - 1)
    end
    return nothing
end

"""
    compute_score_grid!(S, ws, l::CountLatents{T,<:NamedTuple}, i) -> S

Independent double-negative-binomial grid for fixture row `i`. 0 bytes.
"""
function compute_score_grid!(S::Array{Float64,3}, ws::GridWorkspace,
                             l::CountLatents{Float64, <:NamedTuple}, i::Int)
    _tpl_check_target(S, ws, l, i)
    n   = ws.max_goals
    r_h = l.observation_params.r_h
    r_a = l.observation_params.r_a
    @inbounds for k in 1:n_draws(l)
        _tpl_negbin_logpmf!(ws.p_h, r_h[i, k], l.λ_home[i, k], n)
        _tpl_negbin_logpmf!(ws.p_a, r_a[i, k], l.λ_away[i, k], n)
        for j in 1:n
            lpj = ws.p_a[j]
            for i2 in 1:n
                S[i2, j, k] = exp(ws.p_h[i2] + lpj)
            end
        end
    end
    return S
end


# ==============================================================================
# 4. RECOMBINATION FAMILY
# ==============================================================================
#
# Mirrors src/predictions/score_computation/recombination.jl:43-79.
#
# TWO THINGS THE OTHER FAMILIES DO NOT DO, both preserved because they are the legacy
# kernel's and both materially change the answer:
#
#   1. `Poisson(max(1e-6, λ))`. A floor, not the `1e-9` the NegBin path uses.
#   2. RENORMALISATION. Each marginal is divided by its own truncated sum before the
#      outer product, so the 12×12 grid sums to exactly 1 instead of to
#      `P(H ≤ 11)·P(A ≤ 11)`. The other families leave the truncation mass on the
#      floor. That is a genuine modelling inconsistency across the repository, but it
#      is not this prototype's to resolve — changing it here would move recombination
#      prices while claiming to be a container swap.
#
# `sum(p)` is Julia's pairwise summation, not a naive loop, and the two do not agree
# in the last bit. The legacy kernel calls `sum`, so this one calls `sum`.

"""
    compute_score_grid!(S, ws, l::RecombLatents, i) -> S

Discrete convolution of the open-play, penalty and own-goal Poisson channels for
fixture row `i`. 0 bytes.

The convolution is done in closed form: independent Poissons sum to a Poisson of the
summed rate, so `λ_open + λ_pen + λ_og` is convolved by ADDING the three rates, not by
convolving three PMF vectors. Same distribution, `O(1)` instead of `O(M²)` per side,
and it is what the legacy reader does (recombination.jl:36-37) — which is why the
summation ORDER in `recomb_total_home` has to match it exactly.
"""
function compute_score_grid!(S::Array{Float64,3}, ws::GridWorkspace,
                             l::RecombLatents, i::Int)
    _tpl_check_target(S, ws, l, i)
    n = ws.max_goals
    @inbounds for k in 1:n_draws(l)
        _tpl_poisson_pmf!(ws.p_h, max(1e-6, recomb_total_home(l, i, k)), n)
        _tpl_poisson_pmf!(ws.p_a, max(1e-6, recomb_total_away(l, i, k)), n)

        sum_h = sum(ws.p_h)
        sum_a = sum(ws.p_a)
        sum_h > 0.0 && (ws.p_h ./= sum_h)
        sum_a > 0.0 && (ws.p_a ./= sum_a)

        _tpl_outer!(S, ws.p_h, ws.p_a, n, k)
    end
    return S
end


# ==============================================================================
# 5. SMILE FAMILY
# ==============================================================================

"""
    compute_score_grid!(S, ws, l::SmileLatents, i) -> S

Fill fixture row `i`'s baseline count grid, then reweight its anti-diagonals in place so
that the resulting joint tensor has the learned smile totals marginal. 0 bytes.

The baseline kernels mirror `_smile_poisson_grid` and `_smile_negbin_grid`. Within each
total-goals anti-diagonal the reweighting preserves their conditional scoreline split.
When every `φ` is exactly one, the reweighting is skipped and the baseline tensor remains
bit-identical, including its tiny truncation deficit.
"""
function compute_score_grid!(S::Array{Float64,3}, ws::GridWorkspace,
                             l::SmileLatents{Float64, Nothing}, i::Int)
    _tpl_check_target(S, ws, l, i)
    n = ws.max_goals
    @inbounds for k in 1:n_draws(l)
        _tpl_poisson_pmf!(ws.p_h, l.λ_home[i, k], n)
        _tpl_poisson_pmf!(ws.p_a, l.λ_away[i, k], n)
        _tpl_outer!(S, ws.p_h, ws.p_a, n, k)
    end
    _reweight_grid_antidiagonals!(S, l, i, ws)
    return S
end

function compute_score_grid!(S::Array{Float64,3}, ws::GridWorkspace,
                             l::SmileLatents{Float64, <:NamedTuple}, i::Int)
    _tpl_check_target(S, ws, l, i)
    n   = ws.max_goals
    r_h = l.observation_params.r_h
    r_a = l.observation_params.r_a
    @inbounds for k in 1:n_draws(l)
        _tpl_robust_negbin_pmf!(ws.p_h, r_h[i, k], l.λ_home[i, k], n)
        _tpl_robust_negbin_pmf!(ws.p_a, r_a[i, k], l.λ_away[i, k], n)
        _tpl_outer!(S, ws.p_h, ws.p_a, n, k)
    end
    _reweight_grid_antidiagonals!(S, l, i, ws)
    return S
end

"""
`RobustNegativeBinomial`'s PMF at 0:n-1.

TWO DIFFERENCES FROM §3'S KERNEL, BOTH DELIBERATE, BOTH LOAD-BEARING:

  1. The clamp is `1e-6`, not `1e-9`. `RobustNegativeBinomial`'s inner constructor
     (negative_binomial.jl:23) floors at `1e-6`; `DoubleNegativeBinomial`'s
     (double_negative_binomial.jl:17) floors at `1e-9`. The smile NegBin grid goes
     through the former.

  2. This returns PROBABILITIES and the caller multiplies them, where §3 returns
     LOG-probabilities and the caller `exp`s their sum. `exp(a)·exp(b)` and `exp(a+b)`
     are the same real number and DIFFERENT Float64. The two legacy kernels genuinely
     differ here — `_smile_negbin_grid` forms `p_h[i] * p_a[j]` from two `pdf` calls
     (smile_negbin/l02_smile_negbin_predict.jl:75-82), while `negativebinomial.jl:79`
     evaluates the joint `pdf` and gets `exp` of the sum — so a kernel that matched
     one would fail parity against the other. Each mirrors its own.
"""
@inline function _tpl_robust_negbin_pmf!(p::Vector{Float64}, r::Float64, μ::Float64, n::Int)
    rc = max(r, 1e-6)
    μc = max(μ, 1e-6)
    @inbounds for g in 1:n
        p[g] = exp(TPL_MD._nbinom_logpdf_robust(rc, μc, g - 1))
    end
    return nothing
end

"""
    fill_smile_buffers!(λ_tot, φ, l, i) -> nothing

Copy fixture row `i`'s pricing curve into preallocated buffers. 0 bytes.
"""
function fill_smile_buffers!(λ_tot::Vector{Float64}, φ::Matrix{Float64},
                             l::SmileLatents, i::Int)
    nd, nK = n_draws(l), n_strikes(l)
    length(λ_tot) == nd || error("λ_tot buffer is $(length(λ_tot)); expected $nd.")
    size(φ) == (nK, nd) || error("φ buffer is $(size(φ)); expected $((nK, nd)).")
    @inbounds for k in 1:nd
        λ_tot[k] = l.λ_tot[i, k]
        for s in 1:nK
            φ[s, k] = l.φ[i, s, k]
        end
    end
    return nothing
end

# --- Anti-diagonal smile reweighting ------------------------------------------

@inline _smile_nstrikes(φ::AbstractMatrix{Float64}) = size(φ, 1)
@inline _smile_nstrikes(φ::Array{Float64,3}) = size(φ, 2)
@inline _smile_value(φ::AbstractMatrix{Float64}, ::Int, s::Int, k::Int) = φ[s, k]
@inline _smile_value(φ::Array{Float64,3}, i::Int, s::Int, k::Int) = φ[i, s, k]

@inline function _smile_is_identity(φ, i::Int, k::Int)
    @inbounds for s in 1:_smile_nstrikes(φ)
        _smile_value(φ, i, s, k) == 1.0 || return false
    end
    return true
end

function _reweight_draw_antidiagonals!(S::Array{Float64,3}, k::Int, λ_tot::Float64,
                                       φ, i::Int, ws::GridWorkspace)
    n = ws.max_goals
    mass = ws.grid_mass
    ratio = ws.ratio
    n_strikes = _smile_nstrikes(φ)

    fill!(mass, 0.0)
    @inbounds for c in 1:n
        for r in 1:n
            mass[r + c - 1] += S[r, c, k]
        end
    end

    previous_cdf = 0.0
    @inbounds for s in 1:n_strikes
        K = s - 1
        φK = _smile_value(φ, i, s, k)
        smile_cdf = cdf(Poisson(λ_tot * φK), K)
        target_mass = smile_cdf - previous_cdf
        target_mass >= 0.0 || error(
            "anti-diagonal reweighting: the smile curve is not a CDF on draw $k — " *
            "P(total ≤ $K) = $smile_cdf < P(total ≤ $(K - 1)) = $previous_cdf " *
            "at λ_tot = $λ_tot and φ[$s] = $φK. The container is refused rather than clipped.")
        mass[s] > 0.0 || error(
            "anti-diagonal reweighting: the grid holds zero mass on total $K at draw $k " *
            "and cannot be rescaled to $target_mass.")
        ratio[s] = target_mass / mass[s]
        previous_cdf = smile_cdf
    end

    grid_tail = 0.0
    @inbounds for g in (n_strikes + 1):length(mass)
        grid_tail += mass[g]
    end
    target_tail = 1.0 - previous_cdf
    if grid_tail > 0.0
        tail_ratio = target_tail / grid_tail
        @inbounds for g in (n_strikes + 1):length(ratio)
            ratio[g] = tail_ratio
        end
    elseif target_tail > 0.0
        error("anti-diagonal reweighting: the grid holds no mass above total " *
              "$(n_strikes - 1) at draw $k but the smile puts $target_tail there.")
    else
        @inbounds for g in (n_strikes + 1):length(ratio)
            ratio[g] = 0.0
        end
    end

    @inbounds for c in 1:n
        for r in 1:n
            S[r, c, k] *= ratio[r + c - 1]
        end
    end
    return nothing
end

function _check_reweight_inputs(S::Array{Float64,3}, ws::GridWorkspace,
                                n_strikes::Int, n_curve_draws::Int)
    n, n2, nd = size(S)
    n == n2 == ws.max_goals || error(
        "anti-diagonal reweighting: grid is $(size(S)); workspace is for " *
        "$(ws.max_goals) × $(ws.max_goals).")
    n_curve_draws == nd || error(
        "anti-diagonal reweighting: smile curve has $n_curve_draws draws; grid has $nd.")
    1 <= n_strikes < length(ws.grid_mass) || error(
        "anti-diagonal reweighting: $n_strikes strikes leave no grid tail in a $n × $n grid.")
    return nothing
end

"""
    reweight_grid_antidiagonals!(S, λ_tot, φ, ws; identity_shortcut = true) -> S

Rescale each score-tensor draw so `P(total ≤ K)` equals
`cdf(Poisson(λ_tot * φ(K)), K)` at every learned strike. Mass above the last strike is
rescaled proportionally, making every non-identity draw sum to one. A non-monotone set
of per-strike CDF values is refused. The steady-state successful path allocates nothing.
"""
function reweight_grid_antidiagonals!(S::Array{Float64,3},
                                      λ_tot::AbstractVector{Float64},
                                      φ::AbstractMatrix{Float64},
                                      ws::GridWorkspace;
                                      identity_shortcut::Bool = true)
    _check_reweight_inputs(S, ws, size(φ, 1), size(φ, 2))
    length(λ_tot) == size(S, 3) || error(
        "anti-diagonal reweighting: λ_tot has $(length(λ_tot)) draws; grid has $(size(S, 3)).")
    @inbounds for k in 1:size(S, 3)
        identity_shortcut && _smile_is_identity(φ, 0, k) && continue
        _reweight_draw_antidiagonals!(S, k, λ_tot[k], φ, 0, ws)
    end
    return S
end

function _reweight_grid_antidiagonals!(S::Array{Float64,3}, l::SmileLatents,
                                       i::Int, ws::GridWorkspace;
                                       identity_shortcut::Bool = true)
    _check_reweight_inputs(S, ws, n_strikes(l), n_draws(l))
    @inbounds for k in 1:n_draws(l)
        identity_shortcut && _smile_is_identity(l.φ, i, k) && continue
        _reweight_draw_antidiagonals!(S, k, l.λ_tot[i, k], l.φ, i, ws)
    end
    return S
end

"Fill a standard score-grid holder through the existing latent-family kernel."
function compute_score_grid!(g::StandardScoreGrid, ws::GridWorkspace,
                             l::AbstractPosteriorLatents, i::Int)
    compute_score_grid!(g.grid, ws, l, i)
    return g
end

"Fill and anti-diagonal-reweight a smile holder, retaining its source curve buffers."
function compute_score_grid!(g::SmileScoreGrid, ws::GridWorkspace,
                             l::SmileLatents, i::Int)
    compute_score_grid!(g.grid, ws, l, i)
    fill_smile_buffers!(g.λ_tot, g.φ, l, i)
    return g
end


# ==============================================================================
# 6. THE ALLOCATING WRAPPERS
# ==============================================================================

"""
    compute_score_grid(l, i; max_goals = TPL_MAX_GOALS)

Allocating convenience form: builds a destination grid and a workspace, calls the
bang form, returns the grid (or, for `SmileLatents`, a `SmileScoreGrid`).

For anything that loops over fixtures — which is every backtest and every evaluation
pass — call `compute_score_grid!` with one grid and one workspace instead. This form
allocates `max_goals² × n_draws × 8` bytes PER CALL.
"""
function compute_score_grid(l::AbstractPosteriorLatents, i::Integer;
                            max_goals::Integer = TPL_MAX_GOALS)
    ws = GridWorkspace(max_goals)
    S  = alloc_score_grid(l, max_goals)
    return compute_score_grid!(S, ws, l, Int(i))
end

function compute_score_grid(l::SmileLatents, i::Integer;
                            max_goals::Integer = TPL_MAX_GOALS)
    ws  = GridWorkspace(max_goals)
    S   = alloc_score_grid(l, max_goals)
    compute_score_grid!(S, ws, l, Int(i))
    buf = alloc_smile_buffers(l)
    fill_smile_buffers!(buf.λ_tot, buf.φ, l, Int(i))
    return SmileScoreGrid(S, buf.λ_tot, buf.φ, copy(l.strikes))
end


# ==============================================================================
# 7. MARKET PRICING
# ==============================================================================
#
# Every pricer mirrors its `src/predictions/market_inference/` counterpart's
# ACCUMULATION ORDER, not just its predicate. Summing the same 144 numbers in a
# different order gives a different Float64, and a market probability that differs
# from the production one in the last bit is a diff nobody can explain six months
# later.
#
# The market types are the repository's own (`Data.Market1X2` &c). Re-declaring them
# here would produce prices keyed by symbols that look right and compare unequal.

# --- 1X2 ----------------------------------------------------------------------
#
# Mirrors src/predictions/market_inference/1x2.jl:6-59.
#
# The legacy loop walks AWAY columns outermost and, within each column, splits the
# home dimension into three contiguous runs: rows below the diagonal (away win), the
# diagonal cell (draw), rows above (home win). That ordering is not incidental — it is
# what keeps each run a contiguous column-major read — and it fixes the summation
# order, so it is reproduced exactly.

"""
    price_market!(book, S, market) -> book

Write market probabilities for every draw into `book`, a tuple of preallocated
vectors in `market_keys(market)` order. 0 bytes.
"""
function price_market!(book::NTuple{3, Vector{Float64}}, S::Array{Float64,3}, ::Market1X2)
    max_h, max_a, nd = size(S)
    home, draw, away = book
    @inbounds for k in 1:nd
        ph = 0.0
        pd = 0.0
        pa = 0.0
        for c in 1:max_a
            limit_away = min(c - 1, max_h)
            for r in 1:limit_away
                pa += S[r, c, k]
            end
            if c <= max_h
                pd += S[c, c, k]
            end
            for r in (c + 1):max_h
                ph += S[r, c, k]
            end
        end
        home[k] = ph
        draw[k] = pd
        away[k] = pa
    end
    return book
end

# --- BTTS ---------------------------------------------------------------------
#
# Mirrors src/predictions/market_inference/btts.jl:6-45. Plain column-then-row sweep,
# every cell classified by whether both index-minus-one values exceed zero.

function price_market!(book::NTuple{2, Vector{Float64}}, S::Array{Float64,3}, ::MarketBTTS)
    max_h, max_a, nd = size(S)
    yes, no = book
    @inbounds for k in 1:nd
        y = 0.0
        n = 0.0
        for c in 1:max_a
            for r in 1:max_h
                p = S[r, c, k]
                if (r - 1) > 0 && (c - 1) > 0
                    y += p
                else
                    n += p
                end
            end
        end
        yes[k] = y
        no[k]  = n
    end
    return book
end

# --- Over / Under -------------------------------------------------------------
#
# Mirrors src/predictions/market_inference/over_under.jl:6-52.
#
# NOTE THE THIRD BRANCH THAT ISN'T THERE. On an INTEGER line, cells whose total equals
# the line are counted into NEITHER side, so over + under < 1 — the push is dropped
# rather than voided or split. That is the production behaviour and it is preserved
# verbatim; a container swap is not the place to change how pushes settle.

function price_market!(book::NTuple{2, Vector{Float64}}, S::Array{Float64,3}, m::MarketOverUnder)
    max_h, max_a, nd = size(S)
    over, under = book
    line = m.line
    @inbounds for k in 1:nd
        o = 0.0
        u = 0.0
        for c in 1:max_a
            for r in 1:max_h
                p = S[r, c, k]
                total = (r - 1) + (c - 1)
                if total > line
                    o += p
                elseif total < line
                    u += p
                end
            end
        end
        over[k]  = o
        under[k] = u
    end
    return book
end

# --- Score-grid holder routing ------------------------------------------------
#
# Every market, including O/U, reads the joint tensor. For `SmileScoreGrid` that tensor
# has already been anti-diagonal-reweighted, so pricing, evaluation and allocation all
# consume one distribution rather than a grid plus an analytical sidecar.

function price_market!(book::NTuple{N, Vector{Float64}}, g::AbstractScoreGrid, m) where {N}
    return price_market!(book, g.grid, m)
end

"""
    price_market(grid_or_container, market) -> Dict{Symbol, Vector{Float64}}

Allocating form, keyed exactly as `Predictions.compute_market_probs` keys its result,
so a caller can swap one for the other without touching the join.
"""
function price_market(S, m)
    nd   = S isa AbstractScoreGrid ? size(S.grid, 3) : size(S, 3)
    book = alloc_market_book(m, nd)
    price_market!(book, S, m)
    return Dict{Symbol, Vector{Float64}}(k => v for (k, v) in zip(market_keys(m), book))
end
