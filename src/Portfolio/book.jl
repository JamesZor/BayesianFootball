# src/Portfolio/book.jl
#
# Stage A of the pipeline: L1 posterior + market quotes -> MatchBook.
#
# Everything here is a pure function of the data and the BookSpec. Nothing in a PolicySpec can
# reach it, which is what makes `hash(BookSpec)` a sound cache key.

export extract_selections, build_book, build_books, book_cache_key, book_trust, fixture_table,
       is_settled

"Trust key for a selection: `1X2_home`, `O/U 2.5_over_25`, `BTTS_btts_yes`."
selection_family(group::AbstractString, line::Real, sel::Symbol) =
    group == "OverUnder" ? "O/U $(line)_$(sel)" : "$(group)_$(sel)"

_book_market_key(s::Selection) = (s.group, s.line)

function _checked_book_trust(trust::AbstractTrustModel, s::Selection)
    weight = book_trust_for(trust, s)
    (isfinite(weight) && 0.0 <= weight <= 1.0) || error(
        "book-time trust for $(s.family) must be finite and in [0,1], got $weight")
    return weight
end

"A selection-identity-only probe used before prices and model probabilities exist."
function _market_probe(m::AbstractMarket, selection::Symbol)
    group = market_group(m)
    line = Float64(market_line(m))
    return Selection(selection_family(group, line, selection), group, line, selection,
                     2.0, 2.0, 0.5, 0.5)
end

function _market_has_positive_trust(trust::AbstractTrustModel, m::AbstractMarket)
    active = false
    # Do not short-circuit: strict SelectionTrust must validate every declared outcome even when
    # the first one is already positive, otherwise missing-key diagnostics depend on key order.
    for selection in values(outcomes(m))
        active |= _checked_book_trust(trust, _market_probe(m, selection)) > 0.0
    end
    return active
end

"The declared markets that may enter payoff geometry under this BookSpec."
function _effective_markets(spec::BookSpec)
    trust = book_trust(spec)
    trust === nothing && return spec.markets.markets
    active = AbstractMarket[m for m in spec.markets.markets
                            if _market_has_positive_trust(trust, m)]
    isempty(active) && throw(ArgumentError(
        "BookSpec trust excises every declared market; retain at least one market with " *
        "positive trust or use `trust = nothing` for legacy geometry"))
    return active
end

_excise_zero_trust_markets(::Nothing, sels::Vector{Selection}) = sels

"""
Drop whole markets whose admitted selections all have exactly zero book-time trust.

This is deliberately market-level, not selection-level: if one direction has positive trust, all
of that market's columns remain in the payoff matrix. `(group, line)` identifies one market in the
same way quote extraction does. Selection order is preserved exactly for every retained market.
"""
function _excise_zero_trust_markets(trust::AbstractTrustModel, sels::Vector{Selection})
    active = Set{Tuple{String,Float64}}()
    for s in sels
        _checked_book_trust(trust, s) > 0.0 && push!(active, _book_market_key(s))
    end
    return Selection[s for s in sels if _book_market_key(s) in active]
end

_policy_market_active(trust::AbstractTrustModel, market::AbstractMarket) =
    _market_has_positive_trust(trust, market)
_policy_market_active(trust::ScheduledTrust, market::AbstractMarket) =
    any(model -> _policy_market_active(model, market), trust.per_slate)

"Refuse a policy that tries to activate a market the cached book excised."
function _validate_book_policy(spec::BookSpec, policy::PolicySpec)
    trust = book_trust(spec)
    trust === nothing && return nothing
    _effective_markets(spec)  # validates coverage, ranges, non-emptiness and causal trust
    for market in spec.markets.markets
        !_market_has_positive_trust(trust, market) &&
            _policy_market_active(policy.trust, market) && error(
                "PolicySpec trust activates $(market), but BookSpec trust excised that market. " *
                "Use compatible zero-trust patterns or rebuild with `BookSpec(trust = nothing)`.")
    end
    return nothing
end

"""
    extract_selections(odds_df, match_id, spec, model_probs) -> Vector{Selection}

Pull the closing price of every configured market for one match and price it.

A market group is admitted only if **every** one of its outcomes is quoted. This matters more
than it looks: the vig-removal step divides by the sum over whatever legs are present, so a
group missing a leg silently manufactures edge on the survivors -- up to 20% on a 1X2 market
missing one way. On ScottishLower ~70% of O/U 0.5 groups and 2 of 1522 1X2 groups are partial.
"""
function extract_selections(odds_df::DataFrame, match_id::Integer, spec::BookSpec,
                            model_probs::Dict)
    # `Integer`, not `Int`: match ids arrive as Int32 from `ds.matches` and Int64 from a
    # latents frame, and a caller composing the primitives by hand should not have to know which.
    rows = view(odds_df, odds_df.match_id .== match_id, :)
    out  = Selection[]
    isempty(rows) && return out

    for m in spec.markets.markets
        m_str  = string(m)
        grp    = Data.market_group(m)
        line   = Data.market_line(m)
        n_want = length(Data.outcomes(m))
        haskey(model_probs, m_str) || continue

        sub = view(rows, (rows.market_name .== grp) .&
                          isapprox.(rows.market_line, line; atol = 1e-3), :)
        isempty(sub) && continue

        quoted = Dict{Symbol,Float64}()
        for r in eachrow(sub)
            (ismissing(r.odds_close) || r.odds_close <= 1.0) && continue
            quoted[r.selection] = r.odds_close
        end
        (spec.exec.require_complete_markets && length(quoted) != n_want) && continue
        isempty(quoted) && continue

        overround = sum(1.0 / o for o in values(quoted))
        for (sel, o) in quoted
            haskey(model_probs[m_str], sel) || continue
            push!(out, Selection(selection_family(grp, line, sel), grp, line, sel,
                                 o,
                                 settlement_odds(spec.price, o, overround),
                                 mean(model_probs[m_str][sel]),
                                 (1.0 / o) / overround))
        end
    end
    return _excise_zero_trust_markets(book_trust(spec), out)
end

"Date, and final score when the fixture has been played."
const FixtureInfo = @NamedTuple{date::Date, score::Union{Nothing,Tuple{Int,Int}}}

"""
    build_book(spec, latents_row, expr, odds_df, fixtures; require_result = true) -> MatchBook | nothing

Returns `nothing` for any match we cannot stake: unknown fixture, no usable quotes, or a
score-matrix failure. Quotes are checked *before* the score matrix is computed, because that is
the expensive step.

With `require_result = false` an unplayed fixture is built with `settle = nothing`. Such a book
can be staked (that is match-day use) but not simulated -- `simulate` refuses it.
"""
function build_book(spec::BookSpec, latents_row, expr, odds_df::DataFrame,
                    fixtures::Dict{Int,FixtureInfo}; require_result::Bool = true)
    m_id = latents_row.match_id
    haskey(fixtures, m_id) || return nothing
    fx = fixtures[m_id]
    (require_result && fx.score === nothing) && return nothing
    any(==(m_id), odds_df.match_id) || return nothing

    model = _portfolio_model_of(expr)
    score_matrix = try
        Predictions.compute_score_matrix(model,
                                         Predictions.extract_params(model, latents_row))
    catch e
        e isa Predictions.NonMonotoneSmileError && rethrow()
        return nothing
    end

    model_probs = Dict(string(m) => Predictions.compute_market_probs(score_matrix, m)
                       for m in _effective_markets(spec))

    sels = extract_selections(odds_df, m_id, spec, model_probs)
    isempty(sels) && return nothing

    sm_data = Predictions.score_matrix_data(score_matrix)
    max_h, max_a, _ = size(sm_data)
    p_grid = vec(mean(sm_data, dims = 3)[:, :, 1])
    p_grid ./= sum(p_grid)                       # absorb grid truncation


    R   = payoff_matrix(sels, max_h, max_a, spec.exec.commission)
    res = allocate(spec.allocator, p_grid, R, spec.exec)
    k   = shrink_factor(spec.shrink, score_matrix, R, p_grid, spec.allocator, spec.exec;
                        seed_offset = m_id)

    settle = fx.score === nothing ? nothing :
             settle_vector(sels, fx.score[1], fx.score[2], spec.exec.commission)

    return MatchBook(m_id, fx.date, sels, p_grid, R, settle, res.a, k, res.kkt, res.converged)
end

"""
    fixture_table(ds) -> Dict{Int,FixtureInfo}

Kick-off date for every match, plus the final score where one exists. Built once and shared
across the threaded book build.
"""
function fixture_table(ds)
    out = Dict{Int,FixtureInfo}()
    for r in eachrow(ds.matches)
        sc = (ismissing(r.home_score) || ismissing(r.away_score)) ? nothing :
             (Int(r.home_score), Int(r.away_score))
        out[Int(r.match_id)] = (date = Date(r.match_date), score = sc)
    end
    return out
end

"""
    build_books(spec, latents_df, expr, odds_df, ds) -> Vector{MatchBook}

`require_result = false` admits unplayed fixtures, which is what match-day staking needs.

Threaded over matches. Returns books sorted by `(date, match_id)` -- chronological order is
established here, once, so nothing downstream has to remember to sort. Path metrics computed on
an unsorted series are meaningless, and the prototype's `latents.df` order was neither
chronological nor recoverable by sorting on `match_id`.
"""
function build_books(spec::BookSpec, latents_df::DataFrame, expr, odds_df::DataFrame,
                    fixtures::Dict{Int,FixtureInfo}; require_result::Bool = true)
    n   = nrow(latents_df)
    buf = Vector{Union{Nothing,MatchBook}}(undef, n)
    for i in 1:n
        buf[i] = build_book(spec, latents_df[i, :], expr, odds_df, fixtures;
                            require_result = require_result)
    end

    books = MatchBook[b for b in buf if b !== nothing]
    sort!(books, by = b -> (b.date, b.m_id))
    return books
end

"""
    build_books(spec, latents_df, expr, odds_df, ds; require_result = true)

Convenience method deriving the fixture table from a `DataStore`.

**This method can only ever build settled books.** `ds.matches` is the curated store of
*finished* matches, so `fixture_table(ds)` contains no entry whose score is `nothing` and an
upcoming fixture is absent from it entirely -- `build_book` then returns `nothing` for every
one. Passing `require_result = false` here is therefore a silent no-op that yields an empty
vector.

For match-day use pass a `Dict{Int,FixtureInfo}` built from the fixture list directly (see
`MatchDay.fixture_info`), which is the method above.
"""
build_books(spec::BookSpec, latents_df::DataFrame, expr, odds_df::DataFrame, ds;
            require_result::Bool = true) =
    build_books(spec, latents_df, expr, odds_df, fixture_table(ds); require_result = require_result)

"""
    component_hash(x, h = UInt(0)) -> UInt

Content hash of a configuration component: its type name plus its field values, recursively.

Julia's default `hash` for an immutable struct holding a non-isbits field falls back to
`objectid`, which is identity-based -- so two `BakerMcHale()` values built in the same session
hash differently. Hashing a spec directly therefore produces a key that never repeats and a
cache that never hits, silently turning every policy sweep back into a full rebuild.
"""
function component_hash(x, h::UInt = UInt(0))
    h = hash(string(nameof(typeof(x))), h)
    if x isa Union{Number,Symbol,AbstractString,Bool}
        return hash(x, h)
    elseif x isa AbstractDict
        # Dict iteration order is insertion-dependent. A trust table with the same semantic
        # entries must therefore hash the same regardless of how its caller assembled it.
        for key in sort!(collect(keys(x)); by = repr)
            h = component_hash(key, h)
            h = component_hash(x[key], h)
        end
    elseif x isa AbstractArray
        for v in x
            h = component_hash(v, h)
        end
    elseif x isa Tuple
        for v in x
            h = component_hash(v, h)
        end
    else
        for f in fieldnames(typeof(x))
            v = getfield(x, f)
            h = if v isa Union{Number,Symbol,AbstractString,Bool}
                    hash(v, h)
                elseif v isa AbstractArray && eltype(v) <: Union{Number,Symbol,AbstractString}
                    hash(collect(v), h)     # hash(::AbstractArray) is content-based
                else
                    component_hash(v, h)
                end
        end
    end
    return h
end

"""
    book_cache_key(spec) -> UInt

Content hash of everything that can change a `MatchBook`. Use it to name a serialised cache:
a `PolicySpec` sweep must never rebuild books.

Equal specs give equal keys -- asserted in `test/portfolio_tests.jl` for a spec carrying a
`BakerMcHale`, which is the case that breaks under a naive `hash`.
"""
function book_cache_key(spec::BookSpec)
    # Hash the underlying settlement policy, not the compatibility wrapper carrying book trust.
    # Default specs therefore retain their historical key exactly.
    h = component_hash(_book_price(spec.price))
    h = component_hash(spec.allocator, h)
    h = component_hash(spec.shrink, h)
    h = component_hash(spec.exec, h)
    # Trust magnitudes do not change a MatchBook; only the resulting market zero-pattern does.
    # Hashing effective markets keeps trust-weight sweeps cache-free and makes an excised extended
    # spec share the base spec's cache whenever their actual payoff geometry is identical.
    return hash(string.(_effective_markets(spec)), h)
end
