# l02_clv_leadtime.jl — loader for the lead-time curve (REPORT.md §10, r02_clv_leadtime.jl).
#
# Builds on l01 (books, selection frame, bootstrap). Adds: a lead-scaled staleness gate, a
# placebo whose TWA window is anchored to each entry instant (including the per-market
# "earliest" instant), market liquidity at an instant, paired key sets and a signed ASCII bar
# chart. Same filtration contract as l01: nothing at an entry instant looks past it; the close
# is used only for CLV.

include(joinpath(@__DIR__, "l01_clv_napkin.jl"))

"Staleness gate for an entry at `lead_minutes` (< 0): max(90 min, 25% of the lead)."
cl_max_staleness(lead_minutes::Real) = max(90.0, 0.25 * abs(lead_minutes))

"Set of selection keys in a frame."
cl_keys(f::AbstractDataFrame) = Set(zip(f.match_id, f.market_name, f.market_line, f.selection))

"Rows of `f` whose selection key is in `keys`."
cl_restrict(f::AbstractDataFrame, keys::Set) =
    f[[k in keys for k in zip(f.match_id, f.market_name, f.market_line, f.selection)], :]

"""
    cl_placebo_book_at(ds, as_of; span = 155.0) -> (book, refusals)

The l01 placebo re-anchored to an instant: de-vigged TWA of the market's own prices over
[as_of − span, as_of]. At as_of = −25 this is exactly the l01 placebo window [−180, −25].
"""
cl_placebo_book_at(ds, as_of::Real; span::Real = 155.0) =
    cn_placebo_book(ds; window = (Float64(as_of) - span, Float64(as_of)))

"""
    cl_placebo_earliest(ds, early_book; span = 155.0) -> (book, refusals)

Placebo for the per-market "earliest" instant t0: per selection, the time-weighted average of
the last-traded-price step function over [t0 − span, t0] (a tick before the window carries in;
no tick after t0 is read). By construction the last runner's first trade is at t0, so that
runner's TWA is its t0 price; the placebo is degenerate for it.
"""
function cl_placebo_earliest(ds, early_book::AbstractDataFrame; span::Real = 155.0)
    t0 = Dict((r.match_id, r.market_name, r.market_line) => r.entry_minutes for r in eachrow(early_book))
    odds = ds.betfair_odds
    keep = [haskey(t0, (Int(m), String(n), Float64(l)))
            for (m, n, l) in zip(odds.match_id, odds.market_name, odds.market_line)]
    sub = odds[keep, :]
    rows = NamedTuple[]
    for g in groupby(sub, CN_MKEY)
        b = t0[(Int(first(g.match_id)), String(first(g.market_name)), Float64(first(g.market_line)))]
        a = b - span
        for s in groupby(g, :selection)
            ok = (s.minutes_to_kickoff .<= b) .& isfinite.(s.traded_price) .& (s.traded_price .> 1.0)
            any(ok) || continue
            xs = Float64.(s.minutes_to_kickoff[ok]); ps = Float64.(s.traded_price[ok])
            o = sortperm(xs); xs = xs[o]; ps = ps[o]
            num = 0.0; den = 0.0
            for i in eachindex(xs)
                lo = max(xs[i], a)
                hi = i < length(xs) ? min(xs[i + 1], b) : b
                hi > lo || continue
                num += ps[i] * (hi - lo); den += hi - lo
            end
            push!(rows, (; match_id = Int(first(g.match_id)), market_name = String(first(g.market_name)),
                         market_line = Float64(first(g.market_line)), selection = Symbol(first(s.selection)),
                         odds_twa = den > 0 ? num / den : ps[end]))
        end
    end
    book, refused = cn_devig(DataFrame(rows); price_col = :odds_twa)
    return select(book, CN_KEY..., :p_fair => :p_placebo), refused
end

"""
    cl_market_liquidity(ds, as_of; volume_col = nothing, volume_agg = :sum) -> DataFrame

Per market (match, market, line): the number of price rows at or before `as_of` across its
runners (`ticks`) and, when the odds frame carries a volume column, the traded volume up to
`as_of` (`volume`: summed per-tick increments, or the per-runner maximum when the column is a
cumulative total). Nothing after `as_of` is read.
"""
function cl_market_liquidity(ds, as_of::Real; volume_col = nothing, volume_agg::Symbol = :sum)
    odds = ds.betfair_odds
    sub = odds[odds.minutes_to_kickoff .<= as_of, :]
    rows = NamedTuple[]
    for g in groupby(sub, CN_MKEY)
        vol = NaN
        if volume_col !== nothing
            v = coalesce.(g[!, volume_col], 0.0)
            vol = volume_agg == :sum ? Float64(sum(v)) :
                  Float64(sum(maximum(coalesce.(s[!, volume_col], 0.0)) for s in groupby(g, :selection)))
        end
        push!(rows, (; match_id = Int(first(g.match_id)), market_name = String(first(g.market_name)),
                     market_line = Float64(first(g.market_line)), ticks = nrow(g), volume = vol))
    end
    return DataFrame(rows)
end

"Tercile label (1 = thinnest) of `x` given the two cut points."
cl_tercile(x, cuts) = x <= cuts[1] ? 1 : x <= cuts[2] ? 2 : 3

"""
    cl_signed_bars(labels, series; width = 22) -> String

Rows of labels with one signed bar per series (a vector of (name, values, fmt)). Each series
has its own scale; `│` marks zero, bars grow right for positive and left for negative values.
"""
function cl_signed_bars(labels, series; width::Int = 22)
    io = IOBuffer()
    lw = maximum(length.(labels))
    half = width ÷ 2
    header = rpad("", lw) * "  " * join((rpad(name, width + 10) for (name, _, _) in series), "")
    println(io, rstrip(header))
    for (i, lab) in enumerate(labels)
        line = rpad(lab, lw) * "  "
        for (_, vals, fmt) in series
            m = maximum(abs, filter(isfinite, vals); init = 1e-9)
            v = vals[i]
            k = isfinite(v) ? round(Int, abs(v) / m * half) : 0
            left = v < 0 ? " "^(half - k) * "█"^k : " "^half
            right = v > 0 ? "█"^k * " "^(half - k) : " "^half
            line *= left * "│" * right * " " * rpad(isfinite(v) ? fmt(v) : "—", 8)
        end
        println(io, rstrip(line))
    end
    return String(take!(io))
end
