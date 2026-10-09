# Diagnostic for the prescribed 8-week NONCIRCULAR block bootstrap (R07 scheme): does it
# centre on the point estimate? Edge weeks of each season are covered by fewer blocks, so the
# replicate mean can drift if early/late-season deltas differ. Reports, for each 1X2/OU2.5
# pair: point estimate, noncircular replicate mean and 90% CI (must reproduce
# paired_logloss.csv), and a CIRCULAR block bootstrap (uniform week coverage) as a
# sensitivity. Classification in the report remains the prescribed one. Run in the Phase 5
# REPL (needs s5_frames, s5_pairs_spec, s5_ll, s5_week, classify).
using CSV, DataFrames, Dates, Random, Statistics

function bc_bootstrap(fixture_ids, sums, counts; circular::Bool, B = 999, block = 8,
                      seed = 20261009)
    weeks = Dict{Tuple{String,Date},Vector{Int}}()
    for (i, id) in enumerate(fixture_ids)
        push!(get!(weeks, s5_week(id), Int[]), i)
    end
    seasons = Dict{String,Vector{Vector{Int}}}()
    for key in sort!(collect(keys(weeks)))
        push!(get!(seasons, key[1], Vector{Int}[]), weeks[key])
    end
    rng = Random.Xoshiro(seed)
    stats = zeros(B)
    for b in 1:B
        total, n = 0.0, 0.0
        for season in sort!(collect(keys(seasons)))
            ws = seasons[season]
            W = length(ws)
            L = min(block, W)
            chosen = Int[]
            while length(chosen) < W
                if circular
                    s = rand(rng, 1:W)
                    append!(chosen, [mod1(s + k, W) for k in 0:(L - 1)])
                else
                    s = rand(rng, 1:(W - L + 1))
                    append!(chosen, s:(s + L - 1))
                end
            end
            for w in chosen[1:W], i in ws[w]
                total += sums[i]
                n += counts[i]
            end
        end
        stats[b] = total / n
    end
    return (; mean = mean(stats), lo = quantile(stats, 0.05), hi = quantile(stats, 0.95))
end

bc_rows = NamedTuple[]
for (tier, a, b) in s5_pairs_spec, family in ("1X2", "OU2.5")
    fa, fb = s5_ll(s5_frames[a], family), s5_ll(s5_frames[b], family)
    j = innerjoin(select(fa, :match_id, :selection, :ll_model => :la),
                  select(fb, :match_id, :selection, :ll_model => :lb); on = [:match_id, :selection])
    j.d = j.la .- j.lb
    perfix = combine(groupby(j, :match_id), :d => sum => :d, nrow => :n)
    point = sum(perfix.d) / sum(perfix.n)
    nc = bc_bootstrap(perfix.match_id, perfix.d, perfix.n; circular = false)
    cc = bc_bootstrap(perfix.match_id, perfix.d, perfix.n; circular = true)
    push!(bc_rows, (; tier, arm = a, reference = b, market = family, point,
        noncircular_mean = nc.mean, noncircular_lo90 = nc.lo, noncircular_hi90 = nc.hi,
        noncircular_class = classify(nc.lo, nc.hi),
        circular_mean = cc.mean, circular_lo90 = cc.lo, circular_hi90 = cc.hi,
        circular_class = classify(cc.lo, cc.hi)))
end
bc = DataFrame(bc_rows)
CSV.write(joinpath(S5_OUT, "block_bootstrap_check.csv"), bc)
println("BLOCK_CHECK max|noncircular_mean-point|=", maximum(abs.(bc.noncircular_mean .- bc.point)),
        " max|circular_mean-point|=", maximum(abs.(bc.circular_mean .- bc.point)))
println("BLOCK_CHECK_DONE")
