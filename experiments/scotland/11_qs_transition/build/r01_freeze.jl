# Freeze the expanded market table: retain every original wave-2 value exactly.
# %% 1. Runtime / packages
using BayesianFootball, CSV, DataFrames, SHA, Serialization, LinearAlgebra
BLAS.set_num_threads(1)
strip(read(`hostname`, String)) == "mcmc-beast" || error("beast only")

# %% 2. Inputs / data
s2root = dirname(@__DIR__)
s2w2 = normpath(joinpath(s2root, "..", "06_qs_joint_and_market_observation"))
s2old = CSV.read(joinpath(s2w2, "results", "market_rates.csv"), DataFrame)
s2quotes = CSV.read(joinpath(s2root, "eda", "results", "accepted_quotes.csv"), DataFrame)
s2pyramid = Serialization.deserialize("/root/BF_runs/a76a65df/.cache/datastore_ScottishPyramid.jls")
s2matches = filter(r -> r.tournament_id in (56,57) && r.season in ("21/22","22/23","23/24","24/25","25/26"), s2pyramid.matches)
s2byid = Dict(r.match_id => r for r in eachrow(s2quotes))
s2oldbyid = Dict(r.match_id => r for r in eachrow(s2old))

# %% 3. Frozen construction / identity checks
s2rows = NamedTuple[]
for r in eachrow(sort(s2matches, :match_id))
    if haskey(s2oldbyid, r.match_id)
        old = s2oldbyid[r.match_id]
        push!(s2rows, (; match_id=Int(r.match_id), log_lambda_h=old.log_lambda_h,
            log_lambda_a=old.log_lambda_a, full_book=old.full_book))
    else
        r.season == "21/22" || error("non-21/22 row absent from wave-2 table: $(r.match_id)")
        q = get(s2byid, r.match_id, nothing)
        full = q !== nothing && q.n_selections >= 5
        push!(s2rows, (; match_id=Int(r.match_id), log_lambda_h=full ? log(q.lambda_h) : 0.0,
            log_lambda_a=full ? log(q.lambda_a) : 0.0, full_book=full))
    end
end
s2table = DataFrame(s2rows)
all(r -> haskey(s2oldbyid, r.match_id) ?
    isequal(r.log_lambda_h, s2oldbyid[r.match_id].log_lambda_h) &&
    isequal(r.log_lambda_a, s2oldbyid[r.match_id].log_lambda_a) &&
    r.full_book == s2oldbyid[r.match_id].full_book : true, eachrow(s2table)) || error("legacy values changed")
length(intersect(Set(s2table.match_id), Set(s2old.match_id))) == nrow(s2old) || error("lost wave-2 rows")

# %% 4. Artifacts
CSV.write(joinpath(@__DIR__, "market_rates.csv"), s2table)
s2hash = bytes2hex(open(SHA.sha256, joinpath(@__DIR__, "market_rates.csv")))
write(joinpath(@__DIR__, "market_rates.sha256"), s2hash * "\n")
s2coverage = combine(groupby(leftjoin(s2matches[:,[:match_id,:season]], s2table; on=:match_id), :season),
    nrow => :fixtures, :full_book => sum => :full_books)
CSV.write(joinpath(@__DIR__, "market_coverage.csv"), s2coverage)
println("S2_MARKET_FROZEN rows=$(nrow(s2table)) full=$(count(s2table.full_book)) legacy_rows=$(nrow(s2old)) sha=$s2hash")
