# Read-only: why does qs_weak_r's saved config hash differ from its candidate's?
using BayesianFootball, UUIDs
const HDC = Module(:QSHashCandidates)
Base.include(HDC, joinpath(@__DIR__, "candidates.jl"))
hd_db = Training.PostgresStorage(HDC.EXPERIMENT)
hd_runs = Dict("qs_weak_r" => "21f2a9f9-b96f-4034-97de-767704a9d54a",
               "qs_market_r" => "b18ae74b-9bc1-4cfa-b363-a640131adb2d")
for (name, id) in hd_runs
    saved = Training.load_fit(hd_db, UUID(id)).config
    fresh = Harness.fit_config(only(filter(c -> c.name == name, HDC.CANDIDATES));
                               stage = :grid, experiment = HDC.EXPERIMENT)
    tags(c) = join(Training.Inference._db_recipe_tags(c.tags), "|")
    for (part, f) in (("name", c -> c.name), ("model", c -> string(c.model)),
                      ("splitter", c -> string(c.splitter)), ("sampler", c -> string(c.sampler)),
                      ("execution", c -> string(c.execution)), ("tags", tags),
                      ("description", c -> c.description))
        a, b = f(saved), f(fresh)
        if a == b
            println("HASHDIFF $name $part same")
        else
            i = something(findfirst(k -> k > min(length(a), length(b)) || a[k] != b[k],
                                    collect(eachindex(a))), min(length(a), length(b)) + 1)
            lo = max(1, prevind(a, i, 60))
            println("HASHDIFF $name $part DIFFERS at char $i")
            println("  saved: ", a[lo:min(end, nextind(a, i, 120))])
            println("  fresh: ", b[max(1, prevind(b, min(i, lastindex(b)), 60)):min(end, nextind(b, min(i, lastindex(b)), 120))])
        end
    end
end
println("HASH_DIFF_DONE")
