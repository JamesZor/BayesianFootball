#!/usr/bin/env julia
#
# Bytes per compiled gradient, per recipe and fold, with the allocating tape instructions named.
#
#   julia --project -t 1 scripts/tape_allocation_audit.jl <candidates.jl> [<candidates.jl> ...]
#         [--only name,...] [--folds largest,opening] [--profile]
#         [--save-parity FILE] [--compare-parity FILE] [--out FILE.tsv]
#
# For each candidate and each requested fold it:
#   1. builds the fold exactly as the harness does (scoped walk-forward, `create_features`);
#   2. links a seeded prior draw (NUTS's unconstrained space) and records + compiles the tape;
#   3. reports bytes per `gradient!` — the MINIMUM over 20 warmed replays behind a function
#      barrier (src/harness/tape_profile.jl) — and the warmed-minimum latency;
#   4. with `--profile` (implied when bytes > 0) lists every allocating instruction.
#
# `largest` is the last fold (largest training set). `opening` is the first fold of the scope's
# last target season: a GRW there has no observed target steps, so it exercises the no-target
# submodels and, with `target_season_step`, the zero-target boundary path.
#
# PARITY. `--save-parity` writes the log-density, compiled-tape gradient and fresh ReverseDiff
# gradient at the linked prior draw and at θ + δ·sin(i) for δ ∈ (0.003, -0.8, 0.8) — the
# sampler-scale probe of docs/turing_ad_performance_guide.md §10.3. `--compare-parity` reloads such
# a file (written by the PRE-fix code), evaluates the SAME θ points and prints the worst relative
# error. The recipe hash is saved and compared alongside.
#
# This file does not depend on anything added to `src/` by the zero-allocation fix: the profiling
# helpers are included from src/harness/tape_profile.jl by path, so it can be pointed at an older
# checkout with `julia --project=<old checkout> <this checkout>/scripts/tape_allocation_audit.jl`.

using BayesianFootball
using DynamicPPL
using LinearAlgebra
using LogDensityProblems
using Printf
using Random
using ReverseDiff
using Serialization

LinearAlgebra.BLAS.set_num_threads(1)

module TapeProfile
import ReverseDiff
include(joinpath(@__DIR__, "..", "src", "harness", "tape_profile.jl"))
end

const PG = BayesianFootball.Models.PreGame
const SEED = 20260927
const DISPLACEMENTS = (0.0, 0.003, -0.8, 0.8)

function parse_args(args)
    files = String[]
    only = nothing
    folds = [:largest, :opening]
    profile = false
    save_parity = nothing
    compare_parity = nothing
    out = nothing
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--only"
            only = Set(String.(split(args[i + 1], ",")))
            i += 2
        elseif arg == "--folds"
            folds = Symbol.(split(args[i + 1], ","))
            all(in((:largest, :opening)), folds) || error("--folds takes largest,opening")
            i += 2
        elseif arg == "--profile"
            profile = true
            i += 1
        elseif arg == "--save-parity"
            save_parity = args[i + 1]
            i += 2
        elseif arg == "--compare-parity"
            compare_parity = args[i + 1]
            i += 2
        elseif arg == "--out"
            out = args[i + 1]
            i += 2
        elseif startswith(arg, "-")
            error("unknown option $arg")
        else
            push!(files, arg)
            i += 1
        end
    end
    isempty(files) && error("usage: tape_allocation_audit.jl <candidates.jl>... [options]")
    return (; files, only, folds, profile, save_parity, compare_parity, out)
end

function load_candidates(path::AbstractString)
    mod = Module(Symbol("Candidates_", basename(dirname(abspath(path)))))
    Base.include(mod, abspath(path))
    return Base.invokelatest(getproperty, mod, :CANDIDATES)
end

"Scoped fold selection, identical to the harness's `_fold_inputs`."
function fold_feature_set(candidate, raw_ds, which::Symbol)
    ds = BayesianFootball.Data.apply_scope(raw_ds, candidate.scope)
    splitter = BayesianFootball.Data.ScopedWalkForwardCV(candidate.scope)
    bounds = BayesianFootball.Data.create_id_boundaries(ds, splitter)
    index = if which === :largest
        length(bounds)
    else
        season = last(candidate.scope.target_seasons)
        first_of_season = findfirst(b -> b[2].target_season == season, bounds)
        first_of_season === nothing && error("$(candidate.name): no fold in target season $season")
        first_of_season
    end
    fs = first(BayesianFootball.Features.create_features([bounds[index]], ds, candidate.model,
                                                         splitter))[1]
    return (; fs, index, n_folds = length(bounds))
end

relerr(a, b) = norm(a .- b) / max(norm(a), norm(b), 1.0)

"Linked prior draw, compiled tape, bytes, latency — everything behind one function barrier."
function audit_fold(model, fs; profile::Bool)
    tm = PG.build_turing_model(model, fs)
    Random.seed!(SEED)
    vi = DynamicPPL.link!!(DynamicPPL.VarInfo(tm), tm)
    θ = copy(vi[:])
    ld = DynamicPPL.LogDensityFunction(tm, DynamicPPL.getlogjoint_internal, vi)
    f = x -> LogDensityProblems.logdensity(ld, x)
    raw = ReverseDiff.GradientTape(f, θ)
    tape = ReverseDiff.compile(raw)
    g = similar(θ)
    bytes = TapeProfile.compiled_gradient_bytes(tape, g, θ)
    best = minimum(@elapsed(ReverseDiff.gradient!(g, tape, θ)) for _ in 1:100)

    points = [θ .+ δ .* sin.(eachindex(θ)) for δ in DISPLACEMENTS]
    parity = map(points) do p
        compiled = similar(p)
        ReverseDiff.gradient!(compiled, tape, p)
        (; θ = p, logdensity = f(p), compiled, fresh = ReverseDiff.gradient(f, p))
    end
    rows = (profile || bytes > 0) ? TapeProfile.tape_allocation_profile(raw) : NamedTuple[]
    return (; n_rows = length(fs.data[:flat_home_ids]), n_params = length(θ),
              instructions = length(raw.tape), bytes, gradient_ms = 1e3 * best,
              rows, parity, f, tape)
end

"Evaluate the saved θ points under THIS code and return the worst relative errors."
function compare_parity(result, reference)
    worst_ld = 0.0
    worst_compiled = 0.0
    worst_fresh = 0.0
    g = similar(first(reference.points).θ)
    for (i, ref) in enumerate(reference.points)
        length(ref.θ) == length(g) || error("θ length changed: $(length(ref.θ)) → $(length(g))")
        ld = result.f(ref.θ)
        ReverseDiff.gradient!(g, result.tape, ref.θ)
        fresh = ReverseDiff.gradient(result.f, ref.θ)
        worst_ld = max(worst_ld, abs(ld - ref.logdensity) / max(abs(ld), abs(ref.logdensity), 1.0))
        worst_compiled = max(worst_compiled, relerr(g, ref.compiled))
        worst_fresh = max(worst_fresh, relerr(fresh, ref.fresh))
    end
    return (; worst_ld, worst_compiled, worst_fresh)
end

function main()
    opts = parse_args(ARGS)
    reference = opts.compare_parity === nothing ? nothing : deserialize(opts.compare_parity)
    saved = Dict{String,Any}()
    table = NamedTuple[]
    println("git: ", BayesianFootball.Training.git_commit_id(), "  threads: ", Threads.nthreads())
    raw_ds = BayesianFootball.Data.load_datastore_cached(
        BayesianFootball.Data.ScottishPyramid(); max_age_hours = 10_000)

    for file in opts.files, candidate in load_candidates(file)
        opts.only === nothing || candidate.name in opts.only || continue
        hash = BayesianFootball.Harness.recipe_hash(candidate)
        for which in opts.folds
            key = "$(candidate.name)|$(which)"
            fold = fold_feature_set(candidate, raw_ds, which)
            r = audit_fold(candidate.model, fold.fs; profile = opts.profile)
            @printf("AUDIT %-32s %-8s fold=%2d/%-2d rows=%5d params=%5d inst=%4d bytes=%8d grad_ms=%.4f hash=%s\n",
                    candidate.name, which, fold.index, fold.n_folds, r.n_rows, r.n_params,
                    r.instructions, r.bytes, r.gradient_ms, hash[1:12])
            for row in r.rows
                @printf("    #%-4d %8d B  %s\n", row.index, row.bytes, row.description)
            end
            points = [(; θ = p.θ, logdensity = p.logdensity, compiled = p.compiled, fresh = p.fresh)
                      for p in r.parity]
            saved[key] = (; hash, points, bytes = r.bytes, n_params = r.n_params)
            parity = (; worst_ld = NaN, worst_compiled = NaN, worst_fresh = NaN)
            hash_before = ""
            if reference !== nothing
                haskey(reference, key) || error("no reference entry for $key")
                ref = reference[key]
                hash_before = ref.hash
                parity = compare_parity(r, ref)
                @printf("PARITY %-32s %-8s hash_before=%s hash_after=%s same=%s bytes_before=%d bytes_after=%d ld=%.3e compiled=%.3e fresh=%.3e\n",
                        candidate.name, which, ref.hash[1:12], hash[1:12], ref.hash == hash,
                        ref.bytes, r.bytes, parity.worst_ld, parity.worst_compiled, parity.worst_fresh)
            end
            push!(table, (; candidate = candidate.name, fold = String(which), fold_index = fold.index,
                          rows = r.n_rows, params = r.n_params, instructions = r.instructions,
                          bytes = r.bytes, gradient_ms = r.gradient_ms, recipe_hash = hash,
                          recipe_hash_before = hash_before, parity...))
            flush(stdout)
        end
    end
    opts.save_parity === nothing || serialize(opts.save_parity, saved)
    if opts.out !== nothing
        open(opts.out, "w") do io
            println(io, join(keys(first(table)), '\t'))
            for row in table
                println(io, join(values(row), '\t'))
            end
        end
    end
    println("AUDIT_DONE")
end

main()
