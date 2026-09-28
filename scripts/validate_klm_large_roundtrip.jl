#!/usr/bin/env julia
# Build-phase evidence only: reads production, writes/deletes ONLY mcmc_experiments_test.
using BayesianFootball
using BayesianFootball: Training, Harness
using Serialization
using UUIDs

include(joinpath(@__DIR__, "klm_test_db_guard.jl"))

const SOURCE_RUN = UUID("ce7ea22f-61d8-40cc-a505-94d101b98604")
const SOURCE_DB = Training.PostgresStorage("scottish_pyramid_tier_transition_w2")
const TARGET_URL = get(ENV, "BF_EXPERIMENTS_TEST_DB_URL", "")
isempty(strip(TARGET_URL)) && error("Set BF_EXPERIMENTS_TEST_DB_URL to the disposable test DB")
const TARGET_DB = Training.PostgresStorage(TARGET_URL, "klm_large_roundtrip")
assert_klm_test_database!(TARGET_DB)

function serial_size(value)
    io = IOBuffer()
    serialize(io, value)
    return position(io)
end

function peak_rss_kib()
    row = only(filter(line -> startswith(line, "VmHWM:"),
                      split(read("/proc/self/status", String), '\n')))
    return parse(Int, split(row)[2])
end

function main()
    println("Loading read-only source run $SOURCE_RUN")
    source = Training.load_fit(SOURCE_DB, SOURCE_RUN)
    n = length(source.folds)
    raw_fold_bytes = sum(serial_size, source.folds; init = 0)
    target_bytes = ceil(Int, 1.2 * 1024^3)
    copies = max(2, cld(target_bytes + 1, raw_fold_bytes))
    folds = Training.FoldFit[]
    for repeat_idx in 1:copies, fold in source.folds
        push!(folds, Training.FoldFit((repeat_idx - 1) * n + fold.fold,
                                     deepcopy(fold.chain), fold.meta))
    end
    raw_bytes = sum(serial_size, folds; init = 0)
    raw_bytes > target_bytes || error("Only $raw_bytes serialized fold bytes; need > $target_bytes")
    cfg = Training.FitConfig(name = "klm_large_$(uuid4())", model = source.config.model,
        splitter = source.config.splitter, sampler = source.config.sampler,
        execution = source.config.execution, tags = source.config.tags,
        description = source.config.description, save_dir = source.config.save_dir)
    original = Training.Fit(cfg, folds, source.latents, source.diagnostics,
                            source.metadata, source.save_path)
    Training.ensure_schema!(TARGET_DB)
    run_id = nothing
    try
        GC.gc()
        save_start = time()
        run_id = Training.save_fit(original, TARGET_DB; on_duplicate = :error)
        save_seconds = time() - save_start
        conn = Training.Inference._db_connect(TARGET_DB)
        largest_blob, compressed_bytes = try
            rows = Training.Inference._db_rows(conn, """
                SELECT max(octet_length(fold_blob))::bigint AS n,
                       sum(octet_length(fold_blob))::bigint AS total, count(*)::int AS folds
                FROM fit_fold_artifacts WHERE run_id = \$1::uuid;
            """, (string(run_id),))
            rows.folds[1] == length(folds) || error("Stored $(rows.folds[1]) folds, expected $(length(folds))")
            (Int(rows.n[1]), Int(rows.total[1]))
        finally
            close(conn)
        end
        GC.gc()
        load_start = time()
        reloaded = Training.load_fit(TARGET_DB, run_id)
        load_seconds = time() - load_start
        length(reloaded.folds) == length(original.folds) || error("fold count changed")
        for (left, right) in zip(original.folds, reloaded.folds)
            left.fold == right.fold || error("fold index changed")
            Harness._chain_parity(left.chain, right.chain) || error("chain changed on fold $(left.fold)")
            Harness._structural_equal(left.meta, right.meta) || error("fold metadata changed")
        end
        for field in (:config, :diagnostics, :metadata)
            Harness._structural_equal(getfield(original, field), getfield(reloaded, field)) ||
                error("$field changed")
        end
        Harness._structural_equal(original.latents, reloaded.latents) || error("latents changed")
        original.save_path == reloaded.save_path || error("save_path changed")
        println("KLM_LARGE_PASS serialized_fold_bytes=$raw_bytes compressed_fold_bytes=$compressed_bytes " *
                "folds=$(length(folds)) largest_fold_blob=$largest_blob " *
                "save_seconds=$(round(save_seconds; digits=2)) " *
                "load_seconds=$(round(load_seconds; digits=2)) peak_rss_kib=$(peak_rss_kib())")
    finally
        if run_id !== nothing
            conn = Training.Inference._db_connect(TARGET_DB)
            try
                Training.Inference._db_exec(conn,
                    "DELETE FROM runs WHERE run_id = \$1::uuid;", (string(run_id),))
            finally
                close(conn)
            end
        end
    end
end

main()
