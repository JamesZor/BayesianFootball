# r04_extend_grw_spfl_joint.jl — brief 04 step 2: the ONE approved sampling job.
#
# QUESTION
#   Held-out 26/27 predictions for grw_spfl_joint (W1 run 82e01a3b), made the way the W1 folds
#   were made, so r04_live_orderbook.jl can price this season's League One/Two fixtures.
#
# DESIGN
#   model/sampler  the persisted Fit's own `config.model` and `config.sampler` (NUTS 1000 draws,
#                  500 warmup, 4 chains; the W1 artefact was persisted at stride 2, the fresh
#                  folds are kept at full length), `QueuedExecution(16)`.
#   splitter       the run's `ScopedWalkForwardCV` with "26/27" appended to `target_seasons`.
#                  Positions 1–40 are checked to hold out exactly the persisted fixtures; only the
#                  new positions are sampled. As in W1, a fold holds out the next BIWEEKLY clock
#                  step and trains on every in-scope fixture on a calendar day strictly before the
#                  step's first held-out day.
#   data           a DataStore rebuilt from betdb today (out/r04_datastore_ScottishPyramid.jls,
#                  `Data.load_datastore_sql(ScottishPyramid())`); the cached .jls predates 26/27.
#   persistence    NONE in mcmc_experiments. The extension is serialised to
#                  out/r04_ext_grw_spfl_joint.jls on the beast (folds, diagnostics, latents).
#
# USAGE (mcmc-beast, nothing else heavy running):
#   julia --project -t 16 current_development/clv_napkin/r04_extend_grw_spfl_joint.jl \
#       > /root/BF_runs/logs/clv_napkin/r04_extend.log 2>&1

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================

using LinearAlgebra
include(joinpath(@__DIR__, "l04_live_orderbook.jl"))
try
    @eval using ThreadPinning
    pinthreads(:cores)
catch err
    @warn "thread pinning unavailable" err
end
LinearAlgebra.BLAS.set_num_threads(1)
println("threads: ", Threads.nthreads(), "  started ", now())

# %%
# ===================================================================
# 2. The persisted fit and a fresh DataStore
# ===================================================================

db = Training.PostgresStorage(C4_EXPERIMENT)
fit = Training.load_fit(db, C4_RUN)
println("fit: ", fit.config.name, "  folds ", length(fit.folds), "  sampler ", fit.config.sampler,
        "  execution ", fit.config.execution)
raw_ds = deserialize(joinpath(C4_OUT, "r04_datastore_ScottishPyramid.jls"))
println("datastore: ", nrow(raw_ds.matches), " matches, last ", maximum(raw_ds.matches.match_date))

# %%
# ===================================================================
# 3. The plan (no sampling yet) and the filtration checks
# ===================================================================

plan = c4_extension_plan(fit, raw_ds)
println("new positions: ", plan.delta, " of ", length(plan.boundaries))
filtration = c4_filtration(plan)
show(stdout, MIME"text/plain"(), filtration; allrows = true, allcols = true); println()
CSV.write(joinpath(@__DIR__, "out", "r04_folds.csv"), filtration)

# %%
# ===================================================================
# 4. Sample the new folds (the one approved job)
# ===================================================================

ext = c4_extend_inmemory(fit, plan)
println("sampled ", length(ext.folds), " folds in ", round(ext.elapsed / 60, digits = 1), " min")
serialize(joinpath(C4_OUT, "r04_ext_grw_spfl_joint.jls"), ext)

# %%
# ===================================================================
# 5. Convergence (rerun trigger: R̂ > 1.05 or divergences > 0.1%)
# ===================================================================

conv = c4_convergence_table(ext)
show(stdout, MIME"text/plain"(), conv; allrows = true, allcols = true); println()
println("run-level verdict: passed = ", ext.diagnostics.passed, "  failures: ", ext.diagnostics.failures)
CSV.write(joinpath(@__DIR__, "out", "r04_convergence.csv"), conv)
println("held-out latents: ", ext.latents === nothing ? 0 : length(ext.latents.match_ids))
println("DONE r04_extend_grw_spfl_joint ", now())
