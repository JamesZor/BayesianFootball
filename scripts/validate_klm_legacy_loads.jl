#!/usr/bin/env julia
# Read-only build-phase compatibility check. Does not call ensure_schema!.
using BayesianFootball
using BayesianFootball: Training
using UUIDs

const CASES = [
    ("W2 grw_base stride 2", "scottish_pyramid_tier_transition_w2",
     UUID("ce7ea22f-61d8-40cc-a505-94d101b98604")),
    ("W2 grw_step_a2 stride 4", "scottish_pyramid_tier_transition_w2",
     UUID("c4a0fa94-42bf-4c1c-99bd-7511f0272062")),
    ("W0 m12_td", "scottish_lower_joint_player_2426",
     UUID("132df5c2-c742-4e95-8693-3aeb2b2cbaef")),
    ("pre-PR42 negbin", "scottish_lower_negbin_2426",
     UUID("aad544b3-a614-4e5a-9873-6338a77f4ddf")),
]

for (label, experiment, id) in CASES
    db = Training.PostgresStorage(experiment)
    db.dbname == "mcmc_experiments" || error("Read-only legacy source is not production")
    started = time()
    fit = Training.load_fit(db, id)
    dims = [size(f.chain) for f in fit.folds]
    println("KLM_LEGACY_PASS label=$label run=$id folds=$(length(fit.folds)) " *
            "first_chain=$(first(dims)) last_chain=$(last(dims)) " *
            "seconds=$(round(time() - started; digits=2))")
    fit = nothing
    GC.gc()
end
