# Frozen-seed reproduction — Phase 6 PASS

On 2026-10-09, `r06_reproduce.jl` ran in a **fresh** 16-thread Julia REPL on
mcmc-beast, owned session `pi_qsx_repro`, pane `%341`, checkout
`ac04b17315d6a68dfe054dafc1c59b2b1cfb1df2`. Threads were pinned to physical cores;
BLAS used one thread. No fits were saved to the experiment database.

Pinned ScottishLower snapshot SHA256:
`c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`.

## Results

| Arm | Fold | Seed | Draw values compared | Chain/draw equality | Rate-latent equality | Max absolute difference | Seconds |
|---|---:|---|---:|---|---|---:|---:|
| control_grw | 1 | 4001 | 392,000 | exact | exact | 0.0 | 58.4 |
| control_td | 1 | 4004 | 200,000 | exact | exact | 0.0 | 10.3 |
| qs_market_r | 1 | 4002 | 392,000 | exact | exact | 0.0 | 25.7 |
| qs_weak_r | 1 | 4003 | 392,000 | exact | exact | 0.0 | 8.5 |
| market_c0 | 5 | 5611 / 5612 / 5613 / 5614 | 36,000 | exact | exact | 0.0 | 94.2 |
| market_c0 | 33 | 8411 / 8412 / 8413 / 8414 | 36,000 | exact | exact | 0.0 | 91.4 |

Market per-fixture `probabilities.csv` files also matched **byte-for-byte** for
both folds, including all priced selections and rate summaries. Fold 5 originally
ran on two threads; its 16-thread replay proves independence from that thread-count
change. Fold 33 originally ran on 16 threads.

Goal comparison uses equal chain parameter names and `isequal` on the full sampled
parameter arrays, preserving draw and chain order (not rounded posterior summaries).
It also compares both rate matrices after aligning fixture IDs to the saved grid
latents. Market comparison uses the complete unconstrained parameter draws and both
rate matrices, plus raw probability-file bytes. This is equality of numeric output,
not of whole serialized `Fit` files, whose metadata includes timestamps and duration.

## Why fold 1 reproduces with fewer queued tasks

Each arm was seeded before the harness built its inputs in the original grid.
The replay repeats that preamble, then supplies fold 1 alone to `fit_model` with the
unchanged 4 × (500 + 1000) W2 sampler. Its four chain tasks are spawns 1–4, the same
as in the full grid; Julia's task-local RNG splits in deterministic spawn order.
The exact chain comparison confirms this rather than assuming it.

Each market chain and each forecast state draw has an explicit frozen seed in
`market_arm.jl`. Chain seeds are the Phase C C0 seeds [5111, 5112, 5113, 5114]
plus **100 × fold index**. Both folds refit at the original 4 × (2000 + 3000) Phase C budget;
no saved fit was substituted for a fresh refit.

## Evidence and rerun

- Runner: `r06_reproduce.jl` (refuses a nonempty reproduction output directory).
- Machine-readable evidence: `results/reproduction.csv`.
- Beast log: `/root/BF_runs/logs/qs_experiment/phase6_repro.log`.
- Beast artifacts: `/root/BF_runs/qs_experiment_out/repro/`.
- End markers: `REPRO all_identical=true`, `REPRODUCTION_DONE`.

To repeat, use a new owned `pi_qsx_*` REPL with `--project -t 16` and set `QSX_OUT`
to a fresh root containing the original `market_grid/` artifacts (or a symlink to
them); then include `r06_reproduce.jl`. The original reproduction evidence must not
be overwritten.
