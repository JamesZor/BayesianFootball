# Phase 5 — all four frozen-seed reproductions PASS

Source **e7abd34e50bbf6ff0c9a68cff32b5e760846bb6a**; beast detached
checkout `/root/BF_runs/qs_experiment`. Fresh REPL per arm,16threads,pinned
cores,BLAS1, Julia1.12.4 (same observed grid toolchain; full environment audit
in `results/phase5/environment.csv`). No reference refits or new persisted runs.

## Exact scientific payload comparison

`r07_reproduce.jl` adapts wave1 `r06_reproduce.jl`'s chain/latent comparison
path to the pre-grid frozen `Wave2Grid.sample_pending!` queue. One fold per
new arm, **fold1**, attempt0 only, original4×(500warmup+1000retained) sampler.
Fresh scratch checkpoints, not original grid checkpoints. Load saved fits by
explicit final UUID, verify model/sampler equality, construct the same frozen
features, sample with the same per-chain task-local seed, then assemble with
the frozen postprocessing seed. No Harness.grid, save_fit or score writes.

Compare chain parameter names/shapes and posterior parameter arrays (wave1
Array(chain,append_chains=false) convention), plus aligned home/away latent
arrays. Raw Float64 bytes and SHA256 must be equal; max absolute gap must be0.
This is scientific payload equality, not timestamp-bearing Fit metadata or
whole-object serialisation equality. Every channel's two hashes/counts/seeds
are in committed `results/phase5/reproduction.csv` and per-arm CSVs.

| Arm | Fresh pane | UTC start–PASS | Chain values | Latent values | Byte identical | max abs |
|---|---|---|---:|---:|---|---:|
| grw_joint | %403 | 22:43:05–22:44:53 | 400000 | 160000 | true | 0 |
| qs_joint | %404 | 22:44:53–22:46:44 | 400000 | 160000 | true | 0 |
| grw_marketobs | %405 | 22:46:44–22:48:36 | 396000 | 160000 | true | 0 |
| qs_marketobs | %406 | 22:48:36–22:50:31 | 396000 | 160000 | true | 0 |

All on2026-10-09; launcher%402 reports PHASE5_REPRODUCTIONS_PASS and exit0.
No retry, tuning or favourable fold selection after results. Original chain
seed schedule:202700000+100000×arm+100×fold+10×attempt+chain; arm1–4 in frozen
order,attempt0,chains1–4. Postseed202700000+100000×arm+9000+attempt.

| Arm | Chain seeds | Postseed |
|---|---|---:|
| grw_joint | 202800101;202800102;202800103;202800104 | 202809000 |
| qs_joint | 202900101;202900102;202900103;202900104 | 202909000 |
| grw_marketobs | 203000101;203000102;203000103;203000104 | 203009000 |
| qs_marketobs | 203100101;203100102;203100103;203100104 | 203109000 |

## Original-grid and reference preservation

All160 original grid checkpoint SHA256 digests are unchanged, including both
joint arms. Before list: `results/phase4/grid_checkpoint_sha256_before_reproduction.txt`;
after list: `results/phase5/grid_checkpoint_sha256_after_reproduction.txt`.
The files are byte-identical. Reproductions live only under external
`/root/BF_runs/qs_experiment_w2_out/phase5/<arm>/checkpoints/`.

Fresh read-only `r09_final_audit.jl` repeats the Phase3 ownership/UUID/budget/
hard-gate audit into Phase5 outputs (never overwrites Phase3). Its final DB
runs/checks/diagnostic CSVs must match the original Phase3 audit bytes; no
new named grid run, 4runs/160folds/640chains/1000retained,16hard passes.
Final audit acceptance is recorded in `results/phase5/original_grid_unchanged.csv`.

Wave1 tracked files and saved reference outputs untouched. Phase4 exact
reference score/goal-fixture parity and1992-row reference CSV byte parity are
committed. Source provenance discrepancy in the wave1 full scorer is an
explicit separate follow-up in TODO041/DECISIONS_LOG, not patched here.

## Evidence and rerunning

Immutable cleaned logs: `results/gate_logs/phase5_launcher.log` and
`phase5_repro_<arm>.log`; final read-only audit/record log stored separately.
All reproduction REPLs and launcher are closed. To reproduce again requires
manager authorisation and a fresh output/log root: scripts deliberately refuse
existing evidence. Never rerun the grid or overwrite reference/reproduction
outputs. No ROI/staking; GRW ESS review flags remain unchanged.
