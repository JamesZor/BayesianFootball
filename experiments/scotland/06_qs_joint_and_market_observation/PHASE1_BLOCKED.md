# Phase 1 — BLOCKED (2026-10-09)

Source `351b8ddd` (component source `27c53ede`). Fresh owned beast REPL `%360`,
16 pinned threads, BLAS 1. Table preparation failed at 15:47:15 UTC after
50.78477120399475 s include wall time:

```text
LoadError: inversion provenance differs
QSMarketArm.prepare_panel — wave-1 market_arm.jl:57
Wave2MarketTable.prepare — l01_market_table.jl:13
r01_prepare_table.jl:13
```

The guard failed while reading the saved wave-1 panel, before a new frozen
rates table/digest was written. No AD test, allocation test on the new observation,
synthetic recovery, smoke, grid or scoring has run. Phase 1 is not accepted.
No threshold, manifest, reference output or snapshot was changed; no retry.

## Diagnosis for manager

The saved manifest (copied unchanged into `logs/wave1_inversion_manifest.txt`)
contains the pinned snapshot SHA and the representation of:

`Main.QSMarketArm.MarketModel.MarketModelConfig{BayesianFootball.Data.ScottishLower}`.

The new loader includes the same wave-1 module **inside `Wave2MarketTable`**.
The stack identifies `Main.Wave2MarketTable.QSMarketArm`; its configuration type
therefore has a different qualified name. Wave-1 `prepare_panel` compares the
entire `repr(config)` string, including that namespace. This explains a provenance
mismatch without showing any data/prior difference; it is not evidence that the
snapshot changed. Expected/current full configuration strings were not separately
compared, so no broader equivalence claim is made.

Proposed next action for manager: authorise reusing the wave-1 module in its
original namespace, or independently preparing the unchanged pinned panel into a
new wave-2 output directory with all existing count/digest checks retained.
Do not overwrite/bypass the saved wave-1 manifest or tune any prior/seed.

Evidence: `logs/phase1_prepare.summary.txt`, saved manifest copy, and original
raw terminal/launcher logs at `/root/BF_runs/logs/qs_experiment_w2/`.
Both owned idle sessions `%359`/`%360` closed after evidence retrieval.
