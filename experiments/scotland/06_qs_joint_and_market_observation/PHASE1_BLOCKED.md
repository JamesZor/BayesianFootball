# Phase 1 — BLOCKED (2026-10-09)

## Current stop: component gate at ec1a4137

Manager authorised option 1; table preparation passed at 4a71c02e, with the
original Main.QSMarketArm namespace and untouched provenance guard/manifest.
Exact IDs/log rates/full-book values match wave 1 after CSV round-trip; target
counts are 710/595/78/517. Frozen SHA256 is recorded in PROGRESS/DECISIONS_LOG.

Fresh beast REPL %364 (launcher %363), 16 pinned threads, BLAS 1, failed
`test/test_market_rate_observation.jl` at 16:07:56 UTC, 183.3400 s include wall:

| Set | Pass | Fail |
|---|---:|---:|
| Injected feature/builder/likelihood | 22 | 1 |
| GRW real-fold AD | 2867 | 65 |
| QS real-fold AD | 2627 | 305 |
| Synthetic recovery | 15 | 1 |
| Total | 5533 | 372 |

- Likelihood assertion at line 115: density 1.2036465597893726 versus expected
  likelihood + prior 3.7659549021089602 (atol 1e-12). Inspect the standalone
  density/context helper before attributing this to the production likelihood;
  no diagnosis or fix is yet validated.
- All 370 AD failures are central finite-difference checks at line 143, unchanged
  relative tolerance 1e-6. Compiled/fresh ReverseDiff and ForwardDiff comparisons
  passed at all three tested points for both 975-coordinate models. Example
  finite-difference discrepancy: GRW 2.4437e-6, QS 5.8696e-6. These are still hard
  failures; unconditional MARKET_AD PASS print lines are not acceptance evidence.
- Recovery: 13/14 declared parameter intervals cover. GRW dyn.β.σₖ truth 0.024
  falls above the 90% interval [0.011703884611242044, 0.02272934787590222]. All
  frozen truths/seeds/budgets unchanged. Preserve this miss; no seed tuning.
- Julia Test executes all child sets before throwing its aggregate failure, so
  the recovery fits ran within the failed entry point. The shell launcher then
  stopped: allocation, builder and harness gates did not start; no smoke/grid.

Evidence: logs/phase1_market.summary.txt, results/synthetic_recovery.csv,
logs/phase1_tests_launcher.log; raw terminal log retained locally (git-ignored)
and at /root/BF_runs/logs/qs_experiment_w2/phase1_market.log. Table and saved
manifest digests rechecked unchanged after the gate. No retry, source fix or
threshold change after failure. Both owned idle sessions closed.

Await manager review before further Julia execution or modifications. Phase 1
is not accepted; Phase 2+ not started.

## Historical first stop: namespace provenance (resolved by manager ruling)

The following records the first stop; its execution-state statements apply to
that checkpoint only.

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
