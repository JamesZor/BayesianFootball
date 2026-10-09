# Phase 0 — PASS (2026-10-10)

Source `122cbc37`, beast checkout `/root/BF_runs/qs_experiment` (detached at that commit).
Each entry point ran in its own fresh persistent Julia REPL (`claude_qsx3_gate_*`), 16 threads,
pinned cores, BLAS 1, sequentially. 23:16:29–23:46:08 UTC (2026-10-09 UTC). Launcher
`phase0_gate.sh`; raw logs `/root/BF_runs/logs/qs_experiment_w3/phase0/`. All owned sessions
closed; other sessions untouched. No sampling grid, no DB writes.

## Pins (checked in shell before any Julia)

- Snapshot `.cache/datastore_ScottishLower.jls` SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`, mtime `2026-09-25 12:57:15.480765468 UTC` — matches.
- Frozen market table `06_.../results/market_rates.csv` SHA256 `680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549` — matches.

## Gates

| Gate | Pass/total |
|---|---:|
| QualityStyleGRW | 373/373 |
| MarketRateObservation | 3815/3815 |
| MultiScaleGRW | 124/124 |
| Tape allocation (matrix + QS + market obs) | 150/150 + 2/2 + 8/8, 0 B |
| Builder (six testsets) | 106/106 |
| Harness runner | 179/179 |
| Market t01 / t02 / t03 / t04 | 269 / 131 / 131 / 92, all pass |
| t05 fast / reports / workflow / preflight | 17 / 22 / 85 / 11, all pass |
| t05 pooled full-book deterministic | 305/305 |
| t06 (four testsets) | 85/85 |

Excluded by the standing ruling: the C2-pending child set, **29 pass / 10 known failures / 39**,
identical to wave 2.

MarketRateObservation evidence: exact AD max 4.05e-16 (GRW) / 4.11e-16 (QS) on fold 40, 975
parameters; recovery 39/42 (92.9%), as in wave 2. The count is 3815, not wave 2's 3831,
because wave 2's array revalidation set `QSX2_RERUN_RECOVERY_FIRST=true` (three seeds rerun, 42
truth-anchor checks) while this run used the default (first seed read from the immutable CSV,
seeds 2–3 rerun, 28 anchor checks): 14 of the 16 are accounted for this way. The other 2 are not
explained by inspection; there were zero failures or errors.
Per-gate wall times: `results/phase0_gates.csv`.
