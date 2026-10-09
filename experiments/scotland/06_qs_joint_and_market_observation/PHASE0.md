# Phase 0 — PASS (2026-10-09)

Source: `ec96c079`, isolated beast checkout `/root/BF_runs/qs_experiment`.
Every entry point ran in its own fresh persistent Julia 1.12.4 REPL, 16 threads,
pinned cores and BLAS 1. No sampling grid launched. Start 15:08:56 UTC;
all gates complete 15:32:29 UTC. Owned sessions closed; other sessions untouched.

Counts and include wall times are in `results/phase0_gates.csv`; test summaries
are in `logs/*.summary.txt`. Raw terminal logs remain at
`/root/BF_runs/logs/qs_experiment_w2/` (local fetched copies are ignored).

| Gate | Accepted pass/total |
|---|---:|
| QualityStyleGRW | 373/373 |
| MultiScaleGRW | 124/124 |
| Tape allocation matrix + QS | 150/150 + 2/2; 0 B assertions |
| Builder (six testsets) | 106/106 |
| Harness runner | 179/179 |
| Market t01 / t02 / t03 / t04 | 269/269 / 131/131 / 131/131 / 92/92 |
| t05 fast / reports / workflow / preflight | 17/17 / 22/22 / 85/85 / 11/11 |
| t05 pooled full-book deterministic | 305/305 |
| t06 (four testsets) | 85/85 |

The explicitly excluded C2-pending child set remains **29 passes / 10 known
failures / 39 total**. The full-book parent passes; no threshold or test changed.
These deterministic tests do not imply acceptance of unrelated pending C1/C2 fits.

## Joint-observation coverage

Snapshot SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`;
original mtime `2026-09-25 12:57:15.480765468 UTC` preserved.

The exact `MatchProxyXGFeature(k=25, fallback=:none)` extractor was run for
every fold with fold-specific history IDs for the shot-cell fit. Coverage is
both sides available, not merely one nonmissing field. No goals fallback.
`results/proxy_xg_coverage.csv` records all 40 folds × two tournaments ×
training/heldout partitions (160 rows).

| Tournament | Covered training matches per fold | Training fraction range | Heldout covered |
|---|---:|---:|---:|
| 56 (League One) | 180–530 | 50%–100% | 355/355 |
| 57 (League Two) | 180–530 | 50%–100% | 355/355 |

Both tournaments have proxy-xG in **every** training window; all 710 held-out
matches have coverage. Held-out coverage is descriptive only and never a
training observation. Phase 1 may proceed; the pinned snapshot need not change.
