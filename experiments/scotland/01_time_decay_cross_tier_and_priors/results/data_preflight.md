# ScottishAll data preflight

**Run date:** 2026-09-24  
**Scope:** TODO028 data-layer preflight only; fresh read-only SQL collection, no MCMC and no production writes.

## Command

Executed on `mcmc-beast` in the isolated worktree `/root/BayesianFootball-scotland-cross-tier` with the existing main-worktree `.env` loaded only into the process environment (the credential was neither printed nor recorded):

```bash
JULIA_PKG_PRECOMPILE_AUTO=0 \
  /root/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia \
  --startup-file=no --project \
  -e 'include("experiments/scotland/01_time_decay_cross_tier_and_priors/data_preflight.jl")'
```

`data_preflight.jl` calls `Data.load_datastore_cached(Data.ScottishAll(); force=true)`, so the results below are from a fresh SQL fetch rather than a prior cache.

## Runtime and routing

| Check | Measured result | Verdict |
|---|---:|---|
| Remote Julia | 1.12.6 | Recorded |
| `Data.tournament_ids(Data.ScottishAll())` | `[54, 55, 56, 57]` | Pass |
| Finished matches | 4,324 | Pass |
| Unique match IDs | 4,324 | Pass |
| Duplicate match IDs | 0 | Pass |
| Match timestamp missing | 0 | Pass |
| Match date range | 2020-08-01 to 2026-09-18 | Pass |
| Canonical team identities | 46 | Recorded |
| Canonical teams appearing in more than one tier | 29 | Continuity present |

The parent independently ran the focused route test as part of `test/data_tests.jl` locally under Julia 1.12.1: **55/55 passed in 15.1 s**, including the three ScottishAll routing assertions. `ScottishAll` remains qualified as `Data.ScottishAll()` because it is intentionally not exported by `Data`.

## Match inventory by tournament

| Tournament | Tier | Finished matches |
|---:|---|---:|
| 54 | Premiership | 1,224 |
| 55 | Championship | 1,061 |
| 56 | League 1 | 1,019 |
| 57 | League 2 | 1,020 |
| **All** | **SPFL pyramid** | **4,324** |

## Domain coverage

Coverage is distinct source-domain match IDs intersected with the 4,324 finished-match IDs. `outside matches` is the number of distinct source-domain match IDs not found in that match inventory. BBC events are expected to be lower because the raw commentary source begins in 23/24 and only includes matches with shot-bearing live-text events; this is not equivalent to BBC match-page coverage.

| Domain | Rows | Matched fixtures | Pyramid coverage | Outside matches |
|---|---:|---:|---:|---:|
| Odds | 97,548 | 4,318 | 99.86% | 0 |
| Lineups | 163,563 | 4,324 | 100.00% | 0 |
| BBC match commentary / shot counts | 4,253 | 4,253 | 98.36% | 0 |
| BBC shot-bearing live-text events | 46,592 | 2,266 | 52.41% | 0 |

| Domain | 54: matches / coverage | 55: matches / coverage | 56: matches / coverage | 57: matches / coverage |
|---|---:|---:|---:|---:|
| Odds | 1,220 / 99.67% | 1,060 / 99.91% | 1,018 / 99.90% | 1,020 / 100.00% |
| Lineups | 1,224 / 100.00% | 1,061 / 100.00% | 1,019 / 100.00% | 1,020 / 100.00% |
| BBC match commentary / shot counts | 1,201 / 98.12% | 1,045 / 98.49% | 1,003 / 98.43% | 1,004 / 98.43% |
| BBC shot-bearing live-text events | 607 / 49.59% | 550 / 51.84% | 554 / 54.37% | 555 / 54.41% |

## Cache availability

The successful force-refresh wrote the isolated remote cache:

```text
/root/BayesianFootball-scotland-cross-tier/.cache/datastore_ScottishAll.jls
209,304,860 bytes
```

It is available for subsequent smoke work on the beast. Use `force=false` (the default) to consume it; do not treat that cache as evidence of a new SQL pull.

## Findings

- The fresh collection contains all requested tournaments and has no duplicate match IDs or orphaned IDs in odds, lineups, BBC match commentary, or BBC live-text events.
- All four tiers have substantial BBC match-page commentary coverage (98.1–98.5%), including the upper tiers; it is suitable for the staged joint-observation data route subject to the model's normal availability mask.
- Six fixtures have no processed odds: four in tournament 54, one in 55, and one in 56; tournament 57 has complete odds coverage. The domain has no orphaned match IDs. Market-dependent routes must retain their existing per-fixture availability handling.
- Canonical `home_team`/`away_team` labels show 29 teams spanning multiple divisions, confirming that the pooled segment preserves observable cross-tier identities. This is a continuity count, not proof that labels resolve all historical rename variants.
