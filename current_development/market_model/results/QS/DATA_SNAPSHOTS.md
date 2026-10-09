# Cross-league pinned data snapshots

Generated on the laptop on 2026-10-07 from `betdb`, once per segment, using `BayesianFootball.Data.load_datastore_cached(segment; force=true)` in the owned persistent REPL `%72`. No database writes. `MarketModelEnglish` is the stream-local singleton in `l06_qs_eda.jl`, with tournament IDs `[1,2,3,84]`. Copied by explicit-file `rsync` to `/root/BF_runs/market_model_qs/.cache/` on mcmc-beast and verified **all five SHA256s there** against this table before loading with `max_age_hours=10^6`. The first transfer timed out at 180 seconds after IrelandAll completed; the second used partial/append verification and completed, with all five final hashes matching. No `.env` or `Manifest.toml` was transferred.

| Segment | Cache file | SHA256 | Bytes |
|---|---|---|---:|
| MarketModelEnglish | `.cache/datastore_MarketModelEnglish.jls` | `2efe200ca6ed2899a75246570e025d5782af5da945ec26100bfa38331c5168bf` | 971445198 |
| ScottishUpper | `.cache/datastore_ScottishUpper.jls` | `56207e6354f0e9fbf5a6d1e29a277ae49b50769d2975de7a7e7ccca85c93630a` | 140728685 |
| ScottishLower | `.cache/datastore_ScottishLower.jls` | `e894ac6d59f912aceb01f639c80be37555a73f4eab2fa24271cddd801f13006a` | 55903702 |
| IrelandAll | `.cache/datastore_IrelandAll.jls` | `9cc59b002982fc7d313bda0c1d821a31e98dbdd91827bfcdb2ec30dd209cc257` | 114858403 |
| Veikkausliiga | `.cache/datastore_Veikkausliiga.jls` | `e4ef1c33939c28b88571b9d0256a3be8d8b83f88af82a7ff83c85f1315d35b1c` | 61314070 |

Season policy clarified by the manager: retain Scottish League One/Two 25/26 (175/180 fixtures per tier, only the final round missing; published B2 used these same 175 each). Exclude Scottish Championship 25/26 (only 7 of its 175 fixtures have Betfair prices, last close 2025-09-13). Retain Premiership's 198 matches/season, which structurally contain rounds 1–33, not the post-split five rounds. Exclude 26/27 and 2026 seasons while in progress; older seasons without complete Betfair markets are not fitted. A season is excluded when in progress or its Betfair-covered matches stop before the final quarter of the schedule. `season_inventory.csv` will record each selection and reason.

The published Scottish regression uses the **prior** snapshot `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4` separately, before replacing it with the fresh ScottishLower snapshot. Never confuse these two snapshots when claiming reproduction.
