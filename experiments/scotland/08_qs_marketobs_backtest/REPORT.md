# Phase 0 — PASS, historical reproduction accepted

Manager ruling 3 corrected the target: experiment 07's own **m12 / raw /
B0_canonical / prod** cell, not today's serve-console stack. Source `72cd5496`
reproduced **2.515204600600047 unit wealth / 1127 bets / 99 slates**, exactly matching
the committed CSV headline, within the required 2.5152 ±0.00005 gate.
The historical reconstruction has 608 books, converged=true, zero build errors.

The same 710 IDs and T−25 book were required to equal the prior canonical inputs;
both CSVs are also byte-identical locally. All three snapshot books and coverage/
staleness reports are complete, including the preregistered T−24h/360-minute sensitivity.
Static read checks passed; before/after DB snapshots are byte-identical, 10/10 tables
unchanged. SHOW default was off, informational. No refitting, tuning, DB writes,
source allocator changes or later phases. Owned pane %453 is closed.

## T−25m reference results — raw m12, not the final scenario backtest

| Stack / role | Unit wealth | Return % | Bets | Slates | Books | Executed source |
|---|---:|---:|---:|---:|---:|---|
| Historical experiment07 reproduction gate | 2.515204600600047 | 151.52046006000467 | 1127 | 99 | 608 | 72cd5496 |
| canonical, separately reported policy | 2.773493294562897 | 177.3493294562897 | 1132 | 99 | 611 | 8018dacf, retained without rerun |
| option_b_live, separately reported raw policy | 2.207736087761667 | 120.7736087761667 | 1172 | 99 | 611 | 72cd5496 |

Trace: [phase0_policy_reference.csv](results/phase0_policy_reference.csv), new
[reproduction CSV](results/phase0_historical/phase0_reproduction.csv), historical and
Option B daily rows in that directory; prior canonical daily rows remain at
[phase0_canonical_daily.csv](results/phase0_canonical_daily.csv). Every result uses the
existing KellyLogUtility allocator and PerBetCommission(0.02), **not yet a validated
scenario-native allocator**. Historical bankroll is £1000 reporting scale, normalised
by its initial value; other references use unit reporting scale. No bootstrap here.

Saved run: `132df5c2-c742-4e95-8693-3aeb2b2cbaef`, namespace
`scottish_lower_joint_player_2426`; 24/25 + 25/26, 710 held-out prediction IDs.
[Exact recipe with runner/loader/CSV line citations](HISTORICAL_REPRODUCTION_RECIPE.md).
Raw identity rates are unchanged; no fitted calibration container is used.

**Configuration finding:** historical used the 11-direction tradeable book (no
OU0.5), FractionalKelly(0.30), minimum0.001, λ23 and geometric canonical trust.
Contemporary canonical uses the 13-direction canonical book (adds OU0.5),
BakerMcHale and minimum0.0001. The manager's corrected historical stack reproduces;
the earlier canonical FAIL was a wrong target association, not a model-fit failure.
The bundle difference is observed; individual causal effects are not isolated.
The original FAIL rows/log remain preserved and explicitly superseded as a gate.

## Frozen policy inventory — other rulings unchanged

| Setting | canonical | option_b_live | plain_kelly (Phase 3) |
|---|---|---|---|
| Markets | 1X2, BTTS, OU0.5/1.5/2.5/3.5 | Same | 1X2, BTTS, OU2.5 |
| Price / allocator | DeArb / KellyLogUtility | Same | Same |
| Shrink | BakerMcHale defaults | FractionalKelly(0.30) | FractionalKelly(0.25) |
| Home / Under2.5 trust | 0.35 / 0.35 | 1 / 1 | 1 for all selections |
| Draw / Away trust | 0.25 / 0.25 | 1/1.4 / 1/1.4 | 1 |
| Over1.5 trust | 0 | 1/1.4 | Not priced |
| Other trust | 0 | 0 | 1 |
| Drawdown λ | 23 | 28 | 23 |
| Slate / per-bet cap | 0.25 / 0.50 | Same | Same |
| Commission / budget | 0.02 / 0.99 | Same | Same |
| Minimum fraction | 0.0001 | 0.001 | 0.0001 |
| Group / filter | DailySlate / KeepAll | Same | Same |
| Calibration | None, raw | None, raw | None, raw |

BakerMcHale: 128 draws, grid0:0.02:1, seed20260805. Complete markets required,
barrierμ=1e-6; no book-level trust excision. [policy_inventory.csv](results/policy_inventory.csv)
and [DECISIONS_LOG.md](DECISIONS_LOG.md) are the frozen configuration records.
Historical is a validation reference, **not a fourth Phase 3 grid policy**.
Per-model fitted calibration containers remain out of scope for every model/snapshot.

## Snapshot coverage — fixed denominator 710, no favourable subset

| Snapshot | Staleness bound | Any archived fixture | Canonical-market fixture | Complete 1X2 fixture | Archive / canonical selections | Staleness median / p95 (min) |
|---|---:|---:|---:|---:|---:|---:|
| T−25m | 90 | 611 (86.06%) | 611 (86.06%) | 545 (76.76%) | 3723 / 3551 | 4.41 / 61.82 |
| T−6h | 90 | 295 (41.55%) | 291 (40.99%) | 182 (25.63%) | 1062 / 1038 | 28.51 / 80.57 |
| T−24h | 90 | 53 (7.46%) | 53 (7.46%) | 47 (6.62%) | 153 / 153 | 44.57 / 81.14 |
| T−24h sensitivity | 360 | 190 (26.76%) | 190 (26.76%) | 170 (23.94%) | 562 / 562 | 111.26 / 335.18 |

Trace: [overview](results/phase0_historical/phase0_snapshot_overview.csv) and
[per-market coverage](results/phase0_historical/phase0_snapshot_coverage.csv):
36 rows, four snapshot/bound settings, nine source market-lines; include zero rows,
fixture/selection denominators, min/p25/p50/p75/p90/p95/max staleness.
OU4.5/5.5 are archive-only, not canonical staking markets. CorrectScore is excluded
by the book contract/canonical scope (unknown fixed arity; expected selections blank),
reported as zero rather than silently omitted. Canonical selection denominator=710×13.
Book/refusal CSVs are in the same directory. Refusals are builder-global archive
outputs; filter to the 710-ID panel before using them as panel-level reason counts.

**Finding:** even widening T−24h staleness to six hours leaves only 23.94% complete
1X2 coverage. Earlier-book availability is a major limitation, not evidence of edge.
The sensitivity is reported separately, never substituted for the 90-minute primary.

## Safety, verification and limits

- Executed source72cd5496; [phase0_historical.log](logs/phase0_historical.log) contains
  source hashes, static PASS, SHOW off, input equality, both references, snapshots and
  finally audit PASS. [READ_PATH_AUDIT.md](READ_PATH_AUDIT.md) documents reachable reads.
- Before/after counts and available timestamp maxima in
  `results/phase0_historical/phase0_db_{before,after,audit}.csv`: byte-identical,
  10/10 tables unchanged; absent timestamp columns explicitly marked. No exemptions
  for unrelated writers. Counts/maxima alone cannot detect every same-count edit.
- Four negative static fixtures were already rejected in the prior safety verification;
  checker unchanged, actual new entry points pass both locally and on beast.
- [Result checks](logs/historical_result_checks.log): exact committed headline,
  99 daily rows summing1127 bets, terminal unit wealth, snapshot counts/staleness.
  Local cmp also confirms unchanged input IDs/book and before/after DB evidence.
- Only saved cache deserialisation and experiment-DB reads; no betdb connection or SQL
  fallback, no PGOPTIONS debugging. Julia1.12.4, beast persistent REPL, 16 threads,
  pinthreads(:cores), BLAS1. Owned session closed; broader test suite not run, src unchanged.
- **Prices are last-traded archive observations, not verified executable back/lay touch
  or depth/fills.** Reproducing this legacy archive backtest does not establish execution
  feasibility. Historical policy selection reused these seasons; that selection risk
  remains even though the current policies are frozen.

## Boundary

Phase 0 definition of done is met. TODO043 Phase0 criterion checked; overall task
remains IN_PROGRESS/paused for manager. No Phase1 allocator audit, Phase2 change,
Phase3 five-model grid, CLV, liquidity diagnostics or bootstrap intervals ran.
plain_kelly numerical cells remain NOT_RUN by design, not NOT_AVAILABLE. The final
five report questions and whether qs_marketobs makes money are still unanswered.

**PHASEBT_HANDOVER** — no further phase is authorised in this execution.
