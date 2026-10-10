# Phase 0 — BLOCKED at canonical reproduction

Canonical m12 produced wealth **2.7734933** and **1,132 bets** at T−25m.
Required: **2.5152 ±0.00005** and **1,127 bets**. Both parity checks failed.
The 710-fixture universe, 99 daily slates, converged fit and zero build errors are recorded.
Static read-path checks passed; before/after audits are unchanged across all 10 tables.
The actual session default was **off**, accepted under the manager's revised safety ruling.
Option B, T−6h, T−24h and all later phases were not run after the failed gate.
Both policy inventories are reported below; all use raw probabilities, with no fitted calibration.
No retuning, refitting, database writes or source allocator changes occurred.
TODO 043 is BLOCKED; the owned beast pane is closed. Phase 0 is not complete.

## T−25m — reproduction and named policies

| Model | Policy | Final unit wealth | Return % | Bets | Slates | Built books | Status |
|---|---|---:|---:|---:|---:|---:|---|
| m12 | canonical | 2.773493294562897 | 177.3493294562897 | 1132 | 99 | 611 | FAIL required reproduction |
| m12 | option_b_live | — | — | — | — | — | NOT_RUN: canonical failed; stop rule |
| all models | plain_kelly | — | — | — | — | — | NOT_RUN: Phase 3 pending |

Trace: [phase0_reproduction.csv](results/phase0_reproduction.csv) and
[phase0_canonical_daily.csv](results/phase0_canonical_daily.csv). Initial bankroll was
1.0 and commission was 2% per bet. The m12 fit loaded by UUID was
`132df5c2-c742-4e95-8693-3aeb2b2cbaef`, namespace `scottish_lower_joint_player_2426`.
The saved prediction IDs were restricted to 24/25 + 25/26 and required to equal 710;
[phase0_fixture_universe.csv](results/phase0_fixture_universe.csv) records those IDs.
Build report: converged=true, zero build errors. This is an observed **reproduction
failure**, not evidence about whether qs_marketobs beats m12.

The target is the frozen manager ruling, not a newly estimated baseline. The observed
canonical wealth exceeds the target by approximately **0.2582933** unit wealth and its
bet count by **5**. No threshold was relaxed, alternative stack substituted, or retry made.
Experiment 07's historical book used FractionalKelly(0.30), minimum 0.001 and 11 priced
directions, unlike canonical's BakerMcHale/minimum 0.0001/canonical markets. Those
pre-recorded differences remain facts, **not a tested causal attribution of this gap**.
The historical control was removed under the ruling; it was not run on resume.

## Policy inventory — both named stacks, no fitted container

| Setting | canonical | option_b_live | plain_kelly (Phase 3) |
|---|---|---|---|
| Markets priced | 1X2, BTTS, OU 0.5/1.5/2.5/3.5 | Same canonical markets | 1X2, BTTS, OU 2.5 |
| Price / allocator | DeArb / KellyLogUtility | Same | Same |
| Shrink | BakerMcHale defaults | FractionalKelly(0.30) | FractionalKelly(0.25) |
| Home / Under 2.5 trust | 0.35 / 0.35 | 1 / 1 | 1 for all selections |
| Draw / Away trust | 0.25 / 0.25 | 1/1.4 / 1/1.4 | 1 |
| Over 1.5 trust | 0 | 1/1.4 | Not priced |
| Other direction trust | 0 | 0 | 1 |
| Drawdown λ | 23 | 28 | 23 |
| Slate cap / per-bet cap | 0.25 / 0.50 | Same | Same |
| Commission / per-match budget | 0.02 / 0.99 | Same | Same |
| Minimum bankroll fraction | 0.0001 | 0.001 | 0.0001 |
| Group / filter | DailySlate / KeepAll | Same | Same |
| Fitted calibration | None; raw probabilities | None; raw probabilities | None; raw probabilities |

BakerMcHale defaults: 128 draws, grid 0:0.02:1, seed 20260805. All execution specs
require complete markets, barrier μ=1e-6, and use no book-level trust excision.
Direction gating is the downstream policy, not a change to the priced payoff matrix.
[policy_inventory.csv](results/policy_inventory.csv) and [DECISIONS_LOG.md](DECISIONS_LOG.md)
record the settings and sources. Per-model fitted calibration containers are out of scope
for **every** model and snapshot; option_b_live is deliberately a raw-probability policy.

## Database safety — manager ruling 1

| Gate | Evidence | Result |
|---|---|---|
| Static read paths | [READ_PATH_AUDIT.md](READ_PATH_AUDIT.md), runner log/source hashes | PASS |
| Four injected negative static fixtures | Persistence, DML, unguarded SQL and cache fallback rejected | 4/4 PASS |
| Session default record | [phase0_session_default.csv](results/phase0_session_default.csv) | off (informational only) |
| Before/after counts and timestamp maxima | [phase0_db_before.csv](results/phase0_db_before.csv), [phase0_db_after.csv](results/phase0_db_after.csv) | Byte-identical |
| Per-table audit | [phase0_db_audit.csv](results/phase0_db_audit.csv) | 10/10 unchanged |

Audited table-wide counts: runs 347; configs 347; config_registry 459; fold_results
6693; match_latents 131013; fit_artifacts 347; fit_fold_artifacts 464; harness_scores
74550; harness_experiments 30; harness_checks 637. Actual timestamp maxima and absent
created_at/updated_at columns are explicit in the snapshots; runs.finished_at and
harness_checks.at were also recorded. The `finally` audit ran **after** the parity
failure. No namespace exemptions or unrelated-writer exclusions were applied.
No betdb connection was made: the saved cache was deserialised directly, without SQL
fallback. No PGOPTIONS debugging/change was performed on resume.

Counts/maxima cannot detect every same-count mutation on untimestamped tables; they
complement the reviewed read-only call paths rather than constituting a complete write
forensics system. No database write call is reachable from this backtest entry point.

## Snapshot coverage and remaining phases

| Snapshot | Book | Coverage/staleness report |
|---|---|---|
| T−25m | [Saved book](results/phase0_t25_book.csv), [refusals](results/phase0_t25_refusals.csv); 90-minute bound | NOT_RUN: aggregate inventory follows the failed reproduction gate |
| T−6h | NOT_RUN | NOT_RUN: stop rule |
| T−24h | NOT_RUN | NOT_RUN: stop rule; 360-minute sensitivity not assessed |

No Phase 1 allocator audit or Phase 2 implementation ran. No Phase 3 comparisons,
per-market grid breakdown, CLV or bootstrap intervals were computed. None of the
future cells is classified NOT_AVAILABLE: availability has not yet been assessed.
The five final report questions remain unanswered. The gate does not authorise
proceeding simply because the observed canonical wealth is higher than the target.

## Execution, limits and follow-up

- Executed source: `8018dacf`; [logs/phase0_resume.log](logs/phase0_resume.log).
  Beast `/root/BF_runs/qs_backtest_phase0`, detached checkout; owned persistent Julia
  1.12.4 REPL `%452`, 16 threads, core-pinned, BLAS 1. Pane/session now closed.
- Earlier startup/session-default failures are historical: retained attempt 1/2 logs,
  old report at commit `28637800`. They are not the current blocking condition.
- Manager must review the fixed canonical target versus the observed exact canonical
  stack result before authorising further Phase 0 work. Do not silently change policy,
  tolerance, fixture set or source to obtain parity. There is no automatic retry.
- Last-traded archive prices are not guaranteed executable touch prices or fills.
  Existing production policy selection also reused these two seasons; a future full
  backtest must label that in-sample selection risk. No uncertainty estimate is yet available.
