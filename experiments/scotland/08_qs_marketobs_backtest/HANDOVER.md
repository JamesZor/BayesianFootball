# Claude handover — Phase 0 PASS

**PHASEBT_HANDOVER**. Historical reproduction accepted under ruling3; all snapshots
and coverage complete; stop at the Phase0 boundary. No Phase1 or later work ran.

## Decisive evidence

- Source72cd5496 reproduced experiment07 **m12/raw/B0_canonical/prod**:
  **2.515204600600047 wealth / 1127 bets / 99 slates / 608 books**,
  converged=true, zero build errors. This exactly matches the committed CSV headline,
  passing target2.5152±0.00005/1127. Initial £1000 is normalised for unit wealth.
- [HISTORICAL_REPRODUCTION_RECIPE.md](HISTORICAL_REPRODUCTION_RECIPE.md) cites the
  exact runner/loader/CSV lines. Historical uses tradeable11-direction book,
  FractionalKelly0.30/minimum0.001, λ23/cap0.25 and geometric canonical trust.
  Its raw identity rates are unchanged; raw fit used directly, no fitted calibration.
- Same m12 saved UUID/cache/710 prediction IDs and T−25/90-minute book as canonical.
  Runtime assertions passed; fixture-universe and book CSVs are also byte-identical.
- Canonical retained separately, no rerun: **2.773493294562897 / 1132** from8018dacf.
  It uses canonical13-direction book/BakerMcHale/minimum0.0001. Prior target failure
  is superseded, not erased; no single-component causal attribution was made.
- Fixed option_b_live raw m12 reference also ran: **2.207736087761667 / 1172**.
  [phase0_policy_reference.csv](results/phase0_policy_reference.csv) distinguishes
  policy references from the historical gate and records executed source per row.

## Coverage, not executable liquidity

Fixed denominator710; primary max staleness90 minutes:

| Book | Archive fixtures | Canonical fixtures | Complete1X2 | Archive selections |
|---|---:|---:|---:|---:|
| T−25m | 611 (86.06%) | 611 | 545 (76.76%) | 3723 |
| T−6h | 295 (41.55%) | 291 | 182 (25.63%) | 1062 |
| T−24h | 53 (7.46%) | 53 | 47 (6.62%) | 153 |
| T−24h/360-minute sensitivity | 190 (26.76%) | 190 | 170 (23.94%) | 562 |

All per-market fixture/selection denominators and staleness quantiles are committed;
zero rows included, no favourable subset. T−24h is sparse even after six-hour widening.
Archive-only OU4.5/5.5 and unsupported CorrectScore are explicitly labelled, not
added to canonical staking scope. Refusal tables are builder-global; filter to the
fixed710-ID panel before interpreting panel-level reason counts.

**Important:** existing point-in-time builder uses last-traded archive prices,
not verified executable back/lay touch/depth. Do not present these reference returns
or coverage as executable-fill evidence. Policy selection also reused these seasons.

## Safety and ownership

- Static read-path checker PASS locally/on beast; actual SHOW default off recorded
  once. Four negative fixtures were already tested in prior safety work. No PGOPTIONS
  debugging, writes, fitting, SQL cache fallback or betdb connection.
- Before/after audit byte-identical, **10/10** fit/config/harness tables unchanged;
  timestamp absence explicit, no unrelated-writer exemptions. Audit runs in finally.
- Beast `/root/BF_runs/qs_backtest_phase0` @72cd5496 detached; fresh persistent Julia
  1.12.4 REPL, 16 threads, core-pinned, BLAS1. Owned pi_qsbt_phase0 pane **%453 closed**
  and session verified absent. %451/%452 already closed; no unrelated panes touched.
- Checkout initially blocked on nine owned untracked canonical outputs: each matched
  its committed hash, archived in `/root/BF_runs/qsbt_canonical_artifacts_8018dacf`,
  then checkout succeeded. No forced checkout/deletion/merge.
- Datastore SHA256 unchanged:
  `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`.

## Files and current state

- New results: `results/phase0_historical/` — historical target/result/daily, OptionB
  daily, input IDs, all books/refusals, coverage/overview, audit and SHOW record.
- New log: `logs/phase0_historical.log`; local result checks:
  `logs/historical_result_checks.log`. Beast log:
  `/root/BF_runs/logs/qs_backtest/phase0_historical.log`.
- Prior canonical results/audits and attempts1–3 logs remain preserved. Execution
  chronology and arm status CSVs now distinguish superseded failures/current pass.
- [REPORT.md](REPORT.md), [DECISIONS_LOG.md](DECISIONS_LOG.md), [READ_PATH_AUDIT.md](READ_PATH_AUDIT.md).
- TODO043 Phase0 criterion checked; overall IN_PROGRESS, paused for manager.
  Final tracker/diff checks and evidence pushed; see `git log -1` for evidence commit.
  Executed source remains72cd5496, not the subsequent documentation commit.

## Resume only when authorised

Next phase is the brief's Phase1 allocator audit. Rulings1–3 all remain binding:
read-only static review plus per-run before/after audit; named canonical/option_b_live;
raw probabilities for every model, no per-model fitted calibration; plain_kelly0.25
retained. Historical is validation only, **not a fourth Phase3 grid policy**.

No scenario allocator validation/change, five-model×snapshot×policy grid, CLV,
liquidity analysis, bootstrap intervals or final report questions are complete.
Existing KellyLogUtility produced only the Phase0 references. No claims about
qs_marketobs profitability are available. No cells are NOT_AVAILABLE merely because
later phases have not run. Broader test suite not run; no src changes.
