# Claude handover — Phase 2 PASS

**PHASEBT_HANDOVER**. Stop at the Phase2 boundary. Phase1 found missing joint
posterior daily-slate Kelly; the authorised opt-in allocator/adapter is verified.
No Phase3 financial grid, new fits, CLV/liquidity or bootstrap work ran.

## Decisive evidence

| Gate | Executed source | Result |
|---|---|---|
| Prototype, before src graduation | b39db972 | 44/44 PASS |
| Native scenario checks | 13677357 | 51/51 PASS, 0.3s |
| Existing pure portfolio regression | 13677357 | 92/92 PASS, 5.8s |
| Legacy allocation bit parity | 13677357 vs pre-src golden | All3 legs identical |
| Quantitative CSV checks | 13677357 | 5/5 PASS; seed replay difference0 |

- Native beast log: `/root/BF_runs/logs/qs_backtest/phase2_native.log`.
  Retrieved [native log](logs/phase2_native.log), [prototype log](logs/phase2_prototype.log),
  [native quantitative rows](results/phase2/phase2_native_tests.csv),
  [pre-src golden](results/phase2/phase2_legacy_golden.csv).
- Single commission-net Kelly0.1918355076221378 versus analytic0.1918367346938775;
  independent two-bet objective0.03947081362915944 versus brute-grid0.039470768482049584.
  Shared-posterior example chooses total0.1999987862332272 versus analytic0.2,
  not isolated per-match total≈0.4. These are synthetic checks, NOT betting returns.
- Phase1 source/math/line references: [PHASE1_ALLOCATOR_AUDIT.md](PHASE1_ALLOCATOR_AUDIT.md).
  Stock per-match posterior averaging is valid for single-match expected log utility;
  the missing quantity was joint simultaneous daily wealth and shared-draw dependence.

## What changed, and what did not

- Only changed src file: `src/Portfolio/implementations/allocators.jl`.
  ScenarioKelly/ScenarioWeights guard against silent per-match use; shared backend
  retains stock1e-3 starting point, with feasible adaptive start for joint slates.
- User explicitly approved the opt-in experiment adapter; no broader plumbing.
  `l01_scenario_kelly.jl` constructs one commission-net daily scenario matrix,
  solves once and puts raw stakes in NEW books; unchanged stake_slate applies
  production trust/shrink/risk/cap/filter. No defaults/layout changes.
- Fixed1000 maximum scenarios, seed20261010; shared posterior column per scenario,
  independent conditional match scorelines, common scoreline for same-match legs.
  Explicit posterior/fold/chain-layout witness required; unrelated groups fail.
- Existing per-match shrink remains that production overlay, not joint Baker–McHale.
  Existing scalar risk remains sequential/marginal, not shared-posterior risk pricing.
  Final policy-overlaid stakes are not claimed unconstrained joint Kelly optima.
- API/integration limits: [PHASE2_SCENARIO_KELLY.md](PHASE2_SCENARIO_KELLY.md).
  Do not select ScenarioKelly in stock BookSpec: it calls allocation per fixture.
  Use the daily adapter with `native=true` and genuine aligned draw grids/witnesses.

## Coverage — four separate strata, fixed denominator710

| Stratum | Staleness bound | Archive / canonical fixtures | Complete1X2 | Treatment |
|---|---:|---:|---:|---|
| T−25m | 90 min | 611 / 611 | 545 (76.76%) | Reference |
| T−6h | 90 min | 295 / 291 | 182 (25.63%) | Low-coverage subset |
| T−24h primary | 90 min | 53 / 53 | 47 (6.62%) | Low-coverage subset |
| T−24h sensitivity | 360 min | 190 / 190 | 170 (23.94%) | Separately labelled low-coverage subset |

Never pool early subsets with T−25m or substitute360-minute sensitivity for90-minute
primary. Future metrics/intervals/paired m12 comparisons stay within each stratum.
Last-traded archive observations are NOT verified executable touch/depth/fills.

## Safety, ownership and follow-ups

- Synthetic checks only; static PASS; no DB/cache/fits/MCMC/backtest in Phase2.
  DB audit/SHOW N/A (no tables touched), not a new DB-safety pass. Accepted Phase0
  before/after10/10 unchanged/byte-identical audit evidence remains intact.
- Actual runtime: Julia1.12.4/16 threads/core-pinned/BLAS1, package root
  `/root/BF_runs/qs_backtest_phase0`, detached @13677357. Executed source is distinct
  from subsequent evidence commit; see git log for final handover commit.
- Owned `%454`/`pi_qsbt_phase2` CLOSED and verified absent; prior%451–453 closed.
  No remaining owned jobs; unrelated sessions untouched.
- Two non-failing Revise world-age binding warnings for ScenarioKelly/ScenarioWeights
  occurred outside test log. Preserved [REPL capture](logs/phase2_repl_capture.log).
  Fresh-REPL cold-load smoke before authorised Phase3; stricter future Julia/Revise
  may need binding/invokelatest handling. No extra code fix or phase run now.
- Real-fit draw-grid assembly/provenance extraction and support/tail checks are
  Phase3 wiring, NOT_DONE. Missing C0 draw-level data must be honestly NOT_AVAILABLE,
  never a singleton mixture masquerading as a joint posterior. Mixed fold/layouts
  fail; do not silently pair independent posteriors by column number.
- Archived hash-matched owned outputs without force/deletion under `/root/BF_runs/`:
  `qsbt_canonical_artifacts_8018dacf`, `qsbt_phase0_historical_artifacts_72cd5496`,
  `qsbt_phase2_prototype_artifacts_b39db972`.

## Accepted Phase0 retained, no rerun

Source72cd5496 reproduced historical experiment07 m12/raw/B0_canonical/prod:
wealth2.515204600600047 /1127 bets/99 slates/608 books. Canonical separately retained
2.773493294562897/1132 (8018dacf); raw OptionB2.207736087761667/1172. Historical is
validation only, not a fourth grid policy. Exact recipe, IDs, input equality, audits
and coverage remain in REPORT.md/HISTORICAL_REPRODUCTION_RECIPE.md/results/phase0_historical/.

TODO043 first two criteria met; overall IN_PROGRESS/paused for manager. Rulings and
frozen raw/no-fitted-calibration policies remain binding; plain_kelly0.25 retained.
No qs_marketobs profitability claim. Resume Phase3 only when explicitly authorised.
