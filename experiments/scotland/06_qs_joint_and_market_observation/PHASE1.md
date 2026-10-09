# Phase 1 — PASS under manager amendments (2026-10-09)

Source: `153239a7`. Fresh owned beast REPL per entry point, 16 pinned threads,
BLAS 1. Complete at 16:47:05 UTC. Earlier failures remain in PHASE1_BLOCKED.md;
no production likelihood change, prior tuning, seed selection or manifest edit.

## Fresh gates

| Gate | Passed | Include wall seconds |
|---|---:|---:|
| MarketRateObservation, amended | 3787/3787 | 390.34 |
| Legacy tape suite | 150/150 | 347.88 (whole file) |
| QS tape suite | 2/2 | included above |
| GRW/QS market-observation tapes | 8/8, 0 B | included above |
| Builder | 106/106 | 71.68 |
| Harness | 179/179 | 111.74 |

Summaries and launcher evidence are in logs/phase1_*; full raw logs retained
outside the checkout on the beast. No skipped gate counted as a pass.

## Likelihood diagnosis and correction

Diagnosis ran BEFORE correction at bd924940. `_observe` returns its scalar ll;
the production engine accumulates it. The original test evaluated the former
alone, so saw prior 1.2036465597893726 and likelihood 0 at sigma=.12. The
production seam has likelihood 2.5623083423195876, exactly equal to both the
hand-written weighted Normal formula and two-logpdf calculation (all constants
included). Its unlinked joint is 3.7659549021089602. Linked joint is
1.6456913659088697: the difference is the inverse-bijector Jacobian
log(.12)=-2.120263536200091. Fix: test the accumulating production seam, not
change src. Absolute 1e-12 likelihood gate passed. See results/likelihood_audit.csv.

## Exact AD and finite differences

Both models: real fold 40, 975 parameters, all coordinates at three frozen
perturbed points. Relative compiled ReverseDiff versus ForwardDiff maxima
4.0510e-16 (GRW) / 4.1099e-16 (QS); fresh ReverseDiff difference 0. Tightened
1e-10 gate passed. High-precision ForwardDiff versus Float64 exact paths:
4.1583e-16 / 4.5269e-16, also below 1e-10.

The historical single-step failure counts reproduce exactly: GRW [26,21,18]
(total 65), QS [101,101,103] (305). Their coordinate unions are 33 and 112;
with 50 fixed random others each, 83/162 coordinates audited at all three points.
No failed coordinate omitted. Selection manifests: results/fd_coordinates_*.csv.

Both Float64 and 128-bit step curves reported for relative h=1e-3,1e-4,1e-5,1e-6
scaled by max(1,abs(x)). High-precision resolvable orders are 1.999998–2.000000
(GRW) and 1.9999996–2.000000 (QS); other curves reach the recorded rounding
floor (quadratic coordinates often start there). No high-precision trend failure.
Richardson uses h=1e-4 and 5e-5, not a selected favourable step. All 249 GRW and
486 QS extrapolations pass 1e-8; maximum relative errors 4.1924e-16 / 4.2649e-18.
See results/fd_convergence_*.csv, fd_richardson_*.csv and fd_precision_comparison_*.csv.

## Recovery: every interval retained

All **42 intervals**, including each truth, lower/median/upper 90% bounds,
coverage flag, seeds and wall time: [synthetic_recovery_three_seeds.csv](results/synthetic_recovery_three_seeds.csv).
The original 14 intervals in synthetic_recovery.csv remain untouched; only two
additional predeclared seeds per arm were fitted, with identical truths/budget.

| Arm | Seed 1 | Seed 2 | Seed 3 |
|---|---:|---:|---:|
| GRW | 6/7 | 6/7 | 7/7 |
| QS | 7/7 | 6/7 | 7/7 |

Pooled coverage **39/42 = 92.86%**, above 80%; **0/14** parameters miss all
three seeds. Both amended recovery conditions pass. This is a small synthetic
calibration check, not evidence of reliable grid convergence or predictive gains.

## Frozen inputs and next phase

Exact wave-1 panel value/count parity remains passed (710/595/78/517).
Frozen CSV SHA256: `680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549`.
Saved wave-1 manifest SHA256: `57941606a195010fe86cbfda0564d61f456379d05d0e8988dce3bb2edd600fb1`.
Both rechecked unchanged after all Phase 1 gates. Owned panes closed.

Next: declare four candidates, run fresh smoke and report hard checks,
convergence and ETA. Ask manager before any grid. No smoke/grid result claimed here.
