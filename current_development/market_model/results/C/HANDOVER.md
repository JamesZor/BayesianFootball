# PHASEC8_HANDOVER — fast rungs complete; C1 live and unpromoted

2026-10-06, branch `pi/market-model-phase-c`. Read
**PHASE_C_PROGRESS_REPORT.md** first for measured results and limits.
Definition of done for revision8 is met: two fresh accepted C0/H1/H2 runs,
all29 scientific CSVs/PNGs byte-identical; C1 10a/10b launched at prescribed
budgets without promotion; reporting/tracker/verification recorded.
**Full Phase C is NOT complete. C2/PhaseD remain deferred.**

## 1. DO NOT touch these three live runs

Only read panes/logs. Never kill/relaunch, reload/include, edit/check out source,
change package/cache/DB/priors/seeds/budgets, or launch duplicate C1 sampling.

| Run | Pane / session | Frozen source | Checkout |
|---|---|---|---|
| Recovery |%265 /pi_mm_c|6335316f4e93ec91495341fa2f6960a64ce772a9|/root/BF_runs/market_model_c|
| C1 honest10b |%267 /pi_mm_c8_c1_10b|41e28736cdd990d8d701dcb0b48df98196c54674|/root/BF_runs/market_model_c_fast|
| C1 retrospective10a |%268 /pi_mm_c8_c1_10a|41e28736cdd990d8d701dcb0b48df98196c54674|/root/BF_runs/market_model_c_fast|

C1 production started04:50:29Z/04:50:32Z. Recovery process started approximately
03:59:13Z. Each run16 threads/core pinning/BLAS1, fixed4×(2000 warmup+3000
retained),thin1. Production outputs: `results/C/v8_c1_10b/`, `v8_c1_10a/`;
recovery `results/C/v7_recovery/`. Production PROMOTION_PENDING.txt is intentional.

At07:46:59Z, chain1/2/3/4 progress:
- Recovery1000/750/500/750; coarse rates1.37–4.12/min from observed250 at04:44:53Z.
  Individual extrapolations16.2–54.6h; conservative slowest ETA **Oct8 14:25Z**.
  Its last500 marker is stale/censored; re-estimate at the next marker.
- C1 10b1500/750/1000/1000; measured4.53–10.36/min,
  conservative slowest **15.6h /Oct6 23:25Z**.
- C1 10a250/750/500/250; measured2.19–4.66/min,
  conservative slowest **36.1h /Oct7 19:54Z**.

These are uncertain throughput extrapolations, not completion/acceptance claims.
Adaptation/250-step markers/concurrent load affect rates. Load11.14/12.88/13.00.
Exact observed snapshots/rate bases: BEAST_C8_LIVE_RUNS.txt,
LIVE_RUN_ETA_CHECKPOINT.csv. No exception/abort marker is present there.

```bash
ssh root@mcmc-beast 'date -u; tmux capture-pane -t %265 -p -J -S -100'
ssh root@mcmc-beast 'date -u; tmux capture-pane -t %267 -p -J -S -100'
ssh root@mcmc-beast 'date -u; tmux capture-pane -t %268 -p -J -S -100'
```

Logs under `/root/BF_runs/logs/market_model_c/`:
- Recovery `6335316f_recovery1.log`; success **R05_FULLBOOK_RECOVERY_DONE**.
- C1 `c8/41e28736_c1_10b.log` /`_10a.log`, subsequent correctly timestamped
  `_utc.log`; success **R05_C1_PENDING_DRAWS_DONE**, STILL NOT promotion.
- Early `_timed.log` awk prefixes were constant and invalid for rates. Preserved,
  not rewritten. Python UTC logger began05:11:26Z; current valid files end_utc.log.

## 2. Finished fast evidence

Both clean runs source **70336b07a81aac7bc3d7188e7ae2138fb3b59072**, immutable
`/root/BF_runs/market_model_c_fast_final`. Rung budgets/priors/seeds unchanged.
First `%270`:05:26:50→05:46:27Z; second `%271`:07:29:02→07:45:36Z.
Both R05_FAST_RUNGS_DONE. **107/107 engine gates,20/20 convergence rows**;
max Rhat1.000985775031962, min bulk/tailESS6074.903633053705/6129.2431286740775.
**26 CSVs +3 PNGs byte-identical**, both full file sets/checksums verified.

Laptop: `results/C/v8_fast_1/`, `v8_fast_2/`, BYTE_COMPARISON.txt,
BEAST_C8_FAST1/2.txt. Root home_advantage_rungs.csv is a byte-identical convenience
copy. Every measured assertion is sourced in PHASE_C_PROGRESS_REPORT.md.
Fits stay on beast (real timing; NOT byte-identical binary claims).
Fast logs `logs/market_model_c/c8/70336b07_fast1.log` /`fast2.log`.
Own idle fast/test REPLs%270/%271/%66 are closed after evidence capture;
only the three long runs remain live. Never send another include to old IDs.

Interpretation: C0 tracks R6 closely; no H1 forecast gain and its separate
attack/defence HA coefficients are confounded with mu; away suppression is
not established. H2 kappa intervals include zero; no supported quality-linked
HA improvement. HA mixes all12,000 hyperdraws; honest HA training only.
Forecast/path bands are conditional at median theta; full smoothing descriptive.
C8_METHODS.md has formulas, gauge, nonlinear expectation and Ridall2024 p.1/3
context (not a claim to fit that paper's dynamic goal-state formulation).

Final-source tests in LAPTOP_C8_FINAL_SOURCE_TESTS.txt:
report22/22, scalar/native17/17, t05 305/305, workflow85/85,
t04 92/92, t03/t02 each131/131. C2-pending29pass/10fail is excluded.
C1 pre-sampling evidence: each12/12 full Gate1 and53/53 engine/state,
`v8_c1_10a_preflight/`, `v8_c1_10b_preflight/`; not posterior/convergence data.

## 3. Preserved corrections/attempts, not extra reproductions

Manager approved exact Gaussian-only scalar arithmetic after measured dense
C0 speed implied hours. Same native slice sampler/adaptation/RNG;54 dense/scalar
toy/full-real parity gates at unchanged1e-9/1e-8. C1 path/checkouts untouched.
Dense41e attempt%266 stopped solely for speed before any saved fit; remote
checkout/output and `logs/.../c8/dense_attempt_preserved/` remain.
Scalar5ab9980b%269 passed all fits/gates, then plot assertion failed because
Ross County/Airdrie aren't in the fixed56/57 panel. Manager approved annotated
absences, not population expansion/fake paths; Kelty key `kelty-hearts-fc`.
Its checkout and local `v8_scalar_plot_attempt/`/BEAST_C8_PLOT_ATTEMPT.txt remain.
Neither attempt is a clean full reproduction. Failed helper-field test is also
preserved and corrected by reading the actual `udraws` field. No threshold change.
Historical reports/handover/reproduction are preserved as *_V7.md and earlier;
all original failed recovery evidence remains as HANDOVER_V7 records.

## 4. Next actions — no duplicate sampling

1. Monitor the frozen recovery/production panes coarsely. If a root exception
   occurs, preserve expanded Base.current_exceptions()/CompositeException,
   generated panel, root coordinate/log and final post-cancellation accounting.
   Wait for cancellations and STOP; no solver/seed/budget/threshold variation.
2. Accept recovery only if all four truth values(.03,.01,.06,1000) lie inside
   their90% intervals and EVERY parameter passes Rhat<=1.05,bulk/tailESS>=200.
   Generation3962/chains4961–4964; panel517/91, SHA
   `de6a986b1caebc2a2ad1386308c7f4613bf69ea2f71440778783476c348da0d1`.
   C6 log labels do not imply the old solver. No interim accounting exists yet.
3. Preserve completed production draws and ALL-parameter convergence, even if
   recovery still pending. No n/u/shrinkage/forecast interpretation until BOTH
   recovery and relevant production convergence are accepted.
4. Then implement accepted inference/reporting in a NEW loader/runner checkout,
   load existing C1 fits (include l05_pooled first), explicitly call
   require_recovery on the accepted panel/report, retain the promotion gate.
   **Do not include r05_fullbook_core to duplicate already sampled C1 fits.**
   Compute C1 vs accepted C0 on identical pre-week fixtures; full-book
   theta=structure+u shrinkage, n/sigma_u/local noise, properly mixed posterior
   HA/rates/path intervals, all measures/figures. Honest hyperfit training only.
5. Complete accepted C1 full-workflow reproduction and update Phase C report.
   C2/thin pooling/score-grid/PhaseD remain separately deferred.

Verify SOURCE_SHA256SUMS from stream root, SCIENTIFIC_SHA256SUMS from results/C,
`git diff --check`, `./scripts/todo.sh check`; only commit/push authorised branch
`origin pi/market-model-phase-c`. No force/rebase/merge/stash. No source/package/
cache/src/DB scientific changes are authorised beyond the two recorded corrections.
