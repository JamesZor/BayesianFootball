# TODO 039 — Phase C revision 4 implementation checkpoint (2026-10-06)

**HANDOVER, not Phase C completion.** Full-book-only C0/C1/H1/H2 continuation
is authorised by `experiments/pi_market_model_phaseC_v4_scope.md`. C2 is deferred.
Deterministic state engines and preflight pass; **no posterior fit, sampling,
synthetic recovery, forecast evaluation, learned concentration, shrinkage,
ratings figures or production reproduction has been run**. TODO 039 remains
IN_PROGRESS. No beast checkout/session was created.

## 1. Scope and implementation

- Double Poisson, cutoff 10, unchanged density normalisers and grid.
- Same 517 full fixtures / 1034 isolated scoring targets, from `phase_b_panel`.
  `fullbook_markets` asserts IDs/order, full-book type and explicit exclusions.
- `l05_fullbook_engine.jl`, included inside PooledMarket, implements C0/C1/H1/H2.
  C0/H1/H2 use exact joint Gaussian observations. Independent q/s weekly GRWs
  are expressed in raw alpha/beta coordinates; initial independent alpha/beta,
  static mu/gamma and zero-sum observation projection remain those of B2.
- C0 parameters are sigma_obs/sigma_q/sigma_s; C1 sigma_q/sigma_s/sigma_u/n;
  H2 additionally samples kappa directly. The log-coordinate priors include
  Jacobians and normalisers. C1 log n has exactly Normal(log(1000),1.5), with
  **no inherited ±12 bound**. No sampler exists yet.
- H1's missing prior was clarified by the manager via a question: independent
  gamma_att N(.15,.25²) and gamma_def N(0,.25²). Gamma_def is a static state
  with a negative loading on the away rate. Pinning it to zero recovers C0.
  H2 adds kappa times the centred home team's q to its own scoring predictor.
- C1 calls the existing exact book density / joint-mode Laplace update with
  revision-3 mean correction. Its **only** collapsed contribution is the returned
  marginal. Theta moments map back with K=P H'/S, m+=K(b−a), P+=K(V−S)K',
  S=H P H'+sigma_u² I. No full-state precision inverse or extra Gaussian
  likelihood is appended. Raw and clipped curvature remain distinct.
- All forecasts in a week precede every update in that week. Update order is
  the inherited deterministic week/date/ID panel order.
- Stored C1 Gaussian factors match the local marginal and moments. Frozen-factor
  batch checks verify that Gaussian surrogate only, **not an exact globally
  Gaussian representation of the nonlinear Dirichlet model**. RTS is approximate
  for C1 and exact for the two-stage rungs.

`r05_pooled.jl` remains a **preflight runner**, now writing only to
`results/C/v4_preflight/`. It is not a production pipeline or training runner.
Historical v3 tables at the root remain untouched; copies are in `v3_gate/`.
Historical notes are `PHASE_C_V3_REPORT.md`, `REPRODUCIBILITY_V3.md` and
`HANDOVER_V3.md`. DESIGN §5 remains unchanged; the manager's revisions select
these rungs and the Laplace/slice route instead of its old MAP/NUTS formulation.

## 2. Verified evidence

Fresh owned laptop REPL `%47`, Julia 1.12.1, 8 threads, BLAS=1, pinned cache,
existing Manifest. Full output: `LAPTOP_C4_FINAL_PREFLIGHT.txt`.

### Full-book Gate 1

Source: `v4_preflight/laplace_gate.csv` and `laplace_gate_fixture.csv`.
First 30 full books in ID order, n=250/1000/4000, side SD=.05/.20, offsets
0/+.10; 360 fixture-setting rows, **12/12 settings pass**, 48/48 scalar checks.
No type/setting/tolerance was tuned.

| Maximum across the 12 settings | Observed | Limit |
|---|---:|---:|
| Median absolute marginal error | 0.003980333011027959 nats | .01 |
| p95 absolute marginal error | 0.0051705855019272395 nats | .05 |
| Corrected mean error / exact SD | 0.0009284497092163062 | .05 |
| Relative SD error | 0.010184310576972222 | .05 |

The saved order-stability diagnostics remain in the fixture/summary CSVs.
No current full-row join against the v3 archive was certified: the auxiliary
DuckDB comparison command failed twice on unquoted reserved SQL identifiers
(`rows`, then `offset`) and was not retried a third time. This is not a scientific
gate failure or a reason to change numbers; no new byte-identity claim is made.

### State-engine gates and necessary tests

Source: `v4_preflight/state_engine_gates_c.csv`: **53/53** checks pass, including
all 34 inherited B2/B/TODO 023 gates. New checks include:

- C0 matched R6 likelihood error **3.552713678800501e-14**, limit 1e-9;
- H1 pinned gamma_def=0 and H2 kappa=0 likelihood errors **0.0**, limit 1e-9;
- scalar eigen/full-2D mean/covariance/loglik errors at most
  **6.661338147750939e-16**, limit 1e-10;
- independent C0/H1/H2 batch likelihood and RTS means/covariances;
- Gaussian C1 frozen-batch likelihood, RTS means/covariances and exact
  Gaussian observation reduction. These are not posterior convergence gates.

Final t05 necessary suite: **209/209**, `T05_C4_DETERMINISTIC_DONE`, including
first-20 full-book production parity/Hessian/third tensors, conditional-state
identities with singular P, PSD/symmetry, real full-book population/order and
three-real-book nonlinear frozen-factor batch/RTS checks. **Not T05_DONE**:
synthetic parameter recovery is still missing.

Fresh `%46` regressions: t04 **92/92**, t03 **131/131**, t02 **131/131**;
`T04_DONE`, `T03_DONE`, `T02_DONE`. t03's A rerun uses temporary outputs.
Log: `LAPTOP_C4_REGRESSIONS.txt`. An earlier t05 development attempt had
204 passes / one failure: the inherited HalfNormal helper omits constants;
new priors now include their normalisers, fixing the log-prior identity error
5.53723404048801 without changing the specified priors, support or threshold.
That failure is retained in `LAPTOP_C4_FIRST_TESTS.txt`.

### Fixed-parameter full-panel preflight, not fits

Source: `v4_preflight/fixed_parameter_filters.csv`. C0/H1 use
(sigma_obs,sigma_q,sigma_s)=(.07,.03,.01); H2 also kappa=0. C1 uses
(sigma_q,sigma_s,sigma_u,n)=(.03,.01,.06,1000). All 517 fixtures filter and
smooth with finite likelihood and positive smoothed minimum eigenvalue.
These likelihoods have **different observation measures** and are not a model
comparison or evidence that C1 is superior.

| Rung | Fixed collapsed loglik | Minimum smoothed state eigenvalue |
|---|---:|---:|
| C0 | 875.2756941832286 | 1.0707614495298391e-5 |
| C1 (approximate) | 4871.443574381916 | 1.0007842901553005e-5 |
| H1 | 874.8793665077119 | 9.933027762598213e-6 |
| H2 | 875.2756941832286 | 1.0707614495298391e-5 |

Runner reaches `C05_C4_ENGINE_PREFLIGHT_DONE`, **not R05_DONE**. Single warmed
C1 filter took .320236339 s on this laptop (execution metadata in the log,
not a scientific CSV or a sampling benchmark).

## 3. Acceptance status — incomplete

| Requirement | Status |
|---|---|
| Full-book Gate 1, deterministic reductions/batch/projections | Pass at this checkpoint |
| t05 necessary checks and t02–t04 | Pass, excluding explicitly reported C2-pending; full recovery acceptance unbuilt |
| Full-book synthetic parameter recovery | Not implemented/run |
| Four chains, 2000 warmup + 3000 retained, thin 1, all rungs/protocols | No sampler or fits yet |
| Every-parameter convergence | Not run; no posterior promotion |
| Measures, HA posteriors, figures, paired honest forecasts | Not implemented/run |
| Fresh production R05_DONE, byte-identical second beast run | Not implemented/run |
| README, historical preservation, TODO log | Updated; task stays open |

The definition of done is **not met**. This checkpoint establishes the engines,
not that pooling changes rates, improves honest forecasts, learns a useful n,
or answers the human's HA question. No such conclusion is available yet.
All owned panes `%45`/`%46`/`%47` are closed. No `src/`, package/data/grid,
threshold, database, Phase D, later score-grid C2 or unrelated pane changes.

## 4. C2 deferred to the human

**C2 (thin-book pooling) is blocked: Laplace accuracy fails on thin books at realistic-to-wide prediction spreads.**
The integrated 1X2 derivative remains unvalidated. `t05_c2_pending_tests.jl`
runs the unchanged derivative once and reports archived thin Gate 1 checks,
explicitly excluded from C4 acceptance. In `%47`: **29 pass / 10 fail / 39**;
one derivative failure (5.405838441375254e-5, limit 1e-6), nine failed archived
thin settings. It does not regenerate thin quadrature or certify incomplete
1X2 batches. No integrator variation or tolerance change was made.

Options requested by the manager, **none selected**:

| Option | What | Cost | Caveat |
|---|---|---|---|
| (i) exact moment matching (assumed-density filtering) | per-book quadrature of the exact likelihood × prediction: mean, covariance and normaliser, for the ~114 thin books only | slower sampling; feasible on the beast with parallel chains | exact up to quadrature order; needs its own order-stability gate |
| (ii) NUTS on the exact Dirichlet likelihood | the original DESIGN §4.4 route: all books, no approximation | hours of sampling; harder geometry (σ, n, u funnels) | exact; no Kalman collapse |
| (iii) gate at the realistic spread only | re-specify Gate 1 at the C1 posterior's actual prediction spreads | cheap | a threshold or setting change: the human's call |
| (iv) restrict thin-book types | e.g. O/U-only books only (they nearly pass), dropping the ~11 BTTS + O/U books | cheap | changes which data are used: the human's call |
