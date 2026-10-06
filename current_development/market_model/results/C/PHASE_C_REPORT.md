# TODO 039 — Phase C revision 4: recovery solver BLOCKED (2026-10-06)

**Not complete.** C0/C1/H1/H2 full-book scope remains authorised; C2 remains
separately deferred. A dedicated sampler and full-book synthetic generator now
exist, but C1 recovery aborted twice on the same joint-mode line-search failure.
Stop under the two-failure rule; no third numerical variation was attempted.
**No completed posterior fit, retained recovery draws/intervals, convergence,
production forecasts, learned n/u, shrinkage/HA/path posterior figures or two-run
production reproduction exists.** Definition of done is not met. TODO 039 is
BLOCKED on solver robustness at supported parameters, not awaiting scope approval.

Statistical source: `9de98548` (final sampler/engine/tests). Previous source
`c2547742` used finite differences for synthetic books; `19ad4b99` was the
first generator attempt. Engine-only checkpoint documentation is archived as
`PHASE_C_ENGINE_CHECKPOINT_REPORT.md`, `REPRODUCIBILITY_ENGINE_CHECKPOINT.md`
and `HANDOVER_ENGINE_CHECKPOINT.md`. Earlier revision evidence remains untouched.

## 1. Implementation added

`l05_fullbook_sampling.jl` is included inside PooledMarket after the state engine.

- Dedicated coordinate slice chains using the FullBookRung names and normalised
  priors, not R6 fits/priors. Positive physical draws are exponentiated; H2 kappa
  stays untransformed. Four chain-local Xoshiro seeds; 2000 warmup + 3000 retained,
  thin 1. Width adaptation ends at warmup. Frozen seeds: `fullbook_seeds.toml`.
- Explicit representability checks for positive variances and concentration,
  **no inherited whole-target ±12 bound**, especially not on log n. Supported
  mode/PSD failures propagate as errors, never posterior rejections. Failed
  chains log their last evaluated coordinate and cancel peers cooperatively;
  `@sync` waits for every task before returning an error.
- `restrict_fullbook` restricts both fixture pairs and metadata, then joins
  markets by fixture ID. It avoids MID.restrict_panel's retained full metadata
  and any positional reuse of a full-panel market vector in honest training.
- Synthetic state/deviation population on the original 517-fixture / 91-week
  schedule, independent initial raw alpha/beta prior, ordinary q/s innovations,
  static mu/gamma and same zero-sum design. C1 books are independent market
  Dirichlet(n*q) draws on the unchanged production cutoff-10 grid; no thin books,
  inversion gate, seed selection, clipping or redraw. Synthetic `obs_y` holds
  latent generating theta, **not isolated scoring targets**.
- Synthetic probabilities are retained as logp using the exact identity
  Gamma(a)=Gamma(a+1)*U^(1/a) when a<1. This prevents probability underflow from
  becoming an artificial zero-probability observation. Display p can underflow;
  the likelihood consumes logp. Density normalisers and model remain unchanged.
- Primal synthetic density/marginal uses the production probability grid.
  Dual synthetic evaluation uses the existing parity-tested AD grid algebra.
  Synthetic derivative dispatch is AD; real books keep their existing FD
  gradient/Hessian and AD third tensor. **No change** to joint_mode tolerances,
  clipping, Laplace marginal, skewness formula, real-book density or quadrature.

`r05_fullbook_recovery.jl` exposes configuration, model/truth, frozen seeds,
beast-only runtime, pinned population, engine gates, prescribed recovery fit,
convergence/interval gates and evidence in numbered sections. It refuses an
existing completed binary fit and requires loaders before deserialization.
`r05_pooled.jl` is still the deterministic preflight, **not production**.
Production training/evaluation/figures remain unimplemented.

## 2. Verification which passes

### Fresh laptop final-source tests

Owned `%52`, Julia 1.12.1, 8 threads, BLAS=1:

| Check | Result |
|---|---|
| Accepted deterministic t05 | **270/270**, T05_C4_DETERMINISTIC_DONE |
| Separate C2-pending | **29 pass / 10 fail / 39**, explicitly excluded |
| t04 | **92/92**, T04_DONE |
| t03 | **131/131**, T03_DONE; temporary A outputs |
| t02 | **131/131**, T02_DONE |

Logs: `LAPTOP_C4_AD_TESTS.txt`,
`LAPTOP_C4_FINAL_SAMPLING_REGRESSIONS_PREFLIGHT.txt`.
Tests cover deterministic generation, covariance/normalisers, seed repeatability,
synthetic raw/AD derivative parity, underflow-safe logp, exact Beta equivalence,
physical transforms, numerical support without the ±12 bound, error propagation,
ID-safe restriction, and all inherited engine checks. **Not T05_DONE**:
parameter recovery is incomplete. Earlier development attempts are preserved in
`LAPTOP_C4_SAMPLING_DEVELOPMENT.txt` (Float32 indexing typo, then an overbroad
interior-book requirement in a state-covariance ensemble). No gate was weakened.

### Real full-book preflight regenerated without change

Final `%52` reaches `C05_C4_ENGINE_PREFLIGHT_DONE`. All **six** CSVs in
`v4_preflight/` are byte-identical to the preceding engine-only checkpoint;
`diff -rq` found no differences. This checks the synthetic-only dispatch did not
change real-book outputs. It is **not** production reproduction.

- Full Gate 1: **12/12 settings**, 360 fixture-setting rows, same first 30 books,
  n=250/1000/4000, SD=.05/.20, offsets 0/+.10. Maximum median absolute marginal
  error **.003980333011027959** nats (limit .01), maximum p95
  **.0051705855019272395** (limit .05), maximum mean error/SD
  **.0009284497092163062** (limit .05), maximum SD relative error
  **.010184310576972222** (limit .05).
- State gates **53/53**, including 34 inherited B2/B/TODO 023 gates. C0 matched
  R6 likelihood error **3.552713678800501e-14** (limit 1e-9); H1/H2 pinned
  reductions **0.0**; eigen-scalar/full errors <=**6.661338147750939e-16**.
- All four fixed-parameter 517-fixture filters/smoothers remain finite/PSD.
  These are not fitted parameters, posterior comparisons or convergence.

Sources: `v4_preflight/laplace_gate.csv`, `state_engine_gates_c.csv`,
`fixed_parameter_filters.csv`. Frozen-factor C1 batch gates check a Gaussian
surrogate, not globally exact Dirichlet collapse. C1 RTS remains approximate.

## 3. Beast recovery attempts and blocker

Dedicated checkout `/root/BF_runs/market_model_c`, no unrelated pane operated.
Julia 1.12.4, 16 threads, core pinning, BLAS=1, existing Manifest, pinned cache
SHA256 `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`,
metadata-preserving copy, max_age_hours=10^6. Initial load .02; no DB writes.
Truth: (sigma_q,sigma_s,sigma_u,n)=(.03,.01,.06,1000); generation seed **3962**,
chain seeds **4961–4964**, unchanged across attempts.

1. **19ad4b99 / %261:** generation stopped at fixture **15336943**, before
   sampling: ordinary Dirichlet probability storage underflowed. Generating
   log rates **[-.2925837158254979, 2.7903082673178803]**; production-grid
   OU0.5 under probability **5.881035960900128e-7**, concentration shape
   **.0005881035960900128**. Exact log-space Gamma/Dirichlet representation
   fixes this without narrowing the prior or selecting another seed/fixture.
   `BEAST_C4_RECOVERY_FIRST_ATTEMPT.txt`, `v4_recovery_first_attempt/`.
2. **c2547742 / %262:** all synthetic books generated; slice sampling stopped
   on **joint-mode line search stalled**, gradient
   **[-2.533547593941421e-5, 2.8639610683001138e-5]**. No fit was serialized.
   Synthetic FD roundoff was a plausible diagnosis, not established as the sole
   cause. `BEAST_C4_RECOVERY_LOGSPACE_ATTEMPT.txt`, `v4_recovery_logspace/`.
3. **9de98548 / %263:** synthetic-only AD derivatives still fail the **same
   joint-mode line search**. Actual failing chain seed **4964**, warmup
   iteration **8**, last supported coordinate
   **[-3.549527585137839, -4.460929121755582, -2.821549571263347,
   7.9567722491577495]**. Gradient
   **[-1.5699131339808048e-5, 3.249019587192592e-5]**. Seeds 4961/4963 cancel
   at iteration 7, seed 4962 at 8. Full CompositeException expansion confirms
   the root failure, not just the three peer-cancellation exceptions.
   `BEAST_C4_RECOVERY_AD_ATTEMPT.txt`, `v4_recovery_ad/`.

The latter two binary synthetic panels are **byte-identical**, SHA256
`de6a986b1caebc2a2ad1386308c7f4613bf69ea2f71440778783476c348da0d1`.
This proves no synthetic seed/population redraw between derivative attempts;
**it is not two-run fit/production reproduction**. Binaries remain on beast,
loaders required. Every attempt's 53 deterministic engine gates passes.

**Stop:** the same solver check failed twice. Do not try a third numerical
variation, turn supported failures into -Inf, change seeds, narrow priors,
exclude extreme fixtures, adjust grid/data/settings or loosen thresholds.
The remaining dependency is a reviewed robustness diagnosis/remedy for the
joint-mode solver at the frozen supported coordinate. Passing the first 30
real-book Gate 1 settings does not establish robust recovery/sampler support.
No scientific recovery failure/interval is inferred from an aborted solver.

## 4. Acceptance status

| Requirement | Status |
|---|---|
| Full-book Gate 1, deterministic reductions/batch/projections | Pass |
| t05 deterministic checks and t02–t04 | Pass; recovery acceptance incomplete |
| Dedicated sampler/generator/frozen seeds | Implemented, sampling aborted |
| Full-book synthetic parameter recovery | **BLOCKED by repeated mode failure** |
| Prescribed production C0/C1/H1/H2, 10a/10b | Not run/implemented as workflow |
| Every-parameter convergence | Not available; no inference promotion |
| Measures, shrinkage/u smoothing, HA posterior mixing, figures | Unimplemented |
| Fresh production R05_DONE, second-run byte identity | Not available |

No inference answers are available: pooling's forecast benefit, learned book
concentration, retained full-book deviation, ratings and split/quality-linked
HA remain unanswered. H1 states must be reported with state uncertainty and
hyperparameter mixing, not sampled theta columns; conditional u smoothing is
still required. Ridall's source pages have not been checked in this continuation,
so no new citation claim is made. R05_DONE/T05_DONE/PHASEC4_DONE are not printed.
All owned panes %49–%52 and beast %261–%263 are closed; none is running.

## 5. C2 deferred to the human

**C2 (thin-book pooling) is blocked: Laplace accuracy fails on thin books at realistic-to-wide prediction spreads.**
Unchanged C2-pending derivative discrepancy **5.405838441375254e-5**, limit 1e-6;
nine archived thin Gate 1 settings fail. Pending checks execute/report separately,
not accepted or newly regenerated. No C2 numerical investigation was made.
Options requested by the manager, **none selected**:

| Option | What | Cost | Caveat |
|---|---|---|---|
| (i) exact moment matching (assumed-density filtering) | exact-likelihood quadrature mean/covariance/normaliser for ~114 thin books | slower sampling; feasible with parallel chains | needs order-stability gate |
| (ii) NUTS on exact Dirichlet likelihood | original DESIGN route, all books | hours; harder funnels | exact; no Kalman collapse |
| (iii) gate at realistic spread only | C1 posterior's actual spreads | cheap | setting/threshold change, human decision |
| (iv) restrict thin types | e.g. O/U-only, drop ~11 BTTS+O/U | cheap | changes data, human decision |

No Phase D/later score-grid C2, src/package/data/grid/threshold changes,
database writes, merge/rebase/force-push or stash operations. Published A/B/B2/B3
artifacts remain untouched. Source/scientific hashes identify this blocked
checkpoint only. See REPRODUCIBILITY.md and HANDOVER.md for exact evidence paths.
