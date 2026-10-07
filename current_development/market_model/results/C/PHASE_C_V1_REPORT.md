# TODO 039 — Phase C BLOCKED at mathematical preflight (2026-10-06)

**No pooled fit was promoted or launched.** The brief's likelihood/projection
contract needs correction or clarification by the manager. This is a blocker
handoff, not Phase C completion. Double Poisson stays for C; the later score-grid
Phase C2 and Phase D were not attempted. The authorised pooled rung called C2
is distinct from that later phase.

## 1. Evidence and interpretation

Source: `laplace_preflight_summary.csv`. The first 30 accepted books of each
kind in Phase A's match-ID ordering were used, without selection on errors.
Predictions are Gaussian in the two log-rates, centred at the isolated inversion,
with equal-side SD shown below. **n is the PRIOR median times 0.25/1/4, not an
estimated posterior median.** There has been no posterior fit. Consequently
these are necessary-condition diagnostics, not the complete §3 production gate
(which also requires totals-only quadrature at posterior-dependent settings).

Every integration evaluates the **exact** Dirichlet/Beta density with all
normalisers on the unchanged production double-Poisson grid. Gaussian importance
coordinates around the integrand mode improve quadrature resolution; they do
not replace the integrand by a Gaussian. Orders 16 and 32 are independently
evaluated. `laplace_preflight_fixture.csv` preserves both integrals, the literal
prescribed approximation, and the unprojected approximation on every fixture.

| Book | n | Prediction SD | Median absolute error, nats | p95, nats | Fixed 0.01/0.05 limits met? |
|---|---:|---:|---:|---:|---|
| full | 250 | 0.05 | 0.006335 | 0.011610 | yes |
| full | 250 | 0.20 | 0.014120 | 0.024257 | no |
| full | 1000 | 0.05 | 0.010268 | 0.017672 | no |
| full | 1000 | 0.20 | 0.013041 | 0.025335 | no |
| full | 4000 | 0.05 | 0.011706 | 0.021557 | no |
| full | 4000 | 0.20 | 0.012568 | 0.023353 | no |
| 1X2-only, supremacy-only | 250 | 0.05 | 0.047207 | 0.113871 | no |
| 1X2-only, supremacy-only | 250 | 0.20 | 0.333277 | 0.528748 | no |
| 1X2-only, supremacy-only | 1000 | 0.05 | 0.130108 | 0.276218 | no |
| 1X2-only, supremacy-only | 1000 | 0.20 | 0.755114 | 0.933657 | no |
| 1X2-only, supremacy-only | 4000 | 0.05 | 0.331083 | 0.526141 | no |
| 1X2-only, supremacy-only | 4000 | 0.20 | 1.341687 | 1.508102 | no |

Largest quadrature-order difference across the entire table is
**1.0012257689595572e-10 nats**. At n=1000, SD=0.20, retaining both 1X2
directions gives median absolute error **0.0004892284282393966**, versus
**0.7551135925625077** under the prescribed projection. This isolates a
likelihood change from a quadrature-resolution problem. It is NOT a proposal
to retain the biased 1X2 totals in the pooled model.

Source: `totals_projection_preflight.csv`. All **36** no-complete-1X2 books
are retained. Only **7** satisfy the literal rank-one/10-degree requirement.
For example:

- OU2.5-only fixture **12476490** has one retained eigen-direction, at
  **11.21831698761122 degrees** from the equal-weight log-rate direction.
- BTTS+OU2.5 fixture **12476754** has eigenvalues
  **0.034234457618670444 / 1.4021838762587322**, both above the specified
  relative retention threshold. It keeps **two**, not one, directions.
- Some multiple-O/U-only optimisations drift to nearly zero away rates. These
  are diagnostic inversion artefacts, NOT accepted new rate estimates or data
  repairs. They remain visible and were not dropped to obtain a favourable gate.

Single-start totals inversion uses Phase A's first approved start; it is only
an expansion-point diagnostic for books that Phase A refused. No new acceptance
of those isolated rates is claimed. A symmetric start could change a flat
inversion's coordinates; it cannot make a rank-two BTTS+totals likelihood
rank-one or make a curved likelihood globally linear.

## 2. What is wrong with the mathematical contract

### Dirichlet large-n loss has the opposite KL orientation

For a market with S selections, positive p and normalised q,

```
log Dirichlet(p; n q)
 = -n KL(q || p) + (S-1)/2 log n + 1/2 sum_s log q_s
   - sum_s log p_s - (S-1)/2 log(2pi) + O(1/n).
```

Thus Phase A's minimiser of KL(p||q) is **not in general its large-n maximiser**.
The Hessian of the forward cross-entropy is likewise not generally the exact
Dirichlet likelihood curvature. They agree locally when q=p, but full books
have measured grid misfit. The diagnostic tests verify the reverse-KL limit
against the exact density. An exact maximiser is explicitly allowed by the
brief, but changing the centre alone does not correct the stipulated H or
resolve the projection problems below.

### Projecting a 1X2 likelihood changes the density tested by quadrature

For R=(nH)^-1 the full quadratic kernel integrates against N(mean,P) as

```
exp(log L(hat)) (2pi)^(k/2) det(R)^(1/2)
    * Normal(hat; mean, P+R),       k=2.
```

The preflight implements the brief's matching one-direction formula literally:
k=1, observation d' hat, variance d'R d, with the stated peak and determinant
factor. That projected kernel no longer penalises the total direction. The
exact Dirichlet likelihood *does*. Its integral against a Gaussian prediction
therefore differs, even if the unprojected Laplace approximation is excellent.

Integrating the discarded direction against a flat Lebesgue measure is not
integrating it against the Gaussian prediction. The latter depends on the
prediction's total mean/variance and can feed totals information to the state,
contrary to the fixed supremacy-only decision. A projected/powered likelihood
can be a modelling choice, but its appropriate exact reference must be defined
before claiming the requested exact-Dirichlet marginal gate.

### Totals do not define a fixed equal-weight log-rate direction

An O/U market primarily constrains exp(theta_h)+exp(theta_a). Its local normal
in log-rate coordinates is proportional to (lambda_h,lambda_a), not (1,1).
The constant-total contour is curved; a linear projection v' theta is not
invariant along it. A synthetic O/U book with unequal rates already violates
the fixed 10-degree direction requirement, as the diagnostic tests demonstrate
on the unchanged grid. Small cutoff effects do not rescue that requirement.
BTTS plus totals can identify both rates (up to swapping), as the real Hessian
witness above shows. Merely retaining every non-negligible eigen-direction is
therefore incompatible with the instruction that these books inform total only.

## 3. Acceptance evidence — explicitly incomplete

| Brief acceptance item | Status |
|---|---|
| t05 pooled tests and t02–t04 in fresh REPL | **Not met.** Only the new mathematical diagnostic suite ran: 11/11, `T05_PREFLIGHT_DONE`, 3.2 s in fresh owned laptop pane `%39`. No pooled t05/recovery suite exists; prior suites were not rerun. Their code/artifacts were not modified. |
| All §3 gates | **Not met.** Necessary likelihood/projection checks fail as above. C0/R6, scalar/full update and batch-engine gates were not implemented. |
| Convergence everywhere | **Not run.** No C0/C1/C2/H1/H2 or synthetic sampling; no posterior n or sigma_u. |
| Every table and figure | **Not met.** Only the three diagnostic CSVs listed above and this report/reproduction notes exist. No ladder, forecast, shrinkage, convergence, home-advantage or team-path tables/figures are claimed. |
| Fresh beast R05_DONE and second byte-identical production run | **Not run.** No beast checkout/session or production r05 was created. Scientific hashes cover only the diagnostic CSVs; no production reproducibility claim. |
| README and DESIGN mapping | **Met for blocker documentation.** README has Phase C evidence and the human's double-Poisson/C2-later decision. DESIGN §5 is unchanged. |
| TODO 039 dated @pi line and tracker check | Updated to **BLOCKED**, with matching index row and this dependency. Check result is recorded in `REPRODUCIBILITY.md`. |

The preflight loader is `../../l05_laplace_preflight.jl`, runner
`../../r05_laplace_preflight.jl`, mathematical tests
`../../t05_laplace_preflight_tests.jl`. They intentionally do not masquerade as
the requested `r05_pooled.jl` or `t05_pooled_tests.jl`. The loader docstring
contains the likelihood normalisation derivation and its limits.

## 4. Plain words and choices

There is **no evidence yet** about whether pooling changes the rates, improves
honest forecasts, raises thin-book totals by about 0.10, or corrects T015
supremacy. There is no posterior concentration to interpret as book worth, and
no estimate of its ratio to fixture noise. No ratings or HA inference is
available. Fabricating those requested conclusions would hide a failed model
contract.

In-scope choices: test mathematical necessary conditions before expensive fits;
use the pinned close and unchanged grid; deterministic first-in-ID-order books;
explicit prior-centre n and Gaussian prediction spreads; independently increase
quadrature order; retain all totals-book failures. No packages, data, `src/`,
database writes, thresholds or prior-phase artifacts were changed.

The brief supersedes DESIGN's old Turing/static smoke/MAP-fold route with
quality/style GRW and collapsed slice sampling. That mapping is documented,
not silently implemented as the old design. The new likelihood/projection
issues block either inference route; no out-of-scope alternative was fitted.

## Blocked / next action

The manager must reconcile **total/supremacy-only information restrictions**
with the **exact raw-book likelihood used as the gate reference**, and specify
a valid totals-coordinate approximation. For full books, authorise/clarify the
finite-n maximiser and exact curvature instead of the erroneous forward-KL
expansion. Keep the double-Poisson grid and fixed error thresholds.

Then issue a corrected Phase C brief. No attempt was made to tune n, prediction
spreads, starts, fixtures or thresholds until a red gate became green. This
handoff stops before pooled implementation/sampling, with the fixed decisions
and failures intact.
