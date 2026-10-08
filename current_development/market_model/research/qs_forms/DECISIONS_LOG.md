# Quality/style formulations — decisions

## 2026-10-08 — Batch 01 design (before computer results)

- **Phase definition of done:** THEORY.md covers the requested coordinate, hierarchy,
  dynamics and market/truth alternatives; requests/batch_01.md specifies reproducible
  existing-fit diagnostics, output schemas and decision rules; these documents and
  this log are committed. No code, new MCMC, remote execution or final verdict here.
- **Evidence boundary:** trust ASTRA_BRIEF.md; read only the permitted report sections,
  code ranges and CSV headers. The source report specifies *summed marginal*, not
  joint, market forecast scores. Small C0−R6 differences do not establish equivalence.
- **Separate observation coordinates from process covariance.** The 45° transform is
  exactly interpretable even if the process ellipse tilts. R6 already contains every
  unrestricted 2×2 Gaussian step covariance; a learned rotation is not a new model.
- **Correct the meaning of total.** q cancels from log-rate product, not expected total
  goals: lambda_h+lambda_a = 2 exp(M) cosh(D/2). Other-scale coordinates must retain
  this coupling. This is algebra, not an additional empirical finding.
- **Do not infer a factor hierarchy from two indicators.** Unrestricted one-factor
  loadings plus two uniqueness variances are not identified from three covariance
  entries. Unequal scales, covariance tilt and factor loadings are not three separate
  predictive discoveries.
- **Prioritize R01–R03:** exact posterior geometry, posterior level/step comparison,
  and genuinely pre-update predictive residuals. R04–R06 are null-calibrated screens
  for mechanisms, not estimates of alternative fitted models.
- **Avoid smoother-induced discoveries.** Posterior-mean increments attenuate noise;
  FFBS increments inherit the fitted random-walk prior. Dynamics screens require
  simulate → filter/smooth at fixed existing theta → same statistic, without refits.
- **Protect 10b:** training-only theta; forecast the entire week before any update;
  no test-season smoothing used as an honest forecast. Level geometry is 10a only.
- **Treat gauge and selection explicitly:** a uniform league q shock is invisible;
  a uniform s shock aliases the league intercept. Thin-book/market-sharpness claims
  cannot be identified from final inverted rates alone. Missing timestamps or
  liquidity means NOT_IDENTIFIABLE, not permission to fetch data.
- **Keep batch bounded:** eight requests reuse a common draw/path/forecast cache;
  simulations at fixed theta only, no candidate-model fitting. Return partial
  diagnostic failures explicitly rather than expanding the assignment.
- **Provisional recommendation, not a result:** retain interpretable q/s coordinates;
  investigate league-varying r, then add dynamics only if calibrated residual checks
  nominate them. Do not yet recommend OU, mixtures, jumps or stochastic volatility.
- **Phase verification:** THEORY.md has all six required candidate areas/decision
  sections; batch_01.md has R01–R08 with inputs, computations, CSV schemas,
  bounded summaries and decision rules. Staged changes are exactly these three
  requested documents, all under qs_forms. `git diff --cached --check` passed;
  `./scripts/todo.sh check` passed (24 tasks; AGENTS.md 19,673 bytes). No computation
  results are claimed and no final REPORT.md is written before the relay.

## 2026-10-08 — Batch 01 interpreted; focused batch 02 authorized by the brief

- **Relay:** manager supplied batch 01 at `a95ca9ca`; read SUMMARY first, then only
  targeted CSV headers/rows, manifest entries and the document sections to edit.
  R07/R08 remain PARTIAL; do not relabel integrated frozen-style or movers complete.
- **Done for this phase:** update THEORY.md with measured findings and limitations;
  specify only follow-ups that could change the final recommendation; commit the
  updated log/theory and requests/batch_02.md, then return to Claude without polling.
- **45° is retained on stronger evidence:** R01 finds no tilt passing the >10°/0.9
  screen. R03 pooled *joint* C0−R6 score +0.00031 [−0.00018,+0.00138] meets the
  ±0.005 practical-equivalence tolerance. This is not universal league equivalence.
- **Prioritize league r, not a universal 0.3:** R08's English/Scottish tier slopes
  agree in direction in both protocols; SD(log r) is about 0.30. No hierarchical
  model has yet been fitted, so improved pooling prediction remains a future test.
- **Keep style, distinguish its location:** R07's integrated total-goal benefit
  accounts for almost all +0.00671 nats/fixture of style benefit, with Scotland/
  Finland opposite signs. R02 level/step ratios in top flights are far below one.
  R09 will partition common scoring level from centered team style; these are not
  interchangeable, and dynamic common style must not be duplicated by dynamic mu.
- **OU is not yet earned:** R04's EPL/Scottish League Two style slopes/long-lag
  ratios are suggestive, but EPL h=13 has only 48 eligible pairs; Scottish League
  Two h=8 has 21 and h=13 none. R10 changes only eligibility/cohort diagnostics,
  using cached nulls, to distinguish horizon-selection/energy from genuine shape.
- **Do not mistake algebra for a common shock:** centered off-diagonal products
  reduce to negative energy. R09 checks the estimator identity before interpreting
  R05's extreme ranks as any independent dependence finding.
- **Persistent team volatility and global nonlinear style are downgraded:** team
  energy ranks do not persist across halves; selected curvature does not improve
  over a linear predictor robustly. EPL q energy concentration remains a local
  event/noise/SV nomination, not a proven jump mechanism.
- **Scottish tiers are a specific candidate, not a general explanation:** R06 has
  repeated extreme gaps below Rangers, but majority accuracy and the changed 25/26
  gap need checking. R12 examines posterior/rung membership, group-size-aware
  persistence and partial windows, reusing the existing conditional-null search.
- **Audit uncertainty rather than forcing agreeable intervals:** R07 allocation
  means lie above some percentile upper bounds. That is possible without a bug;
  R11 checks moving-block boundary weights and circular/basic-interval sensitivity.
  No claim of failed verification or an invalid score is made from this alone.
- **Scope restraint:** four follow-ups R09–R12, no new fits or simulation panels,
  no integrated-frozen workaround, no quote retrieval or unverified mover joining.
  This is batch 2 of at most 3; expect to finish after it if diagnostics resolve the
  material ambiguities, and retain explicit uncertainty if they cannot.
- **Phase verification passed:** staged whitespace check and `./scripts/todo.sh
  check` (24 tasks; AGENTS.md 19,673 bytes). Exactly THEORY.md, this log and
  requests/batch_02.md changed; all are in qs_forms. R09–R12 each specify inputs,
  formulas, CSVs, ≤10 summary lines and decision rules. No computer code/results
  were edited, no new computation was run, and no final REPORT.md is claimed.
