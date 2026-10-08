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
