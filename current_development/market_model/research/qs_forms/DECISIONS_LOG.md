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

## 2026-10-08 — Batch 02 interpreted; final report, no third batch

- **Relay and definition of done:** manager supplied `4a6d0cd2`; read its SUMMARY
  first, then only the needed tier/projection rows and our edit targets. Finish
  REPORT.md with ≤12 summary lines, evidence-ranked formulations, explicit C0
  comparisons and decisive future tests; correct THEORY.md, log decisions, verify
  scoped changes and commit. No further computer request is necessary.
- **Correct our common-style hypothesis explicitly:** R09 verifies common-mode
  loading ≤1.10e−15 and filtered mean change ≤5.70e−15. C0 zero-sum projects raw
  states before pricing. The proposed raw-state partition was inapplicable, not a
  negative score result. Its checks were NOT_RUN. Active-season cohort mean energy
  is relative to other roster teams, not an observed global scoring factor. A real
  league scoring state requires a future dynamic mu_t fit.
- **Retain 45° on robust predictive evidence:** R11's pooled joint C0−R6 interval
  stays inside ±0.005 for all examined block lengths, methods and pooling weights.
  Circular L8 gives +0.000314 [−0.000590,+0.001185]. No every-league or goal-model
  equivalence is inferred from this pooled market result.
- **Retain style, with league-sensitive shrinkage:** pooled integrated style goal
  gain remains +0.006708 [0.003469,0.009792], mostly totals; Scottish Premiership
  total gain stays negative and Finland positive. Full−frozen-style remains positive
  only as a plug-in deletion comparison, not a refitted/integrated model contest.
- **Bootstrap audit was informative, not a model discovery:** ordinary allocation
  bias −9.27e−5 falls to −1.20e−6 with circular blocks; the old percentile mismatch
  was boundary underweighting. Large score conclusions survive. Do not advertise
  the tiny allocation effect as a new mechanism beyond its Monte Carlo precision.
- **Do not add OU by default:** matched h13 pairs rise to 978 in EPL and 268 in
  Scottish League Two. The latter's normalized style shape now has null rank 0.26,
  undoing the original long-horizon screen; EPL retains rank 0.01, Finland 0.04.
  Low raw style energy and changed time-scale shape are distinct. These local
  nominations require a forward OU/noise-model comparison, not another slope fit.
- **Local tiers are supported, permanent tiers are not:** the named Celtic/Rangers
  upper pair appears in all 128 C0 and R6 paths for 21/22–24/25 but none for 25/26.
  Group-size-aware persistence confirms the earlier pattern and its later change.
  All five seasons share rounds 1–33; do not blame uniquely missing post-split data.
  Conditional-null rank 1 for the multi-season count is post-selected descriptive
  evidence, not a fitted-mixture victory over heavy tails or a universal league law.
- **Final ranking:** projected 45° C0 is the baseline; league log-r pooling and
  separate level/step priors are the strongest design candidates; local quality
  heavy tails/tiers and targeted style-clock/noise fits follow. Stable team-volatility
  types, global nonlinear style and arbitrary coordinate changes are not justified
  as automatic additions. Microstructure and verified division bridges remain open.
- **Stop after two batches:** the remaining decisive work requires new model fits,
  executable quote histories or verified IDs. Batch 03 could not establish those
  within this brief. Partial outputs remain partial; the completed report is a
  bounded research conclusion, not a claim to have completed every proposed test.
- **Final definition of done checked:** REPORT.md has a ten-line summary answering
  all three questions, eight evidence-ranked formulation groups with C0 comparison
  status and decisive future tests, hidden-dynamics findings, Spec A recommendations
  and a partial-work audit. THEORY.md explicitly corrects the R09 hypothesis and
  records the final R10–R12 interpretation. Staged changes are exactly these three
  documentation files in qs_forms; source code, fits and answer artifacts are
  untouched. `git diff --cached --check` and `./scripts/todo.sh check` passed
  (24 tasks; AGENTS.md 19,673 bytes). No third batch or new model fit was launched.
