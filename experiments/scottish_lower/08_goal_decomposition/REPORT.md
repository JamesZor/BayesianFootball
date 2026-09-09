# Goal decomposition: scientific report

## Execution status

Phase 1 incident extraction and initial statistical checks have run. The BBC
referee-source correction has been extracted and verified (47 incident-contract
assertions passed); referee-adjusted inferential analysis is still pending.
The strict four-model Fold 1 smoke passed 64/64, including the six-part
convergence audit, posterior score grids, and fit/portfolio persistence. No
40-fold production or benchmark portfolio result is claimed yet. Measured
evidence belongs in `EMPIRICAL.md`, `STATISTICAL_FINDINGS.md`, and
`results/smoke_fold1_2026-09-09.md`.

## 1. Question and falsifiable hypotheses

Does separating non-penalty, non-own goals from converted penalties and own goals
improve out-of-sample predictions, rather than merely explain observed scores?

- **H1, denoising:** separate component likelihoods improve proper scores against
  `m00_recombined_control`, an identical-prior, identical-latent-structure arm
  whose likelihood sees total goals only.
- **H2, penalty skill:** partially pooled team drawing/conceding effects improve
  predictions relative to a pooled penalty process. Cross-team dispersion alone
  is insufficient: unequal exposure, schedules, division and season must be
  considered, and repeatability is a separate claim.
- **H3, own-goal pressure:** a beneficiary's attacking intensity predicts own-goal
  receipts better than a flat per-side rate.
- **H4, economic value:** predictive improvements survive an identical exchange
  book and portfolio policy. Proper-score gains do not imply profitable edges.

SofaScore `regular` does not identify tactical open play: it may include goals
from corners or free kicks. Accordingly “open play” in the work-package variable
names means **non-penalty, non-own classified goals**, not an independently
observed open-play mechanism.

## 2. Generative model and superposition

For a receiving side, conditional on a posterior draw, let

\[
O\sim\operatorname{Pois}(\lambda_o),\quad
A\sim\operatorname{Pois}(\lambda_p),\quad
C\mid A,k\sim\operatorname{Binom}(A,k),\quad
W\sim\operatorname{Pois}(\lambda_w).
\]

`A` is the number of observed valid in-game penalty attempts (converted plus
missed); it is not independently verified to contain every initial award,
rescinded award, or retake. Shoot-outs must not enter this likelihood.

The probability generating function of converted penalties is

\[
E[z^C\mid\lambda_p,k]
=E[(1-k+kz)^A]
=\exp\{k\lambda_p(z-1)\}.
\]

Under conditional independence of the component processes,

\[
Y=O+C+W\mid\theta\sim\operatorname{Pois}(\Lambda),\qquad
\Lambda=\lambda_o+k\lambda_p+\lambda_w.
\]

Thus each posterior draw supplies a total home and away intensity to the
existing independent-Poisson score-grid kernels. **Recombine each draw first,
then average score probabilities.** Pricing at posterior mean intensities is
not equivalent.

Independence is an assumption, not a consequence of the data labels. Actual
penalties, territorial pressure and match state may share causes. A model
coupling own-goal intensity to latent attack is still conditionally Poisson
if the independent count shocks remain conditional on those latents.

### Posterior prediction is not a single Poisson

After integrating parameter uncertainty,

\[
p(Y=y\mid D)=\int\operatorname{Pois}(y;\Lambda(\theta))p(\theta\mid D)d\theta,
\]

and \(\operatorname{Var}(Y\mid D)=E[\Lambda\mid D]+\operatorname{Var}(\Lambda\mid D)\).
Sharing the conversion parameter across sides also induces posterior-predictive
association after integrating it out. These facts do not invalidate per-draw
Poisson score grids, but do invalidate an unconditional single-Poisson claim.

## 3. Observation likelihood and missing data

On a fully reconciled fixture-side the likelihood is

\[
p(O,A,C,W\mid\theta)
=p(O\mid\lambda_o)p(A\mid\lambda_p)
 p(C\mid A,k)p(W\mid\lambda_w).
\]

Do not append a second likelihood for total goals: the observed total is a
function of these components and doing so double-counts its evidence. Likewise,
use either awarded-plus-conversion or independent converted/missed thinning
likelihoods, not both.

For an unreconciled incident record, a documented total-score likelihood
\(p(Y\mid\Lambda)\) can retain the fixture without inventing component labels.
Its decomposition likelihood must be disabled by a fixed data mask. A missing
feed is not a zero count, including at 0–0: evidence that a feed was collected
is distinct from observing no goals.

Provider own-goal attribution must be established against cumulative score
increments and final score reconciliation. The work package's proposed reversal
is a hypothesis to audit, not ground truth. Ambiguous and unclassified records
must be reported, not silently assigned to the regular component.

## 4. Statistical interpretation of the EDA

### Dispersion and zeros

For any component count, report its sample mean, variance-to-mean ratio (VMR),
observed zero fraction, and the homogeneous-Poisson zero probability
\(\exp(-\bar y)\). These are descriptive checks, not a test that team-conditional
Poisson sampling is wrong. If intensity differs by team, opponent or season,

\[
\operatorname{Var}(Y)=E[\lambda]+\operatorname{Var}(\lambda),\qquad
P(Y=0)=E[e^{-\lambda}]\ge e^{-E[\lambda]}.
\]

Therefore both VMR > 1 and apparent excess zeros arise under an ordinary
heterogeneous Poisson model without a negative-binomial or zero-inflated
observation law. Adjusted dispersion should be assessed against a fitted
team/opponent/home/division/season mean model, ideally by a parametric bootstrap
that refits the same nuisance parameters in each replicate. Sparse-count
asymptotic chi-square approximations require caution.

### Team penalty heterogeneity

A team with twice as many observed matches is expected to draw twice as many
penalties under a homogeneous process. Use counts and exposure, not an ANOVA on
unweighted rates. Under the simplest Poisson null with exposures \(e_t\),
conditioning on the overall count gives

\[
(N_1,\ldots,N_T)\mid N_+\sim\operatorname{Multinomial}
\left(N_+;\frac{e_1}{\sum e_t},\ldots,\frac{e_T}{\sum e_t}\right).
\]

Home/away, division and season can define strata with separately preserved
counts and exposure. Drawing and conceding are two views of the **same**
penalty event, so their p-values are not independent confirmations. A team
random-effect variance at zero lies on the boundary of its parameter space;
a naive ordinary chi-square likelihood-ratio reference is not automatically
valid. Report simulation uncertainty, effect sizes and repeatability rather
than only whether a p-value crosses 0.05.

For repeatability, compare non-overlapping historical blocks after accounting
for exposure and the pooled null. A positive cross-sectional variance can be
real but temporary; a non-significant repeatability estimate can also reflect
low power. Neither licenses the claim that team skill is exactly zero.

### Conversion pooling

Given attempts \(a_t\) and conversions \(c_t\), the pooled likelihood is
\(\prod_t \operatorname{Binom}(c_t;a_t,k)\). A Beta prior gives

\[
k\mid D\sim\operatorname{Beta}(a_0+C,b_0+A-C).
\]

The resulting posterior mean \((a_0+C)/(a_0+b_0+A)\) and interval quantify
pooled conversion uncertainty. Team conversion rates based on a handful of
attempts are intrinsically noisy; teams with no attempts have no observed
conversion rate, not a zero rate. A heterogeneity test conditions on each
team's attempt exposure and estimates the pooled probability under the same
null. Beta-binomial or logistic-normal alternatives must be judged against
that binomial variation rather than the raw variance of team percentages.

### Own-goal pressure and referee identity

Own-goal receipts are attached to the beneficiary. A receiving-side intensity
link can be written

\[
\log\lambda_{w,i}=\mu_w+\rho(\eta_{o,i}-c),
\]

where \(c\) is a fixed, explicitly stated reference log intensity. The flat
model is \(\rho=0\); proportional pressure is \(\rho=1\). Estimating both
\(\mu_w\) and \(\rho\) from a small own-goal count demands shrinkage and a
reported uncertainty interval. A match's realised regular-goal count is a
post-match association variable, **not an available pre-game pressure
predictor**. A predictive test uses frozen, earlier history or fold-trained
latent pressure. Neither an association nor absence of significance establishes
that own goals are purely random.

**Source correction:** the initial audit checked only
`sofascore.matches.raw_data.referee`, not all available referee sources. Its
null result cannot establish absence of referee identity across the segment.
The user identified `bbc.match_officials` (`role = 'referee'`, direct match-ID
join). An independent read-only query verified **2,009/2,019 named fixtures**,
with no duplicate referee rows or conflicting referee IDs on a match. This
source is included in the revised frozen registry
`7571c87570578c63ac9a72c0f24f0d113b082b52b2559b5237485bca47e79951`. The earlier conclusion
that referee effects must be excluded is retracted.

The penalty model will include a centred, partially pooled referee effect,
shared by the two sides of the same match. Estimate referee variation after
considering team, division, season and exposure; a raw high/low rate spread
is neither a causal effect nor proof of predictive value. A reported deviance
reduction of 56.63 on 42 degrees of freedom (p about 0.065) could not be
reproduced on the stated 40-referee cohort. The observed ≥20-match raw cohort
has deviance 54.8497 on 39 df (asymptotic p 0.0475); after reconciliation it
has 39 referees and deviance 50.7590 on 38 df (p 0.0807). All 58 named IDs
give p 0.1083 raw and p 0.1685 reconciled. This sensitivity and sparse counts
support cautious partial pooling, not a conclusive significance claim.

Per the requested prediction policy, a missing or training-unseen referee has
an effect of zero. This is a population-mean **plug-in**, not integration over
uncertainty about a new referee; the distinction must be retained in reports.
Referee IDs are mapped from fitted matches only. Historical named assignments
are not automatically timestamped pre-kickoff observations: absent assignment
publication timestamps, evaluation must explicitly assume the eventual named
referee was known before kickoff, rather than claim verified live availability.

All EDA on the full historical snapshot is exploratory. Selecting model
structure after inspecting seasons 24/25 and 25/26 makes their later benchmark
a retrospective comparison, not an untouched confirmatory holdout. Genuine
confirmation requires a frozen recipe evaluated on subsequent fixtures.

## 5. Model family

The intended open-goal spine is a global intercept and home advantage with
centred, partially pooled team attack and defence effects and 180-day likelihood
time decay. Signs of defence effects must be stated alongside the implementation.
All priors, masks and transforms are frozen before interpreting OOS results.

0. **m00_recombined_control:** the baseline's identical latent components and
   priors, but the component observation mask is forced to zero so every
   fixture uses only its total-score likelihood. This additional arm isolates
   the information supplied by component labels; its component-specific
   parameters may be weakly identified, which must be handled through priors
   and diagnosed rather than hidden.
1. **m01_decomposed_baseline:** global penalty log-rate, separate penalty home
   advantage, hierarchical referee effect, pooled Beta conversion, and a flat
   own-goal receiving rate. The same referee hierarchy is present in the
   identical-spine total-only control.
2. **m02_decomposed_team_penalties:** m01 plus separately shrunk team drawing
   and opponent conceding effects; non-centred, centred team parameters.
3. **m03_decomposed_pressure_own_goals:** m01 with an own-goal pressure link.
   The beneficiary's attacking pressure, not the conceding team's attacking
   intensity, is the relevant predictor. The exact link is recorded with the
   implemented priors. This is **m01 + pressure**, not m02 + pressure, so its
   comparison against m01 does not also change the penalty mechanism.

The existing external `m00_poisson_control` is **not** the same prior/hierarchy
as this suite's regular-goal spine. It is a contextual performance benchmark,
not an identifying H1 ablation. A favourable comparison against that external
run cannot by itself establish that component labels caused the gain.

### Shared referee submodel

For a known fitted referee j, use a non-centred, centred hierarchical effect
\(\gamma_j=\sigma_{ref}(z_j-\bar z)\), with independent standard-normal raw
z values and a proper shrinkage prior on the positive scale. Both sides share
that referee effect:

\[
\log\lambda_{p,h,i}=\mu_p+HA_p+\gamma_{ref(i)}+\alpha_{p,h(i)}+\beta_{p,a(i)},
\quad
\log\lambda_{p,a,i}=\mu_p+\gamma_{ref(i)}+\alpha_{p,a(i)}+\beta_{p,h(i)}.
\]

The team-penalty terms are absent in m00/m01/m03 and present in m02. The
referee term is present throughout, including the identical-spine control.
Unknown/missing referees contribute a structural zero, not a fitted UNKNOWN
category. Names/IDs and their absence are assembled outside the taped model.

### Prior provenance and selection

To avoid full-snapshot prior leakage, rate/conversion centres are to be frozen
using **only usable seasons 20/21 through 23/24**, before either target season.
That historical subset has 1,236 fixtures / 2,472 sides: 3,074 regular goals,
344 attempts, 264 converted penalties, 80 misses and 75 own goals. Its rate
centres are 1.24352751 regular goals, 0.13915858 attempts and 0.03033981 own goals
per side; conversion is 264/344 = 0.76744186. A weak ESS-10 conversion anchor
would therefore be Beta(7.67441860, 2.32558140), **not** the posterior obtained by
reusing all 344 attempts as prior observations.

This is pre-study empirical Bayes: historical matches used to set centres may
also enter training likelihoods, so it is not a wholly independent external
prior. The low pseudo-count strength limits that reuse but does not erase its
provenance. Final implemented prior widths and transforms must be included in
the run manifest. The candidate set and 180-day half-life originate in the
research design/project precedent; pressure and shrinkage prior scales are
research choices, not discoveries from a confirmatory holdout. Full-snapshot
EDA can still influence the later interpretation, so the historical OOS study
remains retrospective.

The m03 pressure predictor is a sampled regular-goal latent intensity, whereas
the EDA uses an earlier-history summary. These are related but different
predictors. Training latents are informed by training outcomes; that is joint
inference, not a pre-game feature observation. Held-out rates must come only
from that fold's training posterior. Report pressure prior/posterior standard
deviations as well as its interval to distinguish learning from prior sampling.

With weights \(w_i\), a powered conversion likelihood and Beta(a,b) prior imply
\(k\mid D\sim\operatorname{Beta}(a+\sum_i w_iC_i,
 b+\sum_i w_i(A_i-C_i))\) conditional on the mask. This supplies an analytic
check on sampled conversion draws. Time decay is a generalized/powered
likelihood, not a claim that old observations were literally fractional trials.
The Beta conjugacy check applies to the classified conversion submodel in
isolation. Once unreconciled fixtures contribute a total-goal likelihood whose
rate contains \(k\lambda_p\), the full posterior for k is **not** conjugate;
it also learns through those totals. A full-model smoke must not incorrectly
require its conversion posterior to equal the isolated Beta update. Compute
the analytic submodel check with the **post-design component mask and actual
decay weights**, not unweighted registry counts. No arbitrary “close enough”
Beta-posterior gate is imposed on the nonconjugate full model.

## 6. Validation and promotion contract

1. Independent likelihood derivation and component conservation on synthetic
   edge cases; score-grid agreement against independent Poisson probabilities.
2. Feature support, typed vectors, stable ID alignment and history-only
   filtration. Modifying held-out incidents cannot change training features.
3. Compiled ReverseDiff versus fresh ReverseDiff and ForwardDiff at recording
   and broad perturbed points, warmed allocation/latency measurements, and
   instruction-count scaling. An allocation-free claim requires measured zero.
4. Production-sized four-chain smoke on one historical fold for each candidate:
   R-hat <= 1.05, minimum bulk/tail ESS >= 200, zero divergences, plus the
   project's remaining six-part audit checks. Failures block promotion.
5. Exact fit/latent database round-trip, canonical recipe registration and
   inference-hash deduplication before sampling; portfolio ledger round-trip.
6. All-fold prepare-only checks before the native queue. Use 16 physical cores
   on mcmc-beast, pin threads and set BLAS threads to one. No competing grid.

A failed gate is a result to investigate. Increasing warm-up/target acceptance
or reparameterizing requires a new recorded recipe, never relaxed thresholds.

## 7. Comparison contract

Published numbers are context, not paired benchmarks. Gen 1's README identifies
`m00_baseline` rather than the prompt's `m01`; Gen 3's 0.6571 belongs to
`m05_joint_production_wealth`, while `m08_joint_composite` reports 0.6572.
Load exact baseline run addresses and recompute on the same fixture/market set.
The external `m00_poisson_control` provides a genuine Poisson benchmark, but
its hierarchy differs from this suite. Attribution to component labels instead
uses the new internal `m00_recombined_control` defined above.

The read-only database audit on 2026-09-08 found an additional hard constraint:
Gen 1's registered `m00_baseline` and `m05_production_wealth` runs carry
`git_commit = synthetic-no-mcmc` and zero runtime. Those imported synthetic
artifacts are **not empirical MCMC comparators**, regardless of their stored
convergence flags. The real Gen 3 controls are:

| Run | UUID | Current coverage |
|---|---|---|
| m05_joint_production_wealth | `5eff755c-3591-48d1-a2cc-5fc2744ddf88` | 40 folds / 710 fixtures |
| m08_joint_composite | `61fc5d87-1bd6-46d1-bb2b-c2aaad39e348` | 40 folds / 710 fixtures |
| m00_poisson_control | `d63f8877-b825-40ae-9ae5-d829e8b8a7f7` | extended to 42 folds / 749 fixtures |

The three share exactly 710 fixture IDs. The original canonical splitter is
`GroupedCVConfig`, tournaments `[56,57]`, two history seasons and target seasons
24/25–25/26. The Poisson control's extra 39 fixtures must not enter the paired
historical comparison. These existing baselines recorded 2, 3 and 6 divergences
respectively under their original gates; they must not be relabelled as passing
this work package's stricter zero-divergence smoke gate. Source evidence:
`results/baseline_audit_runs_2026-09-08.csv` and
`results/baseline_audit_intersections_2026-09-08.csv`.

Report all-match prediction coverage separately from market-specific odds
coverage. Use identical de-vigged Betfair observations for paired proper scores,
with explicitly defined weighting and ECE bins. CRPS requires an ordered scalar
quantity: report marginal home/away goals and/or total-goals CRPS, not an
undefined “scoreline CRPS” or a market-inverted approximation labelled observed
scoreline truth. A 12x12 grid truncates Poisson tails; disclose normalization and
check tail mass.

The requested primary portfolio uses 1X2 + O/U 2.5 and Baker–McHale shrinkage.
Recompute all controls under the same policy and exchange commission. Published
Gen 3 portfolios included BTTS and are not a like-for-like primary comparison.
Moreover, Gen 3's persisted portfolio artifacts used bookmaker `ds.odds`, not
the exchange book used in its published headline. They must not be reused as
Betfair baselines. Returns are historical simulations with optimistic fills,
not prospective profits.

## 8. Incident findings

Initial frozen extraction: 2,019 finished matches, seasons 20/21 through 26/27,
with 5,559 published final-score goals. Of these, **1,992 fixtures reconcile**
with complete classified components; 27 are quarantined. Usable fixtures carry
5,468 goals and 3,984 team-side observations.

### The own-goal reversal in the prompt is wrong for this feed

There are 110 own-goal incidents. Embedded cumulative scores identify the
receiving side unambiguously in 107; **all 107 agree with `is_home` directly**
(51 home, 56 away). Three are ambiguous and their fixtures are quarantined.
The observed feed therefore records the credited/receiving side for these
incidents, not the team committing the own goal. Reversing every own goal would
introduce a systematic attribution error. This conclusion is specific to the
operational feed audited here, not a universal provider API guarantee.

### Reconciled component distribution

| Component | Goals | Share of usable goals | Mean per side | VMR | Zero fraction | Homogeneous Poisson zeros |
|---|---:|---:|---:|---:|---:|---:|
| Regular, non-penalty/non-own | 4,942 | 90.38% | 1.24046 | 1.10462 | 30.6476% | 28.9251% |
| Converted penalties | 419 | 7.66% | 0.10517 | 1.01442 | 90.0853% | 90.0171% |
| Own-goal receipts | 107 | 1.96% | 0.02686 | 1.04817 | 97.4147% | 97.3500% |

Usable records contain 545 penalty attempts: 419 converted and 126 missed,
for a conversion fraction of **76.88%**. Raw incident totals reproduce the
prompt's 5,019 regular, 428 converted, 127 missed, 110 own and 11 unclassified
counts. However, raw goal incidents total **5,568**, not the 5,559 final-score
goals, and the prompt's percentage denominator is inconsistent. Neither raw
incident counts nor its 88.3% regular-goal share should be used uncritically.

Every one of the 2,019 **SofaScore JSON fields queried** lacks referee identity.
The broader conclusion that the matches have no available referee is
**retracted**: `bbc.match_officials` is the newly identified authoritative
source, independently verified to cover **2,009/2,019 fixtures (99.50%)**, with
no duplicated referee assignment rows or conflicting IDs in the join.

The regular component has modest **marginal** excess variation and zeros. This
alone does not establish conditional overdispersion or justify a negative
binomial likelihood; the adjusted tests are required before that conclusion.

## 9. Statistical results and conclusion

Initial tests against the pre-referee registry
`3095efa97d9d3bd9afc12695e5e6ccc9352a5cab3077779bcc464749c72408eb`
completed without MCMC:

| Test | Initial result |
|---|---|
| Conditional regular-goal Poisson refit bootstrap | p = 0.0779 (MC SE 0.0085) |
| Pressure/home/season/division-adjusted penalty drawing heterogeneity | p = 0.0035 (MC SE 0.0013) |
| Corresponding penalty-conceding heterogeneity | p = 0.1594 (MC SE 0.0082) |
| Team conversion heterogeneity, conditioned on attempts | p = 0.0685 (MC SE 0.0056) |
| Pooled conversion, Beta(1,1) prior | posterior mean 0.7678; 95% interval [0.7316, 0.8022] |
| Earlier-history own-goal pressure log-rate coefficient | 0.1264; 95% Wald interval [−0.0613, 0.3140] |

These are **before referee adjustment**, not final evidence for independent
team penalty skill. The chronological-block drawing-rate correlation was
0.0828, a weak descriptive repeatability estimate. See the statistical findings
file for simulation methods and limitations.

### Outcome-dependent quarantine

The 27 quarantined fixtures average **3.3704** final goals, against **2.7450**
in the 1,992 usable fixtures; the fraction with at least four goals is
**51.85% versus 30.17%**. Quarantine is therefore visibly outcome-dependent,
not missing completely at random. It protects against false labels but selects
a lower-scoring component sample. Keeping the quarantined final totals in the
model avoids dropping their score evidence; it does **not** prove that the
component-label missingness mechanism is ignorable. The small fraction
quarantined limits its volume, not automatically the magnitude of bias.

Implementation files for later phases are not validated sampling results.

## 10. Executed deterministic model contract

The real Fold 1 test passed **51/51** on local Julia 1.12.1, with no MCMC.
`results/deterministic_checks_julia_1.12.1.toml` records source hashes and measurements.
All four compiled gradients allocate **zero bytes** after warmup, as do native
in-place 12×12 score grids. Tape lengths are 232/232/307/244 and remain exactly
unchanged when fixture rows are doubled. Forty broad parameter probes agree with
fresh ReverseDiff and ForwardDiff to <2.3e-15 relative error. An independent
scalar distribution-object log joint agrees to <1.7e-15, including the following
exact changes of variables.

### Allocation-free parameterization, unchanged priors

Scalar Normal sites are length-one Normal vectors, so Turing exposes
`TrackedArray` rather than mixing scalar trackers into fused broadcasts.
Counts used in arithmetic are `Float64`, and constant centering projection
matrices replace scalar means in hot broadcasts. These are computational
representations, not changed statistical assumptions.

For a desired HalfNormal scale with standard deviation s, sample u from a
standard Normal base, set sigma = s exp(u), and add

```
log(2) + u - exp(2u)/2 + u^2/2
```

to the log density. This cancels the Normal base and gives exactly
`logpdf(HalfNormal(s), sigma) + log(sigma)`, including the Jacobian.
For the desired Beta(a,b) conversion prior, sample x from a standard Normal
base and let k = logistic(x). Add

```
-a*log1pexp(-x) - b*log1pexp(x) - logbeta(a,b) + x^2/2 + log(2pi)/2
```

to obtain exactly the Beta density plus its logit Jacobian. This is **not** a
logistic-Normal replacement prior. The Binomial likelihood is evaluated directly
from logits with `log1pexp`, remaining finite even when floating-point logistic
rounds to zero or one. No parameter-dependent clamp or guard is taped.

### New teams: explicitly approved prior-predictive extension

Strict fitted-team refusal initially blocked six canonical fixtures: Arbroath
and Inverness in Fold 1, and East Kilbride in Fold 21. The complete audit is
`results/preflight_unseen_team_refusals.csv`. The user explicitly approved
adding teams from the upcoming fixture identities to the hierarchical latent
vocabulary, without reading outcomes or adding observations. New teams receive
sampled attack/defence effects, and sampled drawing/conceding effects in m02;
they are **not** assigned deterministic league-mean rates. Unexpected teams not
declared by that slate still refuse by fixture ID. All four arms use this same
policy; their component-label contrast therefore retains an identical spine.

Fold 1 consequently has 25 teams (23 observed, two prior-only), 39 fitted
referees, 720 training rows and 20 held-out fixtures. Parameter counts are
97/97/149/98. Mutating supplied held-out scores leaves the declaration unchanged.
Perturbing 759 strictly future component/referee rows changes neither fitted
features nor the historical prior anchor. Missing/unseen referees receive
exactly zero effect; the new-team extension does not alter that rule.

Extraction was exercised with two synthetic parameter draws and independently
summed component rates. It matches the native Poisson kernel without renormalizing
its finite-grid tail. These synthetic draws can have large tail mass and are
**not** posterior samples, smoke convergence evidence or benchmark results.
The subsequent real Fold 1 smoke passed those MCMC convergence and exact
fit/portfolio persistence gates for all four models (64/64; zero divergences;
maximum R-hat 1.0051; minimum ESS 1,479). Production comparisons remain a
separate gate. Full smoke diagnostics and immutable run addresses are in
`results/smoke_fold1_2026-09-09.md`.
