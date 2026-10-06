# TODO 039 — Phase B3 report (2026-10-06)

Phase B3 only. Statistical source: `2e02a7854a9ccc59fd8ba4c81aa6cf9ca6f5090a`.
Laptop acceptance tests and both fresh full beast executions pass. All 32
scientific CSVs and all three PNGs reproduce byte-for-byte at the source SHA.
No Phase C, `src/` edits, package/data changes, outside data or database writes.

## Plain-language findings

**Gaussian R6 dependence is adequate for these observable posterior predictive
checks.** There is no extreme PPC evidence for heavy-tailed quality jumps,
directional tail asymmetry or volatility clustering. This is **not proof** of
Gaussianity or absence of tail dependence: this thin market panel leaves many
FFBS steps prior-dominated. Replicates are refiltered specifically to account
for that shrinkage; the check can still have limited power. No alternative step
model was fitted, as the brief explicitly requires when no p-value is extreme.

**Dixon–Coles G1 reproduces the book best among the tested grids.** It substantially
reduces the in-sample draw deficit and BTTS discrepancy. Its pooled heldout-line
absolute-error advantage over G0 is modest, not evidence of decisive forecast
superiority. Bivariate Poisson and Frank improve in-sample KL but have worse
pooled heldout absolute errors than G0.

**G1 largely removes the mean 1X2-only totals bias, but not per-fixture uncertainty.**
The mean shifts from −0.102716 to −0.008941. That remaining mean is small but its
CI excludes zero; the G1 mean absolute bias is still 0.040676. A corrected mean
is not a guarantee that each thin book's total is right.

**There is no supported change in realised outcome log loss.** All paired CIs for
grid versus G0, and for grid versus the available de-vigged close, include zero.
These retrospective close-based rates are not a betting or forecasting promotion.

**R6 looks structurally similar on G1 rates.** Its innovation correlation remains
strongly negative; on the same structural population it slightly worsens the
supremacy point RMSE while improving level RMSE and density. Adding thin books
is informational only and changes the evaluation population.

**The human's fixed decision stands: double Poisson for Phase C; C2 later.**
Nothing here recommends or implements a Phase C grid change. Negative
attack/concessions **team-step** correlation and positive Frank **goal-score**
dependence concern different random variables; their signs are not contradictory.

## 1. Gaussian dependence and tails

Source: `tail_ppc.csv`. Each row uses 200 theta draws and 1,466 active centred
team-week steps; `tail_ppc_draws.csv` saves both statistic distributions and theta
indices. Active means step endpoints from each team's first through last accepted
match week **within a season**, including intervening weeks and first-match weeks
when t>=2. Summer gaps/prior-only teams are not counted. Model simulation still
uses every original fixture/week, including summer.

At every theta draw: observed FFBS path → statistic; independent Gaussian step/noise
replicate conditional on that draw's initial state → refilter → replicate FFBS
path → statistic. Both 10a and 10b PPCs smooth the full panel retrospectively;
10b only restricts theta fitting to the first season. They are not honest forecasts.

P-values are P(T_rep >= T_obs), not a comparison with unfiltered Gaussian theory.
Full-panel 10a distributions below give medians [5%,95%] across FFBS/theta draws:

| Statistic | Observed | Refiltered replicate | PPC p-value 10a / 10b |
|---|---|---|---|
| Quality excess kurtosis | 0.067465 [−0.116453, 0.299440] | −0.022014 [−0.190716, 0.178808] | 0.295 / 0.365 |
| Style excess kurtosis | −0.021486 [−0.190465, 0.221032] | −0.032902 [−0.192287, 0.203270] | 0.470 / 0.470 |
| Kendall tau(alpha,beta) | −0.612739 [−0.671151, −0.559083] | −0.612671 [−0.677586, −0.554307] | 0.520 / 0.630 |
| Joint attack-up/concessions-down 95% events | 0.026603 [0.021794, 0.032094] | 0.026262 [0.021828, 0.030696] | 0.535 / 0.480 |
| Joint attack-down/concessions-up 95% events | 0.026603 [0.021828, 0.032060] | 0.026603 [0.021112, 0.031378] | 0.480 / 0.420 |
| Directional frequency difference | −0.000682 [−0.005491, 0.005457] | 0 [−0.004809, 0.005457] | 0.535 / 0.540 |
| Lag-1 correlation of squared quality steps | 0.022793 [−0.020007, 0.082478] | −0.003227 [−0.046632, 0.042039] | 0.280 / 0.325 |

The observed Kendall-minus-Gaussian benchmark median is −0.000119 (10a) /
−0.003460 (10b), using the draw-specific (2/pi)asin(rho), not the transform of
a marginal median. `tail_ppc.csv` additionally reports alpha/beta kurtosis,
quality/style Kendall and its benchmark difference, both quality/style tail
quadrants, and squared-step lag correlations on all axes. **Every** p-value is
inside the fixed non-extreme region; no subset was chosen to avoid an alternative.

The finite 95% joint events do not estimate asymptotic copula tail coefficients.
The p-values have Monte Carlo resolution from the saved draws; non-extreme
values are not calibrated frequentist uniform tests or Bayes factors.

### Largest quality steps for the human's manager-change check

Source: `largest_quality_steps.csv`, ranking absolute posterior **median signed**
Delta q, with 5/95% path intervals. It lists the largest 15 separately for each
theta protocol. Dates are Monday step endpoints, not inferred managerial dates.
The leading 10a decline is Edinburgh City in the week of 2026-03-30:
Delta q **−0.073527** [−0.121701, −0.016343]; its neighbouring weeks also appear.
The leading positive 10a step is East Kilbride, 2026-04-06:
**0.063324** [0.017192, 0.110697]. These are descriptive state changes, not causal
manager effects; no external information was fetched.

## 2. Score-grid profiles and book reproduction

Sources: `grid_profile.csv`, `grid_fit.csv`. Every grid uses the same accepted
Phase A books, complete markets, KL weights, dispersed multistarts, cutoff 10
and final gates. G2 rates are expected goals lambda1+lambda3 / lambda2+lambda3,
not the idiosyncratic components. Dependence is one global parameter, never per match.

| Grid | Global dependence | Summed KL | Mean KL | Final gates |
|---|---:|---:|---:|---:|
| G0 double Poisson | 0 | 0.600179 | 0.001009 | 595/595 |
| G1 Dixon–Coles | rho=−0.064462526 | 0.228247 | 0.000384 | 595/595 |
| G2 bivariate Poisson | lambda3=0.111325597 | 0.327800 | 0.000551 | 595/595 |
| G3 Frank / Poisson | kappa=0.418541170 | 0.339442 | 0.000570 | 595/595 |

Golden-section refinement follows the best coarse point's neighbours. Every
coarse/refinement evaluation remains in the profile, including failed candidates:
G3 kappa=4 has an unusable optimisation/multistart profile value, not a subsetted
sum. Poor-fit KL/rate failures at other distant candidates are likewise visible.
All **selected** parameters pass every fixture gate. This empirical bounded
profile/multistart check is not a formal proof of a global optimum over all parameters.
Tiny negative KL values in near-exact books are floating-point roundoff, not
negative divergence by construction; they are not hidden or tuned away.

Sources: `grid_line_residuals.csv` overall rows and `grid_heldout.csv` overall rows.
Mean residuals are grid minus market probability; intervals are fixture-bootstrap
95% CIs, conditional on fitted parameters/rates.

| Grid | In-sample draw residual (n=595) | Heldout BTTS-yes residual (n=111) | Pooled heldout MAE (n=271 fixtures) |
|---|---|---|---|
| G0 | −0.013354 [−0.014158, −0.012569] | −0.011972 [−0.014485, −0.009553] | 0.008846 [0.008220, 0.009486] |
| G1 | −0.001370 [−0.001969, −0.000802] | −0.001701 [−0.004170, 0.000687] | 0.008648 [0.008040, 0.009256] |
| G2 | −0.004516 [−0.005130, −0.003903] | 0.004701 [0.002451, 0.006842] | 0.010631 [0.009962, 0.011399] |
| G3 | −0.004806 [−0.005420, −0.004209] | 0.002429 [−0.000036, 0.004742] | 0.010946 [0.010275, 0.011708] |

All OU0.5/1.5/3.5 and BTTS signed/absolute errors and CIs are retained, overall,
by tournament and by book size. Both complement selections contribute; therefore
the pooled **signed** mean cancels by construction. Grid choice uses pooled
absolute selection error, not that uninformative signed mean or outcome losses.
No favourable line subset is used. Different fixtures have different numbers of
heldout lines; the fixture bootstrap retains each fixture's entire line cluster.

**Limitation:** the global parameter is estimated on full books, including the
heldout lines, then held fixed for the 1X2+OU2.5 inversion. This is the specified
heldout-line shape check, not independent fixture CV or a fully heldout parameter
fit. The global grid parameter also uses both seasons in G1's informational R6
10b transformation: honest team-theta fitting is not an honest end-to-end pipeline.

### 1X2-only totals bias

Source: `grid_1x2only_bias.csv`, all 517 books with at least five selections.
Delta is log total from 1X2-only inversion minus full-book inversion at the same
global grid parameter. All inversions retain the same multistart/gates.

| Grid | Mean Delta log total | Fixture-bootstrap 95% CI | q05 / q50 / q95 |
|---|---:|---|---|
| G0 | −0.102716 | [−0.107389, −0.098550] | −0.181382 / −0.104676 / −0.014044 |
| G1 | −0.008941 | [−0.013462, −0.004813] | −0.089339 / −0.010330 / 0.073831 |
| G2 | −0.029419 | [−0.033466, −0.025686] | −0.106972 / −0.030701 / 0.047393 |
| G3 | −0.034759 | [−0.039324, −0.030573] | −0.116224 / −0.037036 / 0.048026 |

The G0 value reproduces the brief's rounded −0.103 without changing data or gates.
G1 satisfies the fixed absolute-mean-bias trigger for including thin books;
therefore its additional informational R6 fit is present, not silently omitted.

### Realised outcome losses

Source: `grid_outcome_logloss.csv`. Absolute grid losses and grid-minus-G0 pairs
use all 517 fixtures. 1X2 close pairs use all 517; OU2.5 close pairs use **353**
quoted closes, exactly the availability documented in Phase B. No price is imputed.
Goal columns are `ds.matches.home_score / away_score` (full-time).

| Grid | 1X2 log loss | Delta vs G0, 95% CI | OU2.5 log loss | Delta vs G0, 95% CI |
|---|---:|---|---:|---|
| G0 | 1.052601 | 0 | 0.692724 | 0 |
| G1 | 1.052011 | −0.000590 [−0.003695, 0.002376] | 0.691350 | −0.001374 [−0.002912, 0.000135] |
| G2 | 1.052480 | −0.000121 [−0.002461, 0.002056] | 0.691620 | −0.001104 [−0.002587, 0.000355] |
| G3 | 1.052391 | −0.000210 [−0.002479, 0.001896] | 0.691730 | −0.000993 [−0.002542, 0.000516] |

For G1 the paired difference versus close is −0.000890
[−0.002643, 0.000788] on 1X2, and −0.000034
[−0.001195, 0.001192] on quoted OU2.5. All other grid-close CIs also include zero.
The quoted OU2.5 close absolute loss, 0.689816, is on a different population than
the all-fixture grid loss; do not subtract those two unpaired means.

### Rate shifts and truncation

Source: `grid_rate_shift.csv`, all accepted fixtures; raw shifts per fixture are
`rate_shift_fixture_G*.csv`. Relative-to-G0 q05/q50/q95:

| Grid | Delta log home | Delta log away | Delta supremacy | Delta level |
|---|---|---|---|---|
| G1 | −0.005972 / 0.003485 / 0.095651 | −0.006492 / 0.003663 / 0.110653 | −0.020023 / −0.000122 / 0.007866 | −0.005702 / 0.003370 / 0.101530 |
| G2 | −0.005161 / 0.010351 / 0.083939 | −0.003188 / 0.016185 / 0.106520 | −0.044923 / −0.006501 / 0.020608 | −0.001729 / 0.014331 / 0.094449 |
| G3 | −0.007579 / 0.007078 / 0.076551 | −0.005678 / 0.012640 / 0.095220 | −0.039645 / −0.006531 / 0.019115 | −0.004449 / 0.010113 / 0.082432 |

Finite cutoff means do not equal untruncated rates exactly. `grid_marginals.csv`
records both means and their errors for every grid/fixture. Maximum absolute
error across either marginal is G0 **0.0100021991**, G1 **0.0099816601**,
G2 **0.0097434675**, G3 **0.0098373533** goals. This is documented truncation under
the brief's explicit exception, not a relaxed assertion of exact 1e-6 agreement.
No tail mass is invented; all grids use the same renormalised support.

## 3. R6 on the best grid (informational)

Source: `r6_on_best_grid.csv`. All five axes and both protocols are saved.
10a smoothed values use the full panel and median theta; 10b smoothing is also
full-panel descriptive, while `n` in this table counts **one-step scored fixtures**.
Nonlinear log total retains B2's seeded MC convention.

| Rates | Protocol | n scored | Supremacy RMSE / logpd / 90% coverage | Level RMSE / logpd / 90% coverage |
|---|---|---:|---|---|
| G0 | 10a | 517 | 0.191684 / 0.266761 / 0.922631 | 0.055179 / 1.452853 / 0.912959 |
| G1 | 10a | 517 | 0.191836 / 0.266058 / 0.924565 | 0.053808 / 1.480990 / 0.918762 |
| G0 | 10b | 273 | 0.190775 / 0.252850 / 0.901099 | 0.053156 / 1.525799 / 0.912088 |
| G1 | 10b | 273 | 0.191056 / 0.249555 / 0.897436 | 0.051795 / 1.555482 / 0.915751 |
| G1 including thin | 10a | 595 | 0.182585 / 0.308373 / 0.922689 | 0.057445 / 1.406480 / 0.904202 |
| G1 including thin | 10b | 291 | 0.187185 / 0.266324 / 0.910653 | 0.052900 / 1.510384 / 0.951890 |

On the same full-panel population, smoothed supremacy/level R² is
**0.988918 / 0.839247** for G0 and **0.989926 / 0.843113** for G1.
The 10a innovation correlation median is **−0.821935** for G0 and **−0.830084**
for G1 (`convergence_G0.csv`, `convergence_G1.csv`); with thin books it is
**−0.777320** (`convergence_G1_including_thin.csv`). The covariance story survives
the score-grid correction. The thin row's changed population prevents a paired
RMSE claim; mean thin-book bias correction does not make those rates noise-free.

## 4. Acceptance evidence

1. **Fresh laptop REPL:** owned pane `%36`, Julia 1.12.1, 8 threads, BLAS=1.
   t04 **92/92**, `T04_DONE`, 35.6 s; t03 **131/131**, `T03_DONE`, 41.2 s;
   t02 **131/131**, `T02_DONE`, 59.3 s. Commands/counts in
   `REPRODUCIBILITY.md`; plain output `LAPTOP_FINAL_TESTS.txt`. t03 additionally
   reaches `R01_DONE` and preserves every published non-runtime Phase A CSV.
2. **Grid gates:** t04 checks mass/positivity, all three independence reductions
   to <=1e-10, synthetic rate and profiled dependence recovery to 1e-5, DC tau
   parity and independent BigFloat Frank rectangles on positive/negative cases.
   Mean truncation is explicitly measured rather than denied. Large-rate G1/G2
   line-search stability and exact Phase B quote availability/loss regression
   are covered. Gaussian simulated-panel PPCs are non-extreme; planted t3 quality
   kurtosis has p-value 0.0 (test evidence, not a real-panel result).
3. **R6 convergence:** **24/24** theta rows pass the unchanged Rhat<=1.05 and
   bulk/tail ESS>=200 gates, both protocols, all three populations. Maximum
   Rhat **1.0010992161573777**, minimum bulk ESS **5822.449706873062**, minimum
   tail ESS **7006.098520527357**, from the three `convergence_*.csv` files.
   Same R6 priors and budgets as B2: 4x(2000+3000), thin 1; seeds 4522/4523.
   No alternative was fitted because no real PPC p-value is extreme.
4. **Artifacts:** all requested tables and figures exist. The three PNGs were
   visually inspected: populated profile curves, residual CIs and both PPC
   distributions. Requested paths are `tail_ppc.csv`, `largest_quality_steps.csv`,
   `grid_profile.csv`, `grid_fit.csv`, `grid_line_residuals.csv`, `grid_heldout.csv`,
   `grid_1x2only_bias.csv`, `grid_outcome_logloss.csv`, `grid_rate_shift.csv`,
   `r6_on_best_grid.csv`; figures `../figures/B3_tail_ppc.png`,
   `B3_grid_residuals.png`, `B3_grid_profile.png`. Raw diagnostic tables are
   retained, not only favourable summaries. Scientific result numbers above
   are rounded from named tables; test/time/pane metadata come from logs.
5. **Fresh top-to-bottom runner:** first fresh beast `%258`, 16 threads, pinned
   cache/Manifest, core pinning and BLAS=1, source SHA above: `R04_DONE` in
   **501.5812849998474 s**. Second fresh `%259`: `R04_DONE` in
   **508.820631980896 s**, same SHA/settings/seeds. All **32 CSVs and three PNGs**
   byte-identical: `SCIENTIFIC_SHA256SUMS.txt` checks 35/35 on the beast and
   laptop copy. Logs `BEAST_RUN1.txt` / `BEAST_RUN2.txt`; original logs and
   first-run snapshots remain in `/root/BF_runs/logs/market_model_b3/`.
   See `REPRODUCIBILITY.md` and `RUN_PROVENANCE.md`.
6. **README/decision:** stream README explains B3 and explicitly records the
   human's double-Poisson-for-C, C2-later decision. No Phase C code or grid change.
7. **Tracking:** TODO 039 and its index have dated @pi B3 verification, with
   Phase B3 checked but the task still IN_PROGRESS for unimplemented/unapproved
   C/D. `./scripts/todo.sh check` and `git diff --check` pass at handoff.

## 5. Choices, corrections and limitations

Full predeclared definitions, profile choices and library citations are in
`METHODS.md`. Literature actually read:

- Frank positive/negative dependence and discrete rectangle construction:
  `mchale2011-modelling-dependence-goals-scored-opposing`, source-PDF **pp. 9–13**
  (CDF p. 11, rectangle p. 12, discrete Kendall caveat p. 13). We use Poisson
  margins, not their negative binomial margins, and do not transfer their fitted
  dependence sign in national-team goals to these market books.
- Discrete copulas need caution/nonuniqueness:
  `genest2007-primer-copulas-count-data`, **p. 1** abstract and **p. 2** explicit
  nonuniqueness discussion. The brief's p. 1 reference is imprecise for that claim.
- Continuous Kendall definition: `ruppert2010-copulas`, **pp. 9–10**; Gaussian
  arcsin identity is on **p. 16**, not pp. 9–10. t-copula degrees of freedom/tail
  dependence: **p. 4**, coefficient/zero Gaussian asymptotic dependence **p. 12**.
  Continuous identities apply to team steps, not discrete goal-score copulas.

DESIGN's old A/B exclusion of alternate grids/outcome scoring is superseded by
this explicit diagnostic B3 authorisation, not by a Phase C decision. Its old
Phase D r04 filename is not this B3 r04. The brief's apparent requirement for
517 paired OU2.5 closes cannot be literal: there are only 353, as already measured
in Phase B. All-fixture grid/G0 losses and quote-paired losses are separately
labelled, with no selected-outcome subset or fabricated price.

Two development failures are retained honestly in beast logs: initial G2 AD
mass validation/line-search underflow, fixed with primal-only validity checks
and cancellation of common exponential factors; then an incorrect assumption
that every structural fixture had OU2.5 quoted, fixed by explicit availability
and a Phase B regression test. Neither statistical gate was loosened. No repeated
identical failing gate was retried three times.

PPCs and interval/bootstrap comparisons condition on estimated market rates and
global grid parameters; rate-inversion/parameter uncertainty is not propagated
into the R6 measurement likelihood. Fixture bootstraps do not account for all
team/temporal dependence. Full-book parameter estimation gives heldout-line and
G1 10b informational comparisons shared-parameter retrospective information.
Non-extreme PPCs and outcome CIs crossing zero must not be promoted to proofs of
Gaussianity, equal predictive skill, causality or betting profitability.

## Blocked / handoff

No unresolved blocker. All Phase B3 acceptance checks pass; this is a handoff
for manager/human review, not Phase C authorisation. All owned laptop/beast
REPL panes are closed; no manager/human session was operated. The dedicated
beast checkout, binary fits and logs remain for review. Results/documentation
are committed and pushed only to `pi/market-model-phase-b3`.
