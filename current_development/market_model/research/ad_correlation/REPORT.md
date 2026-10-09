# Attack/defence correlation and hierarchical team-strength models — literature report

Research agent (Claude Opus 5.5), 2026-10-06, for TODO 039 / James. Sources and depth of reading
are in `BIBLIOGRAPHY.md`; every screened work is in `candidates.csv`; queries are in `SEARCH_LOG.md`.
Sign convention throughout: **β = "concedes more"**, so better teams have α up and β down. Where a
paper uses a defensive *strength*, its correlation is reported here with the sign flipped and
marked **[flipped]**. Statements marked *(inference)* are mine, not a paper's.

## Summary

1. Published football models almost all **assume** independent attack and defence, in levels and in steps (Maher-type static models; Koopman & Lit; Owen; Baio & Blangiardo; Egidi et al.; footBayes). I found **no football paper that estimates an attack–defence innovation correlation**, so R6's −0.82/−0.86 has no published counterpart.
2. The only estimated attack/defence correlation **hyperparameters** found are for US college sports, from unstructured G matrices: ≈ −0.48 to −0.77 [flipped] in American college football, and ≈ +0.3 [flipped] in the NBA (Karl et al. 2014; Broatch & Karl 2017). Practitioners simulate soccer team levels with ρ between −0.4 and −0.8 without an estimate behind it (Benz & Lopez 2021).
3. A level correlation is **not by itself** evidence of a hierarchy. With quality q = (α−β)/2 and style s = (α+β)/2 uncorrelated, corr(α, β) = (r² − 1)/(r² + 1), where r = SD(s)/SD(q). B2's innovation r = 0.315 gives −0.82; the levels' −0.82…−0.86 imply a level r of 0.27–0.31 — **the same geometry at both scales** *(inference)*.
4. Levels correlate in R2 because the data decide levels: teams differ far more in quality than in style. Sum-to-zero constraints create −1/(n−1) correlations **between teams on one side**, never between a team's own α and β; corner constraints and a shared μ only shift levels and cannot create a cross-team correlation *(inference, from Owen 2011 and Knorr-Held 2000)*.
5. One like-for-like out-of-sample test exists: **one strength per team beat separate attack/defence** on RPS (EPL 0.1954 vs 0.1961; internationals 0.1653 vs 0.1656; Ley et al. 2019). This is exactly the σ_s = 0 limit of our rotation. Other 1-D vs 2-D comparisons also change the data used, so they are confounded.
6. **Recommended goal-model change (Spec A):** run `MultiScaleGRW` in rotated quality/style axes, with independent σ_q and σ_s at each time scale and the style/quality ratio's prior centred near 0.3. Spec B (2×2 Cholesky ρ per scale) is the check for a non-zero corr(Δq, Δs). Spec C (correlated level only) is the ablation.
7. Market information has entered published models as **data** (implied rates: Egidi et al. 2018; implied win probabilities in a state-space model: Lopez et al. 2018; odds-driven Elo: Wunderlich & Memmert 2018). It has not entered as a prior on attack/defence covariance; I found no precedent for that.

---

## RQ1. Empirical record

### What has actually been estimated

| Source | Sport, data | What | Level or step | Method | Reported | In our sign |
|---|---|---|---|---|---|---|
| Karl, Yang & Lohr 2014, Table 1 p. 13 | US college football (FBS) 2012, 126 teams, 844 games | **hyperparameter**: G[off, def] in an unstructured 3×3 G | static level, one season | ML, fully exponential Laplace | 0.0473 / √(0.0920·0.1025) = **0.49**; 0.48 in the scores-only model (my arithmetic from G) | **≈ −0.48** [flipped] |
| Broatch & Karl 2017, Table 5 p. 20; text p. 19 | US college football 2005–2013; NBA | **hyperparameter**, same model | static level, per season | ML, Laplace | 2005: 0.71; "ranges from 0.77 for American college football data to −0.3 for … NBA" | **≈ −0.71 to −0.77**; NBA **≈ +0.3** [flipped] |
| Benz & Lopez 2021, appendix p. 22 | soccer simulation, 20 teams | **assumed** level correlation ρ* ∈ {−0.8, −0.4, 0} | static level | simulation design | "reflecting the range of correlations in scoring and defending strength" | same sign (δ = concedes); no flip |
| Benz & Lopez 2021, p. 14 | Bundesliga 2015/16 | correlation of **fitted** posterior means | level | Stan, independent priors | "weak"; no number | — |
| Koopman & Lit 2015, eq. 4 p. 7; p. 8 | EPL 2003/04–2009/10 | innovations **assumed independent** (H diagonal) | step | MC-ML state space | none estimated | — |
| Owen 2011, eqs. 7–8 p. 5 | SPL 2003/04–2005/06 | independent random walks, one common σ² | step | WinBUGS | none estimated | — |
| Baio & Blangiardo 2010, p. 4; pp. 7–8 | Serie A 1991/92, 2007/08 | exchangeable, independent hyperpriors; mixture with sign-truncated group means | level | WinBUGS | Fig. 3 (p. 9) shows the scatter; no number | (their def = concedes) |
| **This project, B2 R6** | Scottish L1/L2 closing odds 24/25–25/26 | **hyperparameter** ρ_αβ | weekly **step** | Kalman/FFBS + MCMC on θ | −0.82 [−0.87, −0.76] (retrospective); −0.86 [−0.94, −0.74] (honest) | native |
| This project, B2 levels | same | correlation of **fitted** season-average levels | level | RTS smoother at median θ | −0.82 to −0.87, R2 included | native |

### Reading the record

- **Hyperparameter versus fitted effects.** Only Karl et al. and Broatch & Karl estimate a correlation hyperparameter. Benz & Lopez assume one in simulation and describe fitted means qualitatively. In the football papers read, neither a hyperparameter nor a correlation of fitted effects is reported as a number.
- **Levels versus innovations.** Every external number is a static level. No published dynamic model I found estimates the correlation of same-team steps. Koopman & Lit (p. 7) and Owen (p. 5) set it to zero by assumption. Koopman & Lit (2019) score-driven updates are block-diagonal, "no spillover effects" (p. 6), so any α–β co-movement in a step comes only through the joint score of each match.
- **Magnitude.** College football's −0.5 to −0.8 brackets our levels' −0.82 to −0.87 only at its top end. Recruiting builds both units (Broatch & Karl, pp. 19–21). The NBA's opposite sign — same players at both ends — shows the sign is not a law *(inference from their discussion)*. For soccer, the only quantitative statement is the unexplained simulation range in Benz & Lopez. **I found no published soccer estimate of the attack–defence level correlation.**
- **Scale of steps.** Koopman & Lit's EPL goal model estimates weekly SDs of 0.0144 (attack) and 0.0120 (defence) (Table 2, p. 16). R6's σ_q = 0.032 and σ_s = 0.010, with corr(Δq, Δs) ≈ 0, imply about 0.034 per side per week, 2.3–2.8 times larger *(inference; different league and data)*. Market-implied states move more than goal-smoothed ones. This matters when centring priors (§Specs).

## RQ2. How the correlation has been modelled

### 2.1 Correlated team effects (bivariate distribution with a learned correlation)

- **Static, frequentist:** mvglmmRank (Karl et al. 2014; Broatch & Karl 2017) puts (offence, defence, win propensity) ~ N₃(0, G*) with G* unstructured, block-diagonal over teams, fitted by Laplace/EM maximum likelihood. There is no prior on the correlation. Variances of offence and defence are similar (0.092 vs 0.103; Karl et al., Table 1).
- **Static, Bayesian, soccer:** none found with a fitted correlation. Thompson's (2018, blog) Stan models and Benz & Lopez's (2021) models use independent non-centred normals. Benz & Lopez's bivariate normal is only their simulation's data-generating process (p. 22).
- **Dynamic:** none found in soccer with a correlated attack/defence step covariance. Crowder et al. (2002) describe attack and defence evolving by "some unobserved bivariate stochastic process" (abstract). I could not read the paper (paywalled), so whether that process is correlated is **unverified**; it is first in `PAYWALLED.md`.
- **LKJ, inverse-Wishart, Cholesky:** no football team-effect paper found using them. The one LKJ use in football found is player-level and across performance metrics (Shaikh 2026, abstract only), not team attack/defence.

### 2.2 Latent factor and rank reduction

- **One factor per team (rank 1).** This is "overall quality" models: Bradley–Terry, Thurstone and Elo models, Glickman & Stern's NFL point-differential state space (pp. 3–5), Lopez et al.'s market-probability state space (pp. 8–10), Koopman & Lit's (2019) dynamic ordered probit (p. 6), and Ley et al.'s **one-strength Poisson**, λ_i = exp(c + (r_i + h) − r_j), λ_j = exp(c + r_j − (r_i + h)) (p. 7). In our notation that last model is α_i = r_i, β_i = −r_i: quality only, s ≡ 0, a correlation of exactly −1 *(inference: the mapping)*.
- **Two factors (rank 2).** Maher-type attack/defence: Ley et al.'s "Def. & Att." variants (p. 8), Koopman & Lit, Owen, Baio & Blangiardo.
- **Our rotation** q = (α − β)/2, s = (α + β)/2 is a fixed (not learned) rotation to supremacy and total axes. It has exact correspondences:
  - corr(Δq, Δs) = 0 with σ_s/σ_q = r is identical to a correlated (α, β) model with σ_α = σ_β and ρ = (r² − 1)/(r² + 1);
  - σ_s = 0 is the one-factor model;
  - σ_s = σ_q is R2.

  *(inference: algebra)*. B2's corr(Δq, Δs) ≈ 0 says the market's innovation ellipse is aligned with these axes, so a diagonal covariance in (q, s) captures R6's structure with two scales rather than three parameters.
- No published football model found uses a learned loading (a factor model with style loadings estimated). The Liga MX "greatness" PCA (Corona et al. 2023, abstract) is a one-factor summary of results, fans and market values, not a goal model.

### 2.3 Groups or mixtures of teams

- **Baio & Blangiardo (2010, §4, pp. 6–9).** Three-component mixtures of non-central t (ν = 4) for att and for def, with **separate** latent labels grp^att(t) and grp^def(t) and Dirichlet(1,1,1) weights (p. 7). The attack/defence anticorrelation enters only through **sign-truncated group means**: bottom μ₁^att ∈ (−3, 0), μ₁^def ∈ (0, 3); top reversed (p. 7). There is no joint tier label and no correlation parameter. The motivation is overshrinkage: in the basic model Inter are predicted at 69 points against 85 observed, and at 76 under the mixture (Table 3, p. 8).
- **Stochastic block model of league tiers** (Basini et al. 2023, abstract only): evidence of a "two-tier league" in the EPL since the early 2000s, from win/draw/loss networks, not goal rates.
- *(Inference)* A mixture on q alone, with s unimodal, would be the tiered version of our rotation. Baio & Blangiardo's separate labels can put a team "top" in attack and "bottom" in defence, which the market's quality-dominated geometry suggests is rare. A t or mixture prior for q addresses overshrinkage of extreme teams, which a Gaussian level prior with correlation does not.

### 2.4 Multilevel by league, tier and season

- **Promoted teams.** Owen (2011, p. 6) gives promoted SPL teams the relegated team's priors. Wunderlich & Memmert (2018, p. 6) have promoted teams "carry on the ratings of the relegated teams". Tsokos et al. (2019) use a "newly promoted" feature (pp. 3, 13). Rue & Salvesen (2000; via Owen 2011, pp. 12–13) set prior means from a prior ranking.
- **Multi-division joint fits.** Crowder et al. (2002) fit all 92 English League teams together (abstract; unread). Ruiz & Pérez-Cruz (2015, abstract) add conference-specific structure in college basketball. Dolores (Constantinou 2019) rates teams across leagues worldwide (title only).
- **Season scale.** Glickman & Stern (1998) and Lopez et al. (2018) use two AR scales, between-season (with regression to the mean, β_s / γ_season) and between-week. Glickman & Stern's posterior σ_s = 2.35 points is much larger than σ_w = 0.88 (Table 3, p. 10). Tsokos et al. add season-specific AR(1) deviations with **one shared SD** for attack and defence (p. 9). Egidi et al. (2018) use seasonal random walks with drift and separate σ_att, σ_def (eq. 3.5).
- **How the correlation behaves across levels:** no paper found estimates attack/defence correlation separately at league, season or week level. *(Inference:)* across tiers, the L1/L2 offset is a direction in which promoted and relegated clubs move mostly in q, so a tier term belongs on q. DESIGN.md §4.5 notes the L1/L2 offset is "a gauge direction with no tier term": league matches alone cannot place the two divisions on one scale, so cup fixtures or transition clubs are needed.

### 2.5 Dynamic versions

| Structure | Papers | Attack/defence coupling |
|---|---|---|
| Independent Gaussian random walks or AR(1) | Owen 2011; Koopman & Lit 2015; Egidi et al. 2018; Macrì Demartino et al. 2026; footBayes (Egidi, Karlis & Ntzoufras 2025, eqs. 4.24–4.25) | none (independent by assumption) |
| Continuous-time Brownian | Rue & Salvesen 2000 (via Owen, p. 12) | common variances, independence (secondary source) |
| Unspecified "bivariate stochastic process" | Crowder et al. 2002 (abstract) | unverified |
| Score-driven (GAS) | Koopman & Lit 2019 (eqs. 11–15) | block-diagonal A, B; coupling only through the score |
| Per-period scale mixtures | Macrì Demartino et al. 2026 (eqs. 10–11) | separate φ_att,τ, φ_def,τ; no covariance |
| 1-D multi-scale AR state space | Glickman & Stern 1998; Lopez et al. 2018 | one dimension |
| Multivariate state space with a full covariance | none found for team attack/defence | — |

## RQ3. Does a correlation imply a hierarchy?

### Candidate artefacts, checked

1. **Sum-to-zero constraints.** A random walk constrained to sum to zero over n teams has step covariance σ²·n/(n−1)·(I − 11ᵀ/n) (Owen 2011, eq. 14), equivalently Q = σ²(I − 11ᵀ/n) (Knorr-Held 2000, p. 6), "a priori negatively correlated" components (Knorr-Held, p. 6). The induced correlation is −1/(n−1) **between different teams on the same side**. Constraints applied to α and β separately leave a team's own α–β covariance untouched. For n = 19–20, −1/(n−1) ≈ −0.05 to −0.06, far from −0.8. `MultiScaleGRW` centres each side's state column after the fact, which is the same projection *(inference: matches Owen's eq. 14 up to the variance factor)*. Knorr-Held adds that in dynamic models the posterior mode is **not invariant** to the choice of constraint (p. 6), so a corner constraint would change state estimates, not just shift them.
2. **Corner constraints and a shared μ.** Both move every team's α (or β) by a common constant. A Pearson correlation **across teams** is invariant to such shifts, so neither can create a cross-team level correlation of point estimates *(inference)*. They can affect within-team posterior draws. If only the sum of per-side means is pinned, the gauge direction (α + c, β − c) leaves every rate unchanged (log λ_h = μ + γ + α_h + β_a); without per-side centring it would put perfectly negative within-team posterior correlation into the draws *(inference)*. Our models centre each side, so this is a check on new code, not an explanation of B2.
3. **Shrinkage.** Shrinkage acts per axis. If style were less precisely observed than quality, posterior means would shrink s more than q, making the fitted level correlation **more** negative than the population value *(inference; direction only, size unknown)*. Baio & Blangiardo's overshrinkage (pp. 6–7) is the same mechanism along q. This is testable by simulating from R2 with known level geometry and refitting.
4. **The population.** Good teams score more and concede less. With corr(q, s) = 0 across teams,

       Var α = Var q + Var s,   Var β = Var q + Var s,   Cov(α, β) = Var s − Var q,
       corr(α, β) = (r² − 1)/(r² + 1),   r = SD(s)/SD(q).

   Any population whose spread is mostly quality has strongly negative attack/defence correlation, whatever prior was used for the steps *(inference: algebra)*.

### Why our levels correlate even with independent innovations

*(Inference throughout; the algebra is checked against B2's numbers.)*

- Inverting the levels table gives level r = √((1 + ρ)/(1 − ρ)) = **0.27–0.31** (R2 10a: −0.835 → 0.30 and −0.863 → 0.27; R6 10a 24/25: −0.820 → 0.315).
- R6's **innovation** r is 0.315 [0.26, 0.37]. Applying the formula with corr(Δq, Δs) = 0 gives ρ = −0.819, the reported −0.822. The honest r = 0.282 gives −0.852 (reported −0.864; corr(Δq, Δs) = −0.18 there accounts for part of the gap).
- So **the cross-section of team levels and the weekly steps have the same quality/style geometry, r ≈ 0.3**. Levels are identified by about 40 weeks of two-axis market data per team-season. Their correlation comes from the data and is nearly unchanged between R2 and R6 (−0.835 vs −0.820, 24/25). Innovations are weakly identified week by week, so the prior's covariance (R2: r = 1 forced) shows up there, and freeing it improves calibration (B2).
- The level correlation therefore says "teams differ mainly in quality", a property of the population, not by itself a latent hierarchy. The **innovation** correlation is a separate finding: **quality also carries most of the week-to-week change**. Agreement of the two ratios is consistent with a single dominant quality factor operating at both scales. That is the "hierarchy" worth encoding, as a rank-dominant structure rather than a discrete grouping.

Caveat: corr(q̄, s̄) across teams was not reported for levels. The level r above assumes it is near zero, as it is for innovations.

### How to test one factor against two

- **Nested zero-variance null (σ_s = 0, or ρ = ±1).** The null sits on the boundary. The likelihood-ratio statistic for one variance is then a 50:50 mixture of χ²₀ and χ²₁ (Self & Liang 1987; Stram & Lee 1994 — standard results, papers not re-read this session). B2 notes that a positive lower quantile under a continuous prior "is not a formal test of that nested null" (PHASE_B2_REPORT, addendum).
- **Make the null interior.** The non-centred signed-SD parameterisation of Frühwirth-Schnatter & Wagner (2010), shown in Huber et al. (2021, eq. 6, pp. 4–5), puts √v (here ±σ_s) into the observation equation as an ordinary coefficient. σ_s = 0 then becomes an interior point that admits a Savage–Dickey Bayes factor or a spike-and-slab or shrinkage prior (Bitto & Frühwirth-Schnatter 2019). *(Inference:)* this fits Spec A directly, because σ_s multiplies standardised style steps.
- **Predictive comparison.** Use leave-future-out CV for walk models (Bürkner, Gabry & Vehtari 2020) rather than PSIS-LOO (Vehtari et al. 2017), which conditions on the future. Football papers compare on RPS or log score out of sample (Ley et al.; Koopman & Lit 2019; Tsokos et al.); none reports a formal one-vs-two-factor test or Bayes factor.
- **In football specifically:** no paper found tests attack/defence independence or one-vs-two dimensions with a likelihood-ratio or Bayes-factor test. Koopman & Lit's (2015) model (d) deletes the opponent's defence rather than imposing α = −β, so its rejection (p. 20) does not bear on dimension *(my reading)*. Maher (1982) may contain reduced-model tests; **unverified**, listed in `PAYWALLED.md`.

## RQ4. Does it help prediction?

| Comparison | Like-for-like? | Result |
|---|---|---|
| Ley et al. 2019, Tables 1–2 (pp. 10–11): Poisson with 1 strength/team vs Poisson with attack and defence, same goal data and time-decay ML | **yes**, the only clean 1-D vs 2-D test found | EPL 2008–2018, 3300 matches: 1-param 0.1953 (biv.) / 0.1954 (ind.) vs att&def 0.1961 / 0.1961. Internationals, 4268 matches: 0.1651 / 0.1653 vs 0.1656 / 0.1656. **1-D wins**; no SEs reported; "parsimony … is important" (p. 10) |
| Koopman & Lit 2019, Tables 3–4 (p. 11): dynamic ordered probit (1-D, outcomes) vs bivariate Poisson / Skellam (2-D, goals) | no: 1-D also discards goals | 2-D better in 5 of 6 leagues (e.g. Spain 0.1916 vs 0.1929, DM 3.09); Netherlands 1-D best (0.1934); attributed to "condensation of data" (p. 10) |
| Tsokos et al. 2019, Table 5 (p. 14): hierarchical Poisson (2-D, goals) vs Bradley–Terry family (1-D, outcomes) | no | 0.2073 (SE 0.0025) vs 0.2079–0.2083: within one SE |
| Ley et al. 2019: Poisson models (goals) vs Bradley–Terry/Thurstone (outcomes) | no | Poisson 0.195–0.196 vs BT/TM 0.1985–0.2003 (EPL) |
| Karl et al. / Broatch & Karl: correlated vs uncorrelated G | partially (they test correlation between win propensity and score effects, not α–β) | joint correlated model significantly improves win/loss log loss in every year (Broatch & Karl, Table 4); effect of the α–β term alone not isolated |
| This project B2: R6 (correlated steps) vs R2 (independent) on market rates, honest 10b | yes, on market data | +0.185 ± 0.038 nats per fixture (sum of axis log densities); coverage 82% → 90% on supremacy |

**Reading** *(inference)*: (i) Using goals beats using only results; that is about data, not dimension. (ii) At fixed data, the one clean test favours **fewer** team parameters, i.e. the σ_s = 0 end. (iii) B2 favours a **non-degenerate** quality-dominated step covariance over independence. Spec A sits between (ii) and (iii): style variance is learned, small, and allowed to shrink towards zero. I found no published out-of-sample comparison of correlated vs independent attack/defence in a soccer goal model; Spec A vs current `MultiScaleGRW` would be the first.

## RQ5. Practice

**Scale priors used.**
- Baio & Blangiardo: Gamma(0.1, 0.1) on precisions (p. 4); Gamma(0.01, 0.01) in the mixture (p. 8).
- Egidi et al. 2018: half-Cauchy(0, 2.5) on σ_att and σ_def (p. 13).
- Lopez et al.: Uniform(0, 1000) on precisions, γ_season ~ U(0, 1), γ_week ~ U(0, 1.5) (p. 11).
- Tsokos et al.: R-INLA defaults (p. 9).
- Owen and Rue & Salvesen: tune one common σ by predictive performance (Owen p. 6, using 2003/04 for optimisation; Rue & Salvesen via Owen p. 12).
- Macrì Demartino et al.: spike-and-slab half-normal mixtures on per-period precisions (eq. 11).
- Thompson (blog): normal(0, 10) on SDs.
- Ours: Gamma(2, θ) per side and scale (`multiscale.jl`).

**Correlation priors.** None found in football team models. The 2×2 LKJ(η) prior is equivalent to (1 + ρ)/2 ~ Beta(η, η) (a standard result, not re-checked against Lewandowski et al. 2009 this session); η = 1 is uniform on ρ. To centre on the market's −0.8 a Beta on (1 + ρ)/2 with mean 0.1 is more direct than LKJ, which is symmetric about 0 *(inference)*.

**Non-centred parameterisation and geometry.**
- Non-centred walks are the norm in Stan/Turing practice (Thompson; footBayes; our `MultiScaleGRW`). Non-centring helps when the data are weak relative to the prior (Papaspiliopoulos, Roberts & Sköld 2007) — the situation for weekly steps on goal data.
- Owen samples n − 1 unconstrained innovations and maps them to the constrained walk (eqs. 16–17); Knorr-Held uses the rank-(n−1) Q directly.
- *(Inference:)* a 2×2 correlation near −0.8 to −0.9 between non-centred α- and β-steps produces a ridge in the posterior of (σ_α, σ_β, ρ). The rotated axes (q, s) are close to the principal axes of that ridge, so NUTS should mix better there; B2's corr(Δq, Δs) ≈ 0 is the evidence that they are.

**State-space and Kalman implementations.**
- Koopman & Lit: importance-sampling MC-ML around a linear Gaussian approximation.
- Glickman & Stern: Gibbs sampling.
- Lopez et al.: Bayesian MCMC with week and season AR scales on market probabilities.
- Our Phase B: exact Kalman/RTS/FFBS on inverted rates, which is cheaper than any of these for the market model.

**Market-implied information in goal models.**
- Egidi, Pauli & Torelli (2018): per-bookmaker implied Poisson rates from 1X2 via the Skellam (eq. 3.2), entered as extra data with truncated-normal noise. The goal rate is a convex combination p θ + (1 − p) λ (eq. 3.3); their probabilities end up "very close to the bookmakers'" (p. 25).
- Lopez et al. (2018): closing money-line probabilities **are** the observations of a 1-D state-space strength model.
- Wunderlich & Memmert (2018): Elo updated with the market's expected result beats goal- and result-based Elo, but not the odds.
- **No paper found uses market-implied attack/defence (or their covariance) as a prior for a goal model's α/β.** The two-stage route — market model → priors on (σ_q, σ_s, r, level geometry) → goal model — has no precedent in the literature I found. Egidi et al. (2018) is the nearest relative and puts the market in the likelihood instead.

---

## Candidate model specs for the goal model

Notation follows `MultiScaleGRW`: three time scales k ∈ {0 (initial level), S (season macro step), K (target-season micro step, normally a match-biweek)}. Teams i = 1…n, standard-normal shapes z, states centred over teams per column (as now). Market numbers are per **week**; ×√2 converts to a biweek. R6 gives σ_q ≈ 0.037–0.045 and σ_s ≈ 0.011–0.014 per biweek (my arithmetic).

**Assumption stated:** the market's dynamic geometry (r ≈ 0.3, ρ ≈ −0.8) applies to true team strength. The overall step **scale** in a goal model is not assumed to equal the market's, since Koopman & Lit's goal-based steps are 2–3 times smaller. Priors below therefore centre the dimensionless ratio or correlation on the market and leave the scale broad.

### Spec A (recommended): rotated quality/style walk with independent axes

**Equations.** For each scale k:

    q-step_{i,k} = σ_q,k · z^q_{i,k},     s-step_{i,k} = σ_s,k · z^s_{i,k},     z ~ N(0, 1) independent
    q_{i,t} = Σ (q-steps to t),  s_{i,t} = Σ (s-steps to t),  each centred over teams per column
    α_{i,t} = q_{i,t} + s_{i,t},   β_{i,t} = s_{i,t} − q_{i,t}
    log λ_home = μ + γ + α_home + β_away,   log λ_away = μ + α_away + β_home   (unchanged)

So log λ_home − log λ_away = γ + 2(q_home − q_away), and log λ_home + log λ_away = 2μ + γ + 2(s_home + s_away). Quality drives supremacy, style drives totals.

**Priors (with reasons).** Parameterise each scale as σ_q,k and r_k = σ_s,k/σ_q,k.
- r_K ~ LogNormal(log 0.3, 0.35), central 90% ≈ [0.17, 0.53]. Centred on B2's 0.28–0.32 (both protocols), widened because market totals may be stickier than true style.
- r_0 ~ LogNormal(log 0.3, 0.35). The levels table implies 0.27–0.31 in both seasons and both R2 and R6.
- r_S ~ LogNormal(log 0.4, 0.6), deliberately weak. The panel has one season transition, so there is no market evidence for the summer step.
- σ_q,k: keep Gamma(2, θ) shapes but rescale so the implied per-side SD σ_q√(1 + r²) has the current prior means for k = 0 and S. For k = K, put the prior mean between Koopman & Lit's goal-based scale (≈ 0.02 per biweek per side) and R6's (≈ 0.047), e.g. Gamma(2, 0.016) for σ_q,K (mean 0.032). This is a judgement call to be checked by prior-sensitivity runs.
- The current asymmetry (defence looser at season boundaries, tighter in-season) cannot be expressed as σ_α ≠ σ_β here. If that asymmetry matters, it is an argument for Spec B.

**Identifiability.** Centre q and s per state column. This is equivalent to centring α and β, since the map is linear and invertible (determinant 2; no Jacobian is needed for a fixed linear map of states built from z). μ and γ unchanged. The H1 home-scoring/away-suppression split stays non-identifiable without neutral venues (MORNING_SUMMARY).

**What would falsify it.**
1. On goal data, posterior r_K or r_0 pushed well above 0.6, towards R2's 1. That would mean the goal process does not share the market's quality dominance.
2. Spec B finds corr(Δq, Δs) clearly non-zero (90% interval excluding 0), i.e. the rotated axes are not principal axes for goals.
3. Leave-future-out log score not better than current `MultiScaleGRW` across the 40 folds (paired, SE across folds), or worse calibration of totals markets (O/U).
4. Supremacy (1X2) unchanged but totals (O/U) worse, which would indicate s is over-shrunk.

**Cost in Turing/NUTS.** Same number of z sites (two shapes per team-step) and three scales per axis, versus three per side now: no parameter growth beyond the r parameterisation. One extra fixed linear map per draw; it fits the accumulator design (two matrix products, then an add/subtract), so the ReverseDiff tape stays allocation-stable *(inference from the file's design notes)*. Geometry: the σ_s,K ↔ z^s funnel is the weakest-identified part; the non-centred form addresses it. Expect wall time within ±10% of the current model *(guess, not measured)*.

**Support.** Ley et al. 2019 (the σ_s = 0 limit is competitive out of sample); B2 R6/R7 (corr(Δq, Δs) ≈ 0, r ≈ 0.3, density and coverage gains); Glickman & Stern 1998 and Lopez et al. 2018 (1-D multi-scale quality walks work); Koopman & Lit 2019 (keep goals, don't condense to results); Baio & Blangiardo 2010 (shrinkage along quality is the known failure mode, so σ_q must not be over-shrunk).

### Spec B: correlated attack/defence steps via a 2×2 Cholesky factor per scale

**Equations.** For each scale k:

    α-step = σ_α,k · z₁,   β-step = σ_β,k · (ρ_k z₁ + √(1 − ρ_k²) z₂),   z₁, z₂ ~ N(0, 1)

This is the non-centred form of (α-step, β-step) ~ N(0, diag(σ) R_k diag(σ)), with R_k the 2×2 correlation matrix. Centring and the likelihood are as now.

**Priors.**
- K and 0: (1 + ρ_k)/2 ~ Beta(2, 16) — mean 0.111, i.e. ρ mean −0.78 with SD ≈ 0.14 — centred on R6 levels and steps but leaving room up to about −0.5.
- S: (1 + ρ_S)/2 ~ Beta(2, 2) (equivalent to LKJ(2): weakly towards 0), since there is no evidence.
- σ_α,k and σ_β,k: the current Gamma priors, kept so the model nests today's at ρ = 0.

**Identifiability.** As now. ρ_k is identified only through co-movement within team-steps. With goal data at scale K the posterior may stay close to the prior; report the prior-to-posterior shrinkage of ρ_K.

**Relation to Spec A.** B with σ_α,k = σ_β,k is exactly A with r_k = √((1 + ρ_k)/(1 − ρ_k)) and corr(Δq, Δs) = 0. B's extra freedom is σ_α ≠ σ_β, which shows up as corr(Δq, Δs) ≠ 0. B therefore tests A's alignment assumption *(inference: algebra)*.

**What would falsify it.**
- ρ_K's posterior concentrates near 0, or reproduces the prior with no data update on goal data.
- No leave-future-out improvement over the current model.
- A dominates B on leave-future-out with B's σ_α ≈ σ_β; then prefer A on parsimony.

**Cost.** One extra parameter per scale. The same number of z sites. A scalar ρ with a Beta prior avoids an LKJ-Cholesky bijector, simpler and AD-friendly *(inference)*. Geometry is worse than A when ρ → −0.9: the (σ_α, σ_β, ρ) ridge. Expect more leapfrog steps; check E-BFMI and divergences.

**Support.** Karl et al. 2014; Broatch & Karl 2017 (unstructured G, levels); B2 R6 (steps); Crowder et al. 2002 ("bivariate stochastic process", unverified).

### Spec C: correlated static level, independent steps

**Equations.** Initial level (α₀, β₀)_i ~ N(0, Σ₀), written in rotated form α₀ = q₀ + s₀, β₀ = s₀ − q₀ with q₀ ~ N(0, σ_q,0²), s₀ ~ N(0, σ_s,0²). Season and micro steps stay independent per side exactly as in today's `MultiScaleGRW`.

**Priors.** r_0 ~ LogNormal(log 0.3, 0.35) as in A; σ_q,0 so that the per-side level SD matches today's prior means. Optionally a t₄ or two-component mixture on q₀ for overshrinkage (Baio & Blangiardo §4), applied to q only.

**Identifiability.** As now.

**What would falsify it.** It is the ablation, not a contender. If C captures most of A's leave-future-out gain, the correlation matters mainly for levels: early season, promoted clubs, thin history. If A ≫ C, the dynamic structure matters, as B2 suggests on market data.

**Cost.** One parameter; trivial.

**Support.** Karl et al.; Broatch & Karl (static correlated levels); B2 levels table; Owen and Wunderlich & Memmert (promoted-team level priors are where level priors bite).

### Suggested comparison protocol *(inference; for James to accept or amend)*

Current `MultiScaleGRW` vs A vs B vs C on the same folds (40 `match_biweek` cutoffs). Compare by:
- paired leave-future-out log score per fixture;
- 1X2 and O/U calibration separately, since A should mainly change totals;
- closing-line divergence: KL from the Phase A market rates, an external benchmark that fits this project;
- prior-sensitivity reruns of A with r centred at 0.3 and at 1.0.

Report posterior r_k or ρ_k against the market values, and corr(q̄, s̄) across teams.

---

## Open questions

1. **Is the market's quality dominance real, or bookmaker microstructure?** Totals lines may be re-priced less often than handicaps, which would understate σ_s in market-implied steps. Spec A's posterior r on goal data is the independent check. No literature found on relative stickiness of totals vs handicap lines.
2. **Can goal data identify style steps at all?** With weekly goal information this weak, σ_s,K may be prior-dominated in any goal model. Then the market prior *is* the model, and prior-sensitivity is the result to report.
3. **Season-scale geometry.** One summer transition in the panel gives no evidence on r_S or ρ_S. Squad rebuilding could load more on style; the current priors make defence looser at season boundaries.
4. **Crowder et al. (2002)** — does their "bivariate stochastic process" include an attack–defence correlation, and how did they link four divisions? Highest-priority paywalled paper.
5. **Tier structure.** Should a promotion/relegation step and the L1/L2 offset act on q only? The data needed to identify the offset are cup ties or transition clubs (DESIGN §4.5).
6. **Shrinkage artefact size.** Simulate from R2 with known level geometry (e.g. r = 0.3 and r = 1), refit, and compare fitted vs true level correlation. That separates the population signal from per-axis shrinkage (RQ3 item 3).
7. **Relation to the project's spread "compression" finding** (an earlier project note: the market wants about 2.4× wider α/β spread than the goal model). Quality-axis overshrinkage under independent priors would produce exactly that. Whether Spec A or C (with a t/mixture on q) reduces it is untested *(inference)*.
8. **Unverified items flagged in `BIBLIOGRAPHY.md`:** journal venue of Broatch & Karl; page range of Lopez et al.; the JRSS D DOI for Knorr-Held (2000); Maher (1982) reduced-model tests; the LKJ ↔ Beta identity was not re-checked against the source this session.
