# What drives the market's prices, and why it is sharper than our goal models

**TODO 023, Phase 2 — Scottish Lower (tournaments 56/57), 24/25 + 25/26**

All numbers come from [`r02_market_feature_attribution.jl`](r02_market_feature_attribution.jl),
run on mcmc-beast on 2026-09-22. The log is
[`results/phase2_run.log`](results/phase2_run.log) and the tables are in
[`results/phase2/`](results/phase2/).

---

## The answer in six points

1. **The market's team-strength edge is mostly two kinds of form.**
   * Recent **goal form**, i.e. results.
   * Recent **proxy-xG form**: chances created and conceded, measured from BBC
     live-text commentary.

   Each accounts for about **30%** of the variation in the market's home-minus-away
   supremacy. Squad wealth adds about 4%, the starting-XI RAPM rating about 3%,
   and rest/travel essentially nothing. About **26–33% is unexplained** by
   anything we have.
2. **Proxy xG is the biggest thing our goal models are missing.** Call "the gap"
   the extra conviction the market has over our pure-goals GRW. Proxy-xG form is
   by far the largest identifiable piece of it:
   * 31–34% of the gap on market favourites;
   * about 20–25% of its variance.

   A team that has been creating more chances than its results show is priced
   by the market as stronger *before* the goals arrive.
3. **The pure-goals model is not badly scaled; it is less informed.**
   * Regress the market on the goal model and the slope is **1.07**. So given
     the goal model's view, the market on average agrees with it.
   * The famous "slope 0.39" (model on market) is simply a correlation of 0.65
     times a spread ratio of 0.60.
   * The model is less *sharp* because the market conditions on information it
     does not see: chance creation, and a residual we cannot attribute.

   The joint Gen 3/4 models (m05, m12) *are* under-scaled: reverse slopes 1.66
   and 1.41.
4. **Once form enters, the market's "team ratings" shrink by 37%.** Put the
   features into the Phase 1 state-space model and the cross-sectional spread of
   the market's latent attack/defence ratings falls:
   * attack 0.167 → 0.106;
   * defence 0.181 → 0.115.

   Almost all of that comes from goal form and proxy-xG form. Wealth and lineup
   move it by nothing. **The market's big team differences are largely
   observable form, not hidden team quality.**
5. **Fixture-level noise is not explained by these features.** The Student-t
   observation model roughly halves σ_obs, from 0.093 to 0.055, with very heavy
   tails (ν ≈ 2.9). The features then move it only to 0.053. The fixture-to-fixture
   price residual is not wealth, lineups, rest or form.
6. **Part of what looked like "market sentiment" is bad inversion.** Two data
   defects are ticketed rather than fixed inline:
   * **[T015](../../docs/tickets/T015-inversion-accepts-books-without-1x2.md):**
     27 of the 623 fixtures have only totals markets. Their inverted supremacy is
     the optimiser's initial guess (+0.4), not the market's.
   * **Thin books:** books with only 3 quoted selections are down-weighted by the
     Student-t model 44% of the time, against about 7% for books with ≥ 5.

   Restricting to well-identified books (1X2 plus ≥ 5 selections) lowers the
   unexplained share from 33% to 26%.

---

## 1. Set-up

**Targets.** Each fixture's market supremacy is
**Δ_mkt = log λ_mkt,h − log λ_mkt,a**, computed from the inverted Betfair
(−20, 0] close (Phase 1 panel, 623 fixtures). The goal model's supremacy,
**Δ_goal**, is the posterior mean of log λ_h − log λ_a from TODO 021's pure-Poisson
GRW `m01` (run `2b42d3bf-…`). Δ_goal is walk-forward: each fixture is priced by
the fold trained before it. The **gap** is Δ_mkt − Δ_goal.

**Features.** All are home-minus-away and all are known before the close.

| Group | Feature | Source |
|---|---|---|
| goal history | Δ_goal (m01 GRW) | `mcmc_experiments`, OOS latents |
| | goal-form supremacy | `Features._pxg_rolling_lookup` on actual goals: strictly earlier days, half-life 16 matches, 3-match league prior |
| | points-per-game gap | this season, strictly earlier days, shrunk by 3 games to 1.37 |
| wealth | production (age-weighted) wealth Δ | point-in-time bridge, feature-compression EDA `c3bdb53a` |
| | raw log-sum squad wealth Δ | same |
| lineup | shots-RAPM XI Δ (starters + 0.10 × bench) | same (history-fitted ratings, played XI) |
| proxy xG | commentary pxG-form supremacy | `PxGFeature(fallback = :none)`: identical windows to goal form, live text only |
| rest & schedule | rest-days Δ (cap 21) | league fixtures only (betdb has no cups) |
| | log travel distance (z) | `DistanceFeature` |

**Methods.** Features are divided by their panel sd but not centred, so 0 means
level teams. The analysis has three parts:

* **(a) Regression attribution.** OLS of Δ_mkt (and of the gap) on the features,
  with exact **Shapley** decompositions over the five groups (all 2⁵
  coalitions), 1,000-rep fixture bootstrap bands, and an honest
  24/25 → 25/26 check.
* **(b) Favourite conviction.** An exact decomposition of the market favourites'
  mean supremacy.
* **(c) The Phase 1 state-space model with the features in the observation
  equation.** The coefficients are static Kalman states (integrated out exactly)
  and the noise is **Student-t** (a per-fixture Gamma scale mixture). It is
  fitted by partially collapsed Gibbs. A new engine gate checks this path
  against the batch joint Gaussian: agreement to 7e-15.

## 2. The conviction gap

| Model (walk-forward) | slope of model on market | R² | sd(model) | slope of market on model = R²/slope |
|---|---:|---:|---:|---:|
| m01 pure-Poisson GRW | 0.389 | 0.417 | 0.255 | **1.07** |
| m05 joint pxG + wealth, time decay | 0.358 | 0.595 | 0.196 | 1.66 |
| m12 joint + wealth + lineup | 0.328 | 0.463 | 0.204 | 1.41 |

sd(market) is 0.423 throughout.

**Why the two slopes disagree.** The slope of model on market is
r × sd_model/sd_market; for m01 that is 0.646 × 0.603. It is below 1 whenever the
model is less informed than the market, even for a perfectly calibrated model.
The slope of market on model asks the calibration question: "given the model
says Δ, where does the market sit?"

* For **m01** it is 1.07, essentially calibrated.
* For **m05 and m12** it is well above 1, so those models are genuinely
  compressed. Their pxG and lineup arms feed a shrunk team latent (compare the
  memory note on the feature-compression EDA).

By market-favourite band, `gap_by_favourite_band.csv` shows the goal model
carrying 41–57% of the market's conviction:

| market favourite p | n | market \|Δ\| | goal model, same direction | gap |
|---|---:|---:|---:|---:|
| < 0.45 | 268 | 0.134 | 0.076 | 0.058 |
| 0.45–0.55 | 216 | 0.414 | 0.201 | 0.214 |
| 0.55–0.65 | 80 | 0.678 | 0.320 | 0.358 |
| 0.65–0.75 | 24 | 1.007 | 0.410 | 0.597 |
| ≥ 0.75 | 8 | 1.436 | 0.611 | 0.826 |

**Caveat.** These bands are *selected on the market*. Even a calibrated,
less-informed model regresses towards zero inside them. The table measures
information difference, not mis-scaling. The reverse slope is the scaling test.

## 3. What the market's supremacy is made of

### 3a. Share of variance (Shapley R²)

The bracketed ranges are 90% bootstrap bands.

| Group | All 623 fixtures | Well-identified 532 |
|---|---:|---:|
| goal history | **30.6%** [27.7, 33.5] | 32.5% |
| proxy-xG form | **28.6%** [25.2, 31.6] | 32.1% |
| wealth | 4.1% [2.6, 6.3] | 4.7% |
| lineup (RAPM) | 3.4% [2.5, 4.4] | 4.0% |
| rest & schedule | 0.2% [0.1, 0.9] | 0.2% |
| **unexplained** | **33.2%** [28.0, 38.0] | **26.4%** |

**Honest check** (fit 24/25, R² on 25/26): all five groups give 0.637, against
0.668 in-sample. Goal history alone gives 0.550, proxy xG alone 0.544, wealth
alone 0.092 and lineup alone 0.032. "All but goal history" gives 0.537, so
**proxy xG plus the rest recovers the market almost as well as all of goal
history does.**

### 3b. Feature by feature

Columns are the Shapley R² share and the stand-alone R².

| Feature | Shapley share | alone R² | OLS coef per sd (t, HC1) |
|---|---:|---:|---:|
| pxG-form supremacy | 21.7% | 0.536 | **+0.208 (12.4)** |
| goal-form supremacy | 16.5% | 0.546 | +0.101 (2.9) |
| points-per-game gap | 11.3% | 0.386 | +0.085 (5.1) |
| Δ_goal (m01 GRW) | 10.7% | 0.417 | +0.010 (0.4) |
| production wealth | 2.4% | 0.106 | +0.057 (2.6) |
| lineup RAPM | 2.4% | 0.095 | −0.045 (−3.8) |
| log-sum wealth | 1.7% | 0.086 | −0.029 (−1.5) |
| log travel | 0.2% | 0.002 | +0.019 (1.6) |
| rest Δ | 0.0% | 0.000 | +0.002 (0.2) |

How to read the columns:

* **The GRW adds nothing on top of simple form.** m01's supremacy aligns *less*
  with the market than a plain exponentially weighted 16-match goal form does
  (R² 0.42 vs 0.55). Given goal form and ppg, its coefficient is zero. The
  market's use of results looks like recent form, and the GRW over-smooths it.
* **Wealth: age-weighted beats raw.** With both in, production (age-weighted)
  wealth is positive and raw log-sum wealth negative. Age-adjusted squad value
  beats raw value, but together they are small.
* **The lineup sign is not a real negative effect.** The lineup coefficient is
  *negative* given everything else. Marginally it is positive (alone R² 0.095).
  The history-fitted RAPM ratings are frozen per season and mostly club identity
  (feature-compression EDA), so conditional on wealth and form the lineup term
  picks up a contrast, not team selection.

## 4. The exact attribution of favourite conviction

The market favourites are the 64 fixtures with max(p_home, p_away) ≥ 0.60. Their
mean supremacy in the direction of the favourite is **0.946 log-rate units**. It
splits exactly into an intercept (the league-average home edge), a Shapley
contribution per group, and a residual. The shares sum to 100%.

**Why the market favours some teams so strongly:**

| Component | log-rate | Share of favourite conviction [90% band] | Well-identified books |
|---|---:|---:|---:|
| 1. Squad wealth | 0.022 | **2.3%** [1.0, 4.5] | 2.5% |
| 2. Starting-XI quality (RAPM) | 0.032 | **3.4%** [2.4, 4.5] | 3.9% |
| 3. Proxy-xG chance creation | 0.256 | **27.0%** [22.6, 30.9] | 29.8% |
| 4. Rest & scheduling | −0.001 | **−0.2%** [−0.4, 0.3] | −0.1% |
| 5. Goal history (GRW, goal form, ppg) | 0.260 | **27.5%** [23.5, 31.4] | 28.6% |
| Home advantage / league mean (intercept) | 0.110 | 11.6% [7.2, 16.4] | 10.2% |
| 6. Unexplained / market residual | 0.268 | **28.3%** [23.3, 33.1] | 25.1% |
| **Total** | **0.946** | **100%** | 100% |

**Why the market is sharper than our goal model.** The same decomposition
applied to the gap, where the favourites' mean gap is 0.521:

| Component | Share of the favourite gap | Well-identified books |
|---|---:|---:|
| Proxy-xG chance creation | **30.7%** [24.9, 36.3] | 34.0% |
| Goal history beyond m01 (goal form, ppg, rescaling) | 8.9% [3.4, 14.6] | 12.7% |
| Starting-XI quality (RAPM) | 2.8% [1.9, 4.1] | 3.6% |
| Squad wealth | 1.6% [0.4, 4.5] | 2.1% |
| Home advantage (intercept) | 4.8% [1.9, 8.4] | 2.6% |
| Rest & scheduling | −0.3% | −0.1% |
| **Unexplained** | **51.4%** [42.3, 60.1] | **45.2%** |

Across all fixtures, the gap's Shapley R² is:

* proxy xG 20.3%;
* goal history 19.8%;
* lineup 1.6%;
* wealth 1.4%;
* rest 0.3%;
* unexplained 56.7%.

The honest out-of-sample R² for the gap is 0.311 with all features and 0.223
with proxy xG alone.

## 5. The feature-augmented state-space model

The model is Phase 1's GRW1 with the features in the observation equation
(antisymmetric: +x on the home rate, −x on the away rate) and Student-t noise.
Each arm had 4 chains × (1,000 warm-up + 2,000 × thin 4) draws. **All 59
parameters pass** R̂ ≤ 1.05 and bulk/tail ESS ≥ 200 (min ESS 1,316, max R̂
1.004).

### 5a. What the features absorb

| Arm | σ_obs | ν | weekly σ_att | α spread | β spread | fixtures with ω < 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| GRW1, Gaussian (Phase 1) | 0.093 | ∞ | 0.027 | 0.167 | 0.181 | — |
| GRW1, Student-t | **0.055** | **2.9** | 0.028 | 0.167 | 0.183 | 56 |
| + goal history | 0.054 | 2.8 | 0.022 | **0.113** | **0.131** | 73 |
| + wealth | 0.055 | 2.9 | 0.028 | 0.168 | 0.184 | 53 |
| + lineup | 0.055 | 2.9 | 0.028 | 0.171 | 0.185 | 59 |
| + proxy xG | 0.054 | 2.8 | 0.024 | **0.126** | **0.128** | 68 |
| + rest & schedule | 0.054 | 2.9 | 0.028 | 0.168 | 0.183 | 61 |
| + all non-goal | 0.054 | 2.8 | 0.022 | 0.127 | 0.129 | 67 |
| + all nine | 0.053 | 2.8 | **0.019** | **0.106** | **0.115** | 75 |

The spread columns are the cross-sectional sd of the centred rating, averaged
over in-season weeks. The 90% bands are in `ss_absorption.csv`.

* **Rating spread.** Form, whether goals or proxy xG, is what the market's team
  differences are made of. Proxy xG alone removes a quarter of the attack spread
  and 30% of the defence spread. Wealth and lineup remove nothing: the latent
  already knows who is rich, and the RAPM XI is club identity.
* **Weekly repricing.** The market's weekly rating step falls by a third
  (0.028 → 0.019) once form is observed. Much of what looked like the market's
  random-walk repricing is the market updating on form it can see.
* **σ_obs.** The 0.093 → 0.055 drop is the Student-t tail handling. It is not
  the features; they add only 0.055 → 0.053.

### 5b. Feature coefficients

These are from the all-nine arm, as the effect on log-rate supremacy
(2w per sd), with 95% posterior intervals.

| Feature | per sd | per unit | 95% excludes 0 |
|---|---:|---:|:---:|
| pxG-form supremacy | **+0.180** | +0.475 per goal of pxG-form edge | yes |
| goal-form supremacy | +0.159 | +0.314 per goal of goal-form edge | yes |
| production wealth | +0.078 | +0.239 | yes |
| raw log-sum wealth | −0.057 | −0.230 | yes |
| points-per-game gap | +0.038 | +0.083 per point/game | yes |
| log travel distance | +0.023 | +0.029 | yes |
| lineup RAPM | +0.027 | +0.072 | **no** |
| rest Δ | −0.008 | −0.003 per day | no |
| Δ_goal (m01 GRW) | −0.007 | −0.028 | no |

On its own (arm s5), pxG form is worth +0.300 per sd, or **+0.79 log-rate
supremacy per goal/match of pxG-form edge**. That is roughly the weight the
market puts on one expected goal per match of chance-creation advantage.

**Favourite conviction inside the state-space model.** For the 64 favourites,
the fitted supremacy in the favourite's direction splits as follows
(`ss_favourite_decomposition.csv`):

| Component | Fitted supremacy |
|---|---:|
| pxG form | 0.274 |
| goal form | 0.232 |
| team latent | 0.206 |
| ppg | 0.043 |
| wealth, net | 0.016 |
| lineup | 0.019 |
| travel and rest | 0.004 |

With every feature in, only about a quarter of the favourites' team edge is left
to the latent "team ability".

### 5c. The fixtures the model refuses to believe

The Student-t model gives 75 fixtures ω < 0.5 (`ss_downweighted_fixtures.csv`).
They are not random:

* 28 of the 64 accepted books with exactly 3 selections are down-weighted,
  against 40 of the 544 books with ≥ 5.
* The worst include every totals-only artefact of T015 that sits away from the
  team ratings. Edinburgh City v Bonnyrigg (ω = 0.017) is one.
* Kelty v Hamilton (ω = 0.014) is the only extreme case with a full,
  cross-checked book.

## 6. So why is the market sharper, and what are we missing?

**What the market prices that our goal models do not, in order:**

1. **Chance creation (proxy xG), as recent form.** It is the single largest
   feature: 22% of variance on its own Shapley share, and 31–34% of the favourite
   gap. Our Gen 3/4 joint models do read pxG, but as an *observation* of the
   same shrunk team latent. That is exactly the channel the compression EDA
   showed the market re-weights by 2.4×. The market treats pxG form as a direct,
   fast-moving signal.
2. **Recent results, weighted like form, not like a slow random walk.** A plain
   16-match exponentially weighted goal form beats the m01 GRW at matching the
   market (R² 0.55 vs 0.42), and it makes m01's own supremacy redundant.
3. **An unattributed residual of about 26–33%.** It is not wealth, lineups, rest
   or travel as we measure them. Candidates: injuries and suspensions beyond the
   XI rating, manager changes, motivation and table stakes (Phase 1 found an
   end-of-season cluster), information from other bookmakers, and the thin-book
   inversion noise of §5c.

**What is *not* the lever:**

* **Squad wealth** carries 2–5% once form is known.
* **The shots-RAPM lineup rating**, as currently built (frozen per season, mostly
  club identity), carries about 3% and nothing significant in the state-space
  model. That does not prove lineups don't matter to the market. It says this
  lineup *measure* does not carry the information. The teamsheet test needs an
  XI-deviation feature: today's XI against the team's usual XI.
* **Rest days** barely vary in a league-only calendar (510 of 623 fixtures have
  Δrest = 0).

**Recommendations:**

1. **Add proxy-xG form as a supremacy covariate** (`PxGFeature`, which already
   exists and is fold-safe) directly in the goal models' linear predictor, *not
   only* as a joint-likelihood observation. The market's weight is about
   +0.48 to +0.79 log-rate per goal of pxG-form edge. That is a prior centre for
   the coefficient.
2. **Shorten the effective memory** of the goal-history term, or add goal form
   as a covariate beside the GRW. The market's goal-history use matches a
   16-match half-life better than the GRW's smoothing.
3. **Fix the inversion gates (T015)** and treat 3-selection books as noisy before
   re-measuring any supremacy slope. The Student-t observation model is the
   right default for anything fitted to inverted closes.
4. **Build an XI-deviation lineup feature**, not a frozen-rating XI sum, before
   concluding anything about lineups.

## 7. Caveats

* **This is attribution, not causation.** Shapley shares split shared variance
  evenly among collinear groups. Goal form, ppg and pxG form are correlated
  (both measure recent quality), so their split is a convention. The *sum*, form
  explaining about 59% and the rest about 8%, is robust.
* **The favourite decompositions are selected on the market.** Their
  "unexplained" share includes the regression-to-the-mean that selecting on
  Δ_mkt induces.
* **Look-ahead in the pxG table.** The shot-xG cell table behind pxG is fitted
  on all commentary shots in the store. It holds no team identity (a
  calibration-scale look-ahead only). The form windows themselves are strictly
  pre-match.
* **Lineup and wealth provenance.** They come from the feature-compression EDA
  panel (`inputs/fce_fixture_panel_c3bdb53a.csv`). The lineup is the played XI,
  which is public before the close.
* **Missing cup matches.** Rest is league-only: betdb holds no cup fixtures.
