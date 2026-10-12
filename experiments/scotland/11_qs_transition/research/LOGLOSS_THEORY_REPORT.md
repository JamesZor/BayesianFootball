# Why QS transition fixes move bias but barely move log loss

**Scope:** theory research and read-only CSV calculations, 2026-10-12. No Julia, database access, fitting, or 26/27 scoring. Pooled-scope results are not assessed. Negative ΔLL means improvement; log losses use natural logarithms. Paths below are relative to `experiments/scotland/11_qs_transition/`.

## Summary — direct answers (10 points)

1. **Log loss is not broken:** a 1–2 pp correction near the true probability buys roughly 0.0002–0.0008 per affected binary forecast, before panel dilution.
2. Transition fixtures are **about 30% of scored data**, not negligible: 281/938 fixtures and 1,336/4,406 selections. Dilution is roughly threefold, not tenfold.
3. Typical changes are smaller than the headline cohort shifts: median fixture-maximum |Δp| is **0.16/0.11/0.72/0.18 pp** for A2/entrant/summer/weekly; rare changes reach 6–24 pp.
4. The actual target all-market ΔLLs are **+0.000313/+0.000212/+0.000267/−0.000343**. Worsening and improving selection-loss contributions offset by 94–98%.
5. **Not every transition fix improves transition scores:** all four first-20 all-market point estimates worsen; weekly's overall gain comes from outside that subset.
6. Paired all-market 80%-power thresholds are about **0.0011–0.0012** under independent-fixture assumptions; transition-only thresholds are approximately **0.0025–0.0037**. Non-significance is not equivalence.
7. A precise market likelihood can overwhelm entry priors **after observations arrive**, but cannot observe the next market re-rating before the cutoff. Fitted grid σ values are not exported; dominance is plausible, not established.
8. Prefer a **historically learned, partially pooled boundary-state transition**, with tier-coordinate identification and explicit jump uncertainty, over simply adding another constant tier term or weakening all market observations.
9. Keep outcome-log-loss non-inferiority, add pre-specified subset proper scores, and use **DESIGN §9 MAE_club** for market-tracking error. MAE to the market is not a proper outcome score or proof of betting value.
10. Transition learning matters when it predicts **new information before the first post-move observation**, improves large club-specific errors rather than just signed averages, and replicates across independent transition cohorts.

## 1. What exactly was compared?

### 1.1 Sources and denominators

Read: `DESIGN.md` §§1–4 and 9; `REPORT_S3.md` §§2–3; `REPORT_S1.md` summary. Calculations use:

- `grid/observations_qs_{control,a2,entrant,summer,weekly}.csv` for paired probabilities and realised losses;
- the corresponding `grid/scores_qs_*.csv` for authoritative subset membership/counts and existing paired bootstrap intervals;
- `grid/bias_game_bins.csv` and `eda/results/q1_club_bias.csv` for signed-bias diagnostics.

**Exclude `scores_qs_a2_initial.csv`:** it is the superseded A2 result, not another replicate. Use `scores_qs_a2.csv` with its matching observation file.

All five current observation files contain exactly **4,406 rows / 938 fixtures**, with no duplicate `(model, match_id, selection)` keys. Each arm pairs with all 4,406 control rows, with identical outcome, market probability and family. The design's **1,070 fixtures** are not the available-price scoring denominator.

| Score panel | Selection rows | Fixtures | First-20 transition rows | First-20 transition fixtures |
|---|---:|---:|---:|---:|
| All markets | 4,406 | 938 | 1,336 | 281 |
| 1X2 | 2,692 | 898 | 806 | 269 |
| OU2.5 | 1,200 | 600 | 340 | 170 |
| BTTS | 514 | 257 | 190 | 95 |

These subsets overlap by market, and transition directions can overlap within a match. Do not add direction counts to estimate unique fixtures. The first-20 all-market relegated/promoted subsets have **94/111 scored fixtures**; the brief's 112 promoted observations also appears as the sum of priced *club-win* game-bin observations. Club observations and unique fixtures are not interchangeable.

### 1.2 A consequential score convention

The exported loss is **binary selection cross-entropy**, even for home/draw/away:

\[
\ell(p,y)=-y\log p-(1-y)\log(1-p).
\]

Direct reconstruction in the control differs from `ll_model` by at most **1.11×10⁻¹⁶**. The headline is a selection-row mean, not an equal-fixture or equal-market-family mean. All-market row weights are approximately 61.1% 1X2, 27.2% OU and 11.7% BTTS. Complementary OU/BTTS rows duplicate the underlying binary event; they are not independent observations.

This score is proper for the individual event probabilities, but is **not** the usual categorical score `−log P(realised home/draw/away)`, nor a joint score for all market outcomes. Keep its existing definition for gate continuity. A future categorical-score diagnostic needs an explicit coherent-vector audit: only 896 fixtures have all three exported selections, and some three-probability sums differ from one (maximum arm discrepancy about 0.0112). I therefore do **not** reinterpret a sum of these exported log probabilities as an official categorical score.

## 2. How much did the forecasts actually move?

### 2.1 Per-fixture Δp distributions

For each paired selection, define `δ_jk = p_arm,jk − p_control,jk`. For an actual **per-fixture** summary use

\[
M_j=\max_{k\in\text{available selections of }j}|\delta_{jk}|.
\]

Every fixture gets one observation here, irrespective of its number of priced markets. Units below are **percentage points**. Source: paired `grid/observations_qs_*.csv`; 938 fixtures per arm.

| Arm | Mean M | Median M | P90 M | P95 M | Maximum M | Fixtures with M > 1 pp |
|---|---:|---:|---:|---:|---:|---:|
| A2 | 0.451 | 0.160 | 0.729 | 1.194 | 19.308 | 6.08% |
| Entrant | 0.296 | 0.112 | 0.353 | 0.840 | 23.981 | 4.48% |
| Summer | 1.037 | 0.719 | 2.387 | 3.165 | 6.260 | 35.50% |
| Weekly | 0.854 | 0.181 | 2.597 | 3.648 | 16.263 | 25.59% |

Selection-level mean |δ| is **0.253/0.174/0.573/0.458 pp**, respectively. Thus “1–2 pp” describes an aggregate/cohort scale, **not a per-fixture upper bound**. The largest paired changes are real in these exports:

| Arm | match_id | Selection | Control p | Arm p | ΔLL for this selection |
|---|---:|---|---:|---:|---:|
| A2 | 12477132 | home | 0.40884 | 0.60192 | +0.39543 |
| Entrant | 12476800 | away | 0.38580 | 0.62561 | +0.49502 |
| Summer | 11395474 | home | 0.39841 | 0.46100 | +0.10987 |
| Weekly | 11395658 | away | 0.31703 | 0.47966 | −0.41409 |

Audit these influential cases before claiming that all changes are small, or interpreting every tiny aggregate difference as structural rather than Monte Carlo variation. No inference about their cause is possible from these files alone.

### 2.2 Second-order sensitivity: the sign and denominator matter

Let the true event probability be `r`. Expected binary loss is

\[
L_r(p)=-r\log p-(1-r)\log(1-p).
\]

At the optimum `p=r`, the expected cost of moving **away** by δ is

\[
L_p(p+\delta)-L_p(p)
=D_{KL}(\mathrm{Bern}(p)\Vert\mathrm{Bern}(p+\delta))
\approx\frac{\delta^2}{2p(1-p)}.
\]

Therefore the brief's `Σ δ²/[2p(1−p)]` is a **total local divergence**, and must be divided by the number/weight of scored observations to obtain mean LL. It is nonnegative, **not a formula asserting every intervention improves LL**. If control has error `e=p−r`, then, near `r`,

\[
L_r(p+\delta)-L_r(p)
\approx\frac{2e\delta+\delta^2}{2r(1-r)}.
\]

A correction `δ=−e` gains approximately `e²/[2r(1−r)]`; an overshoot or wrong-direction correction may lose. These are Taylor expansions of the logarithmic-score/KL relationship [R1], not assumptions that the market equals truth.

**Back of envelope:** at `p=0.40`, a complete correction of a 1 pp error gains about **0.000208** on the affected binary event; 2 pp gives **0.000833**. Applied to 30.3% of scored rows and nowhere else, these become **0.000063 / 0.000253** in the target mean. The nominal `300/1070` fraction would give 0.000058 / 0.000234. Dilution is material but not the whole explanation. Under this simplified equal-event scenario, a target gain of 0.001 requires correcting roughly **4 pp** throughout the affected subset. A coherent win-probability move also redistributes other outcomes, so this is a scale calculation, not an exact 1X2 forecast.

For the **actual** row-wise shifts, compute

\[
Q=\frac1{4406}\sum_{jk}\frac{\delta_{jk}^2}{2p_{jk}(1-p_{jk})}.
\]

| Arm | Actual target ΔLL | Expected local curvature Q | Realised first-order term A | Realised quadratic term B | Taylor remainder |
|---|---:|---:|---:|---:|---:|
| A2 | +0.000313097 | 0.000181424 | +0.000120411 | 0.000182959 | +0.000009728 |
| Entrant | +0.000211672 | 0.000146715 | +0.000041930 | 0.000139310 | +0.000030432 |
| Summer | +0.000266792 | 0.000188344 | +0.000065060 | 0.000200842 | +0.000000889 |
| Weekly | −0.000343339 | 0.000280623 | −0.000610136 | 0.000276119 | −0.000009322 |

Here, directly from the same probability/outcome rows,

\[
A=\overline{\frac{(p-y)\delta}{p(1-p)}},\qquad
B=\frac12\overline{\delta^2\left(\frac{y}{p^2}+\frac{1-y}{(1-p)^2}\right)}.
\]

**The observed score is not flat to first order.** Its first-order outcome term can dominate the positive curvature; weekly illustrates this. Near truth, the first-order term has mean zero but random variation of order `|δ|`, whereas the expected curvature is order `δ²`. That signal/noise mismatch is why small genuine improvements need substantial replication. The entrant arm's larger remainder also warns against applying a small-δ approximation to its extreme tail.

### 2.3 How much ΔLL is actually carried by transition fixtures?

Use the authoritative `transition_any_first20` and `target` score rows. Since every arm has the same paired panel, exact attribution is

\[
\Delta\bar L_{all}=w_T\Delta\bar L_T+(1-w_T)\Delta\bar L_{rest},\quad
w_T=N_{rows,T}/N_{rows,all}.
\]

`w_T=1336/4406=0.3032` for all markets and `806/2692=0.2994` for 1X2. This decomposition respects **the harness's row weights**.

| Arm / market | Target ΔLL | Transition-subset ΔLL | Transition contribution to target | Rest contribution to target | Signed share from transitions |
|---|---:|---:|---:|---:|---:|
| A2 / all | +0.000313097 | +0.000758913 | +0.000230120 | +0.000082978 | 73.5% |
| Entrant / all | +0.000211672 | +0.000862335 | +0.000261480 | −0.000049808 | 123.5% |
| Summer / all | +0.000266792 | +0.000156933 | +0.000047586 | +0.000219206 | 17.8% |
| Weekly / all | −0.000343339 | +0.000480799 | +0.000145789 | −0.000489128 | −42.5% |
| A2 / 1X2 | +0.000323220 | +0.000768082 | +0.000229968 | +0.000093252 | 71.1% |
| Entrant / 1X2 | +0.000470237 | +0.001300718 | +0.000389442 | +0.000080795 | 82.8% |
| Summer / 1X2 | −0.000027658 | −0.000847830 | −0.000253845 | +0.000226187 | 917.8% |
| Weekly / 1X2 | −0.000659091 | +0.000480037 | +0.000143726 | −0.000802817 | −21.8% |

Shares exceeding 100%, or negative shares, are algebraic consequences of offsetting contributions, not proportions of independent explained variation. In particular:

- A2 and entrant concentrate much of their small aggregate **deterioration** in transitions.
- Summer improves transition 1X2 but worsens transition OU2.5 (**+0.002008**) and BTTS (**+0.001107**), leaving all-market ΔLL slightly positive.
- Weekly's aggregate improvement is **not evidence of improved transition outcome prediction**. Its first-20 transition all-market estimate is positive, albeit very uncertain.

### 2.4 Quantifying cancellation — two different phenomena

**Realised loss cancellation.** Sum positive and negative paired selection losses before dividing by 4,406:

| Arm | Sum of worsening ΔLL | Sum of improving ΔLL | Net sum | `1 − |net| / sum|ΔLL|` |
|---|---:|---:|---:|---:|
| A2 | +11.99903 | −10.61953 | +1.37951 | 93.90% |
| Entrant | +8.39049 | −7.45786 | +0.93262 | 94.12% |
| Summer | +26.26977 | −25.09428 | +1.17549 | 97.71% |
| Weekly | +19.36319 | −20.87594 | −1.51275 | 96.24% |

These are descriptive offsets, not independent trials or a variance reduction estimate. Some occur across selections of the same fixture. Aggregating to each fixture's mean loss first still produces approximately **95.4/88.5/97.3/92.7%** cancellation, respectively, although that equal-fixture statistic is not the official row-weighted score.

**Signed probability bias cancellation.** In `grid/bias_game_bins.csv`, control relegation biases are **+0.163, +1.905, −0.373, −0.468 pp** at games 1, 2–5, 6–10, 11–20, with 5/19/22/48 club observations. Their weighted mean is **+0.067 pp**: near-zero bias despite nonzero errors. Even `mean |bin mean|` is about **0.720 pp**, more than ten times `|overall mean|`; it is only a lower-resolution diagnostic, not MAE_club.

The S1 per-club export illustrates cross-club cancellation too: Inverness **−0.359 pp**, Hamilton **+4.049 pp**, Arbroath **+0.871 pp**. These are `qs_marketobs` S1 values on its shorter panel (three relegated club-seasons here), **not a recomputation of the five-club S3 control**. Opposite signed probability errors do not cancel their squared/KL error near truth. It is the old *signed mean bias diagnostic* that cancels mechanically; realised LL changes cancel through outcomes and intervention trade-offs.

## 3. Power: what could this panel detect?

### 3.1 Paired fixture calculation

Do not treat 4,406 correlated selections, or two separately estimated model scores, as independent observations. For fixture `j`, let `S_j=Σ_k Δℓ_jk`, `K_j` its scored-row count, `N` the number of fixtures and `M=Σ_j K_j`. For the official ratio mean `μ̂=Σ S_j/M`, an independent-fixture cluster linearisation gives

\[
\widehat{SE}(\hat\mu)=
\frac{\sqrt{\frac{N}{N-1}\sum_j(S_j-\hat\mu K_j)^2}}{M}.
\]

A two-sided 5% test with approximately 80% power needs

\[
|\Delta|\gtrsim(z_{.975}+z_{.80})SE\approx2.80\,SE.
\]

This is a planning approximation conditional on the current paired forecast differences, not a claim that these arms have an established nonzero effect. Source: paired observation CSVs.

| Arm | All-market SE, N=938 | All-market detectable | 1X2 SE, N=898 | 1X2 detectable |
|---|---:|---:|---:|---:|
| A2 | 0.000395 | 0.001105 | 0.000586 | 0.001640 |
| Entrant | 0.000403 | 0.001128 | 0.000534 | 0.001496 |
| Summer | 0.000393 | 0.001100 | 0.000594 | 0.001662 |
| Weekly | 0.000435 | 0.001219 | 0.000688 | 0.001927 |

Assuming the same variance/composition, required fixture counts scale as

\[
N_{new}\approx N\left(\frac{2.80SE}{\Delta_{desired}}\right)^2.
\]

For all-market improvement **0.001**, approximately **1,134–1,394 fixtures**; for **0.0003**, approximately **12,602–15,492 fixtures**. These are idealised independent-fixture numbers, not guarantees that a future league/season has the same variance. Counting all 1,070 design fixtures as scored would spuriously improve power; even full coverage at unchanged variance would reduce SE only by roughly `sqrt(938/1070)=0.936`.

### 3.2 Transition-only power and dependence

The saved `scores_qs_*.csv` already contain paired fixture-cluster bootstrap intervals for the exact transition subsets. Approximate `SE ≈ (upper−lower)/3.92`, then multiply by 2.80:

| Arm | Transition all-market detectable, N=281 | Transition 1X2 detectable, N=269 |
|---|---:|---:|
| A2 | 0.003655 | 0.005462 |
| Entrant | 0.003397 | 0.004841 |
| Summer | 0.002481 | 0.004046 |
| Weekly | 0.002822 | 0.004672 |

These are **CI-width-derived rough power estimates**, not new bootstrap tests; asymmetric intervals, particularly entrant, limit the normal approximation. For example, A2 transition all-market ΔLL is **+0.000759 [−0.001711,+0.003406]**, and summer is **+0.000157 [−0.001590,+0.001884]**. Small meaningful benefits and harms remain compatible with the data.

Restricting the score removes irrelevant fixtures but also reduces sample size; it is not an automatic power miracle. With effects only on transitions and exactly zero paired differences elsewhere, restriction can leave the signal-to-noise ratio essentially unchanged. Its primary advantages here are **correct attribution and a relevant estimand**; it can improve power when it removes non-transition noise.

Across-fixture dependence from the same clubs, shared fitted folds and seasons is not removed by a fixture bootstrap. The relevant effective replication for generalising transition learning is closer to **5 relegated / 6 promoted club-seasons**, not hundreds of independent promotion experiments. Add club-season sensitivity and temporal/fold blocks, with few-cluster caveats and leave-one-club-season-out results. Two transitioning clubs in a match also share an outcome. Diebold–Mariano's variance uses the **long-run variance of paired loss differences**, not simply the marginal score variance [R2]; Patton stresses defining whether we compare issued forecasts or estimated model procedures [R3]. Do not report a naïve DM p-value on arbitrarily ordered selection rows.

The frozen **+0.002 non-inferiority gate** asks a different question from detecting a 0.0003 gain. Its existing upper bounds are about +0.00110/+0.00106/+0.00103/+0.00051 for A2/entrant/summer/weekly: all pass that global gate. This does not establish transition superiority, nor transition-specific non-inferiority at the same margin.

## 4. Is the market likelihood too strong?

### 4.1 What is, and is not, observed in the records

The context model observes past inverted market log-rates through one fitted Gaussian scale. The grid score/convergence/fold-diagnostic CSVs do **not** export posterior `σ_obs` by fitted arm/fold. Searching the experiment CSVs finds `obs.σ_obs` in **synthetic recovery** outputs, including `pooled/recovery_three_seeds.csv`, with generating value 0.07 and recovered medians about 0.069–0.073. Those are **not empirical grid σ estimates** and must not be used to claim likelihood dominance. Prior-manifest `sigma_q`/`sigma_s` are also not market observation σ.

Therefore an empirical control-versus-arm σ comparison is unavailable here. The structural calculation below is conditional, and its proposed sensitivity tests require a subsequent authorised fitting task.

### 4.2 Precision and prior attenuation

A local scalar Gaussian illustration gives the mechanism exactly:

\[
x\sim N(m_0,\tau^2),\quad z_i\mid x\sim N(x,\sigma^2),\quad
E[x\mid z]=\frac{m_0/\tau^2+\sum_i z_i/\sigma^2}{1/\tau^2+k/\sigma^2}.
\]

Changing the prior mean by `Δm₀` changes the posterior mean by

\[
\Delta E[x\mid z]=\underbrace{\frac{1}{1+k\tau^2/\sigma^2}}_{\text{prior transmission}}\Delta m_0.
\]

QS is multivariate/dynamic, but locally the analogous precision is `P₀⁻¹ + H′R⁻¹H + I_goals`. The relevant `k` is **effective information about this state**, not the number of all historical fixtures: opponent uncertainty, changing states and correlated observations matter.

Using the stated quality/style convention,

\[
\log\lambda_h=\mu+\gamma+(q_h-q_a)+(s_h+s_a),\quad
\log\lambda_a=\mu-(q_h-q_a)+(s_h+s_a).
\]

If the two market-rate errors are conditionally independent with variance `σ²`, one fixture gives local information **2/σ²** about a quality difference (opposite rate derivatives +1 and −1). A prior SD `τ_q` then has transmission approximately `1/[1+2kτ_q²/σ²]`, holding other quantities fixed. **Illustration only:** `τ_q=σ=0.10` gives 1/3 after one paired-rate observation, 1/11 after five. A 0.10 prior shift would transmit only about 0.033 or 0.009 quality units, before propagation through the score grid.

For independent Poisson goals, the corresponding local quality information is `λ_h+λ_a`, so

\[
\frac{I_{market}}{I_{goals}}\approx\frac{2}{\sigma^2(\lambda_h+\lambda_a)}.
\]

At an **illustrative** total rate 2.5, this ratio is 80 for σ=0.10, 8.9 for σ=0.30, and 1.6 for σ=0.70. This demonstrates why σ matters enormously; it does not identify the actual ratio in QS or account for market-error covariance.

### 4.3 Strong likelihood is not the same as a pinned future forecast

The state-space distinction is essential:

\[
x_{t|t-1}=F x_{t-1|t-1}+Bz_{transition},\quad
K_t=P_{t|t-1}H'(HP_{t|t-1}H'+R)^{-1},\quad
x_{t|t}=x_{t|t-1}+K_t(z_t-Hx_{t|t-1}).
\]

Small `R` rapidly corrects states **once the new quote is observed**; a transition term changes the *prediction before that update*. A tight market likelihood is therefore compatible with valuable cold-start transition learning. A small process/jump variance can cause lag, while small measurement variance can reduce it. These are different mechanisms.

The actual target-market quote is not supplied by a past-only likelihood. Remaining discrepancies can arise from cutoff age and intervening news, incomplete old-tier coverage, incorrect transition-state means/variances, shrinkage of unusual clubs, and the score-grid/rate representation's inability to reproduce all market partitions simultaneously. One global σ may also conflate quote noise, latent process misspecification, and correlated home/away inversion error. Outcome randomness remains even if all of these are solved.

A2/entrant's small typical movements are **consistent with** strong historical evidence limiting prior influence, but do not prove it: overlapping parameterisations, a good existing state estimate, gauge changes and limited prior novelty are alternatives. The large tail changes also rule out a universal claim that priors cannot matter.

A useful scale comparison: current control all-market LL is **0.633974876**, versus market **0.632341434**, a realised gap of **0.001633442**. This is not an oracle bound or a statistically established inefficiency gap. As a noise-free *market-as-truth diagnostic*, the same observation rows give

\[
\Delta CE_{market}=\overline{-m\log(p_{arm}/p_0)-(1-m)\log[(1-p_{arm})/(1-p_0)]}.
\]

Values are **+0.0000233/+0.0000201/+0.0001330/−0.0004499** for A2/entrant/summer/weekly. Even on this hypothetical market-truth criterion, only weekly improves globally. Thus “the outcome score cannot see our improvements” is not a sufficient account: some changes do not improve global market-relative KL either. This diagnostic does not substitute for DESIGN §9's club-balanced MAE.

## 5. Ranked recommendations — equations, expected effects and tests

### 1. Separate market tracking, outcome accuracy and economic utility

**Established evaluation principle; application here is recommended.** Keep the frozen all-market non-inferiority gate and report transition-specific paired proper scores by direction and games 1–10/11–20. Freeze outcome-independent cohort rules before evaluation:

\[
S_T(p,y\mid x)=1\{x\in T\}\,\ell(p(x),y).
\]

Conditioning on known transition covariates preserves propriety within that subset. It says nothing about forecasts outside it. Do not select fixtures retrospectively because the club won/lost or because an arm looked bad. “Local score” in scoring-rule theory means dependence on density at the realised outcome; it is not a synonym for “score only transitioning teams.” Outcome-weighted log scores can lose propriety; conditional/censored constructions address that different problem [R1,R4].

Implement DESIGN §9 literally:

\[
MAE_{club}=\frac1C\sum_c\frac1{n_c}\sum_{j\in c}|p_{model,cj}-p_{market,cj}|.
\]

It is **not** `mean_c |mean_j error_cj|`. Retain worst-club absolute signed mean and early/late splits as specified. Expected effect: reveal reductions in individual discrepancies hidden by signed means, without claiming a new outcome score win. Test with the stipulated paired club-season bootstrap, leave-one-club-out sensitivity, and the unchanged frozen gate. With five/six clusters, display every club, not just a significance flag; one entered-SPFL priced cluster cannot yield a meaningful between-club bootstrap interval.

Keep §9 labelled **post hoc** on 23/24–25/26 and pre-registered for the one-time 26/27 test. No further tuning on that holdout. More independent seasons/leagues, not merely more selections per fixture, are needed for strong transition-generalisation claims.

### 2. Learn the boundary transition, not another unrestricted permanent offset

**Established ingredients:** hierarchical partial pooling and dynamic state priors. **Proposal:** this QS-specific transition law. Let `q̃` be strength centred within its tier, and `a_l` the globally anchored tier strength:

\[
\tilde q_{c,new}=\kappa_{d}\tilde q_{c,old}+a_{old}-a_{new}+b_d+u_{c,season},
\quad u_{c,season}\sim N(0,\tau_d^2),
\]
\[
b_d\sim N(b_0,v_b^2),\quad
\tilde s_{c,new}\sim N(\rho_s\tilde s_{c,old}+b_{s,d},\tau_{s,d}^2).
\]

Direction `d` distinguishes promotion, relegation and new entrants; estimate hyperparameters from **earlier transitions only**, preferably across connected tiers. For a global-coordinate q state, remove the `a_old−a_new` coordinate conversion: do not count a tier change twice. Anchor tier levels and centre residual club effects. New entrants need a group-distribution fallback, with genuine uncertainty, rather than invented previous history.

Expected effect: improve the pre-observation boundary forecast and exceptional clubs, with most gains early and aggregate LL often below 0.001. The S1 summary's market re-ratings (+0.212/−0.220 in its D units) and unequal tier gaps support learning direction/scale, but are not ready-made q coefficients: preserve its unit/gauge conversion. This is an extension beyond a fixed A2 term only if the mean, carry and uncertainty actually borrow identifiable information unavailable in the existing arms.

Test: historical expanding-season validation; leave-one-transition-club-season-out; first-quote prediction; games 1–5 versus 11–20; subset proper score plus MAE_club; simulated recovery and gauge identifiability before any grid. Avoid using old quotes once in an empirical prior and again as conditionally independent likelihood evidence. Either fit a coherent joint multilevel model, exclude reused endpoint evidence, or explicitly cross-fit the prior construction.

Literature does not establish a universal promotion coefficient. Owen's Scottish model reused a relegated team's prior for the promoted team and explicitly called that simplistic [R5]. Baio–Blangiardo show why a common strength distribution can overshrink extremes and discuss group/heavy-tailed priors [R6]. These support the mechanism, **not a claim that the proposed QS law will win**. With only five/six target club-seasons, shrink rather than fit many direction×tier×season parameters freely.

### 3. Distinguish transition jumps, ordinary diffusion and observation noise

**Established state-space mechanism; proposed QS specification:**

\[
q_{c,t}=q_{c,t-1}+b_d I_{move,c,t}+\epsilon_{c,t},\quad
\epsilon_{c,t}\sim N(0,Q_{base}\Delta t+Q_{summer}I_{summer}+Q_d I_{move}).
\]

Expected effect: allow a discontinuous boundary move and faster adaptation without making every in-season state volatile. Variance inflation alone changes uncertainty/adaptation, not the correct directional mean; summer's mixed derivative-market results warn against equating “more flexibility” with a fix. Combine with proposal 2 only after separate attribution.

Test: predictive innovation calibration by transition status and forecast age; post-boundary adaptation speed; posterior predictive dispersion and derivative-market scores; one-factor-at-a-time comparison against the existing summer/weekly arms. Weekly already provides some evidence for reducing general forecast staleness, **not yet for improving transition LL**. Pooled historical tiers/cups may strengthen identification, but their running arm supplies no result to this report.

### 4. Diagnose likelihood dominance before changing its weight

**Established precision arithmetic/generalised Bayes; tuning choice is a proposal.** First export posterior market σ, process/jump scales, and source/transition residuals by fold. Estimate home/away residual covariance instead of assuming two independent equally precise measurements. In quality/style coordinates,

\[
z_D=(z_h-z_a-\gamma)/2,\quad z_S=(z_h+z_a-2\mu-\gamma)/2,
\]

with fitted covariance `R_DS`; independent equal original errors imply variances `σ²/2`, but inverted-market errors need not satisfy that assumption.

A sensitivity model is

\[
\pi_w(\theta\mid D)\propto\pi(\theta)L_{goals}(\theta)
\exp\{w_M\log L_{market}(\theta)\}.
\]

For fixed Gaussian σ, curvature is scaled by `w_M` (effective observation variance `σ²/w_M`). Bissiri–Holmes–Walker justify generalised loss updates and emphasise calibrating their scale [R7]. If σ is also re-estimated, do not assume this fixed-σ equivalence describes the resulting fit; compare posterior precision and prediction empirically. Correlated/reused evidence is a better reason for tempering than a desire to make a favourite prior “win.”

Expected effect of downweighting: greater prior/process influence and uncertainty, **possibly worse** market tracking and slower adaptation to genuine re-ratings. Conversely, upweighting transitions puts more attention on those data but creates no new independent transition examples. A transition-dependent scale should describe residual reliability, not encode the desired result.

Test a small, pre-specified weight sensitivity grid only in historical training/validation, plus separate perturbations of jump variance and prior mean. Measure `∂p_pred/∂m_prior`, effective market-versus-goal precision and held-out transition scores. If information is available only after cutoff, changing σ cannot fix its absence. This diagnostic should precede another expensive large grid.

### 5. Decision-focused learning is last, not a repair for an underpowered score

**Established broad framework; football/Kelly application here is a proposal.** A fixed policy maps forecasts and executable odds to stakes `a(p,o)`. A decision objective could be

\[
\min_\theta\sum_j-\log\{1+a(p_\theta(x_j),o_j)'R_j\},
\]

with explicit bankroll/risk/commission/liquidity constraints. Smart Predict-then-Optimize aligns estimation with downstream decision costs [R8], but its general results do not establish calibrated football probabilities or validate this Kelly objective.

Expected effect: possibly better decisions near betting thresholds even when population LL barely changes; a 1 pp correction can remove a bad bet. Risks: noisy realised returns, unstable stakes, policy overfitting, and sacrificing probability calibration. With only a handful of independent transitioning clubs, end-to-end outcome optimisation is particularly fragile.

First test a frozen forecast with conservative staking/shrinkage or a transition uncertainty gate. Compare paired executable-price utility/CLV and drawdown in chronological replay while retaining proper-score and market-tracking guardrails. Do not train on closing prices unavailable at the execution instant, optimise on the held-back 26/27 outcomes, or claim the market-relative MAE result alone establishes an edge.

## 6. Reproduction and limits

All calculations ran in **in-memory DuckDB**; no derived data files or database were written. Examples below use the repository root as working directory. Do not glob the initial A2 score export into the current-arm table.

```sql
-- Run with duckdb -csv -c "..."; paired row ingredients.
WITH o AS (
  SELECT * FROM read_csv(
    'experiments/scotland/11_qs_transition/grid/observations_qs_*.csv')
), d AS (
  SELECT a.model,a.match_id,a.selection,a.family,a.y,
         c.p_model AS p,a.p_model-c.p_model AS dp,
         a.ll_model-c.ll_model AS dll
  FROM o a JOIN o c USING(match_id,selection)
  WHERE c.model='qs_control' AND a.model<>'qs_control'
)
SELECT model,count(*) AS n,count(DISTINCT match_id) AS fixtures,
       avg(dll) AS delta_ll,
       avg(dp*dp/(2*p*(1-p))) AS curvature,
       avg((p-y)*dp/(p*(1-p))) AS first_order
FROM d GROUP BY model ORDER BY model;
```

For the per-fixture table, group `d` by `(model,match_id)` and compute `max(abs(dp))`, then `quantile_cont(...,0.5/0.9/0.95)`. For the paired SE, use `sum(dll)` and `count(*)` in that grouping and the ratio-mean equation in §3. For attribution, select `metric='delta_logloss_vs_control'`, join each arm's `target` and `transition_any_first20` score rows by market, and multiply the latter's `value` by `n_obs_transition/n_obs_target`.

**Deliberate limits / discarded shortcuts:**

- No current-grid posterior σ estimate was located; synthetic recovery σ is not an empirical substitute.
- An attempted join to S1 `q1_club_observations.csv` using `model='td_lower_base'`, `group!='nonmover'`, `game<=20` produced **280** priced 1X2 fixtures, not the harness's **269**. That reconstruction was rejected. All reported transition attributions and power approximations use the authoritative current score subsets; no old-panel mask is silently substituted. Per-fixture Δp distributions are reported for the complete current scored panel, not a falsely matched transition subset.
- No new MAE_club gate/CI is asserted here; the report explains and recommends the exact §9 estimand without fabricating current club-level membership or resampling evidence.
- The sample-size estimates omit across-fixture dependence, future distribution shift, fit-to-fit Monte Carlo uncertainty and multiple-arm selection. They are optimistic planning approximations, not certification of power.
- Library search used local Markdown helpers only, not its PostgreSQL catalogue, to respect the no-DB instruction. Converted texts were searched for `Gneiting`, `Diebold`, `promoted`, `relegated`, `shrink`; relevant pages were read. Unconverted/catalogue-only books were not searched. Web sources below cover scoring theory, forecast comparison, loss weighting and decision-focused learning.

## References

**[R1]** Gneiting, T. & Raftery, A. E. (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*, especially categorical/logarithmic scores and their KL divergence (§§3–4). https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf . Proper scores identify the true distribution in expectation; the second-order expansions here are derived from that score.

**[R2]** Diebold, F. X. & Mariano, R. S. (1995; linked 2002 reprint), *Comparing Predictive Accuracy*, §1.1: paired loss differences and long-run variance. http://eurequa.univ-paris1.fr/membres/Ahamada/cours/article1.pdf .

**[R3]** Patton, A. J. (2014), *Discussion of “Comparing Predictive Accuracy, Twenty Years Later: A Personal Perspective on the Use and Abuse of Diebold–Mariano Tests”*, pp. 1–5. https://public.econ.duke.edu/~ap172/Patton_Diebold_JBES_discussion_21apr14.pdf . Distinguishes issued-forecast comparisons from population-model comparisons and discusses inference under dependence.

**[R4]** Diks, C., Panchenko, V. & van Dijk, D. (2011), *Likelihood-based Scoring Rules for Comparing Density Forecasts in Tails*. Read author presentation, slides 9, 13–16 and 21 (PDF pages 10, 14–17 and 22): weighted-score pitfalls, proper conditional/censored alternatives, HAC comparison. https://indico.ictp.it/event/a10132/session/34/contribution/21/material/0/0.pdf .

**[R5]** Library `owen2011-dynamic-bayesian-forecasting-models-football`, *Dynamic Bayesian forecasting models of football match outcomes with estimation of the evolution variance parameter*, **source-PDF p. 6**: season-start priors from preceding-season data; promoted-team prior borrowed from relegated team as an explicitly simplistic choice.

**[R6]** Library `baio2010-bayesian-hierarchical-model-prediction-football`, Baio & Blangiardo, *Bayesian hierarchical model for the prediction of football results*, **source-PDF pp. 6–7**, §4: overshrinkage and group/heavy-tailed prior alternatives. This is evidence for a modelling issue, not a transition-arm performance guarantee.

**[R7]** Bissiri, P. G., Holmes, C. C. & Walker, S. G. (2016), *A General Framework for Updating Belief Distributions*, §§2.3–3: `posterior ∝ exp(−Σ loss) × prior` and calibration of relative loss scale. https://arxiv.org/html/1306.6430v2 .

**[R8]** Elmachtoub, A. N. & Grigas, P., *Smart “Predict, then Optimize”*, abstract/framework (arXiv version; published Management Science 2022). https://arxiv.org/abs/1710.08005 . Decision-regret objectives and portfolio examples; no direct evidence for this QS transition application.

**Bottom line:** the evidence supports a small, heterogeneous market-tracking problem and insufficient power to establish tiny outcome-score improvements—not a defective log score, a demonstrated over-strong likelihood, or an established need to optimise betting returns directly. Learn the information missing at the transition boundary, preserve temporal/evidence discipline, and evaluate the intended subset without abandoning proper-score guardrails.
