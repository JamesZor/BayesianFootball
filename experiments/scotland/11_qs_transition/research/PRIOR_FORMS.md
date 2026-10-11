# Candidate F1/F2 priors on quality and style

**Recommendation for Stage 1:** compare P1 (cohort baseline), P2 (club carry + empirical transition) and P3 (explicit tier rebase + residual transition) as **prior-only forward predictions**. Choose no numerical hyperparameters from the leaked TODO 029 panel. These are proposed QS adaptations, not published QS formulas or authorization to fit new production models.

Sources below use the fully resolved citations in [RESEARCH_REPORT.md](RESEARCH_REPORT.md); the most direct precedents are Owen's informed season-start means [OW6], Ridall's promoted-club prior/past-season training [RI11], and Dixon–Coles's connected multi-division scale [DC7].

## Shared definitions and non-negotiable construction

Let S be the target season and `H_<S` all permitted observations from **seasons strictly earlier than S**. All quantities below are re-derived from `H_<S` for each fold (or deterministically cached by S/input hash), so the prior is unchanged across folds of S. Historical destination opening quotes are allowed as training labels only when their destination season is `<S`.

Let `x_i=(q_i,s_i)'`, `α_i=q_i+s_i`, `β_i=s_i−q_i`. Positive q means greater quality; positive s raises both scoring and conceding propensity. For comparing movers, use this explicit reporting gauge:

- `q_i` is quality relative to the mean of the club's **current division**.
- `s_i` is style on a **common historical μ reference**, not separately recentered in each league-season. Otherwise a change in the goal environment can masquerade as a style shift.
- `θ_ℓ` is division ℓ's quality offset on the connected common-pyramid scale, with one declared weighted-sum-zero anchor. The production common-scale quality is `q_i^G=q_i+θ_ℓ`. For a lower-scope 56/57 model, map both divisions to its one common reference; do **not** erase their relative offsets by centring each division independently. If offsets are unidentified, flag the pooled-scope mapping as unresolved rather than invent them.

This gauge is for the equations, not an assertion about current QS implementation. A shared shift `q→q−c_q` cancels out of within-scope rate differences. A shared shift `s→s−c_s` requires `μ→μ+2c_s` to preserve rates. Any eventual implementation must apply the same deterministic reference transformation to prior means, covariance and states and preserve the predictors. If a builder already supplies league effects, store only the complementary residual prior: **one tier effect, not two**. [derivations from supplied QS equations; DC7]

### What counts as a historical club estimate?

Write `x̂_i^-` for an old-season endpoint estimate and `V_i^-` for its estimation covariance, including staleness/reliability adjustments. Estimate it from opponent-adjusted market rates, not a raw average of the club's implied goals. One fixture supplies only

\[
D=(\log\lambda_h-\log\lambda_a-\gamma)/2=q_h-q_a,
\qquad U=(\log\lambda_h+\log\lambda_a-2\mu-\gamma)/2=s_h+s_a.
\]

Use a declared connected reconstruction and propagate uncertainty. Fit any recency weighting or reliability calibration on earlier training seasons only. Endpoint/first-quote labels must use information available at their stated date; whole-season smoothing of the destination season is not an opening quote. First-quote validation can instead predict D/U directly, avoiding the fiction of observing an isolated club state. [report §2 derivation; EG market-rate precedent]

Let `g=(origin tier,destination tier,entry type)` with separate types for promoted, relegated, first-SPFL entrant, returning-after-gap and non-mover where supported. Pool sparse cells toward broader **entry-type** parents; do not force promotion/relegation means to be negatives. Empirical covariance must distinguish true between-club variation from `V^-` and noisy opening labels. Uncertain fitted hypermeans/offsets contribute predictive uncertainty too. [adaptation of RI11 and BB6–7]

### Attachment to a zero-centred GRW

Each proposal gives a distribution for a genuinely **new initial node or a season boundary**. Implementationally its Gaussian version can retain

\[
x_{i,0}=m_i+L_i z_i,\quad z_i\sim N_2(0,I),\quad L_iL_i'=C_i,
\qquad x_{i,t}=x_{i,t-1}+\epsilon_{i,t},\ E\epsilon_{i,t}=0.
\]

It changes the initial/boundary **mean**, not the mean of every fortnightly innovation. Non-centred innovations remain zero-centred. Do not reset an incumbent to the same prior at every fold. A boundary transition conditions on the previous state; an entrant prior supplies the new node. [OW6; proposed QS parametrisation]

**Two implementation gates:** (1) a prior using data up to S−1 cannot be placed at the start of a historical multi-season window that precedes those data; (2) market rows summarized by the prior must not be silently counted again in the historical market likelihood. Prefer disjoint pre-window summaries, or sequential boundary propagation followed by only subsequent observations. With overlapping data it is empirical-Bayes regularisation, not an independent Bayes prior; no arbitrary covariance inflation fixes the factorisation. Stage 1 must supply the overlap manifest before Stage 2 chooses a path. [report §3 proposed safeguards]

## P1 — Hierarchically pooled entry-cohort prior (minimal F2 / F1 fallback)

For group g and its broader entry-type parent h(g), use

\[
m_g\mid m_{h(g)},T_{h(g)}\sim N_2(m_{h(g)},T_{h(g)}),
\qquad x_{i,0}\mid g,H_{<S}\ \approx\ N_2(\widehat m_g,\Sigma_g+U_{m_g}).
\]

Here `Σ_g` is intrinsic entrant heterogeneity and `U_mg` uncertainty in the learned group mean. The approximation is an empirical-Bayes/predictive summary; integrating hyperparameters or a cohort bootstrap is preferable to pretending their estimates are exact. A diagonal covariance with pooled scales is the starting simplification; free q/s covariance needs enough historical evidence.

- **Per-fold construction:** form old cohorts' opening labels using the pre-quote opponent/gauge rules above; estimate/shrink means and noise-adjusted dispersion only from destination seasons `<S`. Freeze the hierarchy, pooling strength and uncertainty floor before validating S. Non-movers may inform summer variance but must not overwhelm a mover-specific mean.
- **No-market-history fallback:** this is the fallback itself. Use the most specific sufficiently supported entry cohort; otherwise its pooled entry-type parent. With no earlier relevant cohort at all, use a declared broad `N_2(0,Σ_weak)` sensitivity baseline and explicitly mark the mean as unlearned. Do not substitute future cohorts or TODO 029 constants. A results-only historical cohort is optional and separately labelled, never presented as an observed market prior.
- **Literature basis:** Ridall's common promoted-club distribution and earlier-season hyperparameter training [RI11]; Owen's non-average Scottish entrant means [OW6]; Baio–Blangiardo's warning about one-population overshrinkage [BB6–7]. Our observed entry groups differ from their latent performance mixtures.
- **EDA must estimate:** `m_g`, group counts, pooled `Σ_g`, mean uncertainty, opening-label noise, admissible pooling granularity, and forward error/coverage of the no-data fallback. In particular, how far are relegated/promoted means from zero and is their s component distinguishable from zero?
- **Interpretation:** fixes an inappropriate league-average cold start without claiming useful club carry. It does not identify an absolute tier step or a transition time constant.

## P2 — Reliability-weighted club carry plus empirical mover shift (F1 + F2)

First shrink a noisy old club estimate toward the **old source-cohort** distribution. For positive-definite pooled `C_old,g` and measurement covariance `V_i^-`, the Gaussian working approximation is

\[
W_i=C_{old,g}(C_{old,g}+V_i^-)^{-1},
\quad \widetilde x_i^-=m_{old,g}+W_i(\widehat x_i^- -m_{old,g}),
\]
\[
\widetilde V_i^-=C_{old,g}-C_{old,g}(C_{old,g}+V_i^-)^{-1}C_{old,g}.
\]

Then use a destination-relative transition:

\[
x_{i,0}\mid x_i^-,g\sim N_2(B_gx_i^-+d_g,\Sigma_{jump,g}),
\quad B_g=\operatorname{diag}(\kappa_{q,g},\kappa_{s,g}),
\]
\[
x_{i,0}\mid H_{<S}\ \approx\ N_2\left(
 B_g\widetilde x_i^-+d_g,
 B_g\widetilde V_i^-B_g'+\Sigma_{jump,g}+U_{pred,i}
\right).
\]

`U_pred,i` accounts for uncertainty in carry, transition intercept and source-cohort hyperparameters; it is not necessarily constant across clubs. Joint parameter draws/club-season bootstrap can propagate it without assuming independent estimates. In the diagonal simplification, `W_kk=τ_old,k²/(τ_old,k²+v_i,k)`; this **measurement reliability** is not the same as the biological/seasonal carry `κ_k`.

- **Per-fold construction:** estimate old/new pairs from seasons `<S`; choose endpoint window, source-group pooling and carry restriction using forward inner seasons only. The nested simple case is `B=I` (old strength plus a group shift); compare with `B=0` (cohort-only). Start by exploring `0≤κ≤1`, but report if this restriction systematically misses old-cohort forward errors rather than silently asserting contraction.
- **No-market-history fallback:** with no `x̂^-`, use `m_old,g,C_old,g` as the old distribution, so the new prior mean/covariance are `B m_old,g+d_g` and `B C_old,g B'+Σ_jump,g+U_pred`. If the old source cohort is unsupported (e.g. outside-SPFL entrants), use P1 directly with extra uncertainty. Long gaps add old-state uncertainty before weighting; a zero quote count is not zero uncertainty.
- **Literature basis:** informed historical means and zero-centred residuals [OW6]; group-specific entrant distributions and evidence that lower-tier history need not strongly predict destination performance [RI11]; Bayesian market auxiliary observations [EG §3]. The exact reliability-weighted Gaussian transport above is our conjugate approximation, not a formula claimed from those football papers.
- **EDA must estimate:** old-cohort means/covariance, per-club estimation error and staleness, q/s carry, direction-specific `d_g`, intrinsic residual covariance, parameter uncertainty, and whether old-club history improves on P1 in forward first-quote scoring.
- **Critical interpretation:** because old/new q are each relative to their own division, **`d_g` already includes the net league rebase**, as well as selection/summer residual effects. Do not then add A2's step or P3's rebase to it. If production priors are expressed directly in a common-pyramid gauge, re-estimate/transform this intercept accordingly. P2 is the simplest empirical proposal when precise decomposition of tier and club movement is not identified.

## P3 — Explicit tier rebase plus partially pooled club jump (F1/F2 with F3-compatible accounting)

Let `e_q=(1,0)'` and let `θ_ℓ` be the connected historical quality level of division ℓ. Then

\[
x_{i,0}=B_g\widetilde x_i^- +
 (\theta_{old}-\theta_{new})e_q+\eta_g+\epsilon_i,
\qquad \epsilon_i\sim N_2(0,\Sigma_{jump,g}).
\]

This says: shrink the club's old **within-origin-division deviation**, translate from the old division's reference to the new one, then add a residual summer/selection shift `η_g`. The tier contrast has positive sign for relegation to a weaker league and negative sign for promotion to a stronger league. Carry here is shrinkage toward the **origin league's mean**, not an assertion that an arbitrary global absolute rating contracts toward zero.

The marginal covariance must include old-state uncertainty, residual heterogeneity and uncertainty in the estimated tier contrast/carry/jump. Schematically,

\[
C_{i,0}=B_g\widetilde V_i^-B_g'+\Sigma_{jump,g}
 +\operatorname{Var}_{H_{<S}}\{B_g\widetilde x_i^-+
 (\theta_{old}-\theta_{new})e_q+\eta_g\},
\]

where the last variance is over fitted hyperparameters/offsets **holding the plug-in old estimate fixed**, to avoid duplicating its observation variance. Use joint resampling when old-estimate and hyperparameter errors are correlated; the displayed additive form assumes those errors are separable. Do not merely add independent diagonal errors when the same bridge clubs identify all terms.

- **Per-fold construction:** infer tier contrasts only from older connected cups/movers, with competition/source sensitivity and an explicit anchor. Learn `η_g` from old cohorts **after subtracting** the tier rebase; partially pool it toward zero, especially if cup links are weak. Use non-movers (`θ_old=θ_new`) to learn baseline summer variation. If mover data alone cannot separate offsets and jumps, mark P3 unidentified and use P2 instead, not two freely fitted explanations for the same movement.
- **No-market-history fallback:** when origin tier is known, integrate over the old source-cohort distribution as in P2 rather than assume the mover was an average club of its origin. If its origin lies outside the connected pyramid, use P1's first-SPFL/returning cohort; do not extrapolate an unmeasured “fifth-tier step”. Inflate uncertainty for a missing/weak tier bridge.
- **Literature basis:** inter-division identification through cups and mobility [DC7]; season-boundary uncertainty distinct from ordinary updates [RI8; RI11]; historical informed states [OW6]. Decomposition into an explicit tier rebase and residual q/s jump is our proposal. A2's supplied carry/jump architecture is a project comparator, not independent published evidence.
- **EDA must estimate:** adjacent `θ` contrasts and their **joint** uncertainty, origin-cohort means, carry, residual `η_g` and covariance, connectivity/leave-one-link sensitivity, and forward performance versus P2. Test whether one constant adjacent step is adequate; do not fix 0.47 from the existing panel.
- **Style extension only if supported:** a tier effect is q-only here. A detected style boundary change enters `η_s`, with μ held on its declared scale. If an attack/concession split is later used, rotate it explicitly and remove the corresponding style component from `η_s` to avoid duplication. Under the supplied 48/52 convention a total 0.47 improvement corresponds to `Δq=0.235, Δs=−0.0094`, not a q jump of 0.47. [report §1 conditional derivation]

## Selection and Stage 2 handoff

1. Start with P1 versus P2; retain P3 only if cross-tier linking makes its decomposition useful and stable. Report a simpler no-shift/zero-mean baseline beside all three. This is a recommendation, not a finding.
2. Validate each target season from older seasons only; record the unavoidable scarcity of pre-panel market cohorts. Show prior-only first-quote D/U error and interval coverage, then existing-fit bias diagnostics separately. Do not choose priors by 26/27 ROI or by matching the leaked panel means.
3. Do not force boundary style uncertainty to the supplied within-season ratio ≈0.3. Estimate it with shrinkage, or explicitly retain a simple conservative covariance when not identified. Mean correction and F4 summer variance are different factors; avoid adding the same summer variance twice.
4. Handover parameters must include the information cutoff, exact prior node, common-scope mapping, fallback route, historical likelihood overlap policy, uncertainty components and cohort counts—not just two prior means. Missing chronology/gauge/overlap information blocks implementation of that proposal.

No numerical winner is selected in Stage 0. Literature supports these structures; the Scottish forward EDA must choose their parameters and whether the added complexity is justified.
