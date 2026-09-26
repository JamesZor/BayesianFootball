# Decoupled Generative xG-Primary Funnel — Scottish Lower Decompression

**TODO [025](../../../todos/025_prototype_decoupled_generative_xg_primary_model.md)** ·
namespace `scottish_lower_decoupled_xg` · **COMPLETE — 2026-09-23**

## Scientific Question

Can a decoupled generative chance-creation funnel ($\text{Team Ratings} \to \text{xG} \to \text{Goals}$) decompress favourite pricing (closing the 1.724 slope gap towards 1.00) while preserving totals and BTTS calibration, and does hierarchical team finishing ($\kappa_i$) provide any out-of-sample benefit over a shared league conversion factor ($\kappa$)?

## Decision & Headline Results

**Decision: REJECT both funnel arms. Do not promote. `m02_joint_gamma_poisson` remains the
two-arm standard.** The decoupling costs accuracy and *worsens* the slope it was built to fix.

40 folds · 710 held-out fixtures · 2,899 scored market observations · 4 × (800 warmup + 800 retained).

| Arm | Market on model slope ↓ | Favourite P(win) | Overall LogLoss ↓ | 1X2 LogLoss ↓ | O/U 2.5 LogLoss ↓ | Return | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `m01_poisson_time_decay` (Control 1) | 2.5356 | 48.65% | 0.646788 | 0.620404 | 0.689937 | +135.4% | 1.184 | −23.75% |
| `m02_joint_gamma_poisson` (Control 2) | **1.7214** | 51.96% | **0.643748** | **0.617459** | 0.687049 | +123.4% | **1.238** | **−22.20%** |
| `m03_funnel_shared_kappa` (Candidate 1) | 1.9920 | 47.72% | 0.645216 | 0.619485 | 0.688059 | **+151.7%** | 1.189 | −26.50% |
| `m04_funnel_hierarchical_kappa` (Candidate 2) | 2.0913 | 48.61% | 0.644839 | 0.619125 | **0.686748** | +138.7% | 1.182 | −26.07% |
| Betfair close (Reference) | 1.0000 | 76.25% | 0.641816 | 0.613118 | 0.689878 | — | — | — |

Run UUIDs: `m01` `de7fa956-87e8-418f-afb4-61ce01cb9f7d` · `m02` `97c7a3d9-a05a-4029-90cb-e34279b8c791` ·
`m03` `27d5a9f5-303a-4661-ad3d-464ba776d380` · `m04` `b9d1627c-d7a6-4bc5-ab49-84acad0aafdd`.

### The pre-registered headline: folds 21–40

The chance layer is trained on **proxy xG alone**, and BBC live-text coverage only reaches
100% from fold 21 onward (folds 1–20 average 59.1%, worst fold 50.0%). A cut arm starved of
proxy coverage is starved of *everything*, whereas the goals-trained controls are not, so
folds 21–40 were **pre-registered as the clean comparison before the grid was run**.

ΔLogLoss vs `m02`, 95% CI from 4,000 fixture-clustered bootstrap resamples (positive = worse):

| Contrast | Block | ΔLogLoss | 95% CI | P(Δ<0) |
|---|---|---:|---|---:|
| `m03` − `m02` | folds 21–40 | **+0.0028** | [+0.0001, +0.0055] | 0.021 |
| `m04` − `m02` | folds 21–40 | **+0.0027** | [+0.0002, +0.0050] | 0.017 |
| `m04` − `m03` | folds 21–40 | −0.0002 | [−0.0010, +0.0008] | 0.652 |
| `m03` − `m02` (1X2 only) | folds 21–40 | +0.0045 | [+0.0014, +0.0076] | 0.003 |
| `m04` − `m02` (1X2 only) | folds 21–40 | +0.0042 | [+0.0015, +0.0068] | 0.002 |

On the clean block both funnel arms are **worse than `m02` with the interval excluding zero**.
This is the honest reading, and it is *stronger* evidence against the funnel than the pooled
all-40 number: the cut loses precisely where its input data is best.

The thin block inverts (`m03` 0.6410 vs `m02` 0.6411, `m04` 0.6404) — which is why
pre-registration mattered. Had folds 1–20 been chosen post hoc, the same grid would have
"shown" the funnel winning. That block is a coverage artefact, not a result.

### Answering the two questions directly

1. **Decompression: no — it moves the wrong way.** `m02` sits at slope 1.7214; `m03` regresses
   to 1.9920 and `m04` to 2.0913, both back toward the 2.5356 single-arm control. Cutting
   the goals feedback *removes* the shrinkage that was doing the decompression. The funnel
   also makes favourites **less** likely to win (47.7% / 48.6% vs `m02`'s 52.0%, market 76.3%),
   i.e. it is more overconfident on longshots, not less.
2. **Hierarchical finishing: no measurable benefit.** `m04` − `m03` is −0.0002 [−0.0010, +0.0008],
   P(Δ<0) = 0.65 — indistinguishable from zero after 7h30m of extra compute. Posterior
   $\sigma_\kappa$ averages **0.058** across the 40 folds (range 0.038–0.081) with
   $P(\sigma_\kappa > 0.05) = 0.50$ — the spread of team finishing is a few percent of
   $\kappa \approx 1.10$ and the posterior sits astride the prior's own scale, i.e. **the data
   do not identify team-level finishing deviations** in this cohort. The shared-$\kappa$ arm is
   sampled *exactly*; the hierarchical arm buys nothing but cost.

`m03`'s +151.7% return is the one number favouring a funnel arm, but it comes with the worst
drawdown (−26.50%) and a *lower* Sharpe than `m02` (1.189 vs 1.238) — more leverage on a
worse-calibrated signal, not better edge. It does not overturn a LogLoss regression whose CI
excludes zero.

## Formulation

$$\log \mu_{\text{xg}, h} = \mu_{\text{xg}} + \gamma_{\text{home}} + \alpha_{\text{xg}, h} + \beta_{\text{xg}, a}$$
$$\text{pxg}_h \sim \text{Gamma}(\nu, \mu_{\text{xg}, h} / \nu)$$
$$y_h \sim \text{Poisson}(\lambda_{\text{goal}, h}), \quad \lambda_{\text{goal}, h} = \kappa_h \cdot \mu_{\text{xg}, h}$$

- `m03`: Shared $\kappa \sim \text{LogNormal}(0, 0.20)$
- `m04`: Hierarchical team $\kappa_i = \kappa \exp(\delta_i)$ with $\sum_i \delta_i = 0$

### The cut is structural, not a detached gradient

"Decoupled" is implemented as a **two-stage cut posterior** (modular Bayes), in
[`l15_cut.jl`](l15_cut.jl):

$$p_A(\theta, \nu \mid \text{pxg}) \qquad\text{then}\qquad p_B(\kappa \mid y, \theta)\ \text{at fixed }\theta$$

Stage A fits team ratings against the **Gamma arm only**; Stage B draws $\kappa$ conditional on
those ratings. Row $k$ of the spliced chain carries $\theta$ from Stage A draw $\text{picks}[k]$
**paired with** its own $\kappa$ — draws, not a plug-in point estimate.

Two things this deliberately is *not*:

- **Not a `detach`/`stop_gradient` on the goals term.** NUTS' acceptance ratio reads the full
  log-joint, so a zeroed gradient still lets goals move the ratings through the Metropolis
  correction. Only two separate MCMC runs actually sever the path.
- **Not `m02` with a different prior.** The previous revision of this suite defined `m03` as
  `standard_model(:m03_funnel_shared_kappa, shared_observation())` — *structurally identical to
  `m02`*. The measured "difference" between them was Monte Carlo noise. That defect is what this
  work fixed; the m02≡m03 "null" testset that asserted it has been deleted.

Stage 0 verifies the cut by perturbing the goal vector and measuring the derivative through
the chance layer: **exactly `0.0`**, not merely small.

`m03`'s conditional is available in closed form. Under a flat prior on $u=\log\kappa$,
$p(\kappa)\propto \kappa^{S-1}e^{-\kappa T}$, so $\kappa \mid \theta \sim \text{Gamma}(S, 1/T)$ —
sampled **exactly** by grid inverse-CDF, with no inner MCMC. (The Jacobian here is easy to get
wrong: the naive reading gives $\text{Gamma}(S+1, 1/T)$, off by $S/(S+1)$ ≈ 0.8% at $S≈120$.
A moment test dismisses that as noise; a **deterministic quantile** test has no Monte Carlo
floor and caught it. Measured agreement: 5.4e-7 relative.)

## Verification

- [x] **Stage 0** — 0 heap allocations on compiled ReverseDiff tapes (0.097 ms/gradient);
      goal→rating derivative **exactly 0.0** in both funnel arms; exact-$\kappa$ law verified
      to 5.4e-7 by deterministic quantile comparison; proxy coverage measured per fold.
- [x] **Stage 1** — smoke gate passed on folds 1, 20, 40 for all four arms, 0 divergences.
- [x] **Stage 2** — 40-fold walk-forward grid, 710 OOS fixtures, persisted to
      `scottish_lower_decoupled_xg`. **0 divergences / 128,000 transitions** (`m01`, `m02`) and
      **0 / 48,000** (`m03`, `m04`); R̂ ≤ 1.0255; min ESS 498; score-grid partition ≤ 1.9e-15;
      zero-sum error ≤ 8.6e-16.
- [x] **Stage 3** — supremacy regression, proper scores, fixture-clustered bootstrap and
      portfolio backtest on the common tradeable panel.
- [x] **Stage 4** — recorded here and in the TODO.

### Two gates that had to be fixed, not tuned

Both are cases where the *statistic* was wrong rather than the model:

1. **Stage B R̂ and divergences are gated as RATES, not as a max/sum over runs.** `m04` performs
   `n_conditional` independent inner NUTS runs per fold. A maximum (or a sum) over $N$
   independent runs is an **extreme-value statistic**: it grows with $N$ even when every run is
   healthy, so such a threshold gets *stricter the more thoroughly the fold is sampled*. The
   production grid measured **3 divergences across 48,000 inner runs (76.8M transitions, rate
   3.9e-8)**, concentrated as 2 runs in fold 23 and 1 in fold 39, with every other statistic
   clean (worst inner R̂ 1.029, no fold with any run over 1.05). Gated on the fraction of
   *runs* affected — an upper bound on contaminated retained draws — the worst fold is 0.00167
   against a 0.005 allowance. Pinned by tests in [`test_decoupled_xg.jl`](test_decoupled_xg.jl).
2. **`n_conditional` is the spliced chain's total draw count**, not a "number of extra runs",
   so it must exceed the fit-level ESS ≥ 200 gate with headroom (set to 1,200 in production).

Where a cause *could* be fixed it was: inner acceptance was raised 0.90 → 0.95 rather than
relaxing a divergence gate, because each inner run contributes a retained draw, so a divergence
there biases that draw rather than merely wasting it.

### Provenance note

`m04`'s Stage 2 process was interrupted **after** `save_fit` and **before** its database row, by
the pre-fix summed-divergence gate above. Its chains were re-accepted from disk by
[`r21_resume_m04.jl`](r21_resume_m04.jl) rather than re-sampled: the corrected gate is a
post-hoc acceptance criterion that is not on the sampling path, so re-running 7h30m would have
changed a hash, not a number. The manifest records `sampled_source` and `source` separately and
Stage 3 prints the split; every other arm was sampled and accepted under one fingerprint.

## Reproduce

```bash
julia --project -t 16 experiments/scottish_lower/12_decoupled_generative_xg/r00_preflight.jl
julia --project -t 16 experiments/scottish_lower/12_decoupled_generative_xg/r10_smoke.jl
julia --project -t 16 experiments/scottish_lower/12_decoupled_generative_xg/r20_production_grid.jl   # ~8h, m04 dominates
julia --project -t 16 experiments/scottish_lower/12_decoupled_generative_xg/r30_evaluation.jl
```
