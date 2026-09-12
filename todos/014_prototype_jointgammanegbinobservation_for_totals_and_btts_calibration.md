# 014 — Prototype JointGammaNegBinObservation for Totals and BTTS Calibration

| Field | Value |
|---|---|
| ID | 014 |
| Title | Prototype JointGammaNegBinObservation for Totals and BTTS Calibration |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-11 |
| Updated | 2026-09-12 |
| Related Files / Commits / PRs | `src/models/pregame/builder/{components,builder,engine,equations}.jl`, `current_development/grw_joint_negbin/`, `feat/grw-joint-negbin-observation`, `3ee5de61`, `27e62d76`, `b9ace463`, `3107848b`, PR #31 |

## Context & Problem Statement

In exploratory data analysis of Scottish Lower leagues (tournaments 56/57), goal counts display marginal overdispersion ($\text{Var}(Y) > \text{Mean}(Y)$). While hierarchical models with dynamic team state spaces (`MultiScaleGRW`) absorb substantial variance through latent rate variation, the conditional Poisson assumption on goals ($y \sim \text{Poisson}(\kappa \cdot \mu)$) in `JointGammaPoissonObservation` imposes equidispersion conditional on $\mu$.

In Experiment 02 (`02_negbin_2426_grid`), single-arm Negative Binomial models yielded $\hat{r} \approx 26.0\text{--}26.5$, showing mild residual overdispersion. While 1X2 LogLoss was essentially neutral ($\Delta = +0.0001$), Negative Binomial has distinct theoretical implications for scoreline distributions:
1. Higher probability mass at 0 goals (zero-inflation effect).
2. Lower probability mass at 1 and 2 goals.
3. Fatter right tail on 4+ goals (blowouts).

The hypothesis is that replacing the Poisson goal likelihood in the two-arm joint architecture with a `RobustNegativeBinomial` likelihood (`JointGammaNegBinObservation`) will improve tail pricing, specifically sharpening predictive proper scores (LogLoss, Brier, ECE) on Totals (Over/Under 1.5, 2.5, 3.5) and Both Teams To Score (BTTS).

Constraints:
- ReverseDiff tape compilation safety: zero allocations inside the gradient tape.
- Must execute on the canonical 40-fold walk-forward grid (24/25 + 25/26, 710 matches).
- Evaluate 4-model ladder (`m00`, `m05`, `m10`, `m12`) directly against their Task 013 Poisson counterparts.
- If `GlobalDispersion` demonstrates significant proper score improvement on totals, prototype hierarchical tournament/home-away dispersion.

## Acceptance Criteria

- [x] **Component Implementation**: Implement and wire `JointGammaNegBinObservation` in `current_development/grw_joint_negbin/` (and builder components) with Arm 1 Gamma proxy xG and Arm 2 `RobustNegativeBinomial(r, \kappa \cdot \mu)` using `GlobalDispersion(log_r ~ Normal(3.1, 0.4))`.
- [x] **Score Grid Integration**: Ensure score-grid kernels evaluate bivariate distributions using `RobustNegativeBinomial` PMF on 12×12 grids. *(Routing, not a new kernel — `observation_family = :negbin` reaches the existing zero-alloc double-NegBin kernel. Verified directly by gate G6.)*
- [x] **Smoke Gate (Folds 1–2)**: 2-fold smoke test on `m00_baseline_grw_negbin`, `m05_wealth_grw_negbin`, `m10_lineup_grw_negbin`, `m12_joint_hybrid_synergy_negbin`. Verify gradient tape compilation with ReverseDiff, 0 divergences, R̂ < 1.05, bulk/tail ESS > 400.
- [x] **40-Fold Walk-Forward Grid**: Train all 4 models across Folds 1–40 (seasons 24/25 + 25/26, 710 matches) on `mcmc-beast` with `QueuedNUTSConfig`.
- [x] **Expanded Totals & BTTS Evaluation**: Compute proper scores across 1X2, Over/Under 1.5, Over/Under 2.5, Over/Under 3.5, and BTTS. Run 10,000-sample paired bootstrap significance vs Task 013 Poisson counterparts.
- [x] **Portfolio Backtesting**: Run standard Option B closing-line portfolio simulation and attribution.
- [x] **Hierarchical Extension (Conditional)**: If global dispersion shows statistically significant edge on totals, evaluate hierarchical dispersion. *(Precondition NOT met — 1/56 contrasts significant and it favours Poisson. Not run; `HomeAwayDispersion` is wired and one line away should a future league warrant it.)*

## Ideas & Candidate Solutions

- **Ablation Grid**:
  - `m00_baseline_grw_negbin`: Interception + GRW + HomeAdv + NegBin.
  - `m05_wealth_grw_negbin`: Interception + GRW + HomeAdv + ProductionWealth + JointGammaNegBin.
  - `m10_lineup_grw_negbin`: Interception + GRW + HomeAdv + LineupPillar + NegBin.
  - `m12_joint_hybrid_synergy_negbin`: Interception + GRW + HomeAdv + ProductionWealth + LineupPillar + JointGammaNegBin.
- **Direct Controls**:
  - Task 013 runs: `m00_baseline_grw`, `m05_wealth_grw`, `m10_lineup_grw`, `m12_joint_hybrid_synergy_grw` (all persisted in `mcmc_experiments`).

## Work Log & Progress

- [2026-09-11 @antigravity] Completed `/grill-me` alignment interview with trader. Defined `JointGammaNegBinObservation` specification, 4-model ladder, expanded totals/BTTS evaluation scope, and Task 014 creation.
- [2026-09-11 @antigravity] Branched `feat/grw-joint-negbin-observation` off `feat/grw-player-lineup-hybrid`. Prepared worktrees on archpc and mcmc-beast. Handing off to Claude CLI agent.
- [2026-09-12 @claude] Implemented `JointGammaNegBinObservation` in `src` (commit `3ee5de61`). The struct lives beside the other observations rather than in the prototype loader: `NegBinCountModel`'s `O` type parameter has to be widened to admit it, which is a `src` edit, and a prototype-local struct could not be assembled by `build`. The loader holds the ladder and the gates, as Task 013's does.
- [2026-09-12 @claude] Built the Task 014 prototype: `l01_loader.jl` (ladder, gradient audit, likelihood parity, score-grid gate, dispersion summary), `l02_evaluation.jl` (arms, panel intersection, wide market set, paired bootstrap, Task 012 attribution), and runners `r01`–`r05`.
- [2026-09-12 @claude] Smoke gate PASS 4/4 at the production sampler `4 × (500 + 1000)`, folds 1–2 (commit `27e62d76`). Full tables in `current_development/grw_joint_negbin/README.md` §3.
- [2026-09-12 @claude] 40-fold production grid complete, all four persisted. `m12` needed a second, larger budget after failing on tail ESS alone.
- [2026-09-12 @claude] `r04` proper scores and `r05` Option B portfolio complete. **Result: the hypothesis is not supported** — 1 of 56 matched contrasts significant, favouring Poisson on the thinnest market. Full analysis in `current_development/grw_joint_negbin/README.md` §5–§7.
- [2026-09-12 @claude] Step 5 (hierarchical dispersion) NOT run: its stated precondition was not met.

### Deviations from the work package, and why

| Sketch | Used | Why |
|---|---|---|
| define the struct in the prototype loader | defined in `src/models/pregame/builder/components.jl` | the model struct's type parameter must admit it; acceptance criteria already say "…and builder components" |
| "zero heap allocations inside the gradient tape" | allocation measured and **reported**, not gated | Task 007 established this ReverseDiff stack allocates 35–330 KB per compiled-tape gradient call for *every* model in the repo. What is gated is the property that matters: the tape is exact under perturbation (`0.0e+00`), i.e. it recorded no data-dependent branch |
| smoke at `4 × (500 + 1000)` | same | Task 013 showed smaller budgets fail R̂/ESS as arithmetic artefacts |
| Totals at 1.5 / 2.5 / 3.5 | 1.5 / 2.5 / 3.5 / 4.5 | the work package's own score-grid section lists 4.5, and Betfair quotes it |
| — | the evaluation context is built with an **explicit** market list | `Evaluation.DEFAULT_SCORED_MARKETS` is 1X2 + O/U 2.5 + BTTS only. On the default this study would have reported a verdict on "totals" having priced exactly one totals line |

## Verification & Findings

### Likelihood correctness (G0) — the gate that matters most

`engine.jl` hand-expands the NegBin log-density at `λ = κ·μ`, so one expression carries both
the finishing factor and the dispersion. Its two failure modes — κ leaking into the Gamma arm,
and `r` written against `η` instead of `ζ = η + log κ` — both still sample cleanly and both
still look like a posterior. Checked against an independent reference in `equations.jl` that
builds `NegativeBinomial`/`Gamma` objects and calls `logpdf`:

| arm | points | worst abs | worst rel |
|---|---:|---:|---:|
| `joint_gamma_poisson_td` (existing, harness control) | 4 | 2.27e-13 | 1.42e-16 |
| **`joint_gamma_negbin_td` (new)** | 4 | 1.36e-12 | **8.73e-16** |
| `negbin_td` (existing) | 4 | 0.00e+00 | 0.00e+00 |

### Smoke gate (folds 1–2, production sampler) — PASS 4/4

0 divergences in 8,000 transitions per model; max R̂ 1.0067; min ESS 935; ReverseDiff ==
ForwardDiff to ~1e-14 with the compiled tape exact under perturbation; `save_fit` → `load_fit`
round-trips latents and `observation_params` byte for byte.

### Where the dispersion actually moves the price (G6)

Each model's own 12×12 grid differenced against the **double-Poisson grid at the same posterior
λ draws** — the check that `r` reaches the pricing tensor rather than being sampled and discarded:

| model | mean `r` | Δ BTTS | Δ O/U 2.5 | Δ O/U 3.5 | \|Δ 1X2\| |
|---|---:|---:|---:|---:|---:|
| `m00_baseline_grw_negbin` | 30.3 | **−0.01308** | −0.00564 | −0.00010 | 0.00106 |
| `m05_wealth_grw_negbin` | 31.3 | **−0.01285** | −0.00522 | +0.00037 | 0.00079 |
| `m10_lineup_grw_negbin` | 30.2 | **−0.01317** | −0.00563 | −0.00001 | 0.00112 |
| `m12_joint_hybrid_synergy_negbin` | 30.9 | **−0.01304** | −0.00531 | +0.00035 | 0.00071 |

**BTTS is where the mechanism lives, not O/U 3.5.** A negative binomial's headline effect at a
fixed mean is extra mass at zero on each side; BTTS reads that unopposed and must fall. On a
totals line the extra zeros pull down while the fatter right tail pulls up, and at 3.5 the two
very nearly cancel — the sign is not even stable across models. The work package put O/U 3.5
forward as a headline market; at this league's `r̂` it is the least sensitive one available.
G6 therefore gates on BTTS.

**And the ceiling is low.** The largest grid difference is 1.3 percentage points on BTTS and
half a point on O/U 2.5. That is the size of the effect the 40-fold grid must resolve against
the Betfair closing line over a few hundred quoted fixtures per market.

`r̂ ≈ 28.7–31.3` sits just above Experiment 02's `26.0–26.5` on a TimeDecay state — the expected
direction, since the GRW latent absorbs rate variation that would otherwise remain residual.

### Production grid — all four persisted

| model | max R̂ | ESS bulk / tail | divergences | wall | run UUID |
|---|---:|---:|---:|---:|---|
| `m00_baseline_grw_negbin` | 1.0111 | 834 / 670 | 0 / 160k | 59 min | `0c0da991-7d4c-4f01-90e2-2af157f27aaa` |
| `m05_wealth_grw_negbin` | 1.0102 | 1049 / 807 | 0 / 160k | 57 min | `019d41d4-9e0e-41eb-bbc1-8984263f0f14` |
| `m10_lineup_grw_negbin` | 1.0105 | 1057 / 882 | 0 / 160k | 63 min | `f7fd8385-fa15-4f6a-ae89-e23447907a80` |
| `m12_joint_hybrid_synergy_negbin` | 1.0041 | 1911 / 2518 | 1 / 400k | 112 min | `c3d2aede-ddd5-4590-9c75-5937de4b1bbc` |

`m12` failed the first attempt on **tail ESS alone** (236 at fold 8, gate 400) with R̂ 1.0099 and
zero divergences, and `r02` refused to persist it. Re-run at `4 × (1000 + 2500)`, `persist_stride
= 5`: tail ESS 236 → 2518 while `r̂` was unchanged (29.54 vs 29.50) — the posterior was never
wrong, only its tail under-resolved.

Portfolio runs: `1b02fbec…`, `0a3d5394…`, `e6843844…`, `31ee67e5…` (each ledger reloads identically).

### THE ANSWER: the hypothesis is not supported

Reproduction gate passed exactly — `m12_poisson` scores 0.64437 / ECE 0.0086 on Task 013's own
basis. Panel: 710 fixtures, 4,054 scored rows over six markets.

**1 of 56 matched contrasts is significant**, and it favours the *Poisson* arm on O/U 4.5 Brier
(n = 197, the thinnest market). At 95% over 56 contrasts, ~2.8 false positives are expected.

ΔLogLoss (NegBin − Poisson; negative favours NegBin) — every interval spans zero:

| scope | `m00` | `m05` | `m10` | `m12` | sign |
|---|---:|---:|---:|---:|---|
| 1X2 | +0.00023 | −0.00005 | −0.00015 | +0.00017 | mixed (H0 holds) |
| O/U 1.5 | +0.00051 | +0.00138 | +0.00084 | +0.00061 | worse 4/4 |
| O/U 2.5 | −0.00046 | −0.00056 | −0.00008 | −0.00018 | better 4/4 |
| O/U 3.5 | −0.00058 | −0.00055 | −0.00039 | −0.00016 | better 4/4 |
| O/U 4.5 | +0.00182 | +0.00098 | +0.00143 | +0.00093 | worse 4/4 |
| BTTS | +0.00059 | +0.00044 | +0.00052 | +0.00012 | worse 4/4 |

The middle totals lines gain and the outer ones lose, 8/8 each way. The four contrasts share a
fixture set and overlapping components, so no p-value is claimed for the pattern — it is
reported because it is mechanically what the G6 grid table predicts.

### Why — and it is not "the component doesn't work"

G6 proves the dispersion reaches the score grid and moves it exactly as theory says. The effect
is simply small: `MultiScaleGRW` already absorbs most of the league's overdispersion into latent
team state, leaving `r̂ ≈ 29` — roughly 3–5% excess variance — as the residual.

The Option B ledger makes the cost concrete. Overlap between each pair is 92–94%, and the
differences sit exactly where predicted: extra mass at zero **lowers P(Over 1.5)**, so the NegBin
declines 30–40% of the Over 1.5 bets its Poisson twin takes (33/50, 46/64, 27/47, 45/60) and
those declined bets were **profitable** (ΔROI −3.70, −0.62, −3.93, −0.61). It raises P(Under 2.5)
and takes more of those at slightly worse ROI, and raises P(draw), which is the one book where it
consistently gains. Max drawdown is smaller for the NegBin arm in 4/4 pairs — the only portfolio
statistic with a consistent sign, and the one a fatter-tailed likelihood should move.

So the mechanism is real and directionally correct, and it moves prices **away from where the
edge was**.

### Recommendation

Do not promote `JointGammaNegBinObservation` to the production ladder; `m12_joint_hybrid_synergy`
keeps its Poisson goals arm. The component is retained, tested and documented for leagues with
genuinely higher residual overdispersion, where the machinery applies unchanged.

### Incidental fixes this task forced

* `restrict_latents` refused NegBin containers outright ("Extend this method before using it on a
  negative-binomial posterior"). Task 014 is the first study to restrict a NegBin posterior;
  extended in `b9ace463`.
* `r02`'s report and CSV had fixed filenames, so the `m12` re-run overwrote the record of the
  three models it did not touch. Now stamped by budget and model selection.
* The Option B basket is **five** selections (1X2 home/draw/away, Under 2.5, **Over 1.5**), not
  the "1X2 + O/U 2.5" the work package describes.
