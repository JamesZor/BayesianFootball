# Task 014 — JointGammaNegBinObservation

Does replacing the two-arm joint likelihood's conditional **Poisson** goals density with a
**negative binomial** improve the pricing of Totals and BTTS on Scottish League One/Two
(tournaments 56/57), walk-forward over 24/25 + 25/26?

> **Status: complete. The hypothesis is not supported — see §7.** The component works and
> is fully verified; the effect it produces is real, directionally correct, and too small to
> pay on this league. Nothing here is written that a CSV in `results/` does not contain.

## 1. The question, and why 1X2 is the wrong place to look for it

A negative binomial at a fixed mean is not a different rate, it is a different **shape**.
Against a Poisson of the same mean it puts more mass at 0, less at 1–2, and more at 4+.

1X2 is a sum of scoreline diagonals, and a redistribution that adds mass to 0–0 and to 4–1
alike largely cancels in that sum. Totals and BTTS are not diagonal sums — they are tail
partitions of the same 12×12 tensor, and they read exactly the mass that moved. That is
the entire hypothesis:

| | test |
|---|---|
| **H1** | Totals (O/U 1.5, 2.5, 3.5, 4.5) — NegBin improves LogLoss or ECE |
| **H2** | BTTS — NegBin improves LogLoss or ECE |
| **H0** | 1X2 — no material change; the **control** that says the mean did not move |

**The honest prior is a null.** Experiment 02 fitted a single-arm NegBin on this league and
measured `r̂ ≈ 26.0–26.5` — about 5% excess variance over Poisson — worth Δ LogLoss = +0.0001
on 1X2. `MultiScaleGRW` absorbs rate variation into latent state, so the residual
*conditional* overdispersion this component prices may simply be small. The study is built
so that outcome is reportable rather than embarrassing: every contrast is a matched pair
differing in exactly one component, and the interval is what gets read, not the sign.

## 2. The ladder

Four models. Every component except the goals density is the Task 013 recipe verbatim, so
each rung differs from its control in exactly one slot.

| NegBin rung | Dynamics | Covariates / pillars | Goals density | Task 013 control |
|---|---|---|---|---|
| `m00_baseline_grw_negbin` | `MultiScaleGRW()` | — | NegBin | `m00_baseline_grw` (`158d2a80…`) |
| `m05_wealth_grw_negbin` | `MultiScaleGRW()` | production wealth (Supremacy) | Joint Gamma-NegBin | `m05_wealth_grw` (`b0961bc4…`) |
| `m10_lineup_grw_negbin` | `MultiScaleGRW()` | shots-RAPM lineup, bench 0.10 | NegBin | `m10_lineup_grw` (`b13c8fb9…`) |
| `m12_joint_hybrid_synergy_negbin` | `MultiScaleGRW()` | lineup + wealth | Joint Gamma-NegBin | `m12_joint_hybrid_synergy_grw` (`3a9a4c7e…`) |

`m05` and `m12` isolate the negative binomial **under** the two-arm joint likelihood;
`m00` and `m10` isolate it under a single-arm one.

### 2.1 The component

```
arm 1 (proxy xG)   pxg_s ~ Gamma(ν, μ_s / ν)                   evaluated where the mask is 1
arm 2 (goals)      y_s   ~ RobustNegativeBinomial(r, κ · μ_s)   evaluated everywhere
```

with `log r ~ Normal(3.1, 0.4)` under `GlobalDispersion` — the prior Experiment 02 sampled
under, so `r̂` here is comparable with that study's.

Everything the Gamma arm touches is **shared by construction** with
`JointGammaPoissonObservation`: one `_joint_gamma_poisson_params` submodel (`obs.ν`,
`obs.log_κ`), one `observation_design`, one set of validation rules. A difference between
`m05_negbin` and `m05` is the goal density and nothing else.

**SharedKappa only.** There is no `HierarchicalKappa` counterpart. A per-team κ and a
dispersion `r` both widen the goals arm, and identifying the two against each other on ~40
matches per club is a study of its own, not a free type parameter.

### 2.2 Score-grid integration is one trait, not a new kernel

The work package asks for the 12×12 bivariate grid to use the `RobustNegativeBinomial` PMF.
That kernel already exists and is already zero-allocation; what was missing was the routing.
It is one line:

```julia
observation_family(::JointGammaNegBinObservation) = :negbin
```

which sends the assembled model to `NegBinCountModel <: AbstractNegBinModel` → `latent_family`
→ `NegBinCountFamily()` → `CountLatents{Float64,<:NamedTuple}` carrying `r_h`/`r_a` →
`compute_score_grid!`'s double-negative-binomial kernel. 1X2, every totals line and BTTS are
then three partitions of **one** tensor, so derivative coherence is structural.

Because that routing is invisible to every other gate — a mis-routed model still samples,
converges, extracts finite latents and round-trips through Postgres — the smoke gate checks
it directly (G6 below).

### 2.3 Deviations from the work-package sketch

| Sketch | Used | Why |
|---|---|---|
| define the struct in `current_development/grw_joint_negbin/l01_loader.jl` | defined in `src/models/pregame/builder/components.jl` | `NegBinCountModel`'s `O` type parameter has to be widened to admit the new observation, and that is a `src` edit; a prototype-local struct could not be assembled by `build`. Task 014's own acceptance criteria say "…and builder components", and list `src/.../builder/components.jl` under Related Files. The loader holds the ladder and the gates, as Task 013's does. |
| `NegativeBinomialObservation(GlobalDispersion())` for `m00`/`m10` | same, written `NegativeBinomialObservation(dispersion = GlobalDispersion(log_r = Normal(3.1, 0.4)))` | `GlobalDispersion` takes its prior as a keyword; the value is its own default |
| smoke at `4 × (500 + 1000)` | same | Task 013 established that the smaller sketched budgets fail R̂/ESS as a *budget artefact*; this gate runs at the production sampler so a pass is a statement about the production configuration |
| "zero heap allocations inside the gradient tape" | allocation **measured and reported**, not gated | Task 007 established the installed ReverseDiff stack allocates ~35–330 KB per compiled-tape gradient call for *every* model in this repository, the TimeDecay controls included. A literal zero-allocation gate would fail all of them. What *is* gated is the property that matters: the compiled tape is exact under perturbation (`0.0e+00`), i.e. it recorded no data-dependent branch. |
| `GroupedCVConfig` / 40 folds / 710 fixtures | same | this is the canonical grid every control was scored on |
| Totals scored at 1.5 / 2.5 / 3.5 | **1.5 / 2.5 / 3.5 / 4.5** | the work package's score-grid section lists 4.5, and the Betfair archive quotes it |
| totals and BTTS expected to be where the gain is | **BTTS is where the mechanism is largest; O/U 3.5 is where it is smallest** | measured, not assumed — see the G6 table in §3. The work package's framing put O/U 3.5 forward as a headline; at this league's `r̂` it is the line where the two mass shifts cancel |

### 2.4 One thing the work package did not anticipate

`Evaluation.DEFAULT_SCORED_MARKETS` is **1X2, O/U 2.5 and BTTS**. A `LogLoss()` with no
selection filter prices those three and nothing else, which is what Task 013 and Experiment
06 scored. Building the evaluation context on that default would have tested a hypothesis
about the whole totals ladder on **one line**, and reported a verdict on "totals" without
ever pricing O/U 1.5, 3.5 or 4.5.

Which line carries the effect is not knowable in advance — and the G6 table in §3 shows the
answer is genuinely uneven: at `r̂ ≈ 30` the grid difference is −0.0055 on O/U 2.5 and
±0.0004 on O/U 3.5, because the extra mass at zero and the fatter right tail cancel at
different places on the ladder. A single-line scope would have been a coin flip.

The context is therefore built with an explicit market list (`GJN_MARKETS`). The Betfair
archive quotes O/U 0.5 through 5.5 on this league; 0.5 and 5.5 are deliberately left out
(both are heavily one-sided, and adding markets after seeing which ones moved is how a null
becomes a finding by accident). The set was fixed before any score was computed.

This means **the `all` scope here is not comparable with Task 013's**: it pools 4,054 rows
over six markets, not 2,899 over three. The reproduction gate is run on Task 013's own
three-market basis for exactly that reason.

## 3. Correctness gates (`r01_smoke.jl`, folds 1–2)

Run at the production sampler, `4 × (500 warmup + 1000 retained)`.

**Verdict: PASS 4/4.**

### G0 — likelihood parity against `equations.jl`

The gate that matters most. `engine.jl` hand-expands

```
log NegBin(y; r, λ) = log Γ(y+r) − log Γ(r) − log Γ(y+1) + r·(log r − log(r+λ)) + y·(log λ − log(r+λ))
```

at `λ = κ·μ`, so one expression now carries both the finishing factor and the dispersion.
Its two failure modes — κ leaking into the Gamma arm, and `r` written against `η` instead of
`ζ = η + log κ` — both still sample cleanly and both still look like a posterior. Neither
survives comparison with a reference that builds `NegativeBinomial` and `Gamma` objects and
calls `logpdf`.

| arm | points | worst abs | worst rel |
|---|---:|---:|---:|
| `joint_gamma_poisson_td` (existing, as a control on the harness) | 4 | 2.27e-13 | 1.42e-16 |
| **`joint_gamma_negbin_td` (new)** | 4 | 1.36e-12 | **8.73e-16** |
| `negbin_td` (existing) | 4 | 0.00e+00 | 0.00e+00 |

Machine precision at a prior draw and three displaced points. The reference covers
`TimeDecayDynamics` only, so the parity build uses it; `_observe` receives `η_h`/`η_a` as
opaque vectors and cannot see which dynamics produced them, so a parity pass on TimeDecay is
a parity pass on the likelihood. What it does not cover is the GRW predictor, which Tasks 007
and 013 already checked and which this task did not touch.

### G1 — gradient tape

| model | fold | θ | tape | grad (ms) | alloc (B) | RD vs FD | perturbed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `m00_baseline_grw_negbin` | 1 / 2 | 99 / 159 | 756 / 1330 | 0.74 / 0.78 | 176,816 / 182,592 | 4.0e-14 / 1.2e-14 | **0.0e+00** |
| `m05_wealth_grw_negbin` | 1 / 2 | 102 / 162 | 812 / 1386 | 0.79 / 0.83 | 270,128 / 278,976 | 1.2e-14 / 3.8e-14 | **0.0e+00** |
| `m10_lineup_grw_negbin` | 1 / 2 | 101 / 161 | 781 / 1355 | 0.76 / 0.80 | 223,472 / 230,784 | 2.5e-14 / 1.9e-14 | **0.0e+00** |
| `m12_joint_hybrid_synergy_negbin` | 1 / 2 | 104 / 164 | 837 / 1411 | 0.79 / 0.84 | 316,784 / 327,168 | 5.9e-14 / 1.2e-14 | **0.0e+00** |

Both GRW branches are taped (fold 1 has no target steps, fold 2 has one). ReverseDiff agrees
with ForwardDiff to ~1e-14 and the compiled tape is exact at three perturbed points, which is
the statement that it recorded no data-dependent branch.

### G2–G6 — sampling, latents, persistence, score grid

| model | max R̂ | ESS bulk / tail | divergences | `r̂` median | verdict |
|---|---:|---:|---:|---:|---|
| `m00_baseline_grw_negbin` | 1.0043 | 1149 / 1262 | 0 / 8000 | 28.7 | PASS |
| `m05_wealth_grw_negbin` | 1.0056 | 1681 / 1461 | 0 / 8000 | 29.3 | PASS |
| `m10_lineup_grw_negbin` | 1.0067 | 1108 / 1539 | 0 / 8000 | 28.7 | PASS |
| `m12_joint_hybrid_synergy_negbin` | 1.0048 | 935 / 1363 | 0 / 8000 | 28.7 | PASS |

### G6 in detail — where the dispersion actually moves the price

G6 builds this model's own 12×12 grid and, beside it, the **double-Poisson grid from the
same posterior λ draws**, and differences the market partitions. It exists because a
mis-routed model would pass every other gate in the file: it would sample, converge, extract
finite latents and round-trip through Postgres, and simply not be the model claimed.

Mean signed Δ over folds 1–2 (NegBin grid − double-Poisson grid at identical λ):

| model | mean `r` | Δ BTTS | Δ O/U 2.5 | Δ O/U 3.5 | \|Δ 1X2\| |
|---|---:|---:|---:|---:|---:|
| `m00_baseline_grw_negbin` | 30.3 | **−0.01308** | −0.00564 | −0.00010 | 0.00106 |
| `m05_wealth_grw_negbin` | 31.3 | **−0.01285** | −0.00522 | +0.00037 | 0.00079 |
| `m10_lineup_grw_negbin` | 30.2 | **−0.01317** | −0.00563 | −0.00001 | 0.00112 |
| `m12_joint_hybrid_synergy_negbin` | 30.9 | **−0.01304** | −0.00531 | +0.00035 | 0.00071 |

This is the mechanism behaving exactly as stated, and it is worth reading carefully before
any proper score:

* **BTTS moves most, by an order of magnitude, and always downward (−1.3pp).** A negative
  binomial's headline effect at a fixed mean is extra mass at **zero** on each side, and BTTS
  is the one market that reads that effect *unopposed*: P(both score) must fall.
* **O/U 2.5 moves about half as much (−0.55pp), also downward** — the extra zeros outweigh
  the fatter right tail at this line.
* **O/U 3.5 barely moves at all (±0.0004), and the sign is not even stable across models.**
  This is not noise: at 3.5 the downward pull of the extra zeros and the upward pull of the
  fatter tail very nearly cancel. It is the *worst* available line on which to look for this
  effect — which is why G6 gates on BTTS, and why the work package's emphasis on O/U 3.5
  should be read with this table in hand.
* **1X2 moves least (0.0007–0.0011), as H0 predicts.**

**And read the magnitudes.** The largest of these is 1.3 percentage points on BTTS; on the
totals lines the mechanism is worth half a point or less. That is the size of the effect the
40-fold grid is trying to resolve against a Betfair closing line over a few hundred quoted
fixtures per market. The smoke gate has already told us the ceiling.

`r̂ ≈ 28.7–31.3` sits just above Experiment 02's `26.0–26.5`, in the expected direction: the
GRW latent state absorbs some of the rate variation a TimeDecay model leaves in the residual,
so less conditional overdispersion is left for `r` to price.

## 4. Production grid (`r02_production_grid.jl`)

40 match-biweek folds, 710 held-out fixtures, namespace `scottish_lower_grw_joint_negbin`,
16 pinned threads. Audit on every retained draw; artefact keeps every `persist_stride`-th.

| model | folds | OOS | max R̂ | ESS bulk | ESS tail | divergences | depth | BFMI | wall | run UUID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `m00_baseline_grw_negbin` | 40 | 710 | 1.0111 | 834 | 670 | 0 / 160k | 0.61% | 0.732 | 59 min | `0c0da991-7d4c-4f01-90e2-2af157f27aaa` |
| `m05_wealth_grw_negbin` | 40 | 710 | 1.0102 | 1049 | 807 | 0 / 160k | 0.01% | 0.681 | 57 min | `019d41d4-9e0e-41eb-bbc1-8984263f0f14` |
| `m10_lineup_grw_negbin` | 40 | 710 | 1.0105 | 1057 | 882 | 0 / 160k | 0.55% | 0.744 | 63 min | `f7fd8385-fa15-4f6a-ae89-e23447907a80` |
| `m12_joint_hybrid_synergy_negbin` | 40 | 710 | 1.0041 | 1911 | 2518 | 1 / 400k | 0.05% | 0.623 | 112 min | `c3d2aede-ddd5-4590-9c75-5937de4b1bbc` |

Sampler for the first three: `QueuedNUTS 4 × (500 warmup + 1000 retained)`, δ = 0.80, depth
10, `persist_stride = 2`. `m12` needed `4 × (1000 + 2500)` with `persist_stride = 5` — see
below. All four carry 2,000 persisted draws per fold and all four pass the six-part audit.

### `m12` needed a second, larger budget

At `4 × (500 + 1000)` it failed the audit on **tail ESS alone** — 236 at fold 8 against the
400 gate — and `r02` therefore refused to persist it. Everything else was comfortable: R̂
1.0099 (inside even Task 007's strict 1.01), no divergences in 160,000 transitions, 0.02%
depth saturation, BFMI 0.593. One fold could not resolve the tail of some parameter in 4,000
draws: a **budget** shortfall, not a geometry pathology.

Re-run at `4 × (1000 warmup + 2500 retained)`, `persist_stride = 5` — 10,000 audited draws
per fold, artefact unchanged at the same 2,000 draws the other three carry:

| | max R̂ | ESS bulk | **ESS tail** | divergences | BFMI | verdict |
|---|---:|---:|---:|---:|---:|---|
| `4 × (500 + 1000)` | 1.0099 | 689 | **236** | 0 / 160k | 0.593 | FAIL, not persisted |
| **`4 × (1000 + 2500)`** | 1.0041 | 1911 | **2518** | 1 / 400k | 0.623 | **PASS** → `c3d2aede-ddd5-4590-9c75-5937de4b1bbc` |

Tail ESS rose 236 → 2518, far more than the 2.5× draw increase alone explains: doubling
warmup is what fixed the adaptation. **`r̂` is unchanged — 29.54 against 29.50** — which is
the point worth keeping: the posterior was never wrong, only its tail was under-resolved, and
the gate caught exactly that.

Two things made this recoverable rather than a silently degraded arm in the ladder. `r02`
refuses to persist a model that fails its gate. And the checkpoint directory is stamped with
the budget (`checkpoints_4x1000w2500s`) — without that, `fit_model` would have resumed the
failed run's per-fold checkpoints and handed back the same draws under a new config hash,
reproducing the failure while looking like a fresh result.

The report and CSV are likewise stamped per invocation. They did not used to be, and the
m12 re-run clobbered the three-model record on its way through; the merged four-model table
above is `results/r02_production_runs_ALL.csv`.

### Posterior dispersion `r` (reported, not gated)

| model | median `r` | mean `r` | 90% interval |
|---|---:|---:|---|
| `m00_baseline_grw_negbin` | 28.40 | 30.29 | [17.09, 49.92] |
| `m05_wealth_grw_negbin` | 29.53 | 31.50 | [17.77, 51.93] |
| `m10_lineup_grw_negbin` | 28.99 | 30.91 | [17.46, 50.82] |
| `m12_joint_hybrid_synergy_negbin` | 29.50 | 31.43 | [17.82, 51.61] |

`r̂` is the study's own subject, so it is reported and never gated: a posterior that piles up
at large `r` is a negative binomial saying it is a Poisson, which is a **result** — the latent
state already absorbed the overdispersion — not a failure.

All four sit at `r̂ ≈ 28–30`, a little above Experiment 02's `26.0–26.5` on a TimeDecay state,
in the expected direction. The per-fixture range is wide (`r` spans ~8 to ~178 across draws
and fixtures), so the league-level median understates how much the tail moves on individual
fixtures. At the grid level this is still a small effect: the full-grid G6 figures on the
persisted runs are |Δ BTTS| 0.0128–0.0130, |Δ O/U 3.5| 0.0009–0.0014, |Δ 1X2| 0.0008–0.0011 —
within noise of the folds 1–2 smoke figures, i.e. the mechanism's size is stable across the
whole walk-forward.

## 5. Proper scores (`r04_evaluate.jl`)

710 walk-forward fixtures, **4,054 scored rows** over six markets (1X2 1785, O/U 1.5 430,
O/U 2.5 758, O/U 3.5 528, O/U 4.5 197, BTTS 356) on 630 quoted fixtures, against the
de-vigged Betfair TWA(−20, 0] close.

**Reproduction gate passed exactly**: `m12_poisson` scores LogLoss **0.64437** / ECE
**0.0086** on Task 013's own 1X2 + O/U 2.5 + BTTS basis (2,899 rows) against its published
0.64437 / 0.0086.

### The headline: a null

**1 of 56 matched contrasts is significant at 95%**, and it is `m00_negbin − m00_poisson` on
**O/U 4.5 Brier** — where the negative binomial is **worse** (Δ +0.00057, CI [+0.00005,
+0.00102], n = 197). Over 56 contrasts at 95% roughly 2.8 false positives are expected by
chance, so one — pointing the wrong way, on the thinnest market — is what a pure null looks
like.

ΔLogLoss, NegBin minus its Poisson counterpart (negative favours NegBin), 10,000
fixture-clustered resamples:

| scope | `m00` | `m05` | `m10` | `m12` | sign |
|---|---:|---:|---:|---:|---|
| 1X2 | +0.00023 | −0.00005 | −0.00015 | +0.00017 | mixed |
| O/U 1.5 | +0.00051 | +0.00138 | +0.00084 | +0.00061 | **worse 4/4** |
| O/U 2.5 | −0.00046 | −0.00056 | −0.00008 | −0.00018 | **better 4/4** |
| O/U 3.5 | −0.00058 | −0.00055 | −0.00039 | −0.00016 | **better 4/4** |
| O/U 4.5 | +0.00182 | +0.00098 | +0.00143 | +0.00093 | **worse 4/4** |
| BTTS | +0.00059 | +0.00044 | +0.00052 | +0.00012 | worse 4/4 |

Every interval spans zero. But the **signs are not random**: the negative binomial helps on
the two middle totals lines and hurts on the two outer ones, 8/8 each way. These four
contrasts share a fixture set and overlapping model components, so they are not independent
tests and no p-value is claimed for the pattern — it is reported because it is mechanically
what §3's G6 table predicts, not because it clears a threshold.

### Calibration (ECE) is the one place something moves

| scope | Δ ECE `m00` | `m05` | `m10` | `m12` |
|---|---:|---:|---:|---:|
| 1X2 | −0.0060 | −0.0005 | −0.0004 | −0.0016 |
| BTTS | −0.0209 | −0.0174 | −0.0077 | **+0.0277** |
| O/U 4.5 | +0.0038 | +0.0045 | +0.0063 | +0.0040 |

BTTS ECE improves substantially for three of four rungs and collapses for `m12`. **Do not
lean on this.** ECE over 10 bins and 356 BTTS rows is a very noisy statistic — Task 013 said
the same of its own ECE ranking on 2,899 rows — and a measure that moves by −0.021 on three
arms and +0.028 on the fourth, with no LogLoss or Brier movement anywhere near significance,
is better read as noise than as calibration.

### A caveat on O/U 4.5

The models score **0.33** against a market LogLoss of **0.51** on that line. A model beating
the de-vigged close by 0.18 on a proper score is not plausible as skill; it says the O/U 4.5
closing quotes are thin and badly formed (197 rows, heavily one-sided). The one "significant"
contrast in the study lives on this market, which is a further reason to read it as an
artefact.

### Against the Betfair close

Every arm, NegBin and Poisson alike, sits −0.004 to −0.007 on pooled LogLoss against the
close with intervals spanning zero (e.g. `m05_negbin` −0.00734 [−0.02741, +0.00545] vs
`m05_poisson` −0.00738 [−0.02761, +0.00545]). The two likelihoods are indistinguishable at
this resolution.

## 6. Portfolio and attribution (`r05_portfolio.jl`)

`MatchDay.option_b_system()` on the de-vigged close, 632 fixtures buildable by every arm.

| model | likelihood | return | ROI | Sharpe | max DD | bets | capture |
|---|---|---:|---:|---:|---:|---:|---:|
| `m00_baseline_grw_negbin` | NegBin | +461.4% | 12.05% | 1.353 | **−38.78%** | 1300 | 1.017 |
| `m00_poisson` | Poisson | **+491.6%** | **12.32%** | 1.412 | −38.85% | 1296 | 1.036 |
| `m05_wealth_grw_negbin` | Joint NegBin | +384.6% | 11.64% | 1.434 | **−42.49%** | 1235 | 1.041 |
| `m05_poisson` | Joint Poisson | +385.8% | 11.68% | **1.453** | −42.67% | 1247 | 1.080 |
| `m10_lineup_grw_negbin` | NegBin | **+370.3%** | **11.02%** | 1.243 | **−45.14%** | 1301 | 0.924 |
| `m10_poisson` | Poisson | +348.8% | 10.75% | 1.214 | −48.01% | 1299 | 0.957 |
| `m12_joint_hybrid_synergy_negbin` | Joint NegBin | +297.5% | 10.50% | 1.196 | **−50.23%** | 1244 | 1.007 |
| `m12_poisson` | Joint Poisson | +351.9% | 11.36% | 1.309 | −52.61% | 1253 | 1.043 |

No consistent direction on bankroll: the NegBin arm is worse for `m00` and `m12`, level for
`m05`, better for `m10`. At n ≈ 632 fixtures the bootstrap growth intervals of neighbouring
arms overlap heavily, so this ranking is descriptive.

**Max drawdown is smaller for the NegBin arm in 4 of 4 pairs** (−0.07, −0.18, −2.87, −2.38
points). Small, but the only portfolio statistic with a consistent sign, and it is the one a
fatter-tailed goal likelihood should move: more posterior mass on extreme scorelines means
slightly less confident staking into them.

### The mechanism is visible in the ledger, and it is what costs the money

Ledger overlap is **92–94%**, so the two likelihoods almost always take the same bets — as
expected when the mean is unchanged. The differences concentrate exactly where §3 said they
would:

| selection | `n` bets NegBin vs Poisson | Δ ROI (NegBin − Poisson) |
|---|---|---:|
| **O/U 1.5 over** | 33/50 · 46/64 · 27/47 · 45/60 — **fewer, 4/4** | −3.70 · −0.62 · −3.93 · −0.61 — **worse 4/4** |
| **O/U 2.5 under** | 231/224 · 210/197 · 239/228 · 210/197 — **more, 4/4** | +0.53 · −1.84 · −0.16 · −2.52 |
| 1X2 draw | 300/306 · 244/273 · 315/326 · 260/282 | +0.03 · +1.34 · +1.38 · +0.93 — better 4/4 |

This is the component doing precisely what it was built to do, and being punished for it. The
extra mass at zero goals **lowers P(Over 1.5)**, so the NegBin declines 30–40% of the Over 1.5
bets its Poisson twin takes — and those declined bets were profitable. It **raises P(Under
2.5)** and takes more of those, at slightly worse ROI. It raises P(draw), and the draw book is
the one place it consistently gains.

The finding is therefore sharper than "no effect": on this league the residual conditional
overdispersion is real (`r̂ ≈ 29`) and it moves prices in the theoretically correct direction,
but the direction it moves them is **away from where the edge was**.

## 7. Verdict

**The hypothesis is not supported.** Replacing the two-arm joint likelihood's Poisson goals
density with a negative binomial does not improve predictive proper scores on Totals or BTTS
for Scottish League One/Two, and does not improve closing-line portfolio growth.

1. **H1 (Totals) — not supported.** No significant LogLoss or Brier contrast on any totals
   line. A coherent but non-significant pattern: better on O/U 2.5 and 3.5 (8/8), worse on
   O/U 1.5 and 4.5 (8/8).
2. **H2 (BTTS) — not supported.** ΔLogLoss positive (worse) in 4/4, all intervals spanning
   zero. BTTS ECE improves on three rungs and worsens badly on the fourth; at 356 rows that
   is noise.
3. **H0 (1X2) — held.** Mixed signs, |Δ| ≤ 0.00023, which is the control behaving as stated.

**Why, and it is not "the component doesn't work".** G6 shows the dispersion reaches the
score grid and moves it exactly as theory predicts. The effect is simply small — 1.3pp on
BTTS, ~0.5pp on O/U 2.5, ~0 on O/U 3.5 — because `MultiScaleGRW` already absorbs most of the
league's overdispersion into latent team state, leaving `r̂ ≈ 29` (only ~3–5% excess variance)
as the residual. Experiment 02 measured `r̂ ≈ 26` under a TimeDecay state and found the same
null on 1X2; this study extends that null to the markets that were supposed to be where the
negative binomial paid, and adds the reason it does not.

**Recommendation: do not promote `JointGammaNegBinObservation` to the production ladder.**
`m12_joint_hybrid_synergy` should keep its Poisson goals arm. The component is retained,
tested and documented for leagues with genuinely higher residual overdispersion, where the
same machinery applies unchanged.

**Step 5 (hierarchical dispersion) is not triggered.** Its precondition — "if
`GlobalDispersion` demonstrates significant alpha or improved totals calibration" — is not
met. `HomeAwayDispersion` is wired and one line away (§8) if a future league warrants it;
fitting a second dispersion parameter on evidence this flat would be a search, not a study.

## 8. Hierarchical dispersion — the conditional step

The work package's Step 5 makes a hierarchical dispersion extension conditional: run it only
"if `GlobalDispersion` demonstrates significant alpha or improved totals calibration".

The component is **already wired for it**. `observation_wired(::JointGammaNegBinObservation)`
admits `GlobalDispersion` and `HomeAwayDispersion` on the same rule the single-arm NegBin
uses, `_cb_dispersion_draws` reconstructs both, and `_cb_rates` carries asymmetric `r_h`/`r_a`
into the score grid, which already reads them per side. Adding the arm is a one-line change to
the ladder:

```julia
gjn_dispersion() = HomeAwayDispersion(log_r = Normal(3.1, 0.4), δ_r_home = Normal(0.0, 0.5))
```

It is **not** run unless §5 gives a reason to. Fitting a second dispersion parameter because
the first one was measured, rather than because the measurement asked for it, is how a null
becomes a search. `AdvancedVolatilityDispersion` remains refused at build time for both NegBin
observations — its per-match reconstruction is not AD-safe, which is a pre-existing gap
recorded in `observation_gap`.

## 9. Reproducing

```bash
# on mcmc-beast, from /root/BF_grw_joint_negbin
julia --project -t 16 current_development/grw_joint_negbin/r01_smoke.jl
julia --project -t 16 current_development/grw_joint_negbin/r02_production_grid.jl
julia --project -t 16 current_development/grw_joint_negbin/r04_evaluate.jl
julia --project -t 16 current_development/grw_joint_negbin/r05_portfolio.jl
```

Artefacts land in `results/`; every table above is generated from a CSV there.
