# Task 016 — 1-Parameter Market Smile Spine with MultiScaleGRW

Scottish Lower (tournaments 56 & 57). Six-rung ladder, 43-fold walk-forward, 710-fixture
evaluation panel. Everything below is measured; the run UUIDs are in the tables.

**Verdict in one paragraph.** The 1-parameter spine `log φ(K) = β·(K − 2)` is a well-identified,
cheaper-to-sample model that recovers β = 0.0524 on all 43 folds and fixes ticket T011's staking
defect. It does **not** deliver what the work package hoped: sampling is 5–16% faster rather than
the 2× implied by a ≤ 90-minute target, minimum bulk ESS stays at 482–531 against a ≥ 600 target,
and on the corrected staking route it gives up 1.2–1.4 pp of flat ROI and 0.18 of Sharpe to the
5-parameter smile it replaces. The larger finding is not about the spine at all: **once the
production L2 rate calibrator runs on the tradeable book, the fitted smile shape is redundant and
slightly harmful** (dropping φ beats keeping it for every smile arm), and the smile pillar
actively destroys the two fringe totals lines the baseline profits from.

---

## 1. What was built

| File | Contents |
|---|---|
| `l01_loader.jl` | `MarketSmileSpinePillar` (β scalar), `SpineAnchoredCountModel`, the engine, extraction and pricing routes, G0/G4 gates, and the **anti-diagonal grid reweighting** that resolves T011 |
| `l02_evaluation.jl` | Six-arm ladder across three experiment namespaces, the 0.5–4.5 strike ladder, the two staking routes, and the stake-side coherence gate |
| `r01_smoke.jl` | Folds 1–2: G4a (pre-sampling arithmetic), G0a/b/c, G1, G6, G2, G3, G4b, G4c, G5 |
| `r02_production_grid.jl` | 43 folds × 2 spine rungs; pinned-rung verification; H1/H2 benchmark |
| `r04_evaluate.jl` | Proper scores, per-strike ladder, paired fixture-clustered bootstrap with power controls |
| `r06_portfolio.jl` | Option B closing-line portfolio; both staking routes; T011 measured |
| `r07_t25_portfolio.jl` | T−25 raw vs L2-calibrated, two smile-calibration definitions |
| `r08_trust_sweep.jl` | Trust-pruning sweep, 12 policies × 3 arms × 2 environments |

Task 015's `l01`/`l02`/`l04` are included unchanged and reused throughout. `SpineAnchoredCountModel`
is a **second** wrapper type beside Task 015's rather than a widening of it, because widening the
smile slot's type union would change the type its pinned five-strike artefacts were serialised
with, and r04 must deserialise them.

### The model

```
C1 Supremacy:   η_h − η_a                          ~ Normal(log λ̂_h − log λ̂_a, σ_sup)  × w_sup
C2 SmileSpine:  log κ + log(μ_h + μ_a) + β·(K − 2) ~ Normal(log Λ̂_K, σ_smile), K = 0…4  × w_smile

σ_sup, σ_smile ~ truncated(Normal(0.15, 0.10), lower = 0.02)
β_spine        ~ Normal(0.04, 0.05)
```

φ(2) = 1.000 by construction: at K = 2 the shape term is `β · 0.0`, exactly zero in floating
point for any β. The site is `β_spine`, not `β`, because `MultiScaleGRW` already declares `dyn.β`.

---

## 2. Gates

Every gate that ran and what it returned. Thresholds are the work package's or ticket T011's;
none was changed to make a check pass.

### r01 smoke (folds 1–2, 4 × (500 + 1000)) — PASS 2/2

| gate | result |
|---|---|
| G4a reweighting, synthetic, pre-sampling | totals = smile CDF and Σ = 1 to float precision; φ ≡ 1 shortcut bit-identical; non-monotone curve refused |
| G0a null anchor vs m05 builder model | Δ = **0.0** exactly, four prior draws, both folds |
| G0b spine − base vs independent `logpdf` | worst relative 1.2e-15 (gate 1e-14) |
| G0c spine ≡ five-strike on the line | worst absolute 9.1e-13; exactly 0.0 on three of four rung-folds |
| G1 gradients | RD/FD ≤ 6.9e-16; compiled tape exact under perturbation |
| G4b fitted container | φ(2.5) ≡ 1; typed and legacy O/U routes = reference ≤ 1e-12 |
| G4c staking (T011) | φ ≡ 1 twin ledger bit-identical to grid twin 39/39; staked `p_grid` totals = smile CDF ≤ 3.9e-15; φ changed 38/39 and 37/39 stakes |
| G5 / G6 | PostgreSQL round-trip through the T010 detached-latent path; registry accepted both rungs |

**One gate failed on its first run and was changed, with the reason recorded.** G4a's φ ≡ 1
un-shortcut check used a fixed 1e-6 tolerance and measured 2.23e-4. The cause is the grid's own
truncation mass (goals ≥ 12 per side), which the Σ = 1 reweighting relocates onto totals ≥ 5 — not
a defect. At φ ≡ 1 no cell can move by more than that draw's truncation mass, so the check now
asserts exactly that derived bound (+1e-14 float slack) and measured a worst excess of −8.9e-15.
Every other tolerance is unchanged.

### r02 production (43 folds, 769 held-out) — PASS 2/2

Pinned rungs 1–4 were loaded and proved identical to the recipes `gss_models()` builds, converged,
43 folds, at the production budget, **before** any sampling. All six runs were fitted on
`mcmc-beast`, Julia 1.12.4, 16 threads, so the wall times are comparable.

Per-rung gates: R̂ ≤ 1.05 · ESS ≥ 400 · divergence rate < 0.1% · BFMI ≥ 0.30 · tree-depth < 5% ·
43 folds · OOS set == baseline's · latent audit · PostgreSQL round-trip · smile pricing ≤ 1e-12 ·
reweighting ≤ 1e-9. All passed for both rungs.

### r04 / r06 / r07 / r08 reproduction gates

| gate | result |
|---|---|
| r04 baseline vs Task 013 published | LogLoss **0.64315** = 0.64315, ECE **0.0123** = 0.0123, 2,899 rows — exact |
| r06 P1 baseline vs Task 014 Option B | **+385.78%** / ROI **11.68%** / **1,247** bets on 632 fixtures — exact |
| r07 T1 close/raw vs r06 reweighted | 6 arms, worst \|Δ return\| **0.00e+00** pp, 0 bet-count mismatches |
| r07 T2 baseline vs Task 014 T−25 | raw **+531.78%** / 1,124 bets; calibrated **+245.85%** / 969 bets — exact, on 611 fixtures |
| r07 T3, r08 S2 smile routing + staking | price ≤ 1.9e-15 and stake-side ≤ 3.6e-15 on every smile ledger, fringe strikes included |
| r08 S1 close/P0 vs r06 | reproduced for all three arms |
| r08 S0 zero-trust market inert | **FAILED 6/6** — see §6 and ticket T012 |

---

## 3. H1 — computational efficiency: **not supported as stated**

| rung | wall | max R̂ | min ESS bulk / tail | div | run |
|---|---:|---:|---:|---:|---|
| baseline (Task 013, pinned) | 42 min | 1.0115 | 609 / 616 † | 0 | `b0961bc4` |
| supremacy @0.40 (Task 015, pinned) | 61 min | 1.0105 | 814 / 516 | 0 | `0ee58d18` |
| five-strike @0.20 (Task 015, pinned) | 158 min | 1.0139 | 472 / 498 | 0 | `fcd5e974` |
| five-strike @0.40 (Task 015, pinned) | 185 min | 1.0200 | 431 / 696 | 0 | `30620d3e` |
| **spine @0.20** | **132 min** | 1.0135 | 482 / 854 | 0 | `eaf53852-a078-4190-b744-089966a306f6` |
| **spine @0.40** | **176 min** | 1.0139 | 531 / 922 | 0 | `582035c0-e145-44f7-9f40-89e25388e79a` |

† re-audited on thinned chains by `extend_fit`; not comparable with the others' ESS.

Targets were ≤ 90 min and min bulk ESS ≥ 600. **Neither is met at either weight.** Against the
five-strike rung at the same weight: wall −16% (@0.20) and −5% (@0.40); min bulk ESS +2% and +23%;
min tail ESS +71% and +32%.

**Why the spine does not buy the speed-up that was expected.** Three measurements, none of which
the work package anticipated:

1. **A gradient costs the same.** G1: the spine's tape is one instruction *longer* than the
   five-strike's (844 vs 843 on fold 1) and it allocates ~93 KB *more* per call (488 vs 395 KB),
   because `β_spine .* offsets` records an extra tracked matrix operation where the five-strike
   model adds its row vector directly. Dropping four parameters did not make the tape cheaper.
2. **The pillar weight costs more than the shape dimension.** Spine @0.20 → @0.40 adds 44 min;
   replacing five shape parameters with one saves 9–26 min.
3. **The drag is the pillar's coupling to the team state, not φ's dimension.** Both smile forms
   take 2–3× the supremacy-only wall time. The pillar reads the market's per-strike totals on
   every training match and pulls on `η`, and that is what makes the posterior harder to explore.

**Per-fold mixing is more consistent for the spine but not uniformly better**, and the run-level
summary at @0.20 is misleading:

| pair | median per-fold min bulk ESS, spine / five | median ratio | worst fold | folds spine lower | folds < 600 |
|---|---:|---:|---:|---:|---:|
| @0.20 | 857 / 1097 | 0.83 | 0.38 | 32 / 43 | 5 vs 4 |
| @0.40 | 1087 / 678 | 1.66 | 0.67 | 6 / 43 | 3 vs 13 |

At @0.20 the spine's run-level minimum is slightly *higher* (482 vs 472) while it is lower on 32 of
43 folds — the reference mixes unusually well at that weight (median 1097 vs 678 at @0.40), so the
run-level minimum is set by different folds in the two arms. Reporting only the run-level number
would have inverted this conclusion.

## 4. H2 — parameter recovery: **supported**

β_spine = **0.0524** pooled at both weights ([0.0499, 0.0550] @0.20; [0.0507, 0.0543] @0.40), fold
medians 0.0519–0.0530 over 43 folds, per-fold sd ≤ 0.0018 against a prior sd of 0.05. β itself
mixes well (per-fold ESS ≥ 3,258 bulk / 1,870 tail, R̂ ≤ 1.005), so the worst-mixing site is
elsewhere in the model. β is weight-independent and equals the least-squares slope through Task
015's five-strike medians (0.0525). It sits just above the hypothesised 0.03–0.05.

| | φ(0) | φ(1) | φ(2) | φ(3) | φ(4) |
|---|---:|---:|---:|---:|---:|
| spine (both weights) | 0.900 | 0.949 | **1.000** | 1.054 | 1.111 |
| five-strike @0.40 | 0.843 | 0.976 | 1.001 | 1.026 | 1.069 |

The line cannot bend: it is too high at K = 0 and too high again at K = 4. Both errors are
measurable downstream (§5, §7). Other pillar posteriors: σ_smile 0.062/0.060 (five-strike
0.052/0.050 — the looser fit one parameter implies), κ 1.094/1.092 (1.116/1.115), σ_sup
0.236/0.219 (0.237/0.220).

## 5. H3 — predictive parity: **underpowered on the pooled basis; resolved and opposite at the ends**

Pooled over 1X2 + O/U 2.5 + BTTS, spine − five-strike at the same weight: ΔLogLoss −0.00012
(@0.20) and −0.00014 (@0.40), 95% intervals ±0.001 containing 0. **That is not evidence of
parity.** On the same panel the control contrast spine − baseline (Δ −0.00211 / −0.00237) is *also*
unresolved, so the pooled test cannot resolve a difference that is known to be real. r04 prints
that control beside every parity claim precisely so this cannot be read as confirmation.

The strike ladder is where the restriction shows, and the two ends resolve in **opposite**
directions — the pooled zero is a cancellation, not an absence:

| contrast (spine − five-strike) | ΔLogLoss | interval | verdict |
|---|---:|---|---|
| O/U 0.5 | −0.0046 / −0.0048 | excludes 0 | spine better |
| O/U 2.5 | −0.0008 | includes 0 | unresolved |
| O/U 4.5 | +0.0057 / +0.0055 | excludes 0 | spine worse |

### The per-strike table, which corrects a prediction made before it ran

Mean `p_model` on the UNDER selection, 710-fixture panel:

| line | n | market | baseline | spine | five-strike | realised |
|---|---:|---:|---:|---:|---:|---:|
| Under 0.5 | 149 | 0.0679 | 0.0718 | 0.0893 | 0.0994 | **0.0336** |
| Under 1.5 | 215 | 0.2305 | 0.2423 | 0.2715 | 0.2478 | 0.2279 |
| Under 2.5 | 379 | 0.4702 | 0.4784 | 0.4917 | 0.4784 | 0.4987 |
| Under 3.5 | 264 | 0.6859 | 0.6841 | 0.6733 | 0.6777 | 0.6932 |
| Under 4.5 | 104 | 0.8578 | 0.8397 | 0.8078 | 0.8168 | **0.9038** |

The expectation going in was that the spine would lose at K = 0, since its line cannot reach
φ₀ = 0.843. **The direction is the opposite, and for a reason worth recording:** both smile arms
badly over-price deep Unders and the five-strike over-prices them *more*. At Under 0.5 the realised
rate is 3.4% while the spine says 8.9% and the five-strike 9.9%. So the spine's K = 0 win is **less
damage, not an improvement** — the baseline (LogLoss 0.15797) beats both smile arms (0.17215,
0.17912) and so does the market (0.15574). The same holds at Under 1.5, where the baseline is best.
The smile pillar earns its keep at K = 2–3 only.

This is consistent with `eda/README.md`'s Jensen tail inflation — for an uncalibrated Poisson
posterior `E[e^{−Λ}] ≥ e^{−E[Λ]}`, worst where the function is most convex — rather than with a
defect in the spine. The over-pricing is measured here; the mechanism is not proven.

**Caveat on the Under 4.5 market column.** Every model beats the market's LogLoss there by ~0.30
(0.344–0.352 vs 0.647) while the market's own mean price (0.8578) looks sane against a 0.9038
realised rate. A handful of thin or stale deep quotes dominates that column; treat it as
unreliable. Model-vs-model contrasts on the same rows are unaffected.

## 6. Ticket T011 — the staking fix, measured

T011: `Portfolio` prices a smile container's O/U through `λ_tot·φ(K)` and then solves every stake
off the un-smiled grid. `l01` §8 reweights each posterior draw's 12×12 grid on its anti-diagonals
`G = h + a` so the totals marginal is the smile's, then hands it to the unchanged `_finish_book`.

**The size of the defect.** Under the old route the reported price is exact (≤ 1.9e-15) while the
distribution the Kelly solve read differs from the smile curve by:

| arm | grid route | reweighted |
|---|---:|---:|
| five-strike @0.20 / @0.40 | 4.78e-02 / 4.71e-02 | 3.2e-15 / 3.6e-15 |
| spine @0.20 / @0.40 | 7.06e-02 / 7.03e-02 | 3.4e-15 / 3.0e-15 |

**4.7–7.1 percentage points** of `P(total ≤ K)`, against T011's ≤ 1e-9 acceptance criterion. The
defect is *larger* for the spine because its φ₄ = 1.111 is further from 1 than the five-strike's
1.069, and near `P(total ≤ 4) ≈ 0.85` that moves more mass — the same over-bending that costs it
LogLoss at Under 4.5.

**It is material, not cosmetic.** Correcting it changes 11–16% of each ledger:

| arm | only on grid route | only on reweighted | max shared stake gap | Δ ROI | Δ return |
|---|---:|---:|---:|---:|---:|
| five-strike @0.20 | 193 | 206 | 1.8e-02 | −0.02 pp | +18.0 pp |
| five-strike @0.40 | 202 | 215 | 1.5e-02 | −0.09 pp | +8.5 pp |
| spine @0.20 | 142 | 84 | 1.9e-02 | −0.46 pp | −73.0 pp |
| spine @0.40 | 134 | 97 | 1.9e-02 | −0.48 pp | −69.5 pp |

Task 015's conclusion that φ "changes the reported price and nothing else" is therefore false once
stakes are solved correctly. The `:grid` rows reproduce Task 015's published figures (+588.4% /
+545.0% at ROI 15.71 / 15.82 against its published "+545% to +588%" and 15.8%), which is what
establishes that the reweighted differences are the correction rather than a builder bug.

**A second defect found and NOT fixed inline: ticket [T012](../../docs/tickets/T012-zero-trust-market-reprices-the-portfolio.md).**
r08's S0 gate adds only O/U 4.5 to the book at trust 0 and measures the ledger: it differs 6 times
out of 6 (3 arms × 2 environments), by −22.3 to +10.2 pp of terminal return, with unstable sign.
The payoff matrix widens, `BakerMcHale` returns one `k` per fixture from that widened matrix, and
`k` rescales every stake. Trust 0 zeroes the new bet but not its influence on the geometry the
others were solved in. r08 neutralises it by measuring every addition against P0 on the *same*
extended book.

## 7. H4 — portfolio alpha: **half met, and the spine loses to five parameters**

Option B contract, de-vigged Betfair TWA(−20, 0] close, 632-fixture buildable panel, reweighted
(T011-correct) staking for every arm.

| arm | return | flat ROI | Sharpe | Calmar | max DD | bets (1X2 / totals) |
|---|---:|---:|---:|---:|---:|---|
| baseline | +385.8% | 11.68% | 1.453 | 9.04 | −42.7% | 986 / 261 |
| supremacy @0.40 | +404.6% | 12.75% | 1.551 | 9.49 | −42.6% | 983 / 261 |
| **spine @0.20** | +485.0% | 14.30% | 1.427 | 11.55 | −42.0% | 986 / 239 |
| **spine @0.40** | +469.4% | 14.54% | 1.462 | 11.27 | −41.6% | 995 / 237 |
| five-strike @0.20 | **+606.5%** | 15.69% | 1.611 | 14.92 | −40.7% | 1053 / 200 |
| five-strike @0.40 | +553.5% | **15.72%** | **1.640** | 14.16 | **−39.1%** | 1051 / 199 |

The work package asked the spine to preserve "+19% away-bet ROI and +15.8% flat ROI".

* **Away-bet ROI: met.** 1X2 away ROI is **19.44%** (@0.20) and **20.88%** (@0.40) against the
  baseline's 7.77%. The five-strike arms reach 23.05% / 23.57%.
* **Flat ROI: not met.** 14.30% / 14.54% against the 15.8% target, ~1.2–1.4 pp short, with Sharpe
  0.18 lower and a worse drawdown. The spine's bootstrap growth interval still includes zero
  (lower bound −0.0010) where the five-strike's excludes it (+0.0014).

**Where the shortfall comes from — the totals book.** The five-strike arm prunes totals harder and
concentrates stake in 1X2:

| | totals bets | totals ROI | totals stake share | 1X2 stake share |
|---|---:|---:|---:|---:|
| spine @0.40 | 237 | 7.19% | 22.0% | 78.0% |
| five-strike @0.40 | 199 | **12.67%** | 12.4% | **87.6%** |

The spine keeps 38 more totals bets at 5.5 pp worse ROI. That is the portfolio consequence of §5:
a curve that cannot bend mis-prices the totals ladder, so it fails to decline the bets it should.
The spine pillar still clearly beats the no-smile controls — the **linear restriction**, not the
pillar, is what costs money.

Persisted portfolios (this task's namespace): spine @0.20 `3bd3a461-c505-4ceb-822c-ce1623e0aaf7`,
spine @0.40 `785b5d7f-f596-491b-8442-9d71b5a350e6`, both with identical reloaded ledgers.

**Read all of this as descriptive.** These are closing-line simulations, the prices are the ones
the pillars were fitted against, and at ~630 fixtures neighbouring arms' growth intervals overlap.

## 8. H5 — T−25 calibration and market expansion

### The tradeable book: calibration dominates every pillar choice

| T−25 configuration | return | ROI | Sharpe | max DD |
|---|---:|---:|---:|---:|
| baseline raw | +531.8% | 14.15% | 1.658 | −41.9% |
| supremacy @0.40 raw | **+605.6%** | 16.17% | **1.711** | −38.3% |
| spine @0.40 raw | +457.7% | 16.28% | 1.283 | −43.8% |
| five-strike @0.40 raw | +381.2% | 15.14% | 1.212 | −42.1% |
| baseline calibrated | +245.9% | 17.39% | **1.976** | −22.0% |
| spine @0.40, φ dropped | +221.0% | 21.04% | 1.818 | −16.8% |
| five-strike @0.40, φ dropped | +223.7% | **22.10%** | 1.825 | **−16.7%** |
| spine @0.40, φ kept | +171.1% | 20.55% | 1.633 | −19.9% |
| five-strike @0.40, φ kept | +155.8% | 19.01% | 1.501 | −17.1% |

1. **The L2 calibrator is worth more than any pillar.** It roughly halves drawdown (−42% → −17
   to −22%) and lifts ROI from 14–16% to 17–22% for every arm.
2. **Dropping φ after calibration beats keeping it, for every smile arm** — five-strike @0.40 ROI
   22.10% vs 19.01%, return +223.7% vs +155.8%. The within-arm paired contrast is the nearest thing
   to a resolved result in r07 (p = 0.048 at @0.40). **So the smile pillar's value lies in how it
   shaped the rates during fitting, not in the φ curve used at pricing time**: once the calibrator
   has pooled those rates with the tradeable book, the fitted shape is redundant and slightly
   harmful.
3. **At T−25 raw, both smile forms have markedly worse Sharpe** (1.21–1.31) than the baseline
   (1.658) or supremacy-only (1.711). The smile buys ROI and pays in volatility.
4. **The spine-vs-five-strike sign flips across environments and nothing resolves**: spine loses at
   the close, wins at T−25 raw, loses again calibrated-with-φ-dropped. Every paired slate-growth
   interval spans zero (p 0.18–0.79). At ~600 fixtures these two arms are not separable.

### Market expansion: Under 1.5 and Under 4.5 are accretive for the BASELINE and destroyed by the smile

r04 put a prediction on record in r08's header before the sweep ran: the spine sees a +4.1 pp edge
at Under 1.5 where the realised rate says there is none, so enabling it should lose, and lose more
than the five-strike (+1.7 pp) or baseline (+1.2 pp). Measured:

| close, `+U1.5` | added bets | added ROI | net Δreturn |
|---|---:|---:|---:|
| baseline | 105 | **+37.66%** | +119.3 pp |
| five-strike @0.40 | 128 | −2.73% | −28.6 pp |
| spine @0.40 | **208** | **−6.31%** | **−91.6 pp** |

T−25 repeats it: baseline 76 bets at +3.90%, five-strike 99 at −6.80%, spine 152 at −14.63%. The
bet-count ordering (208 > 128 > 105) is exactly the ordering of the phantom edge, and the ROI
ordering is its reverse. The spine chases a mispricing of its own making, hardest of the three.

`+U4.5`: the smile arms decline the line almost entirely — 0 bets at the close, 1–2 at T−25 —
because they price it *below* the market, while the baseline takes 28 bets at **+18.79% ROI**. The
smile forgoes a line the no-smile model profits from.

**So H5's answer is no, twice over**, and the mechanism is the same distortion in both directions:
the pillar manufactures phantom deep-Under edges and suppresses real high-line ones.

**One correction to the prior.** `eda/README.md` attributes fringe losses partly to capacity
cannibalisation. Under Option B's cap that is not what happens here: core ΔROI is −0.16 to +0.35 pp
across these rows. The damage is the added bets' own ROI. Across the whole sweep only **21 of 66**
addition rows improve terminal return, so the fringe pruning is broadly vindicated; the best
addition differs by arm and environment (`+U1.5` for the baseline at the close, `+BTTS_no` for all
three at T−25), which is not a stable recommendation.

---

## 9. What this means for the stream

1. **Do not graduate the spine as a replacement for the five-strike smile.** It is cheaper to
   sample by 5–16%, not 2×; it does not reach the ESS target; and it costs 1.2–1.4 pp of flat ROI
   at the close. Its one clear advantage is per-fold mixing consistency across weights.
2. **The anti-diagonal reweighting should graduate on its own merits**, independently of the spine.
   It resolves a 4.7–7.1 pp incoherence for *any* smile container, including Task 015's, and it is
   what makes a deep-totals trust study possible at all. T011 option 2 is implemented and measured
   here; promoting it to `src/Portfolio/` is a separate, scoped piece of work.
3. **The smile pillar's φ is not worth carrying to pricing time once L2 calibration is in play.**
   The strongest signal in the task is that dropping φ after calibration beats keeping it. A
   cheaper model than either arm may be "fit with the smile pillar, price without φ" — untested
   here, and the obvious next experiment.
4. **The deep-totals over-pricing is a property of the pillar, not of the parameterisation.** Both
   forms are worse than the baseline at Under 0.5 and Under 1.5. If deep totals are the goal, the
   Jensen correction that `eda/README.md` points at is the thing to build, not a different φ.

## 10. Reproducing

```bash
# mcmc-beast, /root/BF_grw_smile_spine, after `git pull`
julia --project -t 16 current_development/grw_smile_spine/r01_smoke.jl
julia --project -t 16 current_development/grw_smile_spine/r02_production_grid.jl
julia --project -t 16 current_development/grw_smile_spine/r04_evaluate.jl
julia --project -t 16 current_development/grw_smile_spine/r06_portfolio.jl
julia --project -t 16 current_development/grw_smile_spine/r07_t25_portfolio.jl   # needs r06
julia --project -t 16 current_development/grw_smile_spine/r08_trust_sweep.jl     # needs r06
```

`r02` will not resample a persisted recipe; it loads it. Deserialising any artefact from this task
requires `include("l01_loader.jl")` first (the prototype model types live outside `src/`). The
environment was Julia 1.12.4 with the Manifest pinning Distributions 0.25.126, Turing 0.41.4,
DynamicPPL 0.38.10, ReverseDiff 1.17.0, MCMCChains 7.7.0; the DataStore snapshot ends 2026-09-05
and contains no rows from the 2026-09-12 card.

Reports and CSVs are under `results/`: `smoke/4x500w1000s/`, `r02_*`, `evaluation/`, `portfolio/`,
`t25_portfolio/`, `trust_sweep/`.
