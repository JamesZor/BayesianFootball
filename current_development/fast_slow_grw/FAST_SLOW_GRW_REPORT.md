# Fast & Slow GRW — posterior draw mixtures on Scottish Lower (TODO 021)

**Verdict: the fast-slow draw mixture does not decompress the team latent.** The best
loose arm raises the supremacy slope against the Betfair close from 0.399 to 0.488,
where the market sits at 1.0. On market favourites priced ≥ 0.70, the model moves
from 0.550 to at most 0.575 against the market's 0.763. Loosening the walk's priors
widens the spread only as far as the goal data allow; forcing the level wider (m04)
adds spread that does not line up with the market. No mixture improves 1X2 log loss.

All numbers below are measured on the canonical 40-fold walk-forward grid (pooled
tournaments 56/57, target seasons 24/25 + 25/26, 710 held-out fixtures). Raw outputs
are in `results/` (smoke: `results/smoke/4x400w400s/`, grid:
`results/production/`, evaluation: `results/evaluation/`).

## 1. Design as run

Four pure Poisson `MultiScaleGRW` arms (`GlobalInterception`, `GlobalHomeAdvantage`,
`PoissonObservation`; no lineups, wealth or smiles). Only the walk's priors differ:

| Arm | Change from `MultiScaleGRW()` defaults |
|---|---|
| `m01_poisson_grw_tight` | none: the handrail (= Task 013 `m00_baseline_grw`) |
| `m02_poisson_grw_loose_var` | every σ prior scale × 2.5 |
| `m03_poisson_grw_loose_tdist` | z₀, zₛ ~ TDist(4) |
| `m04_poisson_grw_loose_fixed_spread` | σ₀ (att, def) ~ truncated N(0.48, 0.01): the market-implied 2.43 × tight σ₀ |

**Combination.** The work package's element-wise geometric rate pooling was replaced,
at the user's direction, by **posterior draw concatenation**:
`λ_combined = hcat(λ_tight[:, idx₁], λ_loose[:, idx₂])`, with `N₂ = round(ρ · 1600)`
loose draws and the rest tight, at ρ ∈ {0, 0.25, 0.5, 0.75, 1}. The indices are evenly
spaced over each container because draws are stored chain-major. Each draw is still one
coherent score grid, so 1X2, O/U and BTTS stay partitions of one grid per draw. At the
fixture level, a ρ-mixture prices (1 − ρ) P_tight + ρ P_loose, which makes it the
linear probability pool.

**Staking.** `BookSpec(1X2, O/U 2.5, BakerMcHale)`, `PolicySpec(FlatTrust(0.25),
SlateDrawdown(20.0), FixedCap(0.25))`, priced off the de-vigged Betfair (−20, 0] TWA
close. Every mixture stakes the same 622-fixture panel (710 → 635 quoted → 622
buildable by every arm). The supremacy slope uses the 596 fixtures whose close inverts
to (λ_mkt_h, λ_mkt_a) under `Calibration.invert_market_rates`.

## 2. Convergence (Stage 1 smoke, Stage 2 grid)

Smoke (folds 1/20/40, 4 × (400 + 400)): all four arms pass the mechanical gates, which
are exact ReverseDiff tape, 0 divergences, R̂ ≤ 1.025, bulk ESS ≥ 319, exact save/load,
and mixture books equal to the grid's retained Poisson mass to 1.8e-15. m04 failed the
directional supremacy gate (G6); the grid was launched anyway to measure it on 40 folds.

Grid (40 folds / 710 fixtures, 4 × (800 + 800), audit on all 3,200 draws, persisted at
stride 2 to `fast_slow_grw_scottish_lower`):

| Arm | max R̂ | ESS bulk / tail | Divergences | BFMI | σ₀ att / def | Wall | Run |
|---|---:|---:|---:|---:|---:|---:|---|
| m01 tight | 1.0115 | 732 / 692 | 0 / 128k | 0.66 | 0.192 / 0.200 | 38 min | `2b42d3bf-28d7-47ac-8706-88798c9031ac` |
| m02 loose-var | 1.0124 | 682 / 461 | 0 / 128k | 0.68 | 0.209 / 0.214 | 36 min | `27c8f2f2-e130-44ba-a764-2585d39eb2de` |
| m03 loose-t | 1.0140 | 594 / 673 | 0 / 128k | 0.56 | 0.146 / 0.151 | 30 min | `dafbfe00-54c2-42a6-a87f-2c9de470bca2` |
| m04 fixed-spread | 1.0141 | 410 / 791 | 0 / 128k | 0.85 | 0.477 / 0.477 | 32 min | `514c5533-fea4-4786-8a58-30d8262733cf` |

No arm meets the advisory strict R̂ ≤ 1.01; all pass the 1.05 gate.

## 3. Headline metrics

| Mixture | ρ | Sup. slope | R² | Capital ≥ 4.0 | Capital ≤ 1.8 | Max DD | Sharpe | Flat ROI | Total return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m01 tight | 0 | 0.399 | 0.440 | 38.3% | 3.9% | −20.3% | 1.65 | +5.33% | +193% |
| m02 @ 0.25 | 0.25 | 0.421 | 0.455 | 37.6% | 4.2% | −20.4% | 1.62 | +5.99% | +184% |
| m02 @ 0.50 | 0.50 | 0.444 | 0.464 | 36.6% | 4.4% | −19.4% | 1.58 | +6.07% | +172% |
| m02 @ 0.75 | 0.75 | 0.466 | 0.473 | 35.9% | 4.7% | −19.1% | 1.52 | +6.06% | +163% |
| m02 pure | 1 | **0.488** | 0.479 | 34.7% | 5.2% | −19.1% | 1.45 | +6.20% | +153% |
| m03 @ 0.25 | 0.25 | 0.399 | 0.441 | 38.4% | 3.8% | −20.7% | 1.62 | +5.18% | +186% |
| m03 @ 0.50 | 0.50 | 0.400 | 0.441 | 38.4% | 3.7% | −20.8% | 1.67 | +4.87% | +194% |
| m03 @ 0.75 | 0.75 | 0.399 | 0.441 | 38.6% | 3.8% | −21.2% | 1.59 | +4.08% | +180% |
| m03 pure | 1 | 0.399 | 0.439 | 38.6% | 3.7% | −22.0% | 1.54 | +4.55% | +170% |
| m04 @ 0.25 | 0.25 | 0.409 | 0.420 | 37.1% | 4.2% | −20.6% | 1.76 | +5.23% | +233% |
| m04 @ 0.50 | 0.50 | 0.420 | 0.394 | 35.3% | 4.5% | −21.5% | 1.76 | +5.22% | +253% |
| m04 @ 0.75 | 0.75 | 0.430 | 0.366 | 33.7% | 5.1% | −24.0% | **1.82** | +4.61% | +306% |
| m04 pure | 1 | 0.441 | 0.339 | **31.9%** | **5.5%** | −27.7% | 1.75 | +4.80% | +324% |

Capital shares are stake-weighted fractions of the ledger. Flat ROI is one unit on
every bet the policy placed. Every row places 1,314–1,342 bets.

## 4. Favourite tail (1X2, the market favourite's side)

| Market band | n | Market | m01 tight | m02 pure | m03 pure | m04 pure |
|---|---:|---:|---:|---:|---:|---:|
| [0.40, 0.50) | 259 | 0.446 | 0.414 | 0.420 | 0.414 | 0.416 |
| [0.50, 0.60) | 149 | 0.539 | 0.452 | 0.465 | 0.451 | 0.456 |
| [0.60, 0.70) | 46 | 0.636 | 0.484 | 0.502 | 0.485 | 0.490 |
| [0.70, 1.00) | 18 | 0.763 | 0.550 | **0.575** | 0.550 | 0.560 |

Mixtures sit between their arms (full table in `r03_evaluation_report.md`). Even the
best arm closes about a tenth of the gap on the ≥ 0.70 band: 0.550 → 0.575, with the
market at 0.763. Only 18 fixtures fall in that band.

## 5. Derivative markets vs the close

Log loss on settled, quoted fixtures (close log loss in brackets):

| Arm | 1X2 (n = 595) [1.05344] | O/U 2.5 (n = 379) [0.68988] | BTTS (n = 178) [0.68337] | Mean P(over 2.5), close 0.530 |
|---|---:|---:|---:|---:|
| m01 tight | **1.06033** | 0.68756 | 0.69079 | 0.513 |
| m02 pure | 1.06279 | 0.68836 | **0.69003** | 0.514 |
| m03 pure | 1.06129 | 0.68787 | 0.69079 | 0.512 |
| m04 pure | 1.06389 | 0.68889 | 0.69342 | 0.506 |
| best mixture | m01 | m04 @ 0.25: **0.68748** | m02 pure | — |

- **1X2 gets worse in every loose direction.** Log loss rises monotonically with ρ for
  all three loose arms. The extra separation the loose arms buy does not pay on the
  scoring rule.
- **Totals and BTTS are not distorted by the mixture.** Mean P(over 2.5) stays within
  0.506–0.514 and BTTS moves by at most 0.01 against the tight arm. Totals log loss
  changes by at most 0.0015 across all 13 rows. Every Poisson GRW arm beats the close on
  O/U 2.5 log loss (0.6875–0.6889 vs 0.6899) and loses to it on 1X2 and BTTS.

## 6. What the numbers say

1. **The data identify σ₀; the prior barely matters.** A 2.5× wider prior moves the
   posterior attack σ₀ from 0.192 to 0.209, and slope rises by 0.09 on the pure arm.
   Student-t innovations fit a *smaller* σ₀ (0.146), with slope unchanged: the tails
   replace scale instead of adding spread.
2. **Forcing σ₀ to the market-implied 0.48 widens the model in the wrong directions.**
   m04's supremacy slope rises only to 0.441 while R² against the market falls from 0.44
   to 0.34. The added spread is mostly orthogonal to the market's ranking, and 1X2 log
   loss, BTTS log loss and the O/U mean all get worse. So the "2.43×" amplification the
   feature EDA measured is a statement about *which* teams the market separates, not a
   missing variance scale that σ₀ can supply.
3. **The portfolio moves are not decompression.** Loose mixtures cut capital at odds
   ≥ 4.0 from 38% to 32–35%, the intended mechanism, but modestly. m04's higher
   Sharpe and total return (+306% at ρ = 0.75) come with worse log loss on every family,
   a deeper drawdown (−24% to −28%) and lower R². I have not bootstrapped the Sharpe
   differences, so they should not be read as an improvement. m02, the only arm that
   genuinely raises the slope, lowers Sharpe from 1.65 to 1.45.
4. **Draw mixtures are bounded by their arms.** Because a mixture is a linear pool, it
   cannot price outside the pure arms. With no arm near market parity, no ρ can reach it.

## 7. Recommendations for Phase 2

- **Do not carry the fast-slow mixture into the two-arm joint Gamma-Poisson GRW as a
  decompression device.** The binding constraint is information in goal counts, not
  prior shrinkage, and a second likelihood arm changes that only through the xG
  channel.
- **The lever that can reach parity is market information itself.** The generative rate
  calibrator (`GenerativeRateCalibrator`, `calibrate_fit`) already pools posterior
  log-rates with the inverted book and prices through the same kernels. It is the
  principled version of "aim off toward the handrail", with the market as the handrail.
- **If the team latent is to be widened, target the market's direction, not the
  variance.** For example, an informative prior on each team's initial level from the
  previous season's market-implied supremacy, rather than a larger σ₀.
- **Reporting.** The pure Poisson GRW tight arm's slope here (0.399 on 596 fixtures, via
  `invert_market_rates`) is not directly comparable with the 0.316–0.328 quoted for
  m12/m05, which were computed in a different EDA. Recompute those on this panel and
  inversion before comparing.

## 8. Reproduce

```bash
# on mcmc-beast, /root/BF_fast_slow_grw (Manifest copied from BF_grw_player_hybrid; Distributions 0.25.126)
julia --project -t 16 current_development/fast_slow_grw/r01_fast_slow_smoke.jl               # ~6 min
julia --project -t 16 current_development/fast_slow_grw/r02_fast_slow_production_grid.jl     # ~2 h 15 min
julia --project -t 16 current_development/fast_slow_grw/r03_fast_slow_evaluation_and_blend.jl # ~5 min
```
