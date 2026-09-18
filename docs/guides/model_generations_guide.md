# Model generations — Scottish Lower (tournaments 56 / 57)

> Extracted from `AGENTS.md` (formerly §6). `AGENTS.md` keeps the one-row-per-generation
> summary; this guide keeps the Generation 3 and 4 formulations and the Experiment 06 table.
> Measured results and run UUIDs are authoritative in each suite's `README.md`.

Four paradigms, each a full 40-fold walk-forward grid over seasons 24/25 + 25/26
(710 held-out matches, 2,899 scored market observations). Numbers below are the
recorded outcomes in each suite's README, not targets.

| Gen | Suite | Paradigm | Headline |
|---|---|---|---|
| **1** | [`01_poisson_2426_grid/`](../../experiments/scottish_lower/01_poisson_2426_grid/README.md) | Poisson likelihood; baseline, squad wealth, travel distance, joint, age-adjusted production wealth | `m05_production_wealth` LogLoss **0.6597**; Betfair backtest +125% to +140% bankroll |
| **2** | [`02_negbin_2426_grid/`](../../experiments/scottish_lower/02_negbin_2426_grid/README.md) | Negative Binomial; empirical overdispersion | `r̂ ≈ 26.0–26.5` (mild overdispersion); LogLoss **0.6598**, no material gain over Poisson |
| **3** | [`03_joint_gamma_poisson/`](../../experiments/scottish_lower/03_joint_gamma_poisson/README.md) | **Two-arm joint**: shared latent `μ`, Gamma arm on BBC commentary proxy xG, Poisson arm on goals | LogLoss **0.6571** vs Betfair close 0.6568 — the second likelihood is worth ~5× the best covariate |
| **4** | [`05_.../`](../../experiments/scottish_lower/05_player_lineup_and_pxg_fusion/README.md) + [`06_.../`](../../experiments/scottish_lower/06_joint_player_lineup_fusion/README.md) | **Joint + player-lineup hybrid**: `PlayerLineupPillar` (shots-RAPM / pxG-RAPM, starters + bench at fixed `w_bench = 0.10`) composed beside team time decay | `m12` ECE **0.0100** vs Betfair close **0.0139**; +136.6% bankroll, 1.416 annual Sharpe |

## 1. Generation 3 — the two-arm joint observation

One log-intensity `η` per side is read by two densities at once:

```
arm 1 (proxy xG)   pxg_s ~ Gamma(ν, μ_s / ν)     evaluated where the mask is 1
arm 2 (goals)      y_s   ~ Poisson(κ · μ_s)      evaluated everywhere
```

`Gamma(shape = ν, scale = μ/ν)` has mean `μ`, so `ν` is a pure precision and the
proxy measurement is unbiased for the latent by construction; `κ` is the
finishing factor. The proxy arm sharpens `μ` on the seasons that carry BBC live
text; the goals arm carries that sharpened `μ` back across the whole history.
`MatchProxyXGFeature(fallback = :none)` emits an explicit availability mask, so a
match without commentary contributes a finite term multiplied by an exact zero
rather than a fabricated observation.

Identified, not assumed: `κ ≈ 1.13` (~13% more goals converted than the BBC
shot-xG cell table predicts, ~76% prior shrinkage) and `ν ≈ 3.9` with posterior
sd ~0.28 against a prior sd of 1.45.

## 2. Generation 4 — the player-lineup hybrid

```
η_home,i = μ_int + HA + α_home + β_away + L_home,i + Σ_c w_c · x_c,i
η_away,i = μ_int      + α_away + β_home + L_away,i − Σ_c w_c · x_c,i

L_home,i = w_att · R_home,i − w_def · R_away,i
```

`R_s,i` is the aggregated RAPM rating of side `s`'s named teamsheet. RAPM is a
ridge fit — **never sampled** — over each fold's frozen history block
(`fit_on = :history`), so a target fixture never contributes to the ratings that
price it. Covariates enter in the `SupremacyRole()`
(`covariate_sides(SupremacyRole(), q) = (q, −q)`), so a covariate moves the
*result* and holds the *total*.

Experiment 06 (`scottish_lower_joint_player_2426`) measured, over 2,899 scored
observations and 1,455–1,468 bets:

| Model | LogLoss | ECE | Bankroll | Sharpe | Max DD |
|---|---:|---:|---:|---:|---:|
| `m13_joint_composite` (+ distance) | 0.64324 | **0.0088** | **+140.2%** | 1.453 | −21.1% |
| `m12_joint_hybrid_synergy` | 0.64337 | 0.0100 | +136.6% | 1.416 | −20.2% |
| `m05_joint_production_wealth` (control) | **0.64299** | 0.0149 | +131.2% | **1.481** | **−19.1%** |
| `m10_joint_player_shots_bench` | 0.64440 | 0.0090 | +112.2% | 1.217 | −20.0% |
| *Betfair closing line* | *0.64182* | *0.0139* | — | — | — |

**Read this honestly.** The lineup arms do **not** win on LogLoss — the
team-state control is still the sharpest. What they buy is **calibration**: every
lineup arm halves the control's ECE and beats the Betfair closing line's, and
that is what converts into Kelly bankroll growth. The prior EDA (`r59`,
`EDA_FINDINGS.md`) refused H1 (lineups alone improve team state) and H5
(travel), supported H4 (wealth is complementary to RAPM) strongly, and found H2
(bench depth) negligible. The hypotheses were written so they could fail
visibly, and some did.

`m12` is the model the MatchDay consoles load as the hybrid pillar; `m00` and
`m05` are the team-level controls that make a lineup move attributable.
