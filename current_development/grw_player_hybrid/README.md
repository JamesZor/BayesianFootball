# Task 013 — MultiScaleGRW × PlayerLineupPillar

Does replacing the Gen 4 hybrid's `TimeDecayDynamics(180)` team state with the
two-speed `MultiScaleGRW` improve it? Scottish League One/Two (tournaments 56/57),
walk-forward over 24/25 + 25/26, extended into the 2026/27 opening slates.

**Short answer: no, not on this evidence.** The GRW hybrid is the best-calibrated arm
tested (ECE 0.0086 vs 0.0100 for the production hybrid and 0.0139 for the Betfair
close), but it is +0.0010 worse on LogLoss (not significant), and under the production
Option B staking contract it grows the closing-line bankroll less than the TimeDecay
hybrid (+352% vs +607%). Attribution puts the gap on *which* bets it takes, not on how
it sizes them. See §5 for the T−25 order-book result.

## 1. The ladder

Every non-dynamics component is the Experiment 06 `l60` recipe verbatim, so
`m12_joint_hybrid_synergy_grw` differs from production `m12_joint_hybrid_synergy` in one
slot.

| Model | Dynamics | Covariates / pillars | Likelihood |
|---|---|---|---|
| `m00_baseline_grw` | `MultiScaleGRW()` | — | Poisson |
| `m05_wealth_grw` | `MultiScaleGRW()` | production wealth (Supremacy) | Joint Gamma-Poisson |
| `m10_lineup_grw` | `MultiScaleGRW()` | shots-RAPM lineup, bench 0.10, `fit_on = :history` | Poisson |
| `m12_joint_hybrid_synergy_grw` | `MultiScaleGRW()` | lineup + wealth | Joint Gamma-Poisson |

Controls, pinned by UUID: `m05_joint_td_raw` (`ed541a7c…`, Exp 06), `m12_hybrid_td_raw`
(`132df5c2…`, Exp 06, the production hybrid) and `m05_joint_grw_raw` (`f870dbb7…`,
Task 007).

### Deviations from the work-package sketch

| Sketch | Used | Why |
|---|---|---|
| `Data.CVConfig(window_seasons = 3)` | `GroupedCVConfig(history_seasons = 2, dynamics_col = :match_biweek)` | the canonical 40-fold / 710-fixture grid every control was scored on; the GRW micro step *is* the match-biweek |
| `PlayerLineupPillar(rating = :shots_rapm, fit_on = :history)` | `PlayerLineupPillar(feature = ShotsPlusMinusFeature(λ = 1000, half_life_days = 730, fit_on = :history), aggregation = BenchWeightedPlayerAggregation(0.10))` | the sketched keywords do not exist; this is the Exp 06 recipe |
| σ priors `TruncatedNormal`, φ ~ Beta(8, 2) | `MultiScaleGRW()` graduated defaults (Gamma scale priors, no φ) | the component in `src/` has no mean-reversion parameter; its defaults are Task 007's priors |
| `TieredTrust(base = 0.25)`, `FixedCap(0.20)`, Baker-McHale | `MatchDay.option_b_system()` | no `base` constructor exists; Option B is the audited production contract |
| smoke `NUTSConfig(100, 50, 2)` | gate run at the production sampler | see §2 |
| Step 3 unnamed; `r03_evaluate`, `r04_…`, `r05_t25_…` | `r03_extend_2627`, `r04_evaluate`, `r05_portfolio_attribution`, `r06_t25_backtest` | extension got its own runner; files are numbered in execution order |

## 2. Smoke gate (`r01_smoke.jl`, folds 1–2)

| Budget | Divergences | max R̂ | min ESS | Verdict |
|---|---:|---:|---:|---|
| 2 × (50 + 100) — work package | 0 | 1.05–1.13 | 17–51 | FAIL on R̂ (budget artefact: 50 warmup draws do not adapt; 200 draws cannot hold ESS 400) |
| 4 × (400 + 400) — Task 007 preflight | 0 | ≤ 1.017 | 343 | FAIL on ESS for m00, m12 |
| **4 × (500 + 1000) — production** | **0** | **≤ 1.0124** | **886** | **PASS 4/4** |

All budgets pass G1 (compiled tape; ReverseDiff ≡ ForwardDiff to 1e-15; tape correct at
three perturbed points; both the no-target and the target GRW branch taped), G4 (latents)
and G5 (`save_fit` → `load_fit` exact). The lineup pillar adds 2 parameters and 25 tape
instructions; m00/m05 tapes match Task 007 instruction for instruction. Allocation per
gradient: 35 KB (m00) – 181 KB (m12), measured not gated (the TimeDecay control allocates
comparably; see Task 007).

## 3. Production grid (`r02_production_grid.jl`) and extension (`r03_extend_2627.jl`)

`QueuedNUTSConfig(1000 retained, 500 warmup, 4 chains, δ = 0.80, depth 10)`, 16 threads.
Namespace `scottish_lower_grw_player_hybrid`. Audited on all 4,000 draws per fold;
persisted every 2nd draw (a 4,000-draw artefact would exceed PostgreSQL's 1 GB field).

| Model | Folds | OOS | max R̂ | ESS bulk | ESS tail | Divergences | BFMI | Wall | Run UUID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `m00_baseline_grw` | 40 | 710 | 1.0105 | 785 | 947 | 0 / 160k | 0.73 | 28 min | `158d2a80-7ea3-4d6c-b3ab-be62bcf1bc11` |
| `m05_wealth_grw` | 40 | 710 | 1.0101 | 822 | 821 | 0 / 160k | 0.69 | 41 min | `b0961bc4-c40c-4dbe-9c05-57df7ae0839e` |
| `m10_lineup_grw` | 40 | 710 | 1.0093 | 909 | 487 | 0 / 160k | 0.74 | 39 min | `b13c8fb9-ce34-4210-aa3f-9d2ed493c286` |
| `m12_joint_hybrid_synergy_grw` | 40 | 710 | 1.0149 | 678 | 822 | 0 / 160k | 0.61 | 48 min | `3a9a4c7e-378b-45d0-a2d2-c8b69b46786b` |

All four pass the six-part audit (R̂ ≤ 1.05, ESS ≥ 400, divergences < 0.1%, BFMI ≥ 0.30,
depth saturation < 5%). Task 007's strict R̂ ≤ 1.01 (advisory) holds only for m10.

`extend_fit` added folds 41–43 in place (same UUIDs): 43 folds, 769 fixtures (59 in
2026/27), new folds R̂ ≤ 1.0147 and 0 divergences. At the matched 500-draw extension
budget, m00 and m10's new folds reach ESS 320 / 342, so those two runs now carry the
`convergence` flag false at run level; m05 and m12 pass. The 2026/27 folds only feed the
T−25 backtest.

## 4. Proper scores (`r04_evaluate.jl`)

710 walk-forward fixtures, 2,899 scored selections on 627 quoted fixtures, de-vigged
Betfair TWA(−20, 0] close. **Reproduction gates passed**: `m12_hybrid_td_raw` scores
0.64337 / ECE 0.0100 (published 0.64337 / 0.0100); `m05_wealth_grw` vs Task 007's
identical-recipe `m05_joint_grw_raw` ΔLL −0.00001 [−0.00024, +0.00023].

| Model | Dynamics | LogLoss | Brier | RPS (1X2) | ECE |
|---|---|---:|---:|---:|---:|
| `m05_joint_td_raw` | TimeDecay | **0.64299** | **0.22586** | 0.22415 | 0.0149 |
| `m05_wealth_grw` | GRW | 0.64315 | 0.22603 | **0.22383** | 0.0123 |
| `m05_joint_grw_raw` | GRW | 0.64316 | 0.22603 | 0.22385 | 0.0117 |
| `m12_hybrid_td_raw` | TimeDecay | 0.64337 | 0.22605 | 0.22447 | 0.0100 |
| `m00_baseline_grw` | GRW | 0.64433 | 0.22654 | 0.22492 | 0.0184 |
| `m12_joint_hybrid_synergy_grw` | GRW | 0.64437 | 0.22659 | 0.22493 | **0.0086** |
| `m10_lineup_grw` | GRW | 0.64561 | 0.22713 | 0.22631 | 0.0092 |
| *Betfair close* | — | *0.64182* | *0.22529* | *0.21110* | *0.0139* |

Per market (LogLoss / ECE), GRW hybrid vs TD hybrid: 1X2 0.61701 / 0.0140 vs 0.61636 /
0.0155; O/U 2.5 0.68973 / 0.0133 vs 0.68732 / 0.0100; BTTS 0.68496 / 0.0402 vs 0.68520 /
0.0087. Full tables: `results/evaluation/r04_evaluation_report.md`.

**Paired ΔLogLoss**, 10,000 resamples of *fixtures* (a fixture's rows move together):

| Contrast (left − right) | Δ | 95% CI | P(Δ < 0) |
|---|---:|---|---:|
| GRW hybrid − TD hybrid | +0.00100 | [−0.00299, +0.00497] | 0.31 |
| `m05_wealth_grw` − `m05_joint_td_raw` | +0.00016 | [−0.00368, +0.00403] | 0.46 |
| lineup on GRW + joint (`m12` − `m05`) | +0.00122 | [−0.00073, +0.00322] | 0.12 |
| lineup on GRW + Poisson (`m10` − `m00`) | +0.00127 | [−0.00104, +0.00357] | 0.14 |
| GRW hybrid − Betfair close | +0.00255 | [−0.00396, +0.00894] | 0.22 |

No contrast is significant, including every model against the close.

**Reading.** Both of Task 013's mechanisms behave as their parents' studies found, and
they do not stack. The lineup pillar costs a little LogLoss and cuts ECE by 30–50% on
either likelihood (0.0184 → 0.0092 Poisson; 0.0123 → 0.0086 joint) — Exp 06's "calibration,
not sharpness" finding, reproduced on GRW state. The GRW state itself is worth nothing
against this TimeDecay control once the joint likelihood is present (+0.00016), in line
with Task 007's "substitutive, not additive" reading. ECE over 10 bins and 2,899 rows is a
noisy statistic; the ranking of the two hybrids on ECE should not be leaned on alone.

## 5. Portfolios

### 5.1 Closing line, Option B (`r05_portfolio_attribution.jl`)

`MatchDay.option_b_system()` for every arm, 632 fixtures buildable by all seven.
Returns are large because Option B's `SlateDrawdown(8)` is a loose risk budget meant for
T−25 prices *after* calibration shrinks edges; here it stakes raw posteriors at the close.

| Model | Bets | Return | ROI | Growth/slate 95% CI | Sharpe | MDD | Capture |
|---|---:|---:|---:|---|---:|---:|---:|
| `m12_hybrid_td_raw` | 1302 | **+606.5%** | **14.15%** | [−0.0001, +0.0393] | 1.487 | −42.4% | 0.949 |
| `m05_joint_td_raw` | 1280 | +495.5% | 13.57% | [−0.0015, +0.0370] | 1.388 | −40.3% | 0.945 |
| `m00_baseline_grw` | 1296 | +491.6% | 12.32% | [−0.0008, +0.0363] | 1.412 | **−38.9%** | 1.036 |
| `m05_joint_grw_raw` | 1235 | +404.3% | 11.87% | [+0.0002, +0.0328] | **1.498** | −42.8% | **1.094** |
| `m05_wealth_grw` | 1247 | +385.8% | 11.68% | [−0.0004, +0.0323] | 1.453 | −42.7% | 1.080 |
| `m12_joint_hybrid_synergy_grw` | 1253 | +351.9% | 11.36% | [−0.0019, +0.0324] | 1.309 | −52.6% | 1.043 |
| `m10_lineup_grw` | 1299 | +348.8% | 10.75% | [−0.0032, +0.0335] | 1.214 | −48.0% | 0.957 |

Every growth interval but one straddles zero.

**Task 012 attribution, GRW hybrid vs TD hybrid** (bankroll-fraction units):

| | Shared (1,062 bets, 71%) | GRW-only (191) | TD-only (240) |
|---|---|---|---|
| ROI | GRW 13.41% vs TD 12.92% | **−11.88%** | **+29.67%** |
| Win rate / cap-weighted | 32.8% / 35.5% vs 32.8% / 32.9% | 43.5% / 45.2% | 39.2% / 48.0% |
| Capture ratio | 1.087 vs 0.952 | 1.19 | 2.57 |

Sizing ΔPnL on shared bets `Σ(s_grw − s_td)·settle` = −0.032: GRW stakes slightly less
(mean 0.0146 vs 0.0153) at a higher ROI. **The whole closing-line gap is selectivity**:
TimeDecay's exclusive bets are longer-priced (mean odds 3.11 vs 2.44) and returned
+29.7%. The GRW's higher capture ratio — Task 007's finding, reproduced here at 1.04–1.09
for every GRW arm except `m10_lineup_grw` (0.957), vs 0.945–0.949 for TimeDecay — does
not convert into more money under this contract. Other pairs, persisted portfolio UUIDs and per-selection breakdowns:
`results/portfolio/r05_portfolio_report.md`.

### 5.2 2026/27 T−25 order book (`r06_t25_backtest.jl`)

`betfair_live.order_book_1m` at 13:35 UTC, £500 compounding, Option B staking,
provisional XI else `PriorMatchdayXI` (strictly-earlier-day; `LastHistorical` would leak
the played XI). Four slates priced (1, 8, 15 Aug, 5 Sep); 22 Aug refused — no archived
tick before kick-off. Four 1 Aug fixtures are refused for **every** arm because
Airdrieonians and Ross County are absent from fold 41's `team_map`. **Reproduction gate
passed**: all six benchmark tracks match `REPORT_T25_GRW_2627.md` to the penny.

| Arm | TouchOnly final | ROI | LadderSweep final | ROI | Max DD (touch) | Capture |
|---|---:|---:|---:|---:|---:|---:|
| `m05_joint_grw_raw` (Task 007) | **£590.05** | **25.59%** | £600.22 | 23.23% | −0.13% | 1.191 |
| `m05_wealth_grw_raw` | £588.43 | 25.16% | **£600.33** | 23.32% | −0.21% | 1.191 |
| `m05_joint_td_raw` | £569.45 | 19.39% | £587.76 | 20.24% | 0.00% | 1.295 |
| **`m12_hybrid_grw_raw`** | **£557.71** | **16.74%** | **£595.53** | **21.99%** | **0.00%** | 1.217 |
| `m12_hybrid_td_raw` (production) | £551.97 | 13.69% | £575.94 | 16.94% | −2.74% | 1.048 |
| `m00_baseline_grw_raw` | £536.85 | 10.36% | £538.44 | 9.04% | −7.41% | 1.236 |
| `m10_lineup_grw_raw` | £518.58 | 5.44% | £529.02 | 6.98% | −5.88% | 1.161 |
| `m12_hybrid_grw_cal_optB` | £506.77 | 3.88% | £491.12 | −4.21% | −3.63% | 0.901 |

The GRW hybrid beats the production TD hybrid at executable prices (+£5.74 TouchOnly,
+£19.59 LadderSweep) with no drawdown, but it does not beat the team-level GRW it was
meant to improve on (−£32.34 TouchOnly), and Option B calibration removes most of its
edge, as it did for `m12_hybrid_td` in the original report. This is ~62 legs over four
slates: it is the sanity check at tradeable prices, and it points the same way as the
632-fixture closing-line study — **the lineup pillar does not add staking value on top
of the GRW team state**. Ledgers: `results/t25/`.

## 6. Verdict and recommendation

* **Do not promote `m12_joint_hybrid_synergy_grw` over production `m12`.** Better ECE,
  slightly worse LogLoss (n.s.), lower closing-line growth, marginally better at T−25 on
  a sample too small to separate the two.
* **The GRW state and the joint likelihood are substitutes** (ΔLL +0.00016 vs TimeDecay
  with the joint arm present), confirming Task 007.
* **The lineup pillar buys calibration, not sharpness or money**, on either dynamics.
* Worth testing next: where the closing-line gap comes from inside TimeDecay's 240
  exclusive bets (mean odds 3.11 vs 2.44 for GRW's exclusives, +29.7% ROI) — by selection
  family and odds bucket — before attributing it to the dynamics rather than to a few
  longshot winners.

## 7. Regression tests

On mcmc-beast at this branch: `test/test_multiscale_grw.jl` 124/124,
`test/test_player_lineup_dynamics.jl` 149/149 (logs in `results/`). No file under `src/`
was changed by this task.

## 8. Reproduction

```bash
# mcmc-beast, /root/BF_grw_player_hybrid; Distributions pinned at 0.25.126
R=current_development/grw_player_hybrid; J="/root/.juliaup/bin/julia --project -t 16"
R01_SAMPLES=1000 R01_WARMUP=500 R01_CHAINS=4 $J $R/r01_smoke.jl
$J $R/r02_production_grid.jl        # ~2.6 h, resumes from results/<model>/checkpoints
$J $R/r03_extend_2627.jl            # folds 41–43, in place
$J $R/r04_evaluate.jl
$J $R/r05_portfolio_attribution.jl  # writes 4 portfolio_runs rows
$J $R/r06_t25_backtest.jl           # needs BF_DB_URL (betdb) in .env
```
