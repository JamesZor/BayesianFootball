# 015 — Prototype MultiScaleGRW with Market Smile and Supremacy Anchoring

| Field | Value |
|---|---|
| ID | 015 |
| Title | Prototype MultiScaleGRW with Market Smile and Supremacy Anchoring |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-12 |
| Updated | 2026-09-13 |
| Related Files / Commits / PRs | [current_development/grw_market_smile/](../current_development/grw_market_smile/); [docs/tickets/T010](../docs/tickets/T010-postgres-storage-refuses-smile-latents.md); [docs/tickets/T011](../docs/tickets/T011-portfolio-sizes-smile-latents-off-the-grid.md); [src/features/extractors/market_extractors.jl](../src/features/extractors/market_extractors.jl); [src/models/pregame/components/dynamics/multiscale_grw.jl](../src/models/pregame/components/dynamics/multiscale_grw.jl); [src/models/pregame/engines/team_level/time_decay/goals_smile_league.jl](../src/models/pregame/engines/team_level/time_decay/goals_smile_league.jl); [current_development/smile_negbin/](../current_development/smile_negbin/); [current_development/grw_player_hybrid/](../current_development/grw_player_hybrid/) |

## Context & Problem Statement

During live execution on Scottish Lower (tournaments 56 & 57), pure football models exhibit a supremacy compression delusion: home favourites are estimated at 40%–43% win probability while Betfair closing odds price them at 55%–60%. This causes the model to perceive artificial value on away underdogs, resulting in severe losses (e.g. 6 of 6 losing away bets on the 2026-09-12 live slate).

In the Ireland leagues (tournaments 79 & 718), a dual market anchoring mechanism was developed:
1. **Market Supremacy**: Anchoring model log-supremacy $\log \lambda_h - \log \lambda_a$ to de-vigged market supremacy $m_h - m_a$.
2. **Market Smile**: Anchoring total intensity $\log(\lambda_h + \lambda_a) + \log \phi(K)$ to de-vigged market Under lines across strikes $K \in \{0, 1, 2, 3, 4\}$.

When tested previously on Scottish Lower with `TimeDecay`, the smile model failed to improve over baseline because TimeDecay's exponential reversion to league mean fought against the market priors.

With `MultiScaleGRW` (state-space random walk for latent team attack and defence), the latent walk operates with two speeds (slow drift + fast shock) and avoids rigid decay reversion. This should provide the temporal flexibility needed to absorb market-implied trajectories without distorting long-term team fundamentals.

## Acceptance Criteria

- [x] Implement prototype in `current_development/grw_market_smile/`:
  - `l01_loader.jl`: Data loading, feature extraction (including `MarketSmileFeature(Kmax=4)` and closing market supremacy), and Turing engine definitions.
  - `r01_smoke.jl`: 2-fold smoke gate on Scottish Lower (Folds 1–2).
  - `r02_production_grid.jl`: 40-fold walk-forward grid on `mcmc-beast` (-t 16).
  - `r04_evaluate.jl`: Out-of-sample proper scores (LogLoss, Brier, RPS, ECE across 1X2, O/U 2.5, BTTS).
  - `r05_slate_repricing.jl`: Counterfactual re-pricing of 2026-09-12 live card.
  - `README.md`: Comprehensive experimental documentation.
- [x] Implement 3-model ablation ladder on Gen 3 Team-Level Joint foundation (`MultiScaleGRW` + `ProductionWealthCovariate` + `JointGammaPoissonObservation`) — baseline pinned to Task 013 run `b0961bc4` (identical recipe, asserted):
  1. `m05_joint_grw_baseline`: Pure football GRW control (no market anchoring).
  2. `m05_joint_grw_supremacy`: GRW + Market Supremacy pillar.
  3. `m05_joint_grw_smile_supremacy`: GRW + Market Supremacy + Market Smile dual pillar.
- [x] Evaluate 3-point weight tuning grid: Light (0.20 / 0.20), Moderate (0.40 / 0.40 — Ireland default), Strong (0.70 / 0.70). *Evaluated: no weight distinguishable from another (|ΔLogLoss| ≤ 0.0006).*
- [ ] Verify ReverseDiff compiled gradient tape matches ForwardDiff ($\le 10^{-6}$) with zero unexpected heap allocations. *Tape parity MET (≤ 1e-15, exact under perturbation). Zero allocations NOT met: baseline 129 KB, supremacy +47 KB, smile +270 KB per gradient — reported, not tuned away.*
- [x] Verify 6-part convergence audit on all 40 folds on `mcmc-beast` ($\hat{R} \le 1.05$, 0 divergences, ESS $\ge 400$). *All 43 folds, all four rungs; smile_w070 needed 4×(1000+2000) after failing bulk ESS at the production budget.*
- [x] Store fits in `PostgresStorage("scottish_lower_grw_market_smile")`. *Smile runs with the latent panel detached and rebuilt from chains (T010).*
- [x] Counterfactually re-price the 2026-09-12 slate: test whether market anchoring lifts home favourite win probabilities from ~41% toward ~55%–60% and eliminates the 6 losing away bets. *Tested: P(home) rises partly (not to the book) and moves away on two fixtures; five losing away legs survive in every anchored sheet.*

## Ideas & Candidate Solutions

- **Zero-Alloc Strike Evaluation**: Ensure `_alloc_smile_holder` and `SmileScoreGrid` are used in evaluation and portfolio pricing to avoid allocating per-fixture score matrices.
- **De-vigged Close Alignment**: Match closing 1X2 and Over/Under lines using `sofascore.match_odds` and `betfair.markets` through the existing `MarketSmileFeature` extractor.
- **Weight Calibration**: If fixed weights cause over-fitting or under-confidence, consider evaluating an adaptive likelihood scale where $\sigma_{\text{sup}}$ and $\sigma_{\text{smile}}$ have informative priors.

## Work Log & Progress

- [2026-09-12 @antigravity] Aligned task via `/grill-me`. Created task 015, initialized worktree at `/home/james/bet_project/.worktrees/BayesianFootball-grw-market-smile` on branch `feat/grw-market-smile`, prepared work package prompt `GRW_MARKET_SMILE_WORK_PACKAGE.md`, and launched `claude_grw_smile` in tmux.
- [2026-09-13 @claude] Implemented `current_development/grw_market_smile/` (l01–l03, r01, r02, r04, r05, README). Pillars attach through a `MarketAnchoredCountModel` wrapper that reuses the builder's submodels (G0 bit-identity gate against Task 013 `m05`). Baseline rung pinned to Task 013 run `b0961bc4` (identical recipe). Production split is 43 folds (Fold 43 prices the 09-12 card), restricted to the 710 panel for scoring. Found and ticketed T010: `PostgresStorage` cannot round-trip `SmileLatents`; worked around in the loader. First smoke attempt stopped at G0 (displaced test points left GRW scale support — gate defect, fixed with independent prior draws); relaunched.
- [2026-09-13 @claude] r01 smoke: G6 refused the wrapper (`save_model` hard-codes `ComposableCountModel`; recorded in T010, registry now stores base model + full `FitConfig`). At the work-package budget 4×(500+500) the BASELINE failed one tail ESS (363 < 400); at the production budget 4×(500+1000) all three rungs PASS (R̂ ≤ 1.0064, 0 divergences, G0 bit-identity, exact gradients, smile pricing identical across routes, Postgres round-trip identical). Smoke runs `e2152be3…`, `cd886262…`, `5ce9ded8…` in `smoke_grw_smile`.
- [2026-09-13 @claude] r02 43-fold grid (`scottish_lower_grw_market_smile`): supremacy_w040 `0ee58d18-b7e9-4168-8d78-93887b1a8c26` (R̂ 1.0105, ESS 814/516, 60 min), smile_w020 `fcd5e974-9a46-4a10-9828-6b987a5484d6` (R̂ 1.0139, ESS 472/498, 158 min), smile_w040 `30620d3e-e4bd-4c05-b1a1-85cefa36b728` (R̂ 1.0200, ESS 431/696, 185 min) — all PASS, 0 divergences, OOS set == baseline control's 769. smile_w070 FAILED bulk ESS 312 at fold 29 (R̂ 1.0201, 0 div, 266 min); not persisted, re-running at 4×(1000+2000) stride 4. Pillars: σ_sup 0.21–0.24, σ_smile ≈ 0.05, φ = 0.844/0.976/1.001/1.026/1.069 independent of weight. Pre-sampling gates that needed correcting (all before any draw): fold-prefix check read split metadata; baseline control's recorded sampler and fold audit reflect Task 013's thinning and `extend_fit`'s re-audit (evidence: Task 013 committed r02 CSV, 160,000 transitions over 40 folds; `extension.jl:365`); control gate must use the canonical budget, not a re-run's override.
- [2026-09-13 @claude] smile_w070 re-run at 4×(1000+2000) stride 4 PASSED: `32d588f1-d666-4112-a7e1-5c9545fbbe3d` (R̂ 1.0159, ESS 504/1182, 0/344k div, 438 min); pillars unchanged from the failed run.
- [2026-09-13 @claude] r04 evaluation (710 panel, Betfair TWA close, 10k fixture-clustered bootstrap). Reproduction gate exact (baseline 0.64315 / ECE 0.0123 / 2,899 rows). **0 of 84 contrasts significant.** Pooled LogLoss: baseline 0.64315, supremacy_w040 0.64135, smile w020/w040/w070 0.64116/0.64092/0.64097, close 0.64182 — anchored rungs lean better but every interval includes 0, and ECE worsens (0.0123 → 0.0145–0.0198). H1 refuted (sup − base 1X2 Δ −0.00221 [−0.00612, +0.00173]); H2 refuted (smile − sup OU2.5 Δ −0.00097 [−0.01366, +0.01166]; smile worse on BTTS/OU1.5 point-wise); H4 refuted (weights |Δ| ≤ 0.0006). H3 partly: top-bin home-favourite gap −0.121 → −0.091 at best, and in the ≥ 0.50 bins the realised home win rate lies between model and close (0.588 vs close 0.675, n=51; 0.495 vs 0.539, n=107).
- [2026-09-13 @claude] r05 2026-09-12 T−25 counterfactual (replay engine, Fold 43, nothing written). Live reproduction: 11/11 live legs re-priced (+4), max |Δrisk| £3.64, |Δp| 0.023; realised −£45.89 confirmed from `paper_settlements`. Supremacy lifts raw P(home) toward the book on 5 of 9 fixtures but moves away on Stranraer and pushes Queen of the South PAST its 0.392 book (0.488–0.506). Option B full-fill P&L: live −£54.52, baseline −£53.56, supremacy −£39.40, smile w020/w040/w070 −£34.20/−£28.88/−£21.36 — mostly from lower risk (£73.78 → £31.41–£42.44); loss per £ risk only better than live at w070. Five losing away legs survive in every anchored sheet; a new losing QoS home leg appears. Launch fixes: `BF_DB_URL` must be exported (T009) and reached over Tailscale from beast (LAN 192.168.1.88 unreachable). README conclusions written: do not promote.
- [2026-09-13 @claude] r06 Option B closing-line portfolio (`MatchDay.option_b_system()`, de-vigged Betfair TWA(−20,0] close; 710 → 635 quoted → 632 buildable by every arm). Gates: P1 baseline reproduces Task 014 §6 exactly (+385.78% / ROI 11.68% / 1,247 bets); P2 every smile arm's staked totals bets priced through λ_tot·φ(K) to ≤ 1.8e-15; P3 four candidate portfolios persisted and reloaded identically (`1e6b80b0…`, `b8ea28ef…`, `38d3038e…`, `5cff0ccc…`).

  | model | return | flat ROI | max DD | Sharpe | bets (1X2 / totals) | win rate |
  |---|---:|---:|---:|---:|---|---:|
  | baseline `b0961bc4` | +385.8% | 11.68% | −42.67% | 1.453 | 1,247 (986 / 261) | 35.2% |
  | supremacy_w040 `0ee58d18` | +404.6% | 12.75% | −42.64% | 1.551 | 1,244 (983 / 261) | 36.5% |
  | smile_w020 `fcd5e974` | +588.4% | 15.71% | −44.02% | 1.495 | 1,240 (988 / 252) | 34.0% |
  | smile_w040 `30620d3e` | +545.0% | 15.82% | −44.03% | 1.516 | 1,237 (984 / 253) | 34.0% |
  | smile_w070 `32d588f1` | +481.4% | 15.48% | −42.37% | 1.530 | 1,231 (978 / 253) | 35.0% |

  Anchored arms out-return the baseline, but per-slate growth intervals overlap almost entirely (baseline [−0.0004, +0.0323] vs smile_w020 [−0.0003, +0.0390]) and r04 found no proper-score difference. The gain is on 1X2 AWAY (ROI 7.8% → 16.6–19.4%); home ROI falls (20.0% → 13.9–18.5%); the smile shrinks totals stake (24% → 15–18%) without improving totals ROI. Supremacy gains by pruning (declines 173 baseline bets at −25.8% ROI); smile gains by re-sizing shared bets (+2.7 to +4.7 ROI points). Descriptive only — next test is executable T−25 prices, not promotion.
- [2026-09-13 @claude] r07 raw vs calibrated at the close and T−25 (`l04`, `r07`; Task 014 r06 design, `MatchDay.option_b_calibrator()` = InverseGaussianLaw w0.25 s0.35). Gates: close/raw reproduces r06 exactly; T−25 panel is Task 014's 611 and the baseline reproduces its raw (+531.78% / 1,124) and t25_inv (+245.85% / 969) rows; smile ledgers' totals p_model = λ_tot·φ(K) to 1.8e-15. `calibrate_latents` refuses SmileLatents, so two calibrated-smile definitions were run (pooltot: calibrated λ_tot × fitted φ; grid: φ dropped) — they stake BIT-IDENTICAL ledgers because Portfolio sizes from the (λ_h, λ_a) grid and φ only reaches p_model (`pricing.jl:360–375`) → ticket **T011**.

  | T−25 | return | ROI | Sharpe | max DD | bets |
  |---|---:|---:|---:|---:|---:|
  | baseline raw | +531.8% | 14.15% | 1.658 | −41.9% | 1,124 |
  | supremacy_w040 raw | +605.6% | 16.17% | 1.711 | −38.3% | 1,089 |
  | smile_w020 raw | +542.4% | 16.91% | 1.342 | −42.9% | 1,111 |
  | smile_w040 raw | +479.4% | 16.80% | 1.322 | −43.6% | 1,087 |
  | smile_w070 raw | +381.6% | 15.78% | 1.276 | −41.8% | 1,094 |
  | baseline t25_inv | +245.9% | 17.39% | 1.976 | −22.0% | 969 |
  | supremacy_w040 t25_inv | +219.5% | 17.98% | 1.856 | −19.6% | 963 |
  | smile_w020 t25_inv | +242.4% | 21.78% | 1.818 | −18.4% | 955 |
  | smile_w040 t25_inv | +223.7% | 22.10% | 1.825 | −16.7% | 947 |
  | smile_w070 t25_inv | +178.8% | 20.37% | 1.697 | −14.7% | 940 |

  Paired slate log-growth p(better), smile − baseline: close raw 0.72/0.69/0.62; T−25 raw 0.51/0.44/0.33 (supremacy 0.60); T−25 calibrated 0.49/0.41/0.23. Smile keeps a ROI lead (+2.7 raw, +4.4/+4.7 calibrated) but not a bankroll lead; calibrated it is lower exposure and drawdown, not extra edge beyond L2.
- [2026-09-13 @claude] r08 trust-pruning sweep (smile_w040 vs baseline; close and raw T−25; Option B + one fringe line at 1/1.4, then unions; book extended with O/U 4.5). Gates: close/P0 reproduces r06; smile routing on all 732/539 totals bets. Finding: a zero-trust O/U 4.5 market is NOT inert in Option B's solve (−6.6 to −22.3 pp), so every row is measured against P0 on the extended book. Premise corrections: φ₀ = 0.844 RAISES P(Under 0.5) (staked p_model 0.100 vs grid 0.084–0.089, realised 0.000–0.038), and φ never sizes a stake (T011). Under 0.5 loses in all 4 arm × book cells (−42% to −100% ROI); Under 1.5 (+3.9 to +37.7%) and Under 4.5 (+10.4 to +21.8%) positive in all 4 on 23–105 bets; all other lines change sign between close and T−25; all_fringe −159 pp (baseline T−25). Cannibalisation mild under Option B (core stake 0.96–1.00× per line, 0.83–0.95× all_fringe). Smile vs baseline paired p(better) 0.42–0.82 in all 24 cells. Keep EDA pruning; U1.5/U4.5 only candidates for a prospective paper trial.

## Verification & Findings

Not run yet. Record smoke gate passes, 40-fold convergence metrics, paired $\Delta\text{LogLoss}$ bootstrap results across the weight grid, and 2026-09-12 slate counterfactual tearsheet.
