# 016 — Prototype 1-Parameter Smile Spine with MultiScaleGRW

| Field | Value |
|---|---|
| ID | 016 |
| Title | Prototype 1-Parameter Smile Spine with MultiScaleGRW |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | claude |
| Created | 2026-09-13 |
| Updated | 2026-09-17 |
| Related Files / Commits / PRs | [current_development/grw_smile_spine/](../current_development/grw_smile_spine/); [docs/tickets/T010-postgres-storage-refuses-smile-latents.md](../docs/tickets/T010-postgres-storage-refuses-smile-latents.md); [docs/tickets/T011-portfolio-sizes-smile-latents-off-the-grid.md](../docs/tickets/T011-portfolio-sizes-smile-latents-off-the-grid.md); [current_development/grw_market_smile/](../current_development/grw_market_smile/) |

## Context & Problem Statement

In Task 015, the market smile model fitted 5 independent unconstrained strike parameters ($\log \phi \sim \text{Normal}(0, 0.5)^5$) alongside market supremacy anchoring on `MultiScaleGRW`. While the model demonstrated substantial portfolio alpha on the Option B closing line (+545% to +588% bankroll growth and 15.8% flat ROI) by improving 1X2 away-bet sizing and pruning toxic bets, it revealed critical limitations:

1. **Massive MCMC Sampling Drag**: Wall time ballooned from 60 min (supremacy only) to 185 min (`w040`) and 438 min (`w070`), with minimum bulk ESS dropping from 814 down to 312, requiring a $4 \times (1000\text{w} + 2000\text{s})$ doubled budget to clear convergence.
2. **Redundant Degrees of Freedom**: The empirical posterior medians for $\log \phi(K)$ across strikes $K \in \{0, 1, 2, 3, 4\}$ follow an almost perfectly linear monotonic curve centered at $K=2$ (the 2.5 line): $[-0.170, -0.024, +0.001, +0.026, +0.069]$.
3. **Ticket T011 (Portfolio Staking Disconnect)**: `Portfolio.build_books_reported` reports `p_model` using $\phi(K)$, but the Kelly allocator solves the stake vector strictly from the un-smiled $(\lambda_h, \lambda_a)$ Poisson score grid. As a result, $\phi(K)$ never directly sized stakes.

Task 016 prototypes a **1-parameter smile spine**:
$$\log \phi(K) = \beta \cdot (K - 2) \iff \phi(K) = \exp\big(\beta \cdot (K - 2)\big) \quad \text{for } K \in \{0, 1, 2, 3, 4\}$$
with prior $\beta \sim \text{Normal}(0.04, 0.05)$, fixing $\phi(2) = 1.000$ strictly by construction.

In addition, the prototype pricer directly addresses Ticket T011 by reweighting the $12 \times 12$ scoreline grid anti-diagonals (where total goals $G = h + a$) to match the spine's marginal totals CDF before Kelly optimization.

## Acceptance Criteria

- [x] Implement prototype in `current_development/grw_smile_spine/`:
  - `l01_loader.jl`: 1-parameter `MarketSmileSpinePillar` ($\beta$ scalar parameter), log-density, and anti-diagonal grid reweighting.
  - `r01_smoke.jl`: 2-fold smoke gate verifying ReverseDiff compiled tape against ForwardDiff ($\le 10^{-6}$), exact gradients, parameter recovery, and anti-diagonal grid reweighting accuracy.
  - `r02_production_grid.jl`: 43-fold walk-forward grid for `m05_joint_grw_smile_spine_w020` and `m05_joint_grw_smile_spine_w040` on `mcmc-beast` (-t 16).
  - Benchmark computation time and bulk/tail ESS per fold against Task 015's 5-parameter model.
  - `r04_evaluate.jl`: Proper scores (LogLoss, Brier, RPS, ECE) vs Betfair close and Task 015 rungs over the 710 walk-forward matches.
  - `r06_portfolio.jl`: Closing-line Option B portfolio with anti-diagonal reweighted staking (resolving T011).
  - `r07_t25_portfolio.jl`: T−25 tradeable portfolio with and without Option B L2 calibrator (`scot_lower_t25_inv`).
  - `r08_trust_sweep.jl`: Market expansion test evaluating `Under 1.5` and `Under 4.5`.
  - `README.md`: Comprehensive documentation.
- [x] Model Ablation Ladder (Scottish Lower, 43 folds):
  1. `m05_joint_grw_baseline`: Pure GRW control (Task 013 `b0961bc4`)
  2. `m05_joint_grw_supremacy_w040`: Supremacy anchor only (Task 015 `0ee58d18`)
  3. `m05_joint_grw_smile_supremacy_w020`: Full 5-parameter smile @ 0.20 (Task 015 `fcd5e974`)
  4. `m05_joint_grw_smile_supremacy_w040`: Full 5-parameter smile @ 0.40 (Task 015 `30620d3e`)
  5. `m05_joint_grw_smile_spine_w020`: 1-param spine @ 0.20 (new sampling)
  6. `m05_joint_grw_smile_spine_w040`: 1-param spine @ 0.40 (new sampling)
- [x] 6-part convergence audit passed on all 43 folds ($\hat{R} \le 1.05$, 0 divergences, ESS $\ge 400$).
- [x] Store fits in `PostgresStorage("scottish_lower_grw_smile_spine")`.

## Ideas & Candidate Solutions

- **Anti-Diagonal Grid Reweighting (T011 Fix)**: Given joint scoreline probabilities $P_{\text{grid}}(h, a)$, total goals $G = h + a$ has marginal $P_{\text{grid}}(G) = \sum_{h+a=G} P_{\text{grid}}(h, a)$. The spine defines $P_{\text{spine}}(G \le K) = \text{cdf}(\text{Poisson}(\lambda_{\text{tot}} \cdot e^{\beta(K-2)}), K)$. Rescaling anti-diagonals by $P_{\text{spine}}(G) / P_{\text{grid}}(G)$ produces a valid $12 \times 12$ joint scoreline distribution whose totals marginal matches the smile spine, enabling coherent Kelly optimization.
- **Computation Time Reduction**: Dropping from 5 strike parameters to 1 scalar parameter $\beta$ simplifies the NUTS Riemannian geometry and should restore sampling speed close to the ~60-minute supremacy baseline.

## Work Log & Progress

- [2026-09-13 @antigravity] Aligned task via `/grill-me`. Created Task 016, initialized worktree at `/home/james/bet_project/.worktrees/BayesianFootball-grw-smile-spine` on branch `feat/grw-smile-spine`, prepared work package prompt `SMILE_SPINE_WORK_PACKAGE.md`, and launched `claude_smile_spine` in tmux.
- [2026-09-13 @claude] Phase 1 authored, NOT yet executed: `current_development/grw_smile_spine/l01_loader.jl` (includes Task 015's loader unchanged; new `SpineAnchoredCountModel` + `MarketSmileSpinePillar`, site `β_spine`; anti-diagonal reweighting `gss_reweight_grid!` and `gss_build_books_reweighted`) and `r01_smoke.jl` (gates GA, GB, G4a pre-sampling, G0a/b/c, G1, G6, G2, G3, G4b, G4c, G5). Work-package interpretations recorded in the loader header: all 23 anti-diagonals of the 12×12 grid are rescaled (not 12); mass above K = 4 follows the grid's own diagonal proportions; a non-monotone smile draw is refused, not clipped; φ ≡ 1 draws are left bit-identical. G0c added: spine ≡ five-strike model on the line log φ = β(K−2). Smoke budget 4×(500+1000) because Task 015's smoke failed its baseline on tail ESS at 500 draws. Caveat: the line through Task 015's φ medians (β_LS ≈ 0.052) misses K = 0 by ≈ −0.065 in log φ.
- [2026-09-13 @claude] Committed `0286ebd2`, cloned to mcmc-beast `/root/BF_grw_smile_spine` (Task 015's ScottishLower cache of 2026-09-12, `.env` and pinned `Manifest.toml` copied from `/root/BF_grw_market_smile`; Distributions 0.25.126). First r01 launch stopped at G4a before sampling: with φ ≡ 1 and the identity shortcut disabled the reweighted grid moved by 2.23e-4 against a fixed 1e-6 tolerance. Cause is the grid's own truncation mass (goals ≥ 12 per side, ~0.1–0.2% at the synthetic 4.0 rates), which the Σ = 1 reweighting relocates onto totals ≥ 5; shortcut bit-identity, totals-CDF match, mass and non-monotone refusal all passed. With user approval the check now bounds every cell's move by that draw's truncation mass (+1e-14 float slack), a derived bound; other G4 tolerances unchanged.
- [2026-09-13 @claude] r01 smoke PASSED 2/2 at `20b5d18` (mcmc-beast, 16 threads, 4 × (500 + 1000), folds 1–2; report `current_development/grw_smile_spine/results/smoke/4x500w1000s/r01_smoke_report.md`). Run IDs in `smoke_grw_smile_spine`: spine @0.20 `59d3adf2-5bc1-40b1-b89e-8b1934e16335`, @0.40 `ecf43154-8788-4dae-bbdc-5619de431a91`. See Verification & Findings.
- [2026-09-14 @claude] r02 production grid PASSED 2/2 at `d4cceb8` (mcmc-beast, Julia 1.12.4, 16 threads, 43 folds, 769 OOS; report `current_development/grw_smile_spine/results/r02_production_report_4x500w1000s_*.md`). Pinned rungs 1–4 loaded and asserted identical, converged and full-draw audited before sampling; all four were fitted on Julia 1.12.4 / 16 threads. Runs in `scottish_lower_grw_smile_spine`: spine @0.20 `eaf53852-a078-4190-b744-089966a306f6`, @0.40 `582035c0-e145-44f7-9f40-89e25388e79a`. H1 as stated NOT supported (wall ≤ 90 min and min bulk ESS ≥ 600 missed at both weights); H2 supported. See Verification & Findings. First launch attempt did not start: the beast pull refused to overwrite the 10 untracked smoke CSVs, which were byte-identical to the committed copies and were removed. Next: r04, r06, r07, r08 not yet written.

## Verification & Findings

### r01 smoke (folds 1–2, 4 × (500 + 1000)) — PASS 2/2, `20b5d18`

Smoke evidence only; the 43-fold grid (r02) decides H1–H2.

* **G4a** (synthetic, pre-sampling): reweighted totals = smile CDF and Σ = 1 to float precision; φ ≡ 1 shortcut bit-identical; un-shortcut φ ≡ 1 path max |Δ| 2.23e-4 within the per-draw truncation-mass bound (max 1.67e-3, worst excess −8.9e-15); non-monotone curve refused.
* **G0**: null anchor Δ = 0.0 on both folds; spine − base vs independent re-derivation worst rel 1.2e-15; spine ≡ five-strike on the line log φ = β(K−2) worst abs 9.1e-13 (0.0 on three of four).
* **G1**: RD/FD ≤ 6.9e-16, compiled tape exact under perturbation. Per gradient the spine costs the same as the five-strike smile (tape 844 vs 843 fold 1, 0.164–0.168 vs 0.172 ms) and allocates ~93 KB MORE per call (488 vs 395 KB). Any H1 gain must come from sampler geometry, not gradient cost.
* **G2/G3 vs Task 015 smoke at the same budget, host, threads and store**:

| rung | wall | max R̂ | min ESS bulk / tail | div |
|---|---:|---:|---:|---:|
| Task 015 five-strike smile @0.40 | 3.79 min | 1.0064 | 866 / 1383 | 0 / 8000 |
| spine @0.20 | 2.83 min | 1.0065 | 1223 / 1802 | 0 / 8000 |
| spine @0.40 | 3.52 min | 1.0051 | 1544 / 1984 | 0 / 8000 |

  Like for like (@0.40): min bulk ESS +78%, wall −7%.
* **H2 early signal**: β_spine median 0.0528–0.0529 on every rung × fold (sd 0.0012–0.0018 vs prior sd 0.05), weight-independent, equal to the least-squares slope through Task 015's medians (0.0525), slightly above the hypothesised 0.03–0.05. φ = 0.900 / 0.949 / 1.000 / 1.054 / 1.111 against the five-strike 0.843 / … / 1.069: the spine under-bends K = 0 and over-bends K = 4. σ_smile 0.062–0.064 (five-strike smoke 0.053); κ 1.097–1.100 (five-strike smoke 1.131).
* **G4b**: fitted reweighted totals = smile CDF ≤ 4.4e-16, mass ≤ 6.7e-16; O/U through typed and legacy routes = reference ≤ 1e-12. Reweighting moves 1X2: mean Δp_home −0.33 pp on both rungs; Δp_under25 = 0 (φ(2) ≡ 1).
* **G4c (T011)**: φ ≡ 1 twin ledger bit-identical to the grid twin 39/39 on both rungs; staked p_grid totals = smile CDF ≤ 3.9e-15; φ now changes the stake on 38/39 (@0.20) and 37/39 (@0.40) fixtures, where Task 015 measured zero.
* **G5/G6**: registry and PostgreSQL round-trip (T010 detached-latent path) passed.

Environment note: the beast ran Julia 1.12.4 (juliaup default) against a Manifest recording 1.12.1; pins Distributions 0.25.126, Turing 0.41.4, DynamicPPL 0.38.10, ReverseDiff 1.17.0, MCMCChains 7.7.0.

### r02 production grid (43 folds, 4 × (500 + 1000), 769 OOS) — PASS 2/2, `d4cceb8`

All six rungs fitted on mcmc-beast, Julia 1.12.4, 16 threads; wall times are comparable.

| rung | wall | max R̂ (fold) | min ESS bulk / tail | div | BFMI | run |
|---|---:|---:|---:|---:|---:|---|
| baseline (Task 013, pinned) | 42 min | 1.0115 | 609 / 616 (thinned re-audit) | 0 | 0.689 | `b0961bc4` |
| supremacy @0.40 (Task 015, pinned) | 61 min | 1.0105 | 814 / 516 | 0 | 0.683 | `0ee58d18` |
| five-strike smile @0.20 (Task 015, pinned) | 158 min | 1.0139 (34) | 472 / 498 | 0 | 0.623 | `fcd5e974` |
| five-strike smile @0.40 (Task 015, pinned) | 185 min | 1.0200 (39) | 431 / 696 | 0 | 0.665 | `30620d3e` |
| **spine @0.20** | **132 min** | 1.0135 (33) | 482 / 854 | 0 | 0.653 | `eaf53852-a078-4190-b744-089966a306f6` |
| **spine @0.40** | **176 min** | 1.0139 (43) | 531 / 922 | 0 | 0.652 | `582035c0-e145-44f7-9f40-89e25388e79a` |

**H1 (efficiency) — not supported as stated.** Neither target is met at either weight (≤ 90 min; min bulk ESS ≥ 600). Against the five-strike rung at the same weight: wall −16% (@0.20) and −5% (@0.40); run-level min bulk ESS +2% and +23%; tail ESS +71% and +32%. The pillar weight costs more than the shape dimension (spine @0.20 → @0.40 adds 44 min) and the smile pillar still costs 2–3× the supremacy-only wall time, so the drag is the pillar's coupling to the team state, not the number of φ parameters.

Fold by fold (both audited on all 4,000 draws; the pinned baseline is excluded because its folds were re-audited on thinned chains):

| pair | median per-fold min bulk ESS, spine / five-strike | median ratio | worst fold ratio | folds spine lower | folds < 600, spine / five-strike |
|---|---:|---:|---:|---:|---:|
| @0.20 | 857 / 1097 | 0.83 | 0.38 | 32 / 43 | 5 / 4 |
| @0.40 | 1087 / 678 | 1.66 | 0.67 | 6 / 43 | 3 / 13 |

The @0.20 deficit is driven by the reference, not by poor spine mixing: the five-strike model mixes 1.55× better at @0.20 than at @0.40 (median per fold), the spine 0.78×. On the eight worst @0.20 folds (14, 20, 18, 6, 16, 19, 17, 37) the five-strike rung reaches 953–2,215 and the spine 527–978. The spine's per-fold mixing is the more consistent of the two across weights; it is not uniformly better.

**H2 (β recovery) — supported.** β_spine 0.0524 pooled at both weights ([0.0499, 0.0550] @0.20, [0.0507, 0.0543] @0.40); fold medians 0.0519–0.0530 over 43 folds; per-fold sd ≤ 0.0018; β itself mixes well (per-fold ESS ≥ 3,258 bulk / 1,870 tail, R̂ ≤ 1.005) — the worst-mixing site is elsewhere. Weight-independent; equals the least-squares slope through Task 015's medians (0.0525), just above the hypothesised 0.03–0.05. φ = 0.900 / 0.949 / 1.000 / 1.054 / 1.111 vs five-strike 0.843–0.844 / 0.976 / 1.001 / 1.026 / 1.069: the spine under-bends K = 0 and over-bends K = 4. σ_smile 0.062 / 0.060 (five-strike 0.052 / 0.050); κ 1.094 / 1.092 (five-strike 1.116 / 1.115); σ_sup 0.236 / 0.219 (five-strike 0.237 / 0.220).

**Pricer and T011 on the persisted 769-fixture containers (both rungs):** reweighted totals = smile CDF ≤ 4.4e-16, Σ = 1 ≤ 6.7e-16; φ ≡ 1 shortcut bit-identical, un-shortcut path ≤ 1.04e-5 within the truncation-mass bound (≤ 5.94e-5). Reweighting shifts mean 1X2 by draw +0.67 pp, home −0.34 pp, away −0.33 pp — relevant to H4, since Task 015's portfolio gain came from 1X2 away bets.

Provenance: Task 015's three runs recorded `git_commit = unknown` (rsynced checkout); the baseline `037d651c-dirty`.

### r04 proper scores (710-fixture panel, Betfair TWA(−20, 0] close) — `1d9da5e`

Reproduction gate passed EXACTLY: pinned baseline LogLoss 0.64315 / ECE 0.0123 on 2,899 rows, equal to Task 013's published figures. 14 of 96 contrasts resolved. Report `results/evaluation/r04_evaluation_report.md`.

**H3 (predictive parity) — NOT supported as stated, and not refuted either: the pooled test is underpowered.** Spine − five-strike at the same weight, pooled over 1X2 + O/U 2.5 + BTTS: Δ LogLoss −0.00012 (@0.20) and −0.00014 (@0.40), intervals ±0.001 containing 0. But on the same panel the control contrast spine − baseline (Δ −0.00211 / −0.00237) is ALSO unresolved, so the pooled basis cannot resolve a difference we know to be real. "Indistinguishable" here is ignorance, not parity. This is what the runner's paired-control readout was built to expose.

**The ends ARE resolved, and they move in OPPOSITE directions — the pooled zero is a cancellation, not an absence.**

| contrast (spine − five-strike, same weight) | Δ LogLoss | 95% interval | verdict |
|---|---:|---|---|
| O/U 0.5 | −0.0046 / −0.0048 | excludes 0 | spine better |
| O/U 2.5 | −0.0008 | includes 0 | unresolved |
| O/U 4.5 | +0.0057 / +0.0055 | excludes 0 | spine worse |

**The strike ladder (UNDER selection) corrects a prediction I made before seeing it.** I expected the spine to lose at K = 0 because its line cannot bend down to φ₀ = 0.843. The direction is the opposite, because both smile arms over-price deep Unders and the five-strike over-prices them MORE:

| line | n | market | baseline | spine | five-strike | realised |
|---|---:|---:|---:|---:|---:|---:|
| Under 0.5 | 149 | 0.0679 | 0.0718 | 0.0893 | 0.0994 | **0.0336** |
| Under 1.5 | 215 | 0.2305 | 0.2423 | 0.2715 | 0.2478 | 0.2279 |
| Under 2.5 | 379 | 0.4702 | 0.4784 | 0.4917 | 0.4784 | 0.4987 |
| Under 3.5 | 264 | 0.6859 | 0.6841 | 0.6733 | 0.6777 | 0.6932 |
| Under 4.5 | 104 | 0.8578 | 0.8397 | 0.8078 | 0.8168 | **0.9038** |

* **The spine's K = 0 win is less damage, not an improvement.** At Under 0.5 the BASELINE (0.15797) beats the market (0.15574) least badly of the models, and both smile arms are far worse (spine 0.17215, five-strike 0.17912). The smile pillar hurts this line; one parameter hurts it less. Same at Under 1.5, where the baseline is best (0.52070) and the spine worst (0.53207).
* **The smile pillar earns its keep at K = 2–3 only**: at Under 2.5 both smile arms beat the baseline and the market; at Under 3.5 likewise.
* **Consistent with `eda/README.md`'s Jensen tail inflation** — deep Unders systematically over-priced — rather than with a defect in the spine: the anchor is a single global φ per strike against per-fixture market intensities, and E[e^{−Λ}] ≥ e^{−E[Λ]} bites hardest at K = 0. Mechanism not proven here; the measured over-pricing is.
* **Every model beats the market's Under 4.5 LogLoss by ~0.30** (0.344–0.352 vs 0.647) while the market's mean price there (0.8578) looks sane against a 0.9038 realised rate — a handful of thin/mispriced deep quotes dominating that column. Treat the K = 4 market column as unreliable; the model-vs-model contrast on the same rows is unaffected.

**Testable prediction for r08/H5, recorded before the sweep ran**: at Under 1.5 the spine sees a +4.1 pp edge (0.2715 vs market 0.2305) where the realised rate is 0.2279, so `+U1.5` should LOSE and lose more than the five-strike (+1.7 pp) or baseline (+1.2 pp); at Under 4.5 the spine prices below the market so it should decline the bet rather than profit.

Caveat: the three Task 015 arms show `file copy none` — their latent file copies live in the Task 015 checkout, so for those arms the panel rebuilt from persisted chains was verified against the chains but not against a second on-disk copy. Both spine arms verified both ways.

### r06 / r07 / r08 portfolio (Option B; close 632-fixture and T−25 611-fixture panels) — `c992ada`

Full tables and mechanism in [`current_development/grw_smile_spine/README.md`](../current_development/grw_smile_spine/README.md) §6–§8. Every reproduction gate passed exactly: r06 P1 baseline +385.78% / ROI 11.68% / 1,247 bets; r07 T1 close/raw vs r06 worst |Δ| 0.00e+00 over 6 arms; r07 T2 baseline raw +531.78% / 1,124 and calibrated +245.85% / 969; r08 S1 all three arms.

**Ticket T011 measured and fixed in the prototype.** Under the old route the reported price is exact (≤ 1.9e-15) while the distribution the Kelly solve read differs from the smile by **4.71–7.06e-02** — 4.7–7.1 pp of P(total ≤ K) against T011's ≤ 1e-9 criterion. Correcting it changes 11–16% of each ledger (142–215 exclusive bets per arm). Task 015's "φ changes the reported price and nothing else" is false once stakes are solved correctly. The `:grid` rows reproduce Task 015's published +588.4% / +545.0% at ROI 15.71 / 15.82, which establishes the difference is the correction and not a builder bug. The defect is LARGER for the spine (7.0e-2 vs 4.7e-2) because φ₄ = 1.111 is further from 1 than the five-strike's 1.069.

**H4 — half met.** Away-bet ROI target (+19%) MET: 19.44% (@0.20) / 20.88% (@0.40) vs baseline 7.77%. Flat-ROI target (15.8%) NOT met: 14.30% / 14.54%, with Sharpe 0.18 lower and worse drawdown than the five-strike arms (15.69% / 15.72%, Sharpe 1.611 / 1.640). Mechanism: the five-strike prunes totals harder (199 bets at 12.67% ROI, 12.4% of stake) where the spine keeps 237 at 7.19% and 22.0% of stake — the portfolio consequence of a curve that cannot bend. The spine still beats both no-smile controls; the linear restriction is what costs money.

**H5 — no, twice over.** (a) Calibration dominates every pillar choice: L2 halves drawdown (−42% → −17/−22%) and lifts ROI to 17–22%. (b) **Dropping φ after calibration beats keeping it for every smile arm** (five-strike @0.40: ROI 22.10% vs 19.01%, return +223.7% vs +155.8%; p = 0.048, the nearest thing to a resolved contrast in r07) — the pillar's value is in how it shaped the RATES during fitting, not in the φ curve used at pricing time. (c) Under 1.5 and Under 4.5 are accretive for the BASELINE and destroyed by the smile: at the close `+U1.5` gives baseline 105 bets at +37.66% ROI, five-strike 128 at −2.73%, spine 208 at −6.31% (net −91.6 pp) — the bet-count ordering is exactly the phantom-edge ordering r04 predicted and the ROI ordering is its reverse; `+U4.5` sees the smile arms take 0–2 bets where the baseline takes 28 at +18.79%. Only 21/66 addition rows improve terminal return, so `eda/README.md`'s pruning is broadly vindicated — but its capacity-cannibalisation mechanism is NOT what bites here (core ΔROI −0.16 to +0.35 pp); the damage is the added bets' own ROI.

**Spine vs five-strike does not resolve.** The sign flips by environment (spine loses at the close, wins at T−25 raw, loses again calibrated-with-φ-dropped) and every paired slate-growth interval spans zero (p 0.18–0.79). At ~600 fixtures the two arms are not separable; no ranking is claimed.

Persisted portfolios: spine @0.20 `3bd3a461-c505-4ceb-822c-ce1623e0aaf7`, @0.40 `785b5d7f-f596-491b-8442-9d71b5a350e6`, both reloading identical ledgers.

### Defect raised, not fixed inline

[T012](../docs/tickets/T012-zero-trust-market-reprices-the-portfolio.md) — declaring a market in the `BookSpec` widens the payoff matrix and hence `BakerMcHale`'s per-fixture `k`, so a market at **trust 0** still moves every stake: r08's S0 gate failed 6/6 (3 arms × 2 environments) by −22.3 to +10.2 pp of terminal return with unstable sign. Task 015 saw this for one arm and worked around it; this is the generalised measurement. r08 neutralises it by measuring every addition against P0 on the same extended book.

### Recommendation

Do not graduate the spine as a replacement for the five-strike smile. Graduate the **anti-diagonal reweighting** on its own merits (T011 option 2, implemented and measured here) — it fixes a 4.7–7.1 pp incoherence for any smile container including Task 015's. The obvious next experiment is "fit with the smile pillar, price without φ", which §8's calibration result points at directly.
