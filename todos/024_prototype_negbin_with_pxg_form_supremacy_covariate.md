# 024 — Prototype Negative Binomial with Linear Proxy-xG Form Covariate

| Field | Value |
|---|---|
| ID | 024 |
| Title | Prototype Negative Binomial with Linear Proxy-xG Form Covariate |
| Status | COMPLETED |
| Priority | P1 |
| Assignee | pi |
| Created | 2026-09-22 |
| Updated | 2026-09-22 |
| Related Files / Commits / PRs | `current_development/market_inverse_dynamics/PHASE2_FEATURE_ATTRIBUTION.md`, `experiments/scottish_lower/11_decompression_pxg_covariate/`, `f866b2bc-d87b-42d0-8c36-8c8d623d1178` |

## Context & Problem Statement

In Generation 3 (`JointGammaPoissonObservation`) and Generation 4 (`m12_joint_hybrid_synergy`), proxy xG enters via a secondary Gamma observation arm that shares the latent team rating ($\mu_s$). Because $\mu_s$ is regularized by slow-moving hierarchical shrinkage priors and random walks, the model severely dampens the fast chance-creation signal for dominant teams.
Regressing market supremacy on `m05` and `m12` yields reverse slopes of **1.41 to 1.66**, proving severe under-scaling (compression on favourites). In contrast, pure-goal random walk models (`m01`) exhibit an empirical slope of **1.07** (well-scaled, but low-information).

Phase 2 of Market-Inverse Dynamics (`current_development/market_inverse_dynamics/PHASE2_FEATURE_ATTRIBUTION.md`) established:
1. Rolling proxy-xG form alone explains **28.6% of market supremacy variation** (32.1% among well-identified fixtures) and **30.7% of the conviction gap** between market favourites and `m01`.
2. The empirical market pricing weight on rolling proxy-xG advantage is **$+0.48$ to $+0.79$ log-rate supremacy per 1.0 expected goal advantage**.
3. Passing proxy xG through team-level shrunk latents dilutes the signal. Moving proxy-xG form into the linear predictor allows $w_{\text{pxg}}$ to be estimated as a **globally pooled parameter** across all 710 matches, completely bypassing team-level shrinkage.

Option 1 formulates a Negative Binomial goal engine where proxy-xG form enters directly as an antisymmetric supremacy covariate:
$$\log \lambda_{h} = \mu + \gamma_{\text{home}} + \alpha_{\text{att}, h} + \beta_{\text{def}, a} + \frac{1}{2} w_{\text{pxg}} \cdot (\text{pxg\_form}_h - \text{pxg\_form}_a)$$
$$\log \lambda_{a} = \mu + \alpha_{\text{att}, a} + \beta_{\text{def}, h} - \frac{1}{2} w_{\text{pxg}} \cdot (\text{pxg\_form}_h - \text{pxg\_form}_a)$$
$$y_h \sim \text{NegBin}(\lambda_h, \phi), \quad y_a \sim \text{NegBin}(\lambda_a, \phi)$$
with prior $w_{\text{pxg}} \sim \mathcal{N}(0.60, 0.20^2)$ informed by the market-inverse attribution study.

Primary scope: Scottish Lower (tournaments 56/57, seasons 24/25 + 25/26, 40-fold walk-forward grid, 710 matches).

## Acceptance Criteria

- [x] **Stage 0 (Mathematical Formulation & Covariate Design)**:
  - Implement `ProxyXGFormCovariate` in `current_development/scottish_decompression/` (or `experiments/scottish_lower/11_decompression_pxg_covariate/`).
  - Compute rolling proxy-xG form using `Features._pxg_rolling_lookup` over an asymmetric window (e.g. 10–16 matches) with zero future leakage.
  - Formulate antisymmetric supremacy scaling in `CountModelBuilder` and verify ReverseDiff tape compilation and warmed execution allocate 0 B.
- [x] **Stage 1 (Smoke Gate, Folds 1/20/40)**:
  - 4 chains $\times$ 400 warmup + 400 draws on folds 1, 20, 40.
  - Zero NUTS divergences; $\hat{R} \le 1.05$; bulk/tail ESS $\ge 200$.
  - Verify $w_{\text{pxg}}$ posterior concentrates away from zero in the range $[0.40, 0.80]$.
  - Latent extraction, score-grid construction, and `save_fit`/`load_fit` round-trips pass.
- [x] **Stage 2 (40-Fold Walk-Forward Grid on Beast)**:
  - Execute full walk-forward grid (40 folds, 710 fixtures) on `mcmc-beast`.
  - Control arms:
    - `m01_poisson_time_decay`: Baseline Poisson with time decay
    - `m02_joint_gamma_poisson`: Canonical Gen 3/4 two-arm joint observation (the compressed benchmark)
    - `m03_negbin_pxg_covariate`: Candidate Negative Binomial + linear proxy-xG form covariate
  - Persist runs to PostgreSQL `mcmc_experiments` (namespace `scottish_lower_decompression`).
- [x] **Stage 3 (Evaluation, Scaling & Portfolio Benchmark)**:
  - Regress market supremacy on model supremacy ($y = \beta x$); verify $\beta$ decompresses from $1.41\text{--}1.66$ down towards $1.00 \pm 0.15$.
  - Evaluate proper scores (1X2, O/U 2.5, BTTS LogLoss, CRPS, RPS, ECE) vs Betfair closing odds.
  - Full portfolio simulation on the common tradeable panel with `BookSpec(1X2, OU2.5, BakerMcHale)` and `PolicySpec(FlatTrust(0.25), SlateDrawdown(20.0), FixedCap(0.25))`.
- [x] **Stage 4 (Findings Report & Sign-off)**:
  - Deliver findings in `experiments/scottish_lower/11_decompression_pxg_covariate/README.md`.
  - Validate `./scripts/todo.sh check` and `git diff --check`.

## Ideas & Candidate Solutions

- **Antisymmetric vs Attack/Defence Splitting**: Form can be applied symmetrically to team supremacy ($\pm \frac{1}{2} w_{\text{pxg}} \Delta \text{pxg}$) or decomposed into separate attacking chance creation ($\text{pxg}_{\text{for}}$) and defensive chance concession ($\text{pxg}_{\text{against}}$). Symmetrical supremacy is more parsimonious and has a higher signal-to-noise ratio.
- **Negative Binomial Overdispersion**: Using NegBin overdispersion parameter $\phi \sim \text{Exponential}(1.0)$ or $\text{Gamma}(2, 0.1)$ absorbs match-level variance without needing the complex two-arm joint likelihood.
- **Lookback Window Tuning**: Test exponential decay weighting vs rolling 10-match or 16-match boxcar window. Phase 2 showed a 16-match window has $R^2 = 0.54$ against market prices.

## Work Log & Progress

- [2026-09-22 @antigravity] Created specification from market-inverse Phase 2 empirical findings. Allocated to `@pi` using model `openai-codex/gpt-5.6-sol` in dedicated worktree `.worktrees/BayesianFootball-negbin-pxg-covariate`.
- [2026-09-22 @pi] Implemented the exact half-side antisymmetric coefficient, commentary-only 16-match-half-life feature, allocation-free linked-space NegBin engine, deterministic filtration tests, three staged runners, persistence, evaluation and read-only CSV audit.
- [2026-09-22 @pi] Completed smoke and production on `mcmc-beast`: 40 folds / 710 OOS fixtures per arm, 0/128,000 divergences per arm, every R̂/ESS/BFMI/depth gate passed, and all fits round-tripped through `scottish_lower_decompression`.
- [2026-09-22 @pi] Completed the common-panel benchmark and report. The candidate decompressed the reverse slope from 1.7240 to 1.3023 but missed the 0.85–1.15 target and materially underperformed both controls in the portfolio, so it is retained as a research prototype rather than promoted.

## Verification & Findings

- Candidate compiled-tape replay: 0 B on folds 1/20/40; max linked density difference 3.64e-12 and max gradient relative error 2.87e-12 against the composable reference across warmup-scale perturbations.
- Smoke candidate: max R̂ 1.0142, min bulk/tail ESS 754/618, zero divergences; fold posterior means for `w_pxg` 0.780/0.552/0.658 with positive 5th percentiles.
- Production candidate run: `f866b2bc-d87b-42d0-8c36-8c8d623d1178`; 40 folds, 710 fixtures, 3,200 draws/fixture, max R̂ 1.01005, min bulk/tail ESS 1,189/1,258, zero divergences.
- Decompression: market-on-model slope 1.3023 (`m02` 1.7240; `m01` 2.5288), R² 0.6228, favourite probability 56.48% versus market 76.25%.
- Predictive/portfolio result: candidate all-selection LogLoss 0.644755 versus joint 0.643730; return +83.32%, Sharpe 1.057 versus joint +127.97%, 1.270. The candidate does not qualify for production.
- Full protocol, caveats, UUIDs and evidence: `experiments/scottish_lower/11_decompression_pxg_covariate/README.md` and its `verification/` directory.
