# Negative Binomial + Linear Proxy-xG Form Covariate (Scottish Lower Decompression)

> **Work Package**: TODO 024 — Prototype Negative Binomial with Linear Proxy-xG Form Covariate  
> **Target**: Scottish Lower League Football (Tournaments 56 & 57, 40-fold walk-forward cohort, seasons 24/25 + 25/26, 710 fixtures)  
> **Harness**: Pi Solo Agent (`openai-codex/gpt-5.6-sol` with `--thinking high`) in tmux session `agent_pi_negbin_pxg`  
> **Worktree**: `/home/james/bet_project/.worktrees/BayesianFootball-negbin-pxg-covariate`  
> **Branch**: `feat/scottish-lower-negbin-pxg-covariate`  
> **Compute Node**: `mcmc-beast` (AMD Ryzen 9 32-core node, directory `/root/BF_negbin_pxg_covariate`)  

---

## 1. Executive Context & Motivation

Scottish Lower football models suffer from severe **Bayesian shrinkage compression**:
- In Generation 3 (`JointGammaPoissonObservation`) and Generation 4 (`m12_joint_hybrid_synergy`), proxy xG enters via a secondary Gamma observation arm sharing the latent team rating ($\mu_s$).
- Because $\mu_s$ is regularized by slow-moving hierarchical shrinkage priors and random walks, the model severely dampens the fast chance-creation signal for dominant teams.
- Regressing market supremacy on `m05` and `m12` yields reverse slopes of **1.41 to 1.66**, proving severe under-scaling (compression on favourites). In contrast, pure-goal random walk models (`m01`) exhibit an empirical slope of **1.07** (well-scaled, but low-information).

### Grounded Empirical Evidence from Market-Inverse Phase 2
Phase 2 of Market-Inverse Dynamics (`current_development/market_inverse_dynamics/PHASE2_FEATURE_ATTRIBUTION.md`) established:
1. **Rolling proxy-xG form alone explains 28.6% of market supremacy variation** (32.1% among well-identified fixtures) and **30.7% of the conviction gap** between market favourites and `m01`.
2. The empirical market pricing weight on rolling proxy-xG advantage is **$+0.48$ to $+0.79$ log-rate supremacy per 1.0 expected goal advantage**.
3. Passing proxy xG through team-level shrunk latents dilutes the signal. Moving proxy-xG form into the linear predictor allows $w_{\text{pxg}}$ to be estimated as a **globally pooled parameter** across all 710 matches, completely bypassing team-level shrinkage.
4. The Negative Binomial likelihood natively absorbs match-to-match overdispersion without needing an extra observation arm.

### The Breakthrough Formulation
$$\log \lambda_{h} = \mu + \gamma_{\text{home}} + \alpha_{\text{att}, h} + \beta_{\text{def}, a} + \frac{1}{2} w_{\text{pxg}} \cdot (\text{pxg\_form}_h - \text{pxg\_form}_a)$$
$$\log \lambda_{a} = \mu + \alpha_{\text{att}, a} + \beta_{\text{def}, h} - \frac{1}{2} w_{\text{pxg}} \cdot (\text{pxg\_form}_h - \text{pxg\_form}_a)$$
$$y_h \sim \text{NegBin}(\lambda_h, \phi), \quad y_a \sim \text{NegBin}(\lambda_a, \phi)$$
with prior $w_{\text{pxg}} \sim \mathcal{N}(0.60, 0.20^2)$ informed by the market-inverse attribution study.

---

## 2. Experimental Scope & 3-Arm Benchmark

The matched cohort is the standard Scottish Lower 40-fold walk-forward grid over seasons 24/25 and 25/26 (710 fixtures, 2,899 scored market observations).

### The 3 Arms:
1. **`m01_poisson_time_decay` (Control 1)**:
   - Baseline Poisson with time-decay dynamics (`days_half_life = 180.0`).
2. **`m02_joint_gamma_poisson` (Control 2)**:
   - Canonical Gen 3/4 two-arm joint observation (the compressed benchmark, slope 1.41–1.66).
3. **`m03_negbin_pxg_covariate` (Candidate)**:
   - Negative Binomial likelihood + linear antisymmetric proxy-xG form covariate (`w_pxg`).

---

## 3. Agent Operating Rules (CRITICAL)

- **SOLO AGENT EXECUTION**: You are operating as a solo agent with `openai-codex/gpt-5.6-sol` and `--thinking high`. **Do NOT invoke subagents.** Execute all mathematical modeling, Julia/Turing coding, command execution, and verification directly.
- **Deep Mathematical Thinking**: Derive clean, numerically stable, AD-safe Turing components.
- **ReverseDiff AD Performance & Zero Allocations**:
  - The model runs under compiled ReverseDiff gradient tapes.
  - Zero heap allocations in the inner likelihood evaluation.
  - Set `BLAS.set_num_threads(1)` and `ThreadPinning.pinthreads(:cores)`.
- **Database Separation**:
  - `betdb` operational data is on `archpc:5433` (via `BF_DB_URL`).
  - `mcmc_experiments` is on `mcmc-beast:5432` (via `BF_EXPERIMENTS_DB_URL`).
  - Save completed production fits to PostgreSQL namespace `scottish_lower_decompression`.

---

## 4. Execution Stages

### Stage 0: Covariate Extraction & Model Architecture
- Implement `ProxyXGFormCovariate` in `l11_decompression_loader.jl`.
- Rolling proxy-xG calculation: read proxy xG from `bbc.live_text` or `Features._pxg_rolling_lookup` over an asymmetric window (10–16 matches) with zero future leakage.
- Assemble `CountModelBuilder` with `NegativeBinomialObservation()`, `GlobalInterception()`, `GlobalHomeAdvantage()`, `TimeDecayDynamics()`, and `ProxyXGFormCovariate()`.
- Warmed compiled-gradient replay must allocate 0 B.

### Stage 1: Smoke Gate (Folds 1, 20, 40)
- File: `r10_smoke.jl`.
- Budget: 4 chains $\times$ (400 warmup + 400 samples).
- Gates:
  - G1: ReverseDiff gradient tape compiles and evaluates without errors.
  - G2: 0 NUTS divergences across all smoke chains.
  - G3: Max $\hat{R} \le 1.05$ (advisory $\le 1.01$).
  - G4: Bulk & tail ESS $\ge 200$.
  - G5: Score-grid book sums to $1.0$ within machine precision ($\le 1e-12$).
  - G6: Posterior of $w_{\text{pxg}}$ is well-identified and positive ($[0.40, 0.80]$).

### Stage 2: 40-Fold Walk-Forward Production Grid on Beast
- Sync code to `mcmc-beast` (`git push` / `git -C /root/BF_negbin_pxg_covariate pull`).
- File: `r20_production_grid.jl`.
- Execute all 40 folds across the 3 arms on `mcmc-beast`.
- Persist runs to PostgreSQL `mcmc_experiments` in namespace `scottish_lower_decompression`.

### Stage 3: Evaluation, Scaling & Portfolio Benchmark
- File: `r30_evaluation.jl`.
- **Supremacy Slope Regression**: Regress market supremacy on model supremacy ($y = \beta x$). Verify whether $\beta$ decompresses from $1.41\text{--}1.66$ down towards $1.00 \pm 0.15$.
- **Proper Scores**: Compare selection-level binary LogLoss, CRPS, RPS, ECE vs Betfair closing odds.
- **Portfolio Simulation**: Run full portfolio simulation on the common tradeable panel with `BookSpec(1X2, OU2.5, BakerMcHale)` and `PolicySpec(FlatTrust(0.25), SlateDrawdown(20.0), FixedCap(0.25))`.

### Stage 4: Findings Report & Sign-off
- Comprehensive markdown report in `experiments/scottish_lower/11_decompression_pxg_covariate/README.md`.
- Update `todos/024_prototype_negbin_with_pxg_form_supremacy_covariate.md` to `COMPLETED` and verify `./scripts/todo.sh check`.
