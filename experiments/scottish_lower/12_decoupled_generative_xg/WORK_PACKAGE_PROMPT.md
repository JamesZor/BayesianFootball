# Decoupled Generative xG-Primary Funnel (Scottish Lower Decompression)

> **Work Package**: TODO 025 — Prototype Decoupled Generative xG-Primary Model with Subordinate Goals  
> **Target**: Scottish Lower League Football (Tournaments 56 & 57, 40-fold walk-forward cohort, seasons 24/25 + 25/26, 710 fixtures)  
> **Harness**: Pi Solo Agent (`openai-codex/gpt-5.6-sol` with `--thinking high`) in tmux session `agent_pi_decoupled_xg`  
> **Worktree**: `/home/james/bet_project/.worktrees/BayesianFootball-decoupled-xg-funnel`  
> **Branch**: `feat/scottish-lower-decoupled-xg-funnel`  
> **Compute Node**: `mcmc-beast` (AMD Ryzen 9 32-core node, directory `/root/BF_decoupled_xg_funnel`)  

---

## 1. Executive Context & Motivation

In Generation 3 (`JointGammaPoissonObservation`) and Generation 4 (`m12_joint_hybrid_synergy`), proxy xG and goals were implemented as **two parallel sensors reading the exact same latent intensity $\mu$**:
$$\mu_h = \exp(\mu + \gamma_{\text{home}} + \alpha_{\text{att}, h} + \beta_{\text{def}, a})$$
$$\text{pxg}_h \sim \text{Gamma}(\nu, \mu_h / \nu), \quad y_h \sim \text{Poisson}(\kappa \cdot \mu_h)$$

Because $\mu_h$ is deterministic and shared, the sparse, discrete goal likelihood ($y \in \{0, 1, 2\}$) heavily penalizes high values of $\mu_h$, and the zero-mean shrinkage priors on $\alpha, \beta$ pull $\mu_h$ toward the league mean. This causes severe favourite compression: the market-on-model supremacy slope is **1.724** (TODO 024 measured benchmark).

In TODO 024, adding rolling proxy-xG form as a linear supremacy covariate closed 58.1% of the slope gap (1.724 $\to$ 1.302) and gave the best 1X2 LogLoss (0.616690), but switching to a single-arm Negative Binomial degraded Over/Under 2.5 and BTTS pricing, dropping portfolio return from +128% to +83%.

### The Generative Funnel Hypothesis
Option 2 implements a true **hierarchical generative funnel**:
$$\text{Team Ratings } (\alpha_{\text{xg}}, \beta_{\text{xg}}) \longrightarrow \text{Chance Generation } (\mu_{\text{xg}}) \longrightarrow \text{Goal Realization } (\lambda_{\text{goal}} = \kappa \cdot \mu_{\text{xg}})$$

1. **Primary Chance Layer (Continuous)**:
   $$\log \mu_{\text{xg}, h} = \mu_{\text{xg}} + \gamma_{\text{home}} + \alpha_{\text{xg}, h} + \beta_{\text{xg}, a}$$
   $$\text{pxg}_h \sim \text{Gamma}(\nu, \mu_{\text{xg}, h} / \nu)$$
   Because proxy xG is continuous and dense (non-zero every game with large natural spread), team chance ratings $\alpha_{\text{xg}}, \beta_{\text{xg}}$ do not suffer from Poisson goal shrinkage.
2. **Subordinate Goal Layer (Discrete)**:
   $$y_h \sim \text{Poisson}(\lambda_{\text{goal}, h}), \quad \lambda_{\text{goal}, h} = \kappa \cdot \mu_{\text{xg}, h}$$
   Goals are realized conditionally on the chances. Match-day finishing slumps (e.g. 2.8 xG but 0 goals) do not pull down the team's underlying chance-creation rating.

---

## 2. Experimental Scope & 4-Arm Benchmark

The matched cohort is the standard Scottish Lower 40-fold walk-forward grid over seasons 24/25 and 25/26 (710 fixtures, 2,899 scored market observations).

### The 4 Arms:
1. **`m01_poisson_time_decay` (Control 1)**:
   - Baseline Poisson with time-decay dynamics (`days_half_life = 180.0`).
2. **`m02_joint_gamma_poisson` (Control 2)**:
   - Canonical Gen 3/4 two-arm joint observation (compressed baseline, slope 1.724).
3. **`m03_funnel_shared_kappa` (Candidate 1)**:
   - Generative funnel with **shared league-level conversion factor** $\kappa$:
     $$\log \kappa \sim \mathcal{N}(0, 0.20), \quad \lambda_{\text{goal}, h} = \kappa \cdot \mu_{\text{xg}, h}$$
4. **`m04_funnel_hierarchical_kappa` (Candidate 2)**:
   - Generative funnel with **hierarchical team-level finishing factors** $\kappa_i$:
     $$\log \kappa_i = \log \kappa + \sigma_\kappa \cdot (\tilde{\kappa}_i - \text{mean}(\tilde{\kappa})), \quad \tilde{\kappa}_i \sim \mathcal{N}(0, 1), \quad \sigma_\kappa \sim \text{truncated}(\mathcal{N}(0, 0.10), 0, \infty)$$
     $$\lambda_{\text{goal}, h} = \kappa_h \cdot \mu_{\text{xg}, h}$$

---

## 3. Agent Operating Rules (CRITICAL)

- **SOLO AGENT EXECUTION**: You are operating as a solo agent with `openai-codex/gpt-5.6-sol` and `--thinking high`. **Do NOT invoke subagents.** Execute all mathematical modeling, Julia/Turing coding, command execution, and verification directly.
- **ReverseDiff AD Performance & Zero Allocations**:
  - The model runs under compiled ReverseDiff gradient tapes.
  - Zero heap allocations in the inner likelihood evaluation.
  - Set `BLAS.set_num_threads(1)` and `ThreadPinning.pinthreads(:cores)`.
- **Database Separation**:
  - `betdb` operational data is on `archpc:5433` (via `BF_DB_URL`).
  - `mcmc_experiments` is on `mcmc-beast:5432` (via `BF_EXPERIMENTS_DB_URL`).
  - Save completed production fits to PostgreSQL namespace `scottish_lower_decoupled_xg`.

---

## 4. Execution Stages

### Stage 0: Architecture & ReverseDiff Tape Check
- Implement candidate models in `experiments/scottish_lower/12_decoupled_generative_xg/l12_loader.jl`.
- Verify density parity and zero heap allocations on compiled ReverseDiff gradient tapes.

### Stage 1: Smoke Gate (Folds 1, 20, 40)
- File: `r10_smoke.jl`.
- Budget: 4 chains $\times$ (400 warmup + 400 samples).
- Gates: 0 divergences, $\hat{R} \le 1.05$, ESS $\ge 200$, score-grid partition check, DB roundtrips.

### Stage 2: 40-Fold Walk-Forward Production Grid on Beast
- File: `r20_production_grid.jl`.
- Execute all 40 folds across the 4 arms on `mcmc-beast`.
- Persist runs to PostgreSQL `mcmc_experiments` in namespace `scottish_lower_decoupled_xg`.

### Stage 3: Evaluation, Scaling & Portfolio Benchmark
- File: `r30_evaluation.jl`.
- **Supremacy Slope Regression**: Regress market supremacy on model supremacy ($y = \beta x$). Measure whether $\beta$ decompresses from $1.724$ towards $1.00$.
- **Shared vs Hierarchical $\kappa$ Comparison**: Does hierarchical team finishing (`m04`) improve out-of-sample proper scores or portfolio returns over shared $\kappa$ (`m03`)?
- **Proper Scores**: Selection-level binary LogLoss (1X2, O/U 2.5, BTTS), CRPS, RPS, ECE vs Betfair close.
- **Portfolio Simulation**: Common 622-fixture tradeable panel with `BookSpec(1X2, OU2.5, BakerMcHale)` and `PolicySpec(FlatTrust(0.25), SlateDrawdown(20.0), FixedCap(0.25))`.

### Stage 4: Findings Report & Sign-off
- Comprehensive report in `experiments/scottish_lower/12_decoupled_generative_xg/README.md`.
- Update `todos/025_prototype_decoupled_generative_xg_primary_model.md` to `COMPLETED` and verify `./scripts/todo.sh check`.
