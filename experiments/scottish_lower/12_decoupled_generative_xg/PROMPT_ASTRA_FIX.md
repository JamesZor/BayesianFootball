# Decoupled Generative xG-Primary Funnel — Fix & Execution Prompt

> **Task**: Fix Option 2 (TODO 025) Decoupled Generative Funnel in `experiments/scottish_lower/12_decoupled_generative_xg/`  
> **Model**: `openai-codex/gpt-6-astra` with `--thinking high`  
> **Worktree**: `/home/james/bet_project/.worktrees/BayesianFootball-decoupled-xg-funnel`  
> **Branch**: `feat/scottish-lower-decoupled-xg-funnel`  
> **Target Cohort**: Scottish Lower (Tournaments 56 & 57, 40-fold walk-forward, seasons 24/25 + 25/26, 710 fixtures)  
> **Compute Node**: `mcmc-beast` (`/root/BF_decoupled_xg_funnel`)  

---

## 1. The Critical Issue in the Previous Implementation

In the previous run, the agent implemented `m03_funnel_shared_kappa` as:
```julia
standard_model(:m03_funnel_shared_kappa, shared_observation())
```
and `shared_observation()` simply evaluated:
`Turing.@addlogprob! goals_ll + proxy_ll`

The previous agent justified this in a comment by writing:
*"Both have pxg | μ,ν ~ Gamma(ν, μ/ν) and goals | μ,κ ~ Poisson(κ μ), and in an ordinary joint posterior both likelihoods update μ."*

**Why this was rejected (Verdict: REQUEST_CHANGES)**:
In standard Bayesian MCMC, when you write a single joint likelihood log p(goals) + log p(pxg), information flows both ways. When a favourite dominates but fails to score (y = 0), the Poisson goal likelihood heavily penalizes μ and pulls the team rating α downward toward zero. This reproduces the exact same favourite compression (market-on-model slope ≈ 1.72) as Gen 3!

Running `m03` identically to `m02` was a duplicate run that failed to test the decoupling hypothesis.

---

## 2. The Required Architecture: Truly Decoupled Generative Funnel

We want **chance creation capability** to be the primary latent driver, with goals as a subordinate realization, so that low match goal counts do NOT pull down underlying team ratings.

There are two valid formulations to implement this in Turing/ReverseDiff:

### Formulation A: Modular Posterior (Cut Feedback / Two-Stage MCMC)
1. **Primary Chance Creation Layer**:
   Team ratings α_xg,i, β_xg,i are fitted **strictly on continuous proxy xG**:
   log μ_xg,h = μ_xg + γ_home + α_xg,h + β_xg,a
   pxg_h ~ Gamma(ν, μ_xg,h / ν)
   Because proxy xG is continuous and dense, priors on rating innovation are wider (σ_att, σ_def ~ truncated(Normal(0, 0.20), 0, Inf)), allowing favourites to express strong un-shrunk supremacy.
2. **Subordinate Goal Layer (Cut Feedback)**:
   Conditioned on the posterior draws of μ_xg from the chance layer, goals are observed:
   y_h ~ Poisson(κ · μ_xg,h)  (or y_h ~ Poisson(κ_h · μ_xg,h))
   The likelihood of goals estimates conversion factor κ (or κ_i), but has **zero backward gradient** into α_xg, β_xg. Favourites remain strong even after 0-goal games.

### Formulation B: Two-Tier Latent State Space (Match-Level Finishing Shock)
If evaluating in a single Turing model, match-level finishing variance ξ_h,m must be introduced:
   log μ_xg,h,m = μ_xg + γ_home + α_xg,h + β_xg,a
   pxg_h,m ~ Gamma(ν, μ_xg,h,m / ν)
   log λ_h,m = log μ_xg,h,m + log κ + ξ_h,m,  ξ_h,m ~ Normal(0, σ_ξ^2)
   y_h,m ~ Poisson(λ_h,m)
Because ξ_h,m absorbs match-day goal noise / finishing slumps, the underlying team ratings α_xg, β_xg are insulated from Poisson goal shrinkage.

---

## 3. The 4 Benchmark Arms

1. `m01_poisson_time_decay`: Baseline Poisson goals model.
2. `m02_joint_gamma_poisson`: Canonical Gen 3 joint model (compressed baseline, slope ~1.724).
3. `m03_funnel_shared_kappa`: Truly decoupled funnel with shared league conversion κ.
4. `m04_funnel_hierarchical_kappa`: Truly decoupled funnel with hierarchical team finishing κ_i = κ exp(δ_i).

---

## 4. Operating Standards

- **Solo Agent Execution**: Do NOT spawn subagents. Execute Julia/Turing code, tests, and runners directly.
- **ReverseDiff Performance**: Ensure 0 heap allocations during gradient tape evaluation.
- **Verification Gates**:
  1. **Stage 0**: Verify gradient tape compilation with 0 heap allocations on `mcmc-beast`.
  2. **Stage 1 (Smoke Gate)**: Folds 1, 20, 40 on `mcmc-beast` (`r10_smoke.jl`). Confirm 0 divergences, R̂ ≤ 1.02, ESS ≥ 200, score-grid partition check passes.
  3. **Stage 2 (Production Grid)**: All 40 folds across 710 fixtures on `mcmc-beast` (`r20_production_grid.jl`). Persist to namespace `scottish_lower_decoupled_xg`.
  4. **Stage 3 (Evaluation)**: Run `r30_evaluation.jl`. Verify market-on-model supremacy slope (does it decompress towards 1.00?), proper scores (1X2, O/U 2.5, BTTS LogLoss, CRPS, RPS, ECE), and portfolio backtest.
  5. **Stage 4 (Report & Close)**: Update `experiments/scottish_lower/12_decoupled_generative_xg/README.md` and complete `todos/025_prototype_decoupled_generative_xg_primary_model.md`. Ensure `./scripts/todo.sh check` passes.
