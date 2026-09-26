# WORK PACKAGE: Scottish Club Pedigree, Full-Time Status & Tier Priors EDA
# Target Model: openai-codex/gpt-6-astra (thinking: high, browser: enabled)
# Working Branch: feat/scottish-pedigree-fulltime-eda
# Working Worktree: /home/james/bet_project/.worktrees/BayesianFootball-pedigree-fulltime-eda
# Working Directory: experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/
# Tracking TODO: todos/027_eda_scottish_club_pedigree_full_time_status_and_tier_priors.md
# Compute Node: mcmc-beast (ssh root@mcmc-beast) & local archpc
# Operational DB: BF_DB_URL (betdb on archpc:5433)
# Experiments DB: BF_EXPERIMENTS_DB_URL (mcmc_experiments on mcmc-beast:5432)

> **Agent Role**: Principal Quantitative Football Researcher & Bayesian Modeler  
> **Mission**: Investigate, quantify, and model the structural talent gap between Full-Time Professional and Part-Time Semi-Professional Scottish clubs, cross-tier division pedigree, and relegation transition lag across all 4 SPFL leagues. Deliver a definitive research report (`README.md`) with empirical data, verified club operational classifications, and concrete mathematical formulations (informative priors vs linear covariates).

---

## 1. Executive Context & Problem Statement

### A. The Operational Failure Mode (2026-09-19 Live Slate)
In live MatchDay operations on the 2026-09-19 Scottish Lower slate (tournaments 56 & 57):
1. **The ~54% Probability Ceiling on Heavy Favourites**:
   - **Ross County vs Cove Rangers**: Betfair closing implied probability was **67.0%** for Ross County. Production model `m12_joint_hybrid_synergy` predicted only **48.2%** for Ross County and **29.7%** for Cove (market 17.5%).
   - **Hamilton vs Queen of the South**: Betfair closing implied probability was **71.0%** for Hamilton. Model predicted only **54.2%** for Hamilton and **25.2%** for Queen of the South (market 13.5%).
2. **Phantom Kelly Edges on Extreme Underdogs**:
   - Because the model severely underprices the favourites, residual probability mass is forced onto draws and underdogs.
   - On Cove Rangers, the model saw a **+23.8% edge** at decimal odds of 5.70. On Queen of the South, it saw a **+14.4% edge** at odds of 7.40.
   - Fractional Kelly staking dumped **46% of all staked capital** (£20.35 of £43.53 slate budget) onto these two extreme underdogs. Both lost, causing a painful slate drawdown.
3. **The Decoupled xG Funnel Was REJECTED (TODO 025)**:
   - In TODO 025, we tested whether severing goal feedback into team ratings would decompress favourites.
   - The result: **It inverted.** Market-on-model slope worsened to 1.99–2.09, favourite win probability fell to 48%, and LogLoss degraded (+0.0028). The problem is NOT goal feedback!

### B. The Root Cause: Information Asymmetry & Zero-Mean Shrinkage
Why does the closing market price Ross County at 67% and Hamilton at 71%, while our model sits at 48–54%?
1. **Club Pedigree & Professionalism Disparity**:
   - Ross County is a full-time professional club that spent 11 of the last 12 seasons in the Scottish Premiership, with a multi-million-pound payroll, daily training, and elite sports science.
   - Cove Rangers and Annan Athletic are part-time semi-professional clubs whose players work 40-hour day jobs and train on Tuesday/Thursday evenings.
   - When full-time athletes play part-time semi-pros, the market prices an overwhelming physical, conditioning, and depth advantage.
2. **The Model's Blindness**:
   - Our L1 model trains only on 2–3 rolling seasons of League 1 and League 2 (tournaments 56 & 57). It has zero knowledge of the Premiership (54) or Championship (55).
   - Under zero-mean Gaussian shrinkage priors ($\alpha_i \sim \mathcal{N}(0, \sigma^2)$), the model starts Ross County and Hamilton with an expected talent rating of **0.0 (identical to Annan or Clyde)**.
   - Because of small sample sizes (36 matches) and zero-sum identification ($\sum \alpha_i = 0$), the model's net latent supremacy is mathematically capped at ~0.76 log-goals (~55–60% win prob). It takes weeks of matches before ratings can even begin to separate.

---

## 2. Research Mandate & Methodology

The agent must execute a comprehensive 5-part research investigation.

### STAGE 1: Build the Ground-Truth SPFL Operational Status Dataset
- **Objective**: Create a definitive, versioned dataset of operational status (`Full-Time`, `Part-Time`, `Hybrid`) for all 42 SPFL clubs across seasons 21/22, 22/23, 23/24, 24/25, 25/26, and 26/27.
- **Tools**: Use web search / browsing to consult SPFL announcements, club financial statements, manager interviews, and reporting (e.g. Daily Record, BBC Scotland, club official websites) regarding when clubs turned full-time, reverted to part-time, or operated hybrid models (e.g. Falkirk, Dunfermline, Queen's Park, Cove Rangers, Airdrieonians, Hamilton, Kelty Hearts, Inverness CT).
- **Data Proxy Evaluation**:
  - Check whether data already in `betdb` (e.g. `sofascore.match_player_lineups` player market valuations, squad size, age profile, kickoff timing) correlates with or predicts full-time status.
- **Deliverable**: `data/spfl_club_operational_status.csv` + documentation in `README.md`.

### STAGE 2: Quantify Cross-Tier Supremacy Gaps Across All 4 SPFL Tiers
- **Objective**: Measure the empirical talent step between SPFL divisions:
  - Tier 1: Premiership (`tournament_id = 54`)
  - Tier 2: Championship (`tournament_id = 55`)
  - Tier 3: League One (`tournament_id = 56`)
  - Tier 4: League Two (`tournament_id = 57`)
- **Data**: Query `betdb` for all league fixtures plus cross-tier cup matches (SPFL Trust Trophy / Challenge Cup, Scottish League Cup group stages, Scottish Cup) from 2021 to 2026.
- **Metrics to Compute**:
  1. Empirical goal differentials ($\Delta \text{goals}$) and goal-scoring rates per tier.
  2. Proxy xG differentials ($\Delta \text{pxg}$) where available (BBC live text).
  3. Betfair closing implied supremacy ($\text{logit}(P_{\text{home}}) - \text{logit}(P_{\text{away}})$ or log-rate difference).
  4. The estimated step-size between adjacent tiers: $T_1 \to T_2$, $T_2 \to T_3$, $T_3 \to T_4$.

### STAGE 3: Within-Tier FT vs PT Performance Delta (League One Focus)
- **Objective**: Isolate League One where full-time and part-time clubs compete directly in the same division.
- **Analysis**:
  - Compare matches of **FT vs PT**, **FT vs FT**, and **PT vs PT**.
  - Quantify the net goal differential, shot/proxy-xG differential, win rates, and Betfair closing supremacy.
  - Determine: How much of the 1.72 market-on-model slope gap is attributable strictly to the model ignoring FT vs PT status?

### STAGE 4: Relegation / Transition Lag & Financial Cost
- **Objective**: Track the trajectory of relegated and promoted clubs during their first 10 matches in a new division.
- **Case Studies**: Relegated upper clubs (e.g. Hamilton, Dunfermline, Falkirk, Inverness, Ross County) entering League 1.
- **Analysis**:
  - How many matches does it take for a flat-prior ($\mathcal{N}(0, \sigma^2)$) Bayesian model to adjust to the true talent level?
  - What was the model's win probability vs the market closing probability over matches 1–5, 6–10, and 11–20?
  - Simulate the Kelly staking edge and quantify the bankroll drawdown caused by betting on underdogs against these transitioning favourites.

### STAGE 5: Mathematical Modeling Formulations & Architectural Design
- **Objective**: Propose and formulate candidate mathematical mechanisms to solve this in `BayesianFootball.jl`.
- **Formulate Approach A: Tier & Operational Status Informative Priors**:
  - Non-zero prior means: $\alpha_i \sim \mathcal{N}(\mu_{\text{tier}(i)} + \delta_{\text{status}(i)}, \sigma_{\text{team}}^2)$.
  - How to handle zero-sum identifiability ($\sum \alpha_i = 0$ vs reference-tier anchoring).
  - How hierarchical hyperpriors should be structured across tiers.
- **Formulate Approach B: Linear Covariates in the Linear Predictor**:
  - $\log \mu_{h} = \dots + w_{\text{tier}} \Delta \text{Tier} + w_{\text{status}} \Delta \text{Status}$.
  - Verify compatibility with compiled ReverseDiff tapes (0 heap allocations).
  - Address collinearity: How to prevent the covariate from fighting with dynamic team ratings $\alpha_i, \beta_i$.
- **Draft Mock Julia Implementations**: Provide clean, modular Julia component definitions for `src/models/pregame/components/`.

---

## 3. Strict Execution Constraints

1. **NO LARGE MCMC GRIDS**: Do NOT launch multi-fold NUTS or ADVI grids. This is a pure EDA and statistical investigation. Keep compute fast, lean, and reproducible.
2. **Allowed Compute**: SQL extraction via LibPQ, Julia DataFrames, GLM/OLS regressions, Python/Julia statistical scripts, web research. (A single-fold 30-second ReverseDiff tape compilation check is permissible only if validating gradient tape compatibility).
3. **Reproducibility**: All scripts must be saved in `experiments/scottish_lower/13_pedigree_fulltime_and_tier_priors_eda/` with clean numbered filenames (`r01_...jl`, `r02_...jl`, etc.).
4. **Deliverable**: The primary output is `README.md` in this directory, written to the standard of an institutional quantitative research paper.

---

## 4. Work Log & Coordination

- Update `todos/027_eda_scottish_club_pedigree_full_time_status_and_tier_priors.md` as milestones are reached.
- Run `./scripts/todo.sh check` before committing.
- Commit all findings to `feat/scottish-pedigree-fulltime-eda`.
