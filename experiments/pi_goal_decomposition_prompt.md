# WORK PACKAGE: Decomposed Goal Intensity Modeling for Scottish Lower
# Target Model: openai-codex/gpt-6-astra (thinking: high)
# Working Branch: feat/scottish-lower-goal-decomposition
# Working Directory: experiments/scottish_lower/08_goal_decomposition/
# Remote Compute: root@mcmc-beast:/root/BF_goal_decomposition

> **Agent Role**: Principal Bayesian Statistician & Quantitative Football Researcher  
> **Target Model**: `openai-codex/gpt-6-astra` (Reasoning: High, Context: 272K)  
> **Mission**: Formulate, empirically analyze, and benchmark the **Decomposed Goal Intensity Paradigm** ($Y_{\text{goals}} = Y_{\text{opn}} + Y_{\text{pen}} + Y_{\text{own}}$) across the Scottish Lower League walk-forward grid.

---

## 1. Executive Context & Theoretical Foundation

In current production Bayesian models (Generation 1 Poisson `m01`–`m05`, Generation 2 NegBin, and Generation 3 Two-Arm Joint `m08`), total match goals $Y_{\text{goals}}$ are treated as an undifferentiated point process governed by a single team latent intensity $\lambda_s = \exp(\mu + \text{HA} + \alpha_s + \beta_{\text{opp}})$.

However, a football goal is the union of three fundamentally distinct physical mechanisms:
1. **Open-Play Goals ($Y_{\text{opn}}$)**: Driven by continuous territorial dominance, sustained shot creation, tactical dynamics, and attacking/defensive quality. (~88% of all goals).
2. **Penalty Goals ($Y_{\text{pen}}$)**: A discrete two-stage compound process: penalty generation (fouls/handballs in the box) followed by a high-probability spot-kick trial (~77% conversion rate). (~7.5% of all goals).
3. **Own Goals ($Y_{\text{own}}$)**: Rare, highly stochastic misplays by defending players, credited to the opposing side. (~1.9% of all goals).

### The Mathematical Superposition Principle
If $Y_{\text{opn}} \sim \text{Poisson}(\lambda_{\text{opn}})$, $Y_{\text{pen\_awarded}} \sim \text{Poisson}(\lambda_{\text{pen}})$, each penalty converted with probability $k_{\text{pen}} \sim \text{Beta}$, and $Y_{\text{own}} \sim \text{Poisson}(\lambda_{\text{own}})$, then under mutual independence:
$$Y_{\text{pen}} \sim \text{Poisson}(k_{\text{pen}} \lambda_{\text{pen}})$$
$$Y_{\text{goals}} = Y_{\text{opn}} + Y_{\text{pen}} + Y_{\text{own}} \sim \text{Poisson}(\lambda_{\text{total}})$$
where the total predictive intensity is:
$$\lambda_{\text{total}} = \lambda_{\text{opn}} + k_{\text{pen}} \lambda_{\text{pen}} + \lambda_{\text{own}}$$

### Why This Could Provide Alpha
1. **Denoising Core Ratings**: When a match is decided by a dubious penalty or freak own goal, an undifferentiated Poisson model severely shifts team attack/defense ratings ($\alpha, \beta$). Decomposing likelihoods keeps open-play ratings clean and stationary.
2. **Asymmetric Box Skill**: Certain teams specialize in drawing penalties via high box dribble volume or physical play, while others commit frequent box fouls. Modeling $\lambda_{\text{pen}}$ separately captures this asymmetric alpha.
3. **Coherent Downstream Integration**: Because the sum of independent Poissons is Poisson, $\lambda_{\text{total}}$ drops directly into BayesianFootball score-grid kernels (`SmileScoreGrid`, `ScoreGrid`), market evaluation, and the Kelly portfolio allocator with zero structural changes.

---

## 2. Data Landscape & Extraction Guidelines

### A. Database Connection & Incidents Schema
* Operational DB: `BF_DB_URL` (`betdb` on `archpc:5433`).
* Primary incident table: `sofascore.match_incidents`:
  * Open-play goals: `incident_type = 'goal'` AND `data->>'incidentClass' = 'regular'`
  * Penalty goals: `incident_type = 'goal'` AND `data->>'incidentClass' = 'penalty'`
  * Missed penalties: `incident_type = 'inGamePenalty'` AND `data->>'incidentClass' = 'missed'`
  * Own goals: `incident_type = 'goal'` AND `data->>'incidentClass' = 'ownGoal'`
* **CRITICAL OWN GOAL ATTRIBUTION**:
  In SofaScore, an own-goal incident has `is_home` indicating the team that committed the own goal. In the match score, **the goal is credited to the opposing team**. Ensure own-goal counts are properly credited to the receiving team!

### B. Empirical Facts in Scottish Lower (Tournaments 56 & 57)
A preliminary database audit reveals:
* Regular open-play goals: 5,019 (~88.3%)
* Penalty goals: 428 (~7.5%)
* Missed penalties: 127
* Total penalties awarded: 555 (conversion rate $k_{\text{pen}} \approx 77.12\%$)
* Own goals: 110 (~1.9%)
* Unclassified/other goals: 11 (~0.2%)

### C. Missing Referee Data in Scottish Lower
* Audit finding: `sofascore.matches.raw_data->'referee'` is `null` for all 2,019 Scottish Lower matches.
* **Mandate for Agent**: You cannot fit referee fixed/random effects. You must investigate whether $\lambda_{\text{pen}}$ should be modeled via:
  * Team box-attack propensity ($\alpha_{\text{pen}}$) and opponent box-defense fouling ($\beta_{\text{pen}}$) + penalty Home Advantage ($\text{HA}_{\text{pen}}$), OR
  * Division/league-level hierarchical baseline with partial shrinkage, OR
  * Global constants.

---

## 3. Detailed Research Mandate (3-Phase Plan)

### Phase 1: Exploratory Data Analysis & Mathematical Report
1. **Extract Historical Incidents Dataset**:
   * Build an extraction script joining matches, incidents, and teams for Scottish Lower across all available seasons.
   * Generate match-level summary statistics: $(y_{\text{opn}, h}, y_{\text{opn}, a}, y_{\text{pen}, h}, y_{\text{pen}, a}, y_{\text{own}, h}, y_{\text{own}, a})$.
2. **Empirical Distribution & Overdispersion Tests**:
   * Compute variance-to-mean ratios and zero-inflation statistics for each component.
   * Test whether open-play goals $Y_{\text{opn}}$ remain Poisson or exhibit overdispersion.
3. **Team Variance & Repeatability Tests**:
   * Estimate team-level penalty awarded rates and penalty conceded rates.
   * Perform ANOVA or variance-component estimation: is there genuine, statistically significant between-team variance in penalty generation, or is it indistinguishable from homogeneous Poisson noise?
   * Test penalty conversion rate $k_{\text{pen}}$: is there evidence of team-level conversion variance, or does a pooled Beta prior suffice?
   * Analyze own-goal rates: are own goals purely a Poisson rate proportional to opponent attacking pressure, or a flat league rate?
4. **Deliverable**: Create `REPORT.md` and `EMPIRICAL.md` documenting mathematical derivations, likelihood formulations, and empirical findings.

### Phase 2: Turing Model Formulation & Verification Ladder
1. **Formulate the Model Family**:
   * **Candidate 1 (`m01_decomposed_baseline`)**:
     * Open play: $\log \lambda_{\text{opn}, h/a} = \mu_{\text{opn}} + \text{HA}_{\text{opn}} + \alpha_{\text{opn}} + \beta_{\text{opn}}$
     * Penalty: $\log \lambda_{\text{pen}, h/a} = \mu_{\text{pen}} + \text{HA}_{\text{pen}}$, with global $k_{\text{pen}} \sim \text{Beta}(a, b)$
     * Own goals: $\log \lambda_{\text{own}} = \mu_{\text{own}}$
   * **Candidate 2 (`m02_decomposed_team_penalties`)**:
     * Add team penalty-drawing attack $\alpha_{\text{pen}}$ and penalty-conceding defense $\beta_{\text{pen}}$ with hierarchical shrinkage $\sigma_{\text{pen\_att}}, \sigma_{\text{pen\_def}}$.
   * **Candidate 3 (`m03_decomposed_pressure_own_goals`)**:
     * Couple own-goal rate $\lambda_{\text{own}}$ to opponent attacking intensity.
2. **Implement in Turing on ReverseDiff**:
   * Construct clean, typed loader functions in `l08_decomposed_models.jl`.
   * Ensure type stability and compiled ReverseDiff tape compatibility (no runtime allocations in gradient evaluation).
3. **Pass the 6-Part Convergence & Smoke Ladder** (`r08_smoke.jl`):
   * Run a 1-fold test (Fold 1 or Fold 43).
   * Verify AD safety: compiled gradient evaluation succeeds without error.
   * Verify convergence audit: $\hat{R} \le 1.05$, bulk ESS $\ge 200$, tail ESS $\ge 200$, 0 divergences.
   * Verify score-grid generation: in-place 12x12 grid matches $\lambda_{\text{total}}$.
   * Verify serialization: `save_fit` and `load_fit` round-trip.

### Phase 3: Production Walk-Forward Grid & Out-of-Sample Evaluation
1. **Execution on Remote Compute Server (`mcmc-beast`)**:
   * Execute the full 40-fold walk-forward cross-validation grid across seasons 24/25 + 25/26 (710 held-out matches, ~2,900 evaluated market observations).
   * Utilize `QueuedNUTSConfig` or multi-threaded queued execution across the 32 cores on `mcmc-beast`.
2. **Proper Score Benchmarking**:
   * Compare out-of-sample proper scores directly against the canonical standards:
     * **Gen 1 Poisson Baseline**: LogLoss ~0.6597 (`m05`) / ~0.6601 (`m01`).
     * **Gen 3 Two-Arm Joint Baseline**: LogLoss ~0.6571 (`m08`).
     * **Betfair Closing Line**: LogLoss ~0.6568.
   * Measure:
     * 1X2 LogLoss & Brier Score.
     * Over/Under 2.5 LogLoss & ECE.
     * Both Teams to Score (BTTS) LogLoss.
     * CRPS on scorelines.
3. **Betfair Portfolio Simulation**:
   * Run `run_portfolio_simulation` under canonical `BookSpec` (1X2 + O/U 2.5 with Baker-McHale shrinkage) and `PolicySpec`.
   * Compare realized PnL, annual Sharpe ratio, and maximum drawdown against the Gen 1 and Gen 3 benchmarks.

---

## 4. File Layout & Deliverable Standards

All work for this experiment must reside in:
`experiments/scottish_lower/08_goal_decomposition/`

Required files:
1. `REPORT.md`: Comprehensive scientific report (math derivations, EDA results, model specs, convergence audit, out-of-sample proper scores, portfolio results, and conclusion).
2. `EMPIRICAL.md`: Detailed data audit, summary tables, and manifest of extracted incident statistics.
3. `l08_decomposed_models.jl`: Turing model definitions, feature extractors, and score-grid adapters.
4. `r08_eda.jl`: Standalone EDA script extracting incidents and running statistical variance tests.
5. `r08_smoke.jl`: Smoke test runner verifying AD precompilation, single-fold sampling, and convergence.
6. `r08_production_grid.jl`: Production 40-fold walk-forward MCMC grid runner.
7. `r08_portfolio.jl`: Zero-allocation portfolio simulation runner comparing against Gen 1/3 baselines.
8. `README.md`: High-level summary of the experiment suite and headline numbers.

---

## 5. Remote Compute Protocol (`mcmc-beast`)

* **Host**: `root@mcmc-beast` (32 cores, 64 GB RAM).
* **Worktree**: `/root/BF_goal_decomposition` on branch `feat/scottish-lower-goal-decomposition`.
* **Sync Protocol**:
  ```bash
  # 1. Commit and push locally
  git push origin feat/scottish-lower-goal-decomposition

  # 2. Pull on beast
  ssh root@mcmc-beast "git -C /root/BF_goal_decomposition pull origin feat/scottish-lower-goal-decomposition"

  # 3. Run Julia scripts with 32 threads
  ssh root@mcmc-beast "export PATH=/root/.juliaup/bin:\$PATH; cd /root/BF_goal_decomposition && julia --project -t 32 <script.jl>"
  ```
* **Security & Safety**:
  * Never commit, print, or log raw database passwords or `BF_DB_URL`.
  * Never modify `src/Portfolio/` or production MatchDay execution files.
