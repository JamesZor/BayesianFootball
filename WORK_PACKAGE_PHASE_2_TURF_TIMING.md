# Task 008 Phase 2 — Contextual Pitch Surface (Turf) & Match Timing Home Advantage

> **Author**: Antigravity Coordinator (AGY)  
> **Target Agent**: Claude Code subagent running in tmux session `claude_hier_ha`  
> **Worktree**: `/home/james/bet_project/.worktrees/BayesianFootball-hierarchical-home-advantage`  
> **Branch**: `feat/hierarchical-home-advantage`  
> **Execution Host**: `mcmc-beast` (`root@mcmc-beast`, 32 cores, 64 GB RAM) for 40-fold grids; `archpc` for 2-fold smoke gates.

---

## 1. Context & Lessons from Phase 1

In Phase 1, we tested `HierarchicalTeamHomeAdvantage` (club-level random intercepts $u_i \sim \mathcal{N}(0, \sigma_{\gamma}^2)$ alone) on Scottish Lower:
* **Under `TimeDecay(180)`**, $\sigma_{\gamma}$ was not identified: the posterior shrank toward zero ($\sigma_{\gamma} \approx 0.035$, boundary mass 0.28–0.30 vs prior 0.159). TimeDecay's team ratings absorbed whole-venue levels, leaving no variance for unconditioned club intercepts.
* **On the 2026-09-12 live slate**, club random effects moved $P(\text{home})$ by $\le 0.003$ across all nine fixtures. The model backed the same 6 losing away legs against synthetic turf grounds because the random intercepts were completely blind to pitch surface or schedule fatigue.
* **Phase 1 Recommendation**: Phase 2 must incorporate explicit physical covariates: `is_synthetic_pitch` (turf) and match schedule timing (`is_midweek`, rest days).

---

## 2. Ground Truth Pitch Surface Registry

The Scottish Lower stadium geocodes file (`src/features/data/scottish_stadium_geocodes.csv`) has been enriched with the verified `is_synthetic_pitch` boolean column for all 32 clubs:
* **Synthetic 3G/4G Turf Grounds (`is_synthetic_pitch = 1`)**:
  - `airdrieonians` (Excelsior Stadium)
  - `alloa-athletic` (Recreation Park / Indodrill Stadium)
  - `annan-athletic` (Galabank)
  - `clyde-fc` (New Douglas Park groundshare)
  - `cove-rangers` (Balmoral Stadium)
  - `east-fife` (Bayview Stadium — installed 2017)
  - `east-kilbride` (K-Park Training Academy)
  - `edinburgh-city-fc` (Meadowbank Stadium — 4G)
  - `falkirk-fc` (Falkirk Stadium)
  - `forfar-athletic` (Station Park)
  - `hamilton-academical` (New Douglas Park)
  - `kelty-hearts-fc` (New Central Park)
  - `montrose` (Links Park)
  - `queen-of-the-south` (Palmerston Park)
  - `queens-park-fc` (Ochilview Park groundshare period)
  - `stenhousemuir` (Ochilview Park)
  - `the-spartans-fc` (Ainslie Park)
* **Natural Grass Grounds (`is_synthetic_pitch = 0`)**:
  - `albion-rovers`, `arbroath`, `bonnyrigg-rose`, `brechin-city`, `cowdenbeath`, `dumbarton`, `dunfermline-athletic`, `elgin-city`, `inverness-caledonian-thistle`, `partick-thistle`, `peterhead`, `ross-county`, `stirling-albion`, `stranraer`.

---

## 3. Mathematical Formulations & Hypotheses

### A. Core Mathematical Structure
Replace the scalar $\gamma_{\text{home}} \sim \mathcal{N}(0.15, 0.05)$ with `ContextualHomeAdvantage`:
$$\gamma_{ij} = \gamma_{\text{base}} + \beta_{\text{turf\_asym}} \cdot (\text{turf}_i \land \neg \text{turf}_j) + \beta_{\text{turf\_gen}} \cdot \text{turf}_i + \beta_{\text{midweek}} \cdot \text{is\_midweek} + \beta_{\text{rest}} \cdot (\text{rest}_i - \text{rest}_j) + u_i$$
with non-centered stadium random effects:
$$u_i = \sigma_{\text{stadium}} \cdot \tilde{u}_i, \quad \tilde{u}_i \sim \mathcal{N}(0, 1), \quad \sigma_{\text{stadium}} \sim \text{HalfNormal}(0.05)$$
Priors:
* $\gamma_{\text{base}} \sim \mathcal{N}(0.15, 0.05)$
* $\beta_{\text{turf\_asym}} \sim \mathcal{N}(0.05, 0.05)$ (prior expects grass visiting turf to increase home edge)
* $\beta_{\text{turf\_gen}} \sim \mathcal{N}(0.0, 0.05)$ (tests whether home advantage is generally shifted at turf grounds)
* $\beta_{\text{midweek}} \sim \mathcal{N}(0.05, 0.05)$ (Tuesday/Wednesday/Friday evening fatigue on away semi-pro players)
* $\beta_{\text{rest}} \sim \mathcal{N}(0.02, 0.02)$

### B. User EDA Finding: Does Turf Increase Match Pace / Total Goals?
The user noted from prior EDA that *both* teams appeared to benefit or score more on artificial turf (flatter roll, fewer unpredictable bobbles, faster ball speed).
Therefore, also test whether adding $\beta_{\text{turf\_intensity}} \cdot \text{turf}_i$ directly to the match intensity $\log \lambda_{h}, \log \lambda_a$ improves totals calibration!

### C. Hypotheses
* **H1 (Turf Asymmetry)**: $\beta_{\text{turf\_asym}}$ is strictly positive ($P(\beta_{\text{turf\_asym}} > 0) \ge 0.90$), confirming grass teams suffer extra friction at synthetic grounds.
* **H2 (Turf Pace)**: Total scoring on turf grounds is higher than grass grounds ($\beta_{\text{turf\_intensity}} > 0$).
* **H3 (Midweek Timing)**: Midweek fixtures amplify home advantage for semi-pro Scottish Lower clubs ($\beta_{\text{midweek}} > 0$).
* **H4 (Proper Scores)**: Out-of-sample LogLoss improves over the flat `GlobalHomeAdvantage` control (paired bootstrap excludes 0).
* **H5 (2026-09-12 Counterfactual)**: Away underdog win probabilities on turf grounds are reduced, eliminating or downsizing toxic away bets.

---

## 4. Model Ablation Ladder

All models use the Gen 3 Two-Arm Joint Likelihood (`JointGammaPoissonObservation`) and `ProductionWealthCovariate`.

1. **`m05_joint_td_raw` (Control)**: TimeDecay(180) + Flat `GlobalHomeAdvantage` (published benchmark).
2. **`m05_joint_td_turf_asym`**: Control + Asymmetric Turf ($\beta_{\text{turf\_asym}}$) + Stadium RE ($u_i$).
3. **`m05_joint_td_turf_dual`**: Model 2 + General Turf Effect ($\beta_{\text{turf\_gen}}$).
4. **`m05_joint_td_contextual`**: Model 3 + Midweek Timing ($\beta_{\text{midweek}}$) + Rest Differential.
5. **`m12_joint_hybrid_contextual`**: Winning specification promoted to Production `m12` (with PlayerLineup RAPM).

---

## 5. Execution Protocol & Reproduction Gates

### G1: Tape Parity & ReverseDiff Safety
* Verify ReverseDiff compiled tape matches ForwardDiff to $\le 10^{-6}$.
* Non-centered parameterization for all stadium random effects to ensure unconstrained geometry.

### G2: 2-Fold Smoke Gate (`r06_contextual_smoke.jl`)
* Run on `archpc` across Folds 1–2 (Scottish Lower 24/25).
* Budget: $4 \times (500\text{w} + 1000\text{s})$, $\delta = 0.80$.
* Assert $\hat{R} \le 1.05$, 0 divergences, ESS bulk/tail $\ge 400$.

### G3: 40-Fold Walk-Forward Production Grid (`r07_contextual_production.jl`)
* Execute on `mcmc-beast` (`root@mcmc-beast`, 16 threads).
* Target seasons: `["24/25", "25/26"]` (710 matches).
* Persist runs in `mcmc_experiments` under experiment name `scottish_lower_contextual_ha`.

### G4: Evaluation & Proper Scores (`r08_contextual_evaluate.jl`)
* Evaluate against Betfair closing odds over the 710 walk-forward matches.
* Compute paired bootstrap contrasts for LogLoss, Brier, RPS, and ECE.
* Stratify evaluation by pitch surface (Turf Home vs Grass Home) and timing (Midweek vs Saturday).

### G5: 2026-09-12 Counterfactual Slate Repricing (`r09_slate_repricing.jl`)
* Re-price the 2026-09-12 card through the replay engine.
* Compare stake sheet, away risk, and realised P&L vs the live ledger.

---

## 6. Directory Layout for Phase 2

All Phase 2 prototype code should live in `current_development/hierarchical_home_advantage/`:
* `l04_contextual_loader.jl`: Feature extractors, Turing components, model definitions.
* `r06_contextual_smoke.jl`: Smoke test runner.
* `r07_contextual_production.jl`: Production 40-fold grid runner.
* `r08_contextual_evaluate.jl`: Proper scores and parameter credible intervals.
* `r09_slate_repricing.jl`: Slate counterfactual re-pricing.
* Update `README.md` with full Phase 2 findings.
