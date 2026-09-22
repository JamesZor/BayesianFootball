# Claude Phase 2 Prompt: What Drives Market Pricing & Favourite Conviction?

**Task Reference**: [`todos/023_prototype_market_inverse_grw_dynamics.md`](../../todos/023_prototype_market_inverse_grw_dynamics.md)  
**Phase 1 Findings**: [`current_development/market_inverse_dynamics/README.md`](README.md)  
**Branch**: `feat/market-inverse-grw-dynamics`  
**Worktree**: `/home/james/bet_project/.worktrees/BayesianFootball-market-inverse`  
**Tmux Session**: `agent_claude_market_inverse`

---

## 1. User Feedback & The Fundamental Problem

The user reviewed the Phase 1 findings and identified the key unanswered question:

> *"It doesn't really explain what causes the market pricing to favour some teams strongly. What was the whole point of this? I wanted to understand what drives the market prices and why it's always sharper than my models, like if I'm missing features or structural models."*

### Why this matters
In Phase 1, you established that:
- The market's team ratings evolve like a 1st-order random walk (~0.028/week) with zero momentum.
- Volatility models mostly absorb single-fixture outliers as spike-and-revert jumps.
- There is a large unexplained fixture-level variance ($\sigma_{\text{obs}} = 0.093$).
- The cross-sectional spread of market ratings across teams is wide (0.14–0.19 log-rate units).

Meanwhile, in our L1 Bayesian goal models (TODO 021 & 022), team ratings suffer from **Bayesian shrinkage compression** on small goal samples, making them underconfident on heavy favourites (e.g. pricing a 76% market favourite at only 48–56%).

**Phase 2 must answer**:
1. **What features or structural factors drive the market's pricing to favour certain teams so strongly?**
2. **Why is the market consistently sharper than our goal models?**
3. **What are our goal models missing: squad wealth/payroll, player lineups/RAPM, underlying shot/chance dominance (proxy xG), rest, or table stakes?**

---

## 2. Research Plan: Feature Attribution & Sharpness Decomposition

### Step 1: The Conviction Gap Analysis (Market vs Pure Goal Model)
For each of the 623 accepted Scottish Lower fixtures:
1. Compute **Market Supremacy**:
   $$\Delta_{\text{mkt}, m} = \log \lambda_{\text{mkt}, h, m} - \log \lambda_{\text{mkt}, a, m}$$
2. Compute **Goal-Model Supremacy** ($\Delta_{\text{goal}, m}$) from our pure goal-based models (e.g. the baseline Poisson ratings, or `m01` time decay / `m02` GRW from the same panel).
3. Define the **Market Conviction Gap**:
   $$\text{Gap}_m = \Delta_{\text{mkt}, m} - \Delta_{\text{goal}, m}$$
   On heavy favourites, $\text{Gap}_m$ is large: the market is confident, while the goal model is shrunk.
4. Extract features from `DataStore` and regress $\text{Gap}_m$ and the market log-rates against:
   - **Squad Wealth / Market Valuation**: Team squad value ratio $\log(\text{wealth}_h / \text{wealth}_a)$ using `ProductionWealthCovariate` / squad values from `ds.matches` / `Features`.
   - **Player Lineups & RAPM**: Starting XI rating differential $\Delta \text{RAPM}_m$ and bench depth from `ds.match_lineups` / `PlayerLineupPillar` (`shots_rapm`, `pxg_rapm`).
   - **Underlying Shot / Territory Dominance (Proxy xG)**: Commentary-derived proxy xG differential $\Delta \text{pxG}_m$ vs actual goals. Is the market pricing underlying chance creation that hasn't converted into goals yet?
   - **Context & Rest**: Days of rest differential $\Delta \text{rest}_m$, travel distance, and table position/points gap.

### Step 2: Feature-Augmented State-Space Model
Augment the observation model in `l01_market_inverse_loader.jl`:
$$\log \lambda_{\text{mkt}, h, m} = \mu + \gamma_{\text{home}} + \alpha_{\text{att}, h, t} + \beta_{\text{def}, a, t} + \mathbf{\theta}^T \mathbf{X}_m + \epsilon_m$$
$$\epsilon_m \sim \text{Student-}t(\nu, \sigma_{\text{obs}})$$

Use a Student-$t$ observation scale mixture (as recommended in your Phase 1 report §7.1) so outlier fixtures do not distort the feature coefficients $\mathbf{\theta}$.

Measure:
- **Drop in $\sigma_{\text{obs}}$**: How much does unexplained fixture variance decrease from 0.093 when features are conditioned on?
- **Collapse in Rating Spread**: How much does the latent team spread (0.14–0.19) narrow once squad wealth or lineups are included? If latent rating spread shrinks, it proves that the market's apparent team ability differences are directly explained by those observable features!
- **Feature Significance**: Which features have statistically significant, non-zero posterior weights $\mathbf{\theta}$?

### Step 3: Quantified Attribution Breakdown
Deliver a clear decomposition table answering:
- What percentage of the market's favourite conviction is explained by:
  1. **Squad Financial Wealth / Payroll**
  2. **Starting XI Player Quality / Lineup Changes (RAPM)**
  3. **Underlying Shot / Chance Creation Volume (Proxy xG)**
  4. **Rest & Scheduling Context**
  5. **Pure Goal History (the baseline GRW)**
  6. **Unexplained / Market Sentiment Residual**

---

## 3. Deliverables

1. **Loader Extension**: Implement the feature regressors and Student-$t$ observation model in `current_development/market_inverse_dynamics/l01_market_inverse_loader.jl`.
2. **Runner**: Build `current_development/market_inverse_dynamics/r02_market_feature_attribution.jl`.
3. **Findings Document**: Create `current_development/market_inverse_dynamics/PHASE2_FEATURE_ATTRIBUTION.md` explaining in plain English and tables exactly what drives market prices and why the market favours certain teams.
4. **Task Tracking**: Update `current_development/market_inverse_dynamics/README.md` and `todos/023_prototype_market_inverse_grw_dynamics.md`, and verify `./scripts/todo.sh check` is green.
