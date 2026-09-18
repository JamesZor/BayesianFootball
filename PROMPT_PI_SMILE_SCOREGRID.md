# Work Package: Graduate SmileScoreGrid Anti-Diagonal Reweighting and Close Ticket T011

> **Role**: Senior AI Bayesian Research Architect & Core Systems Engineer  
> **Model Target**: `openai-codex/gpt-5.6-sol` (thinking: high)  
> **Assigned Task**: [TODO 017](todos/017_graduate_smilescoregrid_anti_diagonal_reweighting_and_close_t011.md)  
> **Associated Ticket**: [T011](docs/tickets/T011-portfolio-sizes-smile-latents-off-the-grid.md)  
> **Repository Context**: Read `AGENTS.md` before touching Julia code.

---

## 1. Executive Summary & Objective

In BayesianFootball, models with a totals "smile" pillar (`SmileLatents`, e.g. `DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel`) learn local intensities $\Lambda(K) = \lambda_{\text{tot}} \cdot \phi(K)$ at strikes $K \in \{0, 1, 2, 3, 4\}$.

### The Problem (Ticket T011)
Previously, in `src/predictions/score_grids/` and `src/predictions/score_computation/smile_poisson.jl`, the smile was implemented as an un-unified "pricing sidecar":
- `SmileScoreGrid` / `SmileScoreMatrix` held the **un-smiled** bivariate Poisson grid `Array{Float64, 3}` plus side vectors `(\lambda_{\text{tot}}, \phi)`.
- Over/Under markets were priced via the analytical formula $P(N \le K) = \text{cdf}(\text{Poisson}(\lambda_{\text{tot}}\phi_K), K)$.
- Everything else (1X2, BTTS, and critically **Kelly allocation in `src/Portfolio/`**) read the un-smiled grid.
- **Consequence**: The Kelly allocator and Baker-McHale shrinkage sized stakes from the wrong distribution. Totals margins differed from the smile CDF by **4.7–7.1 percentage points**.

### The Solution (Validated in Task 016)
In Task 016 (`/home/james/bet_project/.worktrees/BayesianFootball-grw-smile-spine/current_development/grw_smile_spine/`), anti-diagonal scoreline grid reweighting was developed and proven:
- Rescaling scoreline anti-diagonals ($G = h + a$) to match the smile totals CDF produces a genuine joint probability distribution $P(h, a)$.
- Verified to machine precision: totals gap $\le 4.4 \times 10^{-16}$, total mass $\sum S = 1.0 \pm 7 \times 10^{-16}$.
- Direct Kelly sizing from the reweighted grid lifted 5-parameter smile return from **+588% to +606%**, lowered max drawdown from −44.0% to −40.7%, and boosted Sharpe ratio to **1.64**.

### Your Mission
Graduate anti-diagonal reweighting from the Task 016 prototype into the core production codebase:
1. Establish a clean **`AbstractScoreGrid`** type hierarchy in `src/predictions/score_grids/`.
2. Make **`SmileScoreGrid`** natively produce and encapsulate the anti-diagonal reweighted score tensor.
3. Unify **`price_market!`** to evaluate Over/Under directly by summing anti-diagonals.
4. Eliminate custom smile-routing branches in **`src/Portfolio/pricing.jl`**, enabling Kelly allocation to natively size from `SmileScoreGrid`.
5. Write unit tests, verify zero allocation regressions, and formally close **Ticket T011**.

---

## 2. Mathematical Specification

Given joint scoreline probabilities $P_{\text{grid}}(h, a)$ on a $M \times M$ grid ($M = 12$, $h, a \in 0..11$):
1. **Grid Anti-Diagonal Mass**:
   $$\text{mass}[G] = \sum_{h + a = G} P_{\text{grid}}(h, a) \quad \text{for } G \in 0..2M-2$$
2. **Smile Target PMF**:
   For learned strikes $K \in \{0, 1, \dots, K_{\max}\}$ ($K_{\max} = 4$):
   $$F_{\text{smile}}(K) = \text{cdf}\Big(\text{Poisson}\big(\lambda_{\text{tot}} \cdot \phi(K)\big), K\Big)$$
   $$P_{\text{smile}}(G = K) = F_{\text{smile}}(K) - F_{\text{smile}}(K - 1) \quad (\text{with } F_{\text{smile}}(-1) = 0)$$
   Monotonicity invariant: require $P_{\text{smile}}(G = K) \ge 0$. If $F_{\text{smile}}(K) < F_{\text{smile}}(K-1)$, throw an error (non-monotone curve refused).
3. **Rescaling Learned Strikes ($0 \le G \le K_{\max}$)**:
   $$\text{ratio}[G] = \frac{P_{\text{smile}}(G)}{\text{mass}[G]}$$
4. **Tail Redistribution ($G > K_{\max}$)**:
   Let $\text{grid\_tail} = \sum_{G = K_{\max}+1}^{2M-2} \text{mass}[G]$.
   Let $\text{target\_tail} = 1.0 - F_{\text{smile}}(K_{\max})$.
   $$\text{tail\_ratio} = \frac{\text{target\_tail}}{\text{grid\_tail}}$$
   $$\text{ratio}[G] = \text{tail\_ratio} \quad \text{for all } G > K_{\max}$$
   This uniformly rescales the tail and relocates any missing grid truncation mass onto $G > K_{\max}$, ensuring $\sum_{h, a} S_{\text{smile}}[h+1, a+1] = 1.000$ strictly.
5. **Scoreline Rescaling**:
   $$S_{\text{smile}}[h+1, a+1] = S[h+1, a+1] \times \text{ratio}[h + a]$$
6. **Identity Shortcut**:
   If $\phi(K) == 1.0$ for all strikes $K$, skip reweighting (grid remains bit-identical).

Reference prototype: See `_gss_reweight_draw!` and `gss_reweight_grid!` in:
`/home/james/bet_project/.worktrees/BayesianFootball-grw-smile-spine/current_development/grw_smile_spine/l01_loader.jl` (lines 650–740).

---

## 3. Implementation Plan

### Step 1: Types & Workspaces (`src/predictions/score_grids/types.jl`)
- Define `abstract type AbstractScoreGrid end`.
- Define `struct StandardScoreGrid <: AbstractScoreGrid` (wraps `Array{Float64, 3}`).
- Refactor `SmileScoreGrid <: AbstractScoreGrid`:
  ```julia
  struct SmileScoreGrid <: AbstractScoreGrid
      grid::Array{Float64, 3}      # Reweighted joint tensor [max_goals x max_goals x n_draws]
      λ_tot::Vector{Float64}
      φ::Matrix{Float64}           # [n_strikes x n_draws]
      strikes::Vector{Float64}
  end
  ```
- Update `GridWorkspace` (or create `SmileGridWorkspace`) to preallocate anti-diagonal accumulation vectors:
  - `grid_mass::Vector{Float64}` of length `2 * max_goals - 1`
  - `ratio::Vector{Float64}` of length `2 * max_goals - 1`
  This guarantees **zero allocations** in inner loops.

### Step 2: Kernels & In-Place Reweighting (`src/predictions/score_grids/kernels.jl`)
- Implement `reweight_grid_antidiagonals!(S, λ_tot, φ, ws)` using the mathematical specification.
- In `compute_score_grid!(S, ws, l::SmileLatents, i::Int)`:
  1. Compute the baseline double-Poisson PMF into `S`.
  2. Populate `λ_tot` and `φ` for fixture `i`.
  3. Execute `reweight_grid_antidiagonals!(S, λ_tot, φ, ws)`.
- Update `price_market!(book, g::AbstractScoreGrid, m::MarketOverUnder)`:
  - Over/Under is evaluated by summing anti-diagonals of `g.grid` (for any `AbstractScoreGrid`, including `SmileScoreGrid`).
  - Eliminate the analytical Poisson CDF branch in `price_market!`, as the tensor already embodies the smile.

### Step 3: Portfolio Layer Simplification (`src/Portfolio/pricing.jl`)
- Remove ad-hoc smile branching in `BookWorkspace` and `_fill_extra!`.
- Ensure `extract_selections` and `_finish_book` read `p_grid` directly from `w.S`.
- Verify that `CountLatents` portfolios are completely unchanged.

### Step 4: Verification & Regression Ladder
1. **Create `test/test_score_grids.jl`**:
   - Verify `AbstractScoreGrid` hierarchy and dispatch.
   - Verify total mass sums to $1.0 \pm 10^{-14}$.
   - Verify totals marginal matches smile CDF to $\le 10^{-9}$.
   - Verify $\phi \equiv 1$ identity shortcut is bit-identical.
   - Verify forced un-shortcut path is bounded by draw truncation mass ($1 - \sum \text{grid}$).
   - Verify non-monotone $\phi$ is refused.
2. **Run Julia Test Suite**:
   `julia --project -e 'using Pkg; Pkg.test()'`
3. **Ticket Close-Out**:
   - Update `docs/tickets/T011-portfolio-sizes-smile-latents-off-the-grid.md`: set `Status: closed`, record resolution and test findings.
   - Update `docs/tickets/README.md`.
   - Update `todos/017_graduate_smilescoregrid_anti_diagonal_reweighting_and_close_t011.md`: set `Status: COMPLETED`, log work and verification.
   - Run `./scripts/todo.sh check` to verify index consistency.

---

## 4. Execution Guidelines

- Do not break existing non-smile models (`CountLatents`, `PlayerLineupLatents`).
- Ensure no memory leaks or inner-loop allocations.
- Commit clean, logical units of work (`feat: ...`, `test: ...`, `docs: ...`).
- When completed, ensure working tree is clean and push to `origin/feat/smile-scoregrid-reweighting`.
