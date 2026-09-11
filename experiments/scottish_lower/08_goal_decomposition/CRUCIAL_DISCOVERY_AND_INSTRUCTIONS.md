# Crucial Discovery & Science Review Instructions for 08_goal_decomposition

## 1. Context
The earlier resume message was prematurely truncated by terminal submission. Here is the complete discovery and instructions for Phase 2 and Phase 3.

---

## 2. CRITICAL DATA UPDATE — REFEREE DATA IS FULLY AVAILABLE (99.5% COVERAGE)

### The Cause of the Previous Null Result
Your earlier query on `sofascore.matches.raw_data` returned null because SofaScore does not populate the referee name in that top-level JSON key.

### The Authoritative Source: `bbc.match_officials`
Referee names and official IDs are fully tracked in the BBC dataset in PostgreSQL:
* **Table**: `bbc.match_officials`
* **Schema**:
  - `match_id` (integer, joins directly to `sofascore.matches.match_id`)
  - `role` (character varying, filter by `role = 'referee'`)
  - `name` (text, e.g. "Ross Hardie", "Steven Reid", "David Dickinson")
  - `bbc_official_id` (character varying)
* **Coverage**: **2,009 out of 2,019** Scottish Lower finished matches across tournaments 56 and 57 (seasons 20/21 through 26/27) have the named referee (**99.5% coverage**). Only 10 matches lack a record.
* **SQL Join**:
  ```sql
  SELECT m.match_id, m.tournament_id, s.year AS season, m.home_team, m.away_team,
         m.home_score, m.away_score, m.start_timestamp, m.raw_data::text AS raw_match,
         o.name AS referee_name, o.bbc_official_id AS referee_id
  FROM sofascore.matches AS m
  JOIN sofascore.seasons AS s ON s.season_id = m.season_id
  LEFT JOIN bbc.match_officials AS o ON o.match_id = m.match_id AND o.role = 'referee'
  WHERE m.status_type = 'finished' AND m.tournament_id = ANY(ARRAY[56, 57])
  ORDER BY m.start_timestamp, m.match_id;
  ```

### Empirical Significance of Referee on Penalties
* Across 40 referees with $\ge 20$ Scottish Lower matches, penalty award rates exhibit a **4.5× spread**:
  - Low: Steven Reid at 0.107 penalties/match
  - High: Ross Hardie at 0.475 penalties/match
  - Poisson deviance reduction: $\Delta D = 56.63$ on 42 d.f. ($p \approx 0.065$).
* Referee identity has a demonstrable physical impact on penalty award frequency.

### Required Actions for Referee Data:
1. **Update `l08_incident_data.jl`**:
   - Update `_raw_frames()` to perform the `LEFT JOIN bbc.match_officials o ON o.match_id = m.match_id AND o.role = 'referee'`.
   - Populate `:referee_name` and `:referee_id` in the match table. For missing records (10 matches), assign `"UNKNOWN"` or missing.
   - Regenerate the registry artifacts in `results/`.
2. **Update Documentation**:
   - Retract the statement in `EMPIRICAL.md` and `REPORT.md` that referee data was absent.
   - Document the 99.5% coverage and the empirical penalty award spread.
3. **Model Formulation**:
   - Incorporate a hierarchical referee effect $\gamma_{\text{ref}} \sim \operatorname{Normal}(0, \sigma_{\text{ref}})$ into the penalty sub-model:
     $$\log(\lambda_{\text{pen}, i}) = \mu_{\text{pen}} + \text{HA}_{\text{pen}} + \gamma_{\text{ref}(i)} + \alpha_{\text{pen}, \text{home}(i)} + \beta_{\text{pen}, \text{away}(i)}$$
   - For unseen referees or missing records in test folds: set $\gamma_{\text{ref}} = 0$ (the population mean).

---

## 3. INCORPORATE SCIENCE REVIEW ACTION ITEMS (from claude-opus-5 review)

1. **Add `m00_recombined_control`**:
   - Add an identical-spine model that has `complete = 0` (or fits only total goals likelihood on the sum of rates $\lambda_{\text{total}}$).
   - This provides a controlled within-spine baseline to isolate the exact value of component decomposition versus total-goals modeling, free of prior mismatch confounds.
2. **ReverseDiff AD Safety for Binomial Likelihood**:
   - Do NOT instantiate `Distributions.Binomial` objects inside the ReverseDiff model loop.
   - Use closed-form binomial log-density:
     $$\log p(c \mid a, k) = c \log(k) + (a - c) \log(1 - k) + \log \binom{a}{c}$$
     (using `SpecialFunctions.logabsbinomial(a, c)[1]` or precomputed $\log \binom{a}{c}$ since attempts $a$ and conversions $c$ are observed data).
   - This ensures the compiled ReverseDiff tape has 0 allocations and is $O(1)$ in model evaluation.
3. **Refuse Unseen Teams**:
   - When encountering an unseen team in a test fold, refuse the fixture rather than pricing at league mean, strictly adhering to `AGENTS.md` §7.4.
4. **Per-Draw Score Tensor Averaging**:
   - When computing market probabilities (1X2, O/U 2.5, BTTS), calculate the 12×12 score grid for each posterior draw's $\Lambda^{(s)}$, then average probabilities across draws:
     $$P(\text{score}) = \frac{1}{S} \sum_{s=1}^S P(\text{score} \mid \Lambda^{(s)})$$
   - Never price at posterior mean intensity $\bar{\Lambda}$ as that destroys Jensen convexity.

---

## 4. EXECUTION ROADMAP

1. **Regenerate & Test Locally**:
   - Update `l08_incident_data.jl` and run `test08_incident_contract.jl`.
   - Run `r08_smoke.jl` on Fold 1 to pass the 6-part convergence audit ($\hat{R} \le 1.05$, ESS $\ge 200$, 0 divergences) and verify 0 allocations on the ReverseDiff tape.
2. **Phase 3 Production Grid on `mcmc-beast`**:
   - Commit and push changes to `feat/scottish-lower-goal-decomposition`.
   - On `mcmc-beast` (`/root/BF_goal_decomposition`), pull the changes.
   - Run the 40-fold walk-forward grid with 16 physical cores pinned:
     `JULIA_PKG_PRECOMPILE_AUTO=0 julia --project -t 16 r08_production_grid.jl`
   - Run evaluation (`r08_evaluate.jl`) and portfolio backtests (`r08_portfolio.jl`) benchmarked against `m05_joint_production_wealth` and `m00_recombined_control`.

---

## 5. STRICT DIRECTIVE: TERMINATE ALL SUBAGENTS TO PREVENT TOKEN BURN

1. **Close All 6 Active Subagents Immediately**:
   - The session has 6 subagents running (`models`, `eda`, `runners`, `baseline-evidence`, `evaluation`, `statistics`).
   - `eda`, `statistics`, and `baseline-evidence` finished their tasks hours ago and are obsolete.
   - You MUST terminate/close all subagents immediately to stop token consumption and context bloat.
2. **Execute Directly as a Single Agent**:
   - Do NOT spawn any new subagents.
   - Execute all remaining tasks directly in this single agent session:
     a) Run `r08_smoke.jl` on Fold 1 locally.
     b) Git commit and push all suite 08 files to `origin/feat/scottish-lower-goal-decomposition`.
     c) Pull to `root@mcmc-beast:/root/BF_goal_decomposition` and launch `r08_production_grid.jl` with 16 physical cores pinned.
     d) Run evaluation (`r08_evaluate.jl`) and portfolio backtesting (`r08_portfolio.jl`).
