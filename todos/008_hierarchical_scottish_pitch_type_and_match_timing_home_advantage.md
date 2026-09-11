# 008 — Hierarchical Scottish Pitch Type and Match Timing Home Advantage

| Field | Value |
|---|---|
| ID | 008 |
| Title | Hierarchical Scottish Pitch Type and Match Timing Home Advantage |
| Status | BACKLOG |
| Priority | P2 |
| Assignee | unassigned |
| Created | 2026-09-10 |
| Updated | 2026-09-10 |
| Related Files / Commits / PRs | [src/models/pregame/components/home_advantage.jl](../src/models/pregame/components/home_advantage.jl); [src/models/pregame/components/interfaces.jl](../src/models/pregame/components/interfaces.jl); [src/features/extractors/](../src/features/extractors/); [experiments/scottish_lower/](../experiments/scottish_lower/) |

## Context & Problem Statement

Current pregame models use a single global home advantage parameter $\gamma_{\text{home}} \sim \mathcal{N}(0.15, 0.05)$ across all Scottish Lower matches. In Scottish Championship, League 1, and League 2, ground conditions and schedules are notoriously heterogeneous:
1. **Pitch Surface (Synthetic Turf vs Natural Grass)**: Multiple clubs play on 3G/4G artificial turf (e.g. Alloa Athletic, Falkirk, Queen of the South, Hamilton Academical, Airdrieonians, Cove Rangers), while others play on grass. Grass-based squads visiting synthetic grounds face altered ball bounce, pace, and unfamiliar footing, creating asymmetric home advantage.
2. **Match Timing & Recovery**: Friday night fixtures (e.g., BBC Alba/BBC Scotland broadcasts), midweek Tuesday night cross-country journeys (e.g. Stranraer traveling to Peterhead or Elgin) where semi-professional squads travel after work, versus standard Saturday 3pm kickoffs.

With ReverseDiff's fast gradient tape, we can replace the crude scalar $\gamma_{\text{home}}$ with a hierarchical contextual home advantage pillar that pools ground-specific and situational effects with Bayesian shrinkage.

## Acceptance Criteria

- [ ] Add stadium pitch surface metadata (`is_synthetic_pitch` boolean) and match scheduling metadata (day of week, evening kickoff flag, rest days) to Scottish Lower match feature extraction.
- [ ] Implement `HierarchicalContextualHomeAdvantage` in `src/models/pregame/components/home_advantage.jl`:
  $$\gamma_{ij} = \gamma_{\text{base}} + \beta_{\text{turf\_asym}} \cdot (\text{turf}_i \land \neg \text{turf}_j) + \beta_{\text{midweek}} \cdot \text{is\_midweek} + u_i$$
  with ground random effects $u_i \sim \mathcal{N}(0, \sigma_{\text{stadium}}^2)$.
- [ ] Ensure non-centered parameterization for $u_i$ to prevent NUTS geometry bottlenecks.
- [ ] Implement zero-allocation feature tensor extraction compatible with compiled ReverseDiff tapes.
- [ ] Smoke test on 2-fold CV to verify tape compilation and posterior parameter recovery.
- [ ] Execute full 40-fold walk-forward grid on `mcmc-beast` (-t 16).
- [ ] Audit posterior credible intervals: verify whether $\beta_{\text{turf\_asym}}$ Credible Interval excludes zero.
- [ ] Benchmark out-of-sample proper scores (LogLoss, Brier, RPS) against the baseline `GlobalHomeAdvantage`.

## Ideas & Candidate Solutions

- **Surface Asymmetry Directionality**:
  - Asymmetric indicator: $(\text{turf}_i == 1 \land \text{turf}_j == 0)$, testing whether grass teams visiting turf grounds suffer extra penalty.
  - Symmetrical interaction: Also test whether turf teams visiting grass grounds suffer an equivalent penalty.
- **Semi-Pro Midweek Travel Interaction**:
  - In Scottish Leagues 1 and 2, players are semi-professional. A Tuesday evening away trip involving $>100\text{ km}$ of travel represents severe fatigue. Include an interaction feature: $\text{midweek} \times \log(1 + d_{ij})$.
- **Stadium-Level Random Intercepts**:
  - Individual stadium random effects $u_i \sim \mathcal{N}(0, \sigma_{\text{stadium}}^2)$ with a tight hyperprior $\sigma_{\text{stadium}} \sim \text{HalfNormal}(0.05)$ to prevent small-sample overfitting (each club only plays ~18 home games per season).
- **Friday Night Broadcast Effect**:
  - Isolated indicator for televised Friday evening fixtures, which often have higher attendance and different pre-match preparation routines.

## Work Log & Progress

- [2026-09-10 @antigravity] Created task in BACKLOG following user proposal and Scottish Lower domain analysis. Defined surface asymmetry and scheduling specifications.

## Verification & Findings

Not run yet. Record commands, pass/fail, wall time, posterior credible intervals on β_turf and β_timing, and comparative proper scores vs GlobalHomeAdvantage.
