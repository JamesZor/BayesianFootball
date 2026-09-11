# 009 — Hierarchical Multi Tournament Pooling Across Scottish Segments

| Field | Value |
|---|---|
| ID | 009 |
| Title | Hierarchical Multi Tournament Pooling Across Scottish Segments |
| Status | BACKLOG |
| Priority | P2 |
| Assignee | unassigned |
| Created | 2026-09-10 |
| Updated | 2026-09-10 |
| Related Files / Commits / PRs | [src/Data/fetchers/segments.jl](../src/Data/fetchers/segments.jl); [src/models/pregame/components/interception.jl](../src/models/pregame/components/interception.jl); [src/models/pregame/components/interfaces.jl](../src/models/pregame/components/interfaces.jl); [experiments/scottish_lower/](../experiments/scottish_lower/) |

## Context & Problem Statement

Currently, tournament segments are modeled as isolated silos (e.g., `ScottishLower` covers Tournaments 56 [League 1] and 57 [League 2], while Scottish Championship [Tournament 55] is evaluated separately). This creates two major blindspots:
1. **Promoted/Relegated Team Discontinuity**: When a club transitions between divisions (e.g., Falkirk promoted from League 1 to Championship, or Hamilton bouncing between tiers), their posterior talent history is disconnected, requiring weeks of new observations before ratings stabilize.
2. **Ignored Cross-Tier Information**: Scottish football features frequent cross-tier bridge matches—SPFL Trust Trophy (Challenge Cup), Scottish League Cup group stages, and Scottish Cup ties—where Championship, League 1, and League 2 teams face each other in competitive fixtures.

With ReverseDiff's computational capacity, we can build a joint hierarchical segment model that estimates tier supremacy differentials ($\Delta_{\text{tier}}$) and tournament baseline rates ($\mu_{\text{tier}}$) with hierarchical shrinkage toward a shared Scottish national prior:
$$\mu_{\text{tier}} \sim \mathcal{N}(\bar{\mu}_{\text{Scotland}}, \tau^2), \quad \alpha_{i} \sim \mathcal{N}(\text{TierSupremacy}_{\text{tier}(i)}, \sigma_{\text{team}}^2)$$

## Acceptance Criteria

- [ ] Define an expanded segment in `src/Data/fetchers/segments.jl` (e.g., `ScottishCombined(tiers = [:championship, :league1, :league2])`) including domestic cup bridge matches.
- [ ] Implement a `TierSupremacyPillar` or hierarchical intercept component in `src/models/pregame/components/` that estimates inter-division supremacy offsets.
- [ ] Implement reference-tier anchoring (e.g. Championship = 0.0 baseline, League 1 $\approx -0.30$, League 2 $\approx -0.60$) with half-Normal shrinkage priors on tier dispersion.
- [ ] Ensure ReverseDiff tape compilation succeeds on the multi-tier dataset without allocations.
- [ ] Run walk-forward CV evaluating prediction quality specifically on:
  - Promoted/relegated teams during their first 10 fixtures in a new tier.
  - Cross-tier cup fixtures (SPFL Trust Trophy).
- [ ] Verify proper score performance (LogLoss, Brier, RPS) is equal or superior to disconnected single-segment models.

## Ideas & Candidate Solutions

- **Bridge Fixture Weighting**:
  - Cup matches may exhibit different rotation policies or motivational dynamics than league matches. We can introduce a cup variance inflation factor or down-weighting factor: $\sigma_{\text{cup}}^2 = \omega_{\text{cup}} \sigma_{\text{league}}^2$.
- **Seamless Rating Continuity**:
  - A team’s latent talent $\alpha_i$ remains continuous across the summer break. When transitioning from League 1 to Championship, the effective team supremacy relative to opponents automatically adjusts by the estimated tier gap $(\Delta_{\text{Champ}} - \Delta_{\text{L1}})$, eliminating the "cold-start" problem.
- **Hierarchical Goal Distribution Overdispersion**:
  - Test whether overdispersion $r_{\text{tier}}$ or proxy-xG precision $\nu_{\text{tier}}$ varies systematically across divisions (e.g. higher variance in lower tiers).

## Work Log & Progress

- [2026-09-10 @antigravity] Created task in BACKLOG following user proposal. Outlined cross-tier supremacy modeling and promotion/relegation continuity.

## Verification & Findings

Not run yet. Record commands, pass/fail, wall time, estimated tier supremacy posteriors, and prediction accuracy on promoted/relegated clubs.
