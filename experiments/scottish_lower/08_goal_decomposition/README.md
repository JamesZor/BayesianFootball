# Scottish Lower 08 — goal decomposition

**Status: BBC referee registry regenerated and incident tests passed (47/47).
Model/AD verification is in progress. No MCMC or production result yet.**

Investigates separate non-penalty/non-own goals, penalty attempts and conversion,
and own-goal receipts. Total intensities are recombined per posterior draw and
priced through the existing Poisson score grids; production Portfolio and
MatchDay code remain unchanged.

## Deliverables

- [`REPORT.md`](REPORT.md): derivations, hypotheses, validation and comparison contract.
- `EMPIRICAL.md`: measured incident audit and statistical evidence.
- `l08_incident_data.jl` / `r08_eda.jl`: reproducible extraction and EDA.
- `l08_decomposed_models.jl`: three requested candidates, an identical-spine
  total-goals-only control, and feature/latent adapters.
- `l08_model_checks.jl`: deterministic model and AD verification.
- `test08_incident_contract.jl`: synthetic attribution/missingness and actual
  registry conservation checks (**47/47 passed**, local Julia 1.12.1).
- `l08_eda_statistics.jl` / `r08_eda_statistics.jl`: adjusted statistical tests
  against the frozen registry, without MCMC.
- `l08_workflow.jl`: experiment registration, persistence and orchestration helpers.
- `r08_smoke.jl`: strict single-fold promotion ladder.
- `r08_production_grid.jl`: prepare-only checks and remote walk-forward queue.
- `r08_portfolio.jl`: paired exchange backtests, not reused bookmaker artifacts.

The registry can be reconstructed directly from the committed CSVs with an
identical verified SHA-256; the ignored `.jls` binary is optional. No SQL refresh
is needed to reproduce this snapshot. Prototype types must be included before
deserializing their fits. Commands,
immutable run UUIDs and measured headline tables will be recorded here after
execution. A written runner is not evidence that its verification passed.

## Measured incident findings

- 2,019 finished matches; 1,992 reconcile, 27 quarantined.
- Own-goal `is_home` identifies the **recipient** for all 107 unambiguous
  incidents; three are unresolved. The prompt's reversal would be incorrect.
- **Referee correction:** `bbc.match_officials` verifiably names 2,009/2,019
  referees, with no duplicate assignment rows. The earlier blanket absence
  claim is retracted; the BBC-backed registry is now used.
- Reconciled goals: 4,942 regular, 419 penalties, 107 own goals.
- Penalty attempts: 545, of which 419 converted (76.88%).

These are descriptive data findings, not evidence of predictive improvement.

## Scientific safeguards

Own-goal attribution is determined from score reconciliation, not assumed from
the work package. Missing incident feeds never become zero-count observations.
“Regular” goals may include set pieces and are not tactical open-play labels.
Historical benchmark scores are recomputed on identical fixtures and markets;
conditional Poisson superposition is not an unconditional Poisson claim.

## Remote execution

Worktree: `/root/BF_goal_decomposition`, branch
`feat/scottish-lower-goal-decomposition`. Use explicit Julia 1.12.6, 16 physical
CPU threads, pinned cores, and single-thread BLAS. Other operators' existing
REPLs are left untouched. Cache input is copied separately from Git; credentials
remain in the existing ignored environment configuration, never in artifacts.
