# Scottish Lower 08 — goal decomposition

**Status: incident tests 47/47; real Fold 1 deterministic model tests 51/51;
strict four-model Fold 1 smoke 64/64. All four compiled gradients allocate zero
bytes and all four posterior fits passed the six-part convergence gate. The
40-fold production grid is pending.**

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
- `l08_model_checks.jl` / `test08_model_contract.jl`: executed deterministic
  model, exact-prior/Jacobian, AD, filtration, extraction and native-grid verification.
- `test08_incident_contract.jl`: synthetic attribution/missingness and actual
  registry conservation checks (**47/47 passed**, local Julia 1.12.1).
- `l08_eda_statistics.jl` / `r08_eda_statistics.jl`: adjusted statistical tests
  against the frozen registry, without MCMC.
- `l08_workflow.jl`: experiment registration, persistence and orchestration helpers.
- `r08_smoke.jl`: strict single-fold promotion ladder.
- `r08_sampling_budget_benchmark.jl`: matched Fold 1 NUTS budget/target-acceptance benchmark (TODO 001).
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

## Executed deterministic gates

`julia --project -t 8 experiments/scottish_lower/08_goal_decomposition/test08_model_contract.jl`
passed **51/51** on local Julia 1.12.1 (2m14s). The source-hashed evidence is
`results/deterministic_checks_julia_1.12.1.toml`.

| Arm | Parameters | Tape instructions | Warm gradient allocations |
|---|---:|---:|---:|
| m00 recombined control | 97 | 232 | 0 bytes |
| m01 decomposition | 97 | 232 | 0 bytes |
| m02 team penalties | 149 | 307 | 0 bytes |
| m03 own pressure | 98 | 244 | 0 bytes |

All 40 broad probes agreed with fresh ReverseDiff and ForwardDiff (maximum
relative error <2.3e-15). Doubling fixture rows left tape lengths unchanged.
The independent distribution-object density, including exact HalfNormal/Beta
Jacobians, agreed within 1.7e-15 relative error. Perturbing 759 future registry
rows changed neither fitted features nor prior anchors. Synthetic extraction
matched independently summed rates and native 12×12 kernels; these two synthetic
parameter draws are **not posterior samples or convergence evidence**. Their
reported truncated-grid tail mass is not normalized away.

## Executed Fold 1 smoke

`L08_RUN_SMOKE=true julia --project -t 8 experiments/scottish_lower/08_goal_decomposition/r08_smoke.jl`
passed **64/64** on archpc (Julia 1.12.1). All four fits used 4 chains × 1,000
warmup × 1,000 retained draws, had zero divergences, built finite per-draw 12×12
score grids, and passed exact fit/latent and portfolio-ledger round trips.
Detailed diagnostics and immutable run UUIDs are recorded in
`results/smoke_fold1_2026-09-09.md`.

| Arm | max R-hat | minimum ESS | divergences |
|---|---:|---:|---:|
| m00 recombined control | 1.0041 | 2,104 | 0/4,000 |
| m01 decomposition | 1.0044 | 1,758 | 0/4,000 |
| m02 team penalties | 1.0051 | 1,659 | 0/4,000 |
| m03 own pressure | 1.0042 | 1,479 | 0/4,000 |

### Approved new-team policy

The canonical 40-fold audit identified six initially refused fixtures:
Arbroath and Inverness in Fold 1, East Kilbride in Fold 21; see
`results/preflight_unseen_team_refusals.csv`. The user explicitly approved
**prior-only hierarchical effects for teams declared from upcoming fixture
identities** so all 710 fixtures remain in scope. Declaration reads no outcomes,
adds no fitted rows, and samples attack/defence uncertainty (also penalty effects
in m02), rather than plugging in league-average rates. Unexpected, undeclared
teams still refuse by fixture ID. Missing/unseen referees retain exactly zero
effect, not a fitted UNKNOWN group.

## Fold 1 sampling-budget benchmark (TODO 001)

`r08_sampling_budget_benchmark.jl` ran on mcmc-beast (Julia 1.12.6, 16 pinned threads)
on 2026-09-10. It compared A (4 × 1,000/1,000, δ 0.95, the production sampler), B
(500/500, δ 0.95) and C (500/500, δ 0.90) on m00 and m01, with four independent 4-chain
fits per cell (96 chains). Measured values, per-fit gates and raw-artefact paths are in
[`results/sampling_budget_fold1_2026-09-10.md`](results/sampling_budget_fold1_2026-09-10.md).
Headline results:

- B costs 0.59–0.61× A's core time, not 0.5×. ESS/draw is unchanged or lower, so min
  ESS per wall-second is highest under A in both arms.
- Every Fold 1 fit clears ESS ≥ 400. Scaled by the worst-fold/fold-1 ratio of the 40
  m00 grid checkpoints (0.34), B and C project to about 190–340 at the worst fold.
- One divergence each occurred under A (m00) and C (m00), so the zero-divergence gate
  fails the control itself on 1 of 4 fits. Tree depth never exceeded 7 (cap 10).
- Recommendation: keep A. The larger lever is runtime throughput (TODO 003): chains in
  the lightly loaded queue tail ran up to 2× faster than chains under full load.

## Remote execution

Worktree: `/root/BF_goal_decomposition`, branch
`feat/scottish-lower-goal-decomposition`. Use explicit Julia 1.12.6, 16 physical
CPU threads, pinned cores, and single-thread BLAS. Other operators' existing
REPLs are left untouched. Cache input is copied separately from Git; credentials
remain in the existing ignored environment configuration, never in artifacts.
