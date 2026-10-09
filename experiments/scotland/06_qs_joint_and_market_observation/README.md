# 06 — QS joint goals+xG and builder market-rate observations

> **Status:** blocked on Phase 2 recipe loading; Phase 1 passed. Grid NOT authorised.
> **Experiment:** `scottish_lower_qs_wave2_2426`; **TODO:** 041.
> **Control:** `grw_joint`; predictive run UUID pending.
> **Scorecard:** same versioned harness/scoring path as wave 1; no scores yet.

## Questions and fixed panel

1. Does weak-r QualityStyleGRW help the production-like joint goals+proxy-xG observation?
2. Can GRW/QS observing full-book market log rates match saved wave-1 market C0?

Same pinned ScottishLower snapshot, tournaments 56/57, 40 walk-forward folds,
710 held-out fixtures in 24/25+25/26. `lower` scope, two history seasons,
match-biweek dynamics clock. No ROI/staking, reference refits or prior tuning.
Executable source of truth: [candidates.jl](candidates.jl).

| Candidate | Role | Dynamics | Observation |
|---|---|---|---|
| grw_joint | control | MultiScaleGRW | Joint goals+proxy-xG, shared kappa |
| qs_joint | candidate | QS, all r priors LogNormal(0,1) | Same joint observation |
| grw_marketobs | candidate | MultiScaleGRW | Full-book Normal log rates |
| qs_marketobs | candidate | Same weak-r QS | Same market observation |

All arms use GlobalInterception and GlobalHomeAdvantage. Joint observation is
experiment 04's unchanged no-league-delta specification, k=25, fallback=:none,
shape prior truncated Normal(4,1) to [.5,Inf), log-kappa Normal(0,.2).
Market sigma prior is C0 HalfNormal(.20). Predictions remain double Poisson.
W2 grid budget: 4 chains x (500 warmup + 1000 retained), accept .65, depth 10.

## Gates and provenance

[PHASE0.md](PHASE0.md): fresh branch-health gates and joint-feature coverage.
[PHASE1.md](PHASE1.md): component 3787/3787, 0 B tapes, builder/harness,
likelihood decomposition, amended AD/FD and three-seed recovery (39/42).
Manager amendments and frozen seeds in [DECISIONS_LOG.md](DECISIONS_LOG.md).
Earlier failures preserved in PHASE1_BLOCKED.md, not silently removed.

Frozen market_rates.csv exactly matches wave 1's prepared panel, including
CSV round-trip and target counts 710/595/78/517. SHA256:
`680bc2283e9c0fc7a896bf01b4ba209cb3aae06f50d00667c98b02f379872549`.
Original provenance guard and saved wave-1 manifest unchanged. Feature injection
aligns by match ID; held-out matches never appear in their fold's training rows.

## Execution record

- Phase 1 verified source 153239a7; acceptance evidence pushed at e86a03fe.
- Phase 2 source 73147ce0: first grw_joint entry point fails at candidates.jl:5
  (include undefined in the bare candidates module), before any check/sampling.
  No UUID/ETA/convergence; see PHASE2_BLOCKED.md. Await manager loading-fix approval.
- Phase 2 method: fresh owned beast REPL per arm via phase2_gate.sh, 16 pinned threads,
  BLAS 1, pinned ScottishLower input (not generic CLI's ScottishPyramid cache).
- Harness smoke: folds 1 and 40, 2x(200+200), all required hard checks, persistence
  parity and standard 16-chain largest-fold performance probe. Root seeds 20261040–43
  in declared candidate order. Convergence and performance flags are recorded review,
  not suppressed or recast as a hard failure.
- Canonical recipes registered before sampling; completed grid config hashes checked.
- Outputs outside checkout: /root/BF_runs/qs_experiment_w2_out/phase2/.
  Raw logs: /root/BF_runs/logs/qs_experiment_w2/.
- ETA will extrapolate the largest-fold full-concurrency probe to
  40x4x1500 iterations. This excludes grid/scoring overhead and is not a guarantee.
- **Ask manager after all four smoke hard passes, before ANY grid.**

## Decision and planned comparisons

Pending. Primary A: qs_joint−grw_joint. Primary B: each market-observation arm
versus saved C0 and grw_joint. Secondary: joint versus saved Poisson controls,
qs_marketobs−grw_marketobs, and every arm versus the close. Same outcome/goal
scores, slopes, cohorts and three bootstrap paths as wave 1. Classification uses
the prescribed noncircular interval; flag circular disagreement. Report all fold
r and sigma posteriors and convergence reruns. No predictive result claimed yet.
