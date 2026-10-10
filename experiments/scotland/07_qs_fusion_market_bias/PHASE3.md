# Phase 3 — 40-fold grid (attempt a1)

`phase3_gate.sh a1` at `dde8a441` on mcmc-beast, 2026-10-10 01:04–03:48 UTC. Pins matched
(snapshot `c786e2fc…`, mtime 2026-09-25 12:57:15.480765468 UTC; market table `680bc228…`).
Fake-chain queue check `t05_grid_queue.jl` PASS 10/10 before any sampling. Arms ran in the
approved order in fresh REPLs, W2 sampler 4×(500+1000), accept 0.65, depth 10, seeds as
pre-declared in `DECISIONS_LOG.md` (base 203200000). Every harness grid hard check passed
(`grid_stage` errors otherwise).

| Arm | Final run (= primary) | Wall | max R̂ | Reruns | min bulk ESS | min tail ESS | Tail < 400 | Divergences |
|---|---|---:|---:|---|---:|---:|---|---:|
| fusion_qs_bias (control) | `26de19bf-fc29-4818-bfdf-7bf76535901f` | 54 min | 1.0194 | none | 251.2 | 330.7 | fold 40 | 0 |
| fusion_qs_nobias | `5afdfeb3-abad-4d23-82fb-1bfe6babe797` | 53 min | 1.0123 | none | 409.3 | 764.6 | none | 0 |
| fusion_grw_bias | `b2e5a527-e51e-4913-b7bb-e1b94a4d1ff2` | 54 min | 1.0147 | none | 483.1 | 882.3 | none | 0 |

- **R̂ reruns:** no fold in any arm exceeded 1.05, so no `__rhat_rerun` was sampled.
- **Divergences:** 0 in all 120 fold fits (gate ≤ 0.1%).
- **Tail ESS vs 400 (review, not a gate):** one fold below, control fold 40 (330.7; bulk 251.2,
  R̂ 1.0194). Fold 40 has the most data; the control's posterior there is the least
  efficiently explored of any fold. Its Phase 4 fold-40 scores and bias posteriors should be
  read with that in mind; nothing is rerun (R̂ is within threshold).
- **Control forwarding:** both non-control arms verified the control (summary = receipt =
  completed DB run with 40 folds) before sampling. Control score parity after each non-control
  grid: 664 rows, 9960 cells, **exact** (max abs diff 0.0), both times.
- **Wall time:** 2.7 h in total against the 13.343 h probe ETA. The 16-chain short probe
  overstates the 4-chain-per-fold queue cost by about 5×; the ETA was an upper bound, not a
  budget, and nothing about the sampler changed.

Evidence: `results/phase3/` (diagnostics, grid summaries, rerun selections, receipts, control
score snapshots and parity), `results/gate_logs/phase3_launcher_a1.log`. Beast logs:
`/root/BF_runs/logs/qs_experiment_w3/phase3/a1/`. No `claude_qsx3_*` sessions remain.
