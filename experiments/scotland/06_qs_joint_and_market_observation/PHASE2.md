# Phase 2 — all four hard smokes PASS; grid approval required

2026-10-09. Joint results retained at ac30c4d7; ONLY market arms rerun at
**d0d933ae** after array-feature revalidation (b006743f). No harness changes.
Original dictionary failure, UUID and check rows remain unchanged.

## Complete four-arm summary

Smoke budget 2×(200 warmup + 200 retained), folds 1/40, pinned ScottishLower.
Wall is harness invocation time, including probe; raw include windows are longer.

| Arm | Hard checks | Wall s | max Rhat | min bulk/tail ESS | Divergences | Grid ETA h |
|---|---:|---:|---:|---:|---:|---:|
| grw_joint (retained) | 6/6 | 170.2 | 1.05414 | 182.89 / 69.66 | 0 | 1.9374 |
| qs_joint (retained) | 6/6 | 177.5 | 1.07449 | 133.14 / 79.94 | 0 | 1.9550 |
| grw_marketobs (new) | 6/6 | 228.8 | **1.13787** | **16.10 / 57.47** | 0 | **4.1991** |
| qs_marketobs (new) | 6/6 | 227.3 | 1.06250 | 67.14 / 74.33 | 0 | 4.2278 |

**Full four-arm probe ETA: 12.3192 h (~12 h 19 min)**, excluding setup/scoring,
checkpoint overhead and any diagnostic reruns. Largest-fold 16-chain 50+50 probe
sampling window extrapolated by (40×4×1500)/(16×100); not a runtime guarantee.

**These are hard-check passes, NOT convergence passes.** All short smokes retain
convergence fail/review flags. In particular, the manager-highlighted original
GRW market Rhat **1.14**, bulk ESS **16**, ETA **4.219 h** persists in the new
smoke: identical sampler diagnostics, new probe ETA **4.199 h (~4.2 h)**.
No longer-budget convergence claim. Full-grid tail ESS is reviewed against 400;
all these short-chain minima are below it.

Market performance is also flagged: GRW GC 35.8%, GC-stall 57.3%, efficiency
4.725; QS GC 36.2%, GC-stall 52.2%, efficiency 4.561. Both compiled gradient
tapes remain 0 B; the reports diagnose non-tape overhead, not tape allocation.

## Persistence and eligibility

- grw_marketobs UUID `30ad2ca7-ccd4-4cc0-9973-81142b67dcb3`.
- qs_marketobs UUID `3ec74b0f-ee29-4bef-9ed9-dea5ceb8bba5`.
- Retained joint UUIDs `e169aa8e-2302-4c9c-9e39-6c3b1e93218b` /
  `032bd044-1f10-4e71-9876-f586a04eaa48`.
- Market fit_parity passes actual database save/load, not abstention. All six
  required hard checks present/pass; snapshot/table/priors/seeds unchanged.
- Separate read-only fresh REPL %387 confirms UNCHANGED has_passing_smoke returns
  true for every current recipe. No sampling, database writes, history deletion,
  relabel or eligibility bypass. Canonical component display gives revised inputs
  their own content/prior identity; AbstractDict gap remains TODO follow-up.

## Traceability / approval gate

`results/phase2_arrays/four_arm_smoke_summary.csv` combines unchanged joint CSVs
with new market CSVs. All individual checks/summaries, read-only eligibility
receipt and exact audit source are in the same directory. Performance reports:
`results/phase2_arrays/smoke_perf/`. Full console text, byte-exact compressed raw
logs and PASS receipts: `results/gate_logs/phase2_smoke_arrays_*` and
`phase1_smoke_eligibility*`; launcher receipt `phase2_arrays_launcher.log`.

Manager must authorise BEFORE any grid. Proposed unchanged order:
grw_joint → qs_joint → grw_marketobs → qs_marketobs.
W2 4×(500+1000), resume/checkpoints, one fixed-seed rerun for Rhat>1.05,
divergences <=0.1%, tail-ESS 400 review gate. No grid, predictive scoring, ROI,
staking or reference refits have run. All owned panes are closed.
