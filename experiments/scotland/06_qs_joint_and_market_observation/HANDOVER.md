# Wave 2 — continuation brief (2026-10-09)

**Phase 3 is explicitly approved and RUNNING, not blocked on approval.**
Read PROGRESS.md and DECISIONS_LOG.md first. Do not relaunch or redo passes.

- Branch exp/qs-goal-vs-market, TODO 041 IN_PROGRESS, assignee pi.
- Frozen grid source 8f285a6a in `/root/BF_runs/qs_experiment`.
  All Julia on root@mcmc-beast, -t16, pin cores, BLAS 1. Do not update
  this checkout during the grid; never touch other sessions.
- OPEN launcher `pi_qsx2_grid_launcher_v1` %390; first-arm REPL
  `pi_qsx2_phase3_grid_v1_grw_joint` %391. As of beast 19:21 UTC,
  grw_joint has 2/40 initial checkpoints (folds 2/36); no failure or UUID yet.
  Check launcher then list owned pane IDs: later arms get new panes.
- Launcher log `/root/BF_runs/logs/qs_experiment_w2/phase3_launcher_v1.log`;
  arm logs `phase3_grid_v1_<arm>.log`; checkpoint directory
  `data/checkpoints/scottish_lower_qs_wave2_2426/<arm>/` in checkout.
  Outputs `/root/BF_runs/qs_experiment_w2_out/`.
- Approved order grw_joint → qs_joint → grw_marketobs → qs_marketobs,
  W2 4×(500+1000), no tuning. Mock-chain durable queue validation passed
  checkpoint resume and exact seeded replay before sampling.
- l05_grid.jl uses unchanged core sample_fold, 16 concurrent chain tasks,
  atomic complete-fold checkpoints and fixed task-local per-chain seeds:
  202700000+100000*arm+100*fold+10*attempt+chain. Arms 1–4, folds 1–40,
  attempts 0/1, chains 1–4. Immutable manifests reject changed recipes.
- Unchanged Harness.grid consumes checkpoints/persists and checks parity.
  Preserve primary UUID/diagnostics. Every initial Rhat>1.05 fold gets ONE
  attempt-1 rerun under <arm>__rhat_rerun; unaffected folds copied intact.
  Always select the rerun even if worse, never choose better/third attempt.
  Finite-Rhat/divergence <=0.1% final gate; tail ESS reviewed against 400.
- If a launcher gate fails: stop/report, retain pane/evidence. If resuming,
  phase3_gate.sh needs a fresh QSX2_GRID_ATTEMPT label (logs never overwritten);
  checkpoints and UUID receipts handle resume. No retry without diagnosis.
- Poll every 5–6 min on Codex, 20 min on Claude route.
- Phase 1 array feature/canonical identity b006743f revalidated:
  component 3831/3831, tapes 160/160 at 0 B, all three recovery seeds
  rerun with 39/42 coverage/zero all-seed misses. Evidence d0d933ae.
  src/harness untouched; Dict comparator gap recorded separately.
- Four hard smokes PASS, evidence a7b332a6. Joint passes retained ac30c4d7,
  market-only d0d933ae real DB parity. GRW market Rhat 1.13787/bulk16.10,
  QS market 1.06250/bulk67.14; zero divergences, review flags retained.
  Full ETA 12.3192 h before overhead/reruns disclosed before approval.
- Frozen table/manifest/snapshot digests in PROGRESS.md. Historical failures
  retained; no history deletion, reference refits, ROI/staking or prior tuning.
- Phase 3 evidence/UUIDs still to collect and commit. Phase 4 scoring/report
  and Phase 5 byte-for-byte frozen-seed reproduction remain; use the exact
  work-package metrics/pairs/bootstrap classes and saved wave-1 references.
