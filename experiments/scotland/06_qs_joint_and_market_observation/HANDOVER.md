# Wave 2 — active v2 continuation brief (2026-10-09)

**Phase3 resumed; grw_marketobs running at081ba5a6.** Read PROGRESS.md,
DECISIONS_LOG.md, CONTROL_FORWARDING_REVALIDATION.md. TODO041 IN_PROGRESS.

- Branch exp/qs-goal-vs-market; frozen running source081ba5a6 on beast
  /root/BF_runs/qs_experiment. Do not update this checkout during grid.
  Julia beast only, -t16, pin cores, BLAS1. Never touch other sessions.
- OPEN v2 launcher pi_qsx2_grid_launcher_v2 %394 and fresh grw_marketobs
  pi_qsx2_phase3_grid_v2_grw_marketobs %396 (started20:32:29UTC). Later
  qs_marketobs gets a new pane; verify owned pane IDs before driving.
- V2 launcher /tmp/qsx2_phase3_resume_v2.sh skips standalone GRW entirely,
  runs qs_joint → grw_marketobs → qs_marketobs in fresh REPLs, stops on FAIL.
  Evidence copy results/gate_logs/phase3_resume_v2.sh. Original tracked
  phase3_gate.sh unchanged. Watch phase3_launcher_v2.log/arm logs in
  /root/BF_runs/logs/qs_experiment_w2/. Poll every5–6min on Codex,20min Claude.
- Manager authorises l05/r05-only explicit final GRW RunRef forwarding to
  primary/rerun for every non-control arm. checked_control verifies fixed
  UUID98447840-e95c-420f-92b4-2db6545506a3, experiment/name/status completed,
  all40 folds before sampling. No src/harness changes.
- Manager clarifies unchanged H.grid's INTERNAL paired-control scoring is
  allowed (no new GRW fit/run). Every recomputed GRW score field must match
  existing rows exactly/absolute1e-12. Wrapper saves before-snapshot and
  comparison CSV per stage, refuses candidate receipt on mismatch.
- Fresh offline regression %393 PASS57/57 (8ownership+7comparison+
  42forwarding/negative checks), no DB/datastore/MCMC. Pane closed.
- GRW complete UUID98447840-e95c-420f-92b4-2db6545506a3, all40 folds,
  Rhat1.01008/bulk641.78/tail324.47, zero divergences/no reruns. KEEP one
  tail ESS400 review flag in report. V1 include2008.6993s, never refit.
- QS initial fit persisted before v1 scoring failure atUUID
  2a5eee90-3402-40ce-a5e3-f4c6b57fc8ea. V2 resumed that exact completed
  recipe, no resampling, scoring/receipt complete20:32:29UTC in %395 (closed).
  Rhat1.00937/bulk832.35/tail761.02, zero divergences/no reruns. Its internal
  GRW score parity664rows/9960fields EXACT, maxdiff0. All80 initial joint
  checkpoint SHA256 hashes unchanged, evidence committed. Both joint fits
  now grid-arm complete. Historical PHASE3_BLOCKED.md/v1 logs retained.
- Budget each arm4×(500+1000), all priors/thresholds/seeds frozen. Queue
  seed202700000+100000*arm+100*fold+10*attempt+chain (arms1–4,folds1–40,
  attempt0/1,chains1–4), atomic complete-fold checkpoints/immutable manifests.
  Every initial Rhat>1.05 fold gets exactly ONE seeded attempt1 rerun;
  always select it even if worse, keep primary UUID/diagnostics and selection.
  Never third attempt/favourable subset. Final divergence<=0.1%/finiteRhat,
  report residual Rhat flags and tail ESS review400.
- Checkpoints checkout data/checkpoints/scottish_lower_qs_wave2_2426/<arm>/;
  outputs /root/BF_runs/qs_experiment_w2_out/phase3/. Preserve summaries,
  UUID receipts, diagnostics and rerun-selection ledgers. Local evidence
  results/phase3/ and results/gate_logs/. Stop/report gate failures.
- Previous component array feature3831/3831, tapes160/160at0B, recovery39/42
  zero all-seed misses; all four six-hard-check smokes PASS. Original GRW
  market smokeRhat1.14/bulk16 and full12.3192h probe ETA disclosed before
  grid approval. Frozen input digests in PROGRESS.md unchanged.
- Phase3 not complete until both market arms finish. Then commit evidence,
  Phase4 exact package metrics/pairs/bootstrap classes using SAVED wave1
  references (no refits), and Phase5 fresh frozen-seed byte reproduction.
  No ROI/staking, tuning, history deletion or reference changes.
