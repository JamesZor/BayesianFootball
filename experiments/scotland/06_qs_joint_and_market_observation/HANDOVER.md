# Wave 2 — blocked continuation brief (2026-10-09)

**Phase 3 stopped; no retry pending manager review.** Read PROGRESS.md,
DECISIONS_LOG.md and PHASE3_BLOCKED.md. TODO 041 BLOCKED, assignee pi.

- Branch exp/qs-goal-vs-market; frozen grid source 8f285a6a on beast
  `/root/BF_runs/qs_experiment`. No src/harness changes. Julia beast only,
  -t16, pinned cores, BLAS 1. Other sessions never touched.
- Approved order grw_joint → qs_joint → grw_marketobs → qs_marketobs,
  W2 4×(500+1000). Launcher stopped at QS, 20:14:20 UTC.
- GRW COMPLETE UUID 98447840-e95c-420f-92b4-2db6545506a3: 40 folds,
  Rhat 1.01008, bulk641.78/tail324.47, zero divergences/no reruns.
  One tail ESS<400 review flag retained. Include 2008.6993 s.
- QS PERSISTED (not scoring-complete) UUID 2a5eee90-3402-40ce-a5e3-f4c6b57fc8ea:
  40 folds, four hard checks including DB parity pass, convergence review
  passes; Rhat1.00937, bulk832.35/tail761.02, zero divergences/reruns due.
  Include FAIL at 1962.5365 s during harness scoring, after save/check writes.
- Error: expected exactly one role=:control; found 0. l05_grid.grid_stage
  has control=nothing, but run never supplies the completed GRW control.
  Harness.grid correctly self-controls GRW, not role=:candidate QS/market arms.
- Proposed experiment-only correction (NOT applied): construct explicit
  RunRef from grw_joint final UUID in its grid_summary CSV; thread into
  primary/rerun grid_stage calls for non-control arms. Preflight completeness;
  add offline forwarding regression. No core/harness changes or new sampling.
- After manager authorisation, fresh phase3 attempt label/REPLs can resume:
  GRW receipt returns saved fit; QS completed DB fit/checkpoints avoid refit,
  complete missing scoring/receipt then advance to market arms in fixed order.
  Neither market arm started. No paired predictive conclusions yet.
- ALL owned panes CLOSED: launcher %390, successful GRW %391, failed QS %392
  after read-only diagnostic audit. No Julia workers remain owned by us.
  Both DB fits/all 80 initial checkpoints retained.
- Durable queue seeds frozen before sampling:
  202700000+100000*arm+100*fold+10*attempt+chain (arms1–4, folds1–40,
  attempt0/1, chains1–4). Atomic complete-fold checkpoints and immutable
  manifests validated before real runs. Do not change budgets/seeds/priors.
  Every initial Rhat>1.05 gets ONE attempt-1 rerun, always selected even if
  worse; no third attempt/favourable subset. Divergence<=0.1%/finite Rhat
  final gate, tail ESS review400.
- Checkpoints in checkout: data/checkpoints/scottish_lower_qs_wave2_2426/.
  Logs: /root/BF_runs/logs/qs_experiment_w2/phase3_launcher_v1.log and
  phase3_grid_v1_<arm>.log. Outputs: /root/BF_runs/qs_experiment_w2_out/phase3/.
  Local committed evidence: results/phase3/ and results/gate_logs/phase3_*.
- Phase 1 array feature/canonical identity b006743f revalidated component
  3831/3831, tapes160/160 at0B, recovery39/42 zero all-seed misses;
  evidence d0d933ae. Four six-hard-check smokes PASS, evidence a7b332a6,
  retaining joint passes and all review flags. 12.3192h ETA / GRW market
  Rhat1.14 bulk16 warning disclosed before explicit grid approval.
- Frozen table/manifest/snapshot digests in PROGRESS.md unchanged. Historic
  failures preserved. No reference refits, ROI/staking or prior tuning.
- Phase 3 incomplete; Phase 4 full prescribed report/bootstraps/leaderboard
  and Phase 5 byte-for-byte fresh-REPL frozen-seed reproduction remain.
