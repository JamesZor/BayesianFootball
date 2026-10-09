# Wave 2 — manager handover (2026-10-09)

**BLOCKED on component gate at ec1a4137. Phase 1 not accepted.**

- Branch exp/qs-goal-vs-market, TODO 041 BLOCKED, assignee pi.
- Phase 0 remains PASS; its complete source/evidence is in PHASE0.md.
- Manager option 1 implemented at 4a71c02e: Main.QSMarketArm reused, exact
  table-value parity added; original guard and saved manifest untouched.
- Table preparation PASS; target counts 710/595/78/517, all IDs/log rates/flags
  exactly match wave-1 prepared panel after CSV round-trip. Frozen table/hash
  and parity evidence pushed at ec1a4137; SHA256 in DECISIONS_LOG/PROGRESS.
- Fresh component gate 5533 pass / 372 fail: standalone density assertion,
  370 finite-difference assertions (GRW 65, QS 305), GRW defence micro-scale
  recovery miss. Compiled/fresh ReverseDiff and ForwardDiff comparisons pass,
  but AD is NOT accepted. Unconditional MARKET_AD PASS print lines are misleading.
- All frozen priors/truths/seeds/budgets unchanged; all 14 recovery intervals
  preserved in results/synthetic_recovery.csv (13 cover). No retry/tuning/fix.
- Allocation/builder/harness gates did not start after the component failure.
  No smoke, grid, scoring, ROI/staking or wave-1 reference refit.
- Next manager review: inspect density/context helper for standalone likelihood
  discrepancy, assess finite-difference failures without loosening 1e-6, and
  address the predeclared recovery miss without tuning seeds/truths. See
  PHASE1_BLOCKED.md and logs/phase1_market.summary.txt for exact evidence.
- Beast checkout /root/BF_runs/qs_experiment detached at ec1a4137; environment
  symlinks/cache unchanged. Table and saved manifest digests rechecked unchanged.
  Raw logs in /root/BF_runs/logs/qs_experiment_w2/. All owned sessions closed;
  other sessions untouched. No further Julia until manager authorisation.
- Read PROGRESS.md and DECISIONS_LOG.md first on resumption. Historical first
  provenance stop retained in PHASE1_BLOCKED.md; resolved only by manager ruling.
