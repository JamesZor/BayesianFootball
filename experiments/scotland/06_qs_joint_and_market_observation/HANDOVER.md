# Wave 2 — manager handover (2026-10-09)

**Phase 1 PASS; Phase 2 loader fix/retry authorised and applied, pending fresh smoke.**

- Branch exp/qs-goal-vs-market, TODO 041 IN_PROGRESS, assignee pi.
- Phase 0 PASS, evidence in PHASE0.md. Phase 1 source 153239a7, evidence PHASE1.md.
- Main.QSMarketArm reused. Exact wave-1 table parity/counts passed; original
  provenance guard/manifest untouched. Frozen SHA256 in PROGRESS/DECISIONS_LOG.
- Likelihood diagnosis ran BEFORE correction: _observe returns ll and production
  parent accumulates it. Standalone test seam was wrong; src unchanged. Normal
  likelihood equals independent constants-inclusive hand formula at 1e-12.
  Linked versus unlinked density difference equals log(sigma), as required.
- Exact AD <=4.2e-16 at 1e-10; all historical failing coordinates plus 50 others
  per model included in three-point FD audits. All convergence/Richardson gates
  pass; both arithmetic precisions and floor estimates preserved.
- Recovery 39/42=92.86%; no parameter misses all three seeds. All 42 intervals,
  truths/seeds and coverage flags saved; first-seed CSV untouched, no tuning.
- Component 3787/3787; tapes 150/150+QS 2/2+MRO 8/8 (0 B); builder 106/106;
  harness 179/179. Earlier failures remain archived in PHASE1_BLOCKED.md.
- Phase 2 source 73147ce0: candidates.jl:5 include undefined in the bare
  Wave2Candidates module. No checks, fits, registration or DB writes started;
  other three arms did not run. See PHASE2_BLOCKED.md.
- Manager authorised one-line Base.include(@__MODULE__, ...) loading fix.
  Applied; next commit/push and fresh smoke with v2 evidence names. No frozen prior/seed/input or
  threshold changes. Report hard checks, review flags and ETA; ASK BEFORE GRID.
- No smoke, grid, predictive scoring, ROI/staking or wave-1 reference refit yet.
- Beast source 73147ce0; environment symlinks/cache unchanged. All owned panes
  closed after evidence retrieval; other sessions untouched. Raw logs outside
  checkout at /root/BF_runs/logs/qs_experiment_w2/.
- Read PROGRESS.md and DECISIONS_LOG.md first on resumption.
