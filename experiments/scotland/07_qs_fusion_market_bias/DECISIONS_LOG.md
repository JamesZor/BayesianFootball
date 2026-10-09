# Wave 3 — decisions

## 2026-10-10 — scope and Phase 0

- Scope set by James 2026-10-10 (brief `experiments/claude_qs_fusion_bias_brief.md`). Implementer: Claude CLI; manager: the launching Claude session.
- Wave-2 manager rulings carry over: C2-pending t05 child set excluded from acceptance; FD audit = step-size convergence with Richardson ≤ 1e-8 plus exact AD ≤ 1e-10; recovery = 3 seeds, no parameter misses all three, pooled coverage ≥ 80%; control forwarding; deterministic control rescoring with exact-equality audit; wave-1 market module loaded at top level (namespace provenance).
- Phase 0 reuses wave 2's gate list, adds `test/test_market_rate_observation.jl` (its outputs redirected to `/root/BF_runs/qs_experiment_w3_out/phase0/<gate>` through `QSX2_TEST_OUTPUT`, so wave-2 outputs are untouched) and drops the xG coverage gate (unchanged snapshot, already audited for all 40 folds in wave 2). Pins are checked in shell before any Julia.
- Phase 0 PASS at `122cbc37` (`PHASE0.md`): all 16 gates pass; C2-pending 29/39 excluded as ruled; pins match. Owned panes closed.
