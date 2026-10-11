# Wave 5 progress (TODO 044, Claude CLI)

Brief: `experiments/claude_qs_wave5_config_brief.md`. Sentinels: W5_HANDOVER / W5_BLOCKED / W5_DONE.

## Phase 0 — setup and gate (PASS, accepted by manager)
- [x] Folder, TODO 044 (registry row), DECISIONS_LOG (rulings 1–10 + implementation decisions).
- [x] Runner: `l00_wave5.jl` (module QSWave5), `l01_phase0_gate.jl`, entry `r00_phase0_gate.jl`.
- [x] Beast checkout `/root/BF_runs/qs_wave5` at `21f3514f`; datastore cache copied (sha256 c786e2fc…, same as wave 4).
- [x] Gate PASS (runner `21f3514f`, log `logs/phase0.log`, outputs `results/phase0/`):
  - m12 × canonical × T−25m 2.773493294562897 / 1132, bit-identical; qs × SK (all legs) 2.154029744590488 / 952, bit-identical.
  - Per-season MDD (reset to 1.0): m12 24/25 −13.5952, 25/26 −16.1523; qs SK 24/25 −7.8066, 25/26 −10.4742 (all within 0.01 pp).
  - 24/25-only runs equal the full runs' first 49 slates exactly (bankroll path, MDD, bets: m12 2.08953 / 575, qs SK 1.65279 / 503).
  - Basket column-subset == problem rebuilt on leg-filtered books (B1, B2; 6 days each).
  - Seal refuses the evaluation half. DB audit unchanged (10 tables).

## Phase 1 — selection half, 24/25 only (DONE, awaiting manager release)
- [x] Runner `l02_phase1_select.jl` + `r01_phase1_select.jl`; final run `6d779be2` (log `logs/phase1.log`;
  attempt 1 `ccddaa79` stopped on a B4-filter bug, `logs/phase1_attempt1.log`). DB audit unchanged both runs.
- [x] Per-direction table `results/phase1/phase1_directions.csv` (ungated flat 0.25, canonical SK, m12).
- [x] λ curves for 6 baskets `phase1_lambda_curve.csv`; matched λ + selection `phase1_selection.csv`;
  sensitivity `phase1_sensitivity.csv`; per-day solve diagnostics `phase1_basket_diag.csv`.
- [x] Selected **B1_canonical at λ 12** (g 17.520×10⁻³, MDD −13.07%); fallback B1 at λ 23. `FREEZE.md` pushed.
- [x] Manager released FREEZE `04fdba71`.

## Phase 2: evaluation, decision, paper-trading spec (DONE)
- [x] Runner `l03_phase2_eval.jl` + `r02_phase2_eval.jl` (`716c84a1`), the only `release!` caller. DB audit unchanged.
- [x] Results `results/phase2/` (`0f600b92`): summary (all wave-4 columns, both bootstrap intervals,
  paired d), directions, markets, daily, bets (gz), solve diag, decision.
- [x] Ruling 8: **paper-trade the frozen config** (25/26 g 9.70×10⁻³, MDD −19.26% vs bound −20.39%).
- [x] `REPORT.md`, `PAPER_TRADING_CONFIG.md`.
- [x] TODO 044 closed; beast pane %458 closed.

## Beast
- Checkout: `/root/BF_runs/qs_wave5` (detached worktree of `/root/BayesianFootball`).
- Logs: `/root/BF_runs/logs/qs_wave5/`.
- REPL: tmux session `claude_w5_repl`, pane **%458** (julia --project -t 16, OPENBLAS 1; runner sets BLAS 1 + pinthreads).
