# Wave 4 progress — Phase 3 running (Claude CLI)

- Handover from pi (Codex fault + weekly quota): Phases 0–2 accepted; Claude CLI runs Phases 3–4 under `experiments/claude_qs_backtest_phase34_brief.md`. Rulings 1–3 binding.
- Phase 3 design frozen in DECISIONS_LOG (2026-10-10, before any grid number): gate first, 5 models × 4 snapshots × 5 policies, CLV vs napkin close, 8-week block bootstrap and paired m12 differences.
- Runner `l02_phase3_grid.jl` / `r02_phase3_grid.jl` (`f059662b`) launched in owned beast REPL `jl-claude_qsbt_p3`; log `/root/BF_runs/logs/qs_backtest/phase3_grid.log`.
- Cold-load smoke of native ScenarioKelly in a fresh REPL: 51/51 PASS.
- C0 has draw-level latents (512 per fixture), so all its cells are computable.
