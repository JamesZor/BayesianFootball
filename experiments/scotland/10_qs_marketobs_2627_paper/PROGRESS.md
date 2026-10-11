# 26/27 paper trade progress (TODO 045, Claude CLI)

Brief: `experiments/claude_qs_2627_paper_brief.md`. Sentinels: P27_HANDOVER / P27_BLOCKED / P27_DONE.

## Phase 0 — setup, data audit, gates, plan (IN PROGRESS)
- [x] Folder, TODO 045 (README row), DECISIONS_LOG (rulings 1–9), PROGRESS.
- [ ] Beast checkout `/root/BF_runs/qs_2627`, `jl` REPL `claude_p27`.
- [ ] DataStore rebuilt read-only from betdb; sha256; 79 fixtures + results.
- [ ] Coverage table by match day (79 / 70 / 60 at T−25m).
- [ ] Odds frames at T−25m (back / mid / LTP) + mid TWA close.
- [ ] Gates: inversion reproduction (r3), engine reproduction (r9), m12 fold inventory (r4), qs extension dry run.
- [ ] Sampling plan (folds, ETA, memory).

## Beast
- Checkout: `/root/BF_runs/qs_2627`. Logs: `/root/BF_runs/logs/qs_2627/`. Outputs: `/root/BF_runs/qs_2627/out/`.
- REPL: `jl` session `claude_p27` (tmux `jl-claude_p27`).
