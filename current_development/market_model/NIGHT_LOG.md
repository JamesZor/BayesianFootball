# Night log, TODO 039 (2026-10-06)

The manager's running record while the human sleeps. If the manager's context is compacted,
**read this file first**: it holds the current state, the next action and every answer given on
the human's behalf.

## Human instructions (2026-10-06, before sleeping)

- **Scope:** run Phase B3, then Phase C on the manager's defaults.
- **Phase B3:**
  - (a) a tail/asymmetry check on R6's quality/style steps;
  - (b) score-grid comparison: double Poisson (baseline), Dixon–Coles global ρ, bivariate
    Poisson λ₃, and a Frank copula with Poisson margins.
- **Phase C:**
  - **double Poisson kept regardless of B3**; any better grid goes to a later Phase C2 that the
    human reviews first;
  - 1X2-only books inform supremacy only;
  - quality/style GRW with independent steps; independent equal-side u_m;
  - Laplace–Kalman engine (Gaussian approximation of each book's Dirichlet/Beta likelihood,
    learned concentration n);
  - totals-only books inform the total only.
- **Authority:** the manager answers pi's in-scope questions and logs each below. Anything that
  changes scope, `src/`, thresholds or data waits for the human.
- **Limits:** on a quota limit or stall, wait for the reset and resume the same model
  (`openai-codex/gpt-6.1-sol`).
- **Monitoring:** watcher scripts plus the manager; no subagent.
- **Context:** a fresh pi session per phase. If pi's context passes ~75% mid-phase, pi writes
  `HANDOVER.md` and is restarted fresh.
- **Git:** merge reviewed phases into `feat/market-model-inversion-eda` locally (the stream branch
  is not pushed); pi pushes only its own branch.
- **Morning:** `MORNING_SUMMARY.md` plus a chat summary.
- **Don't** start Phase D or Phase C2. Don't touch the human's sessions: `pi_julia_mm_b` (their
  Claude pane), `agents`, `0`, `library`.

## State

| Time | Event |
|---|---|
| 2026-10-06 start | Stream branch at `19b83990` (A, B, B2 merged and reviewed). |

## Answers given on the human's behalf

None yet.

## Next action

Write the B3 brief, create worktree `BayesianFootball-market-model-pi-b3`, launch pi, start the
watcher.
