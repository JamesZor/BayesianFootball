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
| B3 launched | Brief `97a2b201`; worktree `BayesianFootball-market-model-pi-b3`, branch `pi/market-model-phase-b3`; pi pane `%33` (session `agent_pi_market_model_b3`), gpt-6.1-sol; codex quota 30% (5h) / 18% (7d) at launch. Watcher: `scratchpad/watch_pi_night.sh %33 B3`. |

## Human requests during the night

- **2026-10-06, mid-setup:** "can the home advantage be correlated with the attack defense of
  the team". Added to the Phase C brief as §2.4 side rungs, two-stage on isolated rates against
  C0:
  - H1 split HA (γ_att on the home rate, γ_def off the away rate);
  - H2 quality-linked HA (κ·q_home).

  Added unless the human objects. It doesn't change C1/C2.

## Answers given on the human's behalf

None yet.

## Next action

1. While B3 runs, draft the Phase C brief.
2. When the watcher fires:
   - on QUESTION, LIMIT or IDLE, handle it;
   - on the sentinel, review B3: fresh laptop tests, beast rerun plus hash check, then merge.
3. After B3 is merged, launch Phase C.
