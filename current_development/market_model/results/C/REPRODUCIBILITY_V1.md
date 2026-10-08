# Phase C blocker preflight — reproduction, not production acceptance

This is a **single fresh laptop diagnostic execution**, not two fresh beast
production runs. No `R05_DONE`, posterior, converged rung, synthetic recovery or
byte-identical production rerun is claimed. `SCIENTIFIC_SHA256SUMS.txt` covers
only the three diagnostic CSVs. `SOURCE_SHA256SUMS.txt` identifies the diagnostic
code independently of later documentation commits.

## Inputs and environment

- Base reviewed B3 commit: `d287c7454f78a48b8567401c41aa6c913c455d2e`.
- Branch: `pi/market-model-phase-c`, dedicated task worktree.
- Pinned cache: `.cache/datastore_ScottishLower.jls`, SHA256
  `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`.
- `MarketModelConfig`: `scottish_lower_2425_2526()`. Phase A rates and book gates
  unchanged; cache loaded with `max_age_hours=10^6`. No new book/fixture exclusion.
- Julia 1.12.1, Distributions 0.25.126, 8 threads, BLAS=1; existing Manifest.
  No package installation/update. Initial package precompilation is not an update.
- Owned fresh laptop session `pi_julia_mm_c`, pane `%39`; no beast session.
  Only that owned pane was operated, then closed after evidence capture.

## Fresh owned laptop REPL

From the task worktree root, after verifying that the chosen session name is
unused (never operate a similarly named human/manager session):

```bash
PANE=$(tmux new-session -d -P -F '#{pane_id}' -s pi_julia_mm_c -c "$PWD" \
  'env JULIA_PKG_PRECOMPILE_AUTO=0 GKSwstype=100 julia --project -t 8')
printf 'owned pane: %s\n' "$PANE"
tmux send-keys -t "$PANE" -l -- \
  'include("current_development/market_model/t05_laplace_preflight_tests.jl"); include("current_development/market_model/r05_laplace_preflight.jl")'
sleep 0.15
tmux send-keys -t "$PANE" Enter
# Wait for the anchored markers before capture/cleanup.
tmux capture-pane -t "$PANE" -p -J -S -200
# After output is saved and verified:
tmux kill-pane -t "$PANE"
```

Recorded outcomes in `LAPTOP_PREFLIGHT.txt`:

- 11/11 **diagnostic** mathematical tests, 3.2 s: `T05_PREFLIGHT_DONE`.
- Preflight runner: `C_PREFLIGHT05_BLOCKED`.
- Explicit environment line: Julia/Distributions/thread/BLAS versions above.
- No retry, subset change or tolerance adjustment. The two quadrature orders are
  part of the original diagnostic, not attempts to make a failed model pass.

These are not `t05_pooled_tests.jl` or `r05_pooled.jl`. All missing production
acceptance items and the required contract corrections are in `PHASE_C_REPORT.md`.
The n grid is fixed at the prior centre and its quarter/fourfold values, not a
fabricated posterior. Predictions are centred at the isolated rates with
specified side spreads. Totals diagnostics retain every gate-2 refused book;
the first approved Phase A optimisation start is used without selecting on fit.

## Hash and repository checks

```bash
cd current_development/market_model
sha256sum -c results/C/SOURCE_SHA256SUMS.txt
cd results/C
sha256sum -c SCIENTIFIC_SHA256SUMS.txt
```

Before handoff: `./scripts/todo.sh check` passes (24 tasks; AGENTS.md under its
size budget), and `git diff --check` passes. TODO 039/index say BLOCKED, not
completed. Existing A/B/B2/B3 artifacts and all `src/` files remain untouched.
This evidence justifies stopping, not promoting Phase C.
