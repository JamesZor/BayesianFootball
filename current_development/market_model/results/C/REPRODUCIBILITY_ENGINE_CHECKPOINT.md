# Phase C revision 4 — deterministic checkpoint, not production reproduction

Full-book continuation authorised; C2 deferred. See PHASE_C_REPORT.md for
what passes and what is missing. No sampler, posterior/synthetic fit, beast
session or production byte-identity proof exists. No R05_DONE/T05_DONE claim.

## Snapshot and source

Branch `pi/market-model-phase-c`, dedicated laptop worktree. Existing Manifest,
Distributions 0.25.126, Julia 1.12.1. No package updates or database writes.
Pinned `.cache/datastore_ScottishLower.jls` SHA256:
`c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`.
Load `max_age_hours=10^6`; do not refresh the cache.

`SOURCE_SHA256SUMS.txt` identifies current local implementation/dependencies;
`SCIENTIFIC_SHA256SUMS.txt` identifies historical and new deterministic CSVs.
Neither is evidence of two successful production executions. Source must be
frozen again before future sampling, with fresh seed and run manifests.

From the stream directory:

```bash
sha256sum -c results/C/SOURCE_SHA256SUMS.txt
(cd results/C && sha256sum -c SCIENTIFIC_SHA256SUMS.txt)
```

V3 historical root tables remain unchanged. Copies in `v3_gate/` preserve those
files and their old manifests. Old source manifests retain historical paths;
check old source at `8591099b`, not against revised loaders. Historical notes:
PHASE_C_V3_REPORT.md, REPRODUCIBILITY_V3.md, HANDOVER_V3.md. V1/V2 remain preserved.

## Owned laptop verification

Only `%45`, `%46`, `%47` were created/operated; all closed. No unrelated session
was operated. New deterministic logs:

- `LAPTOP_C4_FIRST_TESTS.txt`: development failure using inherited unnormalised
  HalfNormal helper, 204 pass/one fail; fixed by including prior constants.
- `LAPTOP_C4_REGRESSIONS.txt`: fresh `%46`, t05 205/205 before four additional
  engine assertions; C2-pending 29 pass/10 fail; t04 92/92, t03 131/131,
  t02 131/131, all regression markers.
- `LAPTOP_C4_FINAL_PREFLIGHT.txt`: fresh `%47`, final-source t05 209/209,
  separate C2-pending 29 pass/10 fail, r05 full-book gate + engines + all fixed
  full-panel filters reaches `C05_C4_ENGINE_PREFLIGHT_DONE`.

`t03` preserves published A outputs using a temporary regression directory.
`r05` writes only `v4_preflight/`, not historical root C tables. No current-row
join or two-run production CSV equality is claimed (the auxiliary SQL comparison
command had two parser errors and was stopped; scientific gates did pass).

## Rerun, REPL only

Verify the session name is unused first. Start from worktree root; target the
returned pane ID ONLY. No one-shot Julia or package updates.

```bash
PANE=$(tmux new-session -d -P -F '#{pane_id}' -s pi_julia_mm_c -c "$PWD" \
  'env JULIA_PKG_PRECOMPILE_AUTO=0 GKSwstype=100 julia --project -t 8')
printf 'owned pane: %s\n' "$PANE"
tmux send-keys -t "$PANE" -l -- \
  'include("current_development/market_model/t05_pooled_tests.jl")'
sleep 0.15
tmux send-keys -t "$PANE" Enter
# Wait for the ACCEPTED suite summary and C2_PENDING_REPORTED separately.
tmux capture-pane -t "$PANE" -p -J -S -2000
```

Expect accepted **209/209** and `T05_C4_DETERMINISTIC_DONE`, then explicitly
excluded C2-pending failures and `C2_PENDING_REPORTED`. This is NOT complete
synthetic recovery acceptance. The pending set runs an unchanged integrated
check and evaluates archived thin setting flags, labelled NOT regenerated.
No C2 numerical investigation is authorised by this checkpoint.

In separate commands, wait for each result before continuing:

```julia
include("current_development/market_model/t04_copula_grid_tests.jl") # 92, T04_DONE
include("current_development/market_model/t03_covariance_tests.jl") # 131, T03_DONE
include("current_development/market_model/t02_two_stage_tests.jl")   # 131, T02_DONE
include("current_development/market_model/r05_pooled.jl")
```

The runner replaces six CSVs in `v4_preflight/`: book inventory, Gate 1 summary,
fixture diagnostics, scalar Gate 1 checks, state-engine gates, fixed-parameter
filter summaries. Preserve them before diagnostic changes. It does NOT sample
or evaluate, and its final marker is `C05_C4_ENGINE_PREFLIGHT_DONE`.
Close only your owned pane after evidence capture: `tmux kill-pane -t "$PANE"`.

## Future beast runs

Not yet executable as production. Implement sampler/recovery/evaluation and
complete acceptance first. Read the remote guide; use a dedicated checkout
`/root/BF_runs/market_model_c`, owned session `pi_mm_c`, returned pane ID only,
16 Julia threads, core pinning, BLAS=1; logs `/root/BF_runs/logs/market_model_c/`.
Inspect load and session/path ownership, copy pinned cache preserving metadata,
use existing Manifest and never operate human/manager panes. No database writes.
Both full beast runs must use identical frozen source/seeds and reproduce every
scientific CSV byte-for-byte before completion. Existing hashes are preflight
and historical evidence only.
