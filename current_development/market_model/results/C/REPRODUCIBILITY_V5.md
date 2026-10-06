# Phase C revision 5 — blocked exact-mode verification, not production reproduction

Branch `pi/market-model-phase-c`; statistical source
**120ac906922e7be7ea175ba9763920be4b875085** (later documentation-only commit
changes no statistical code). Source file identity: `SOURCE_SHA256SUMS.txt`. Read `PHASE_C_REPORT.md` before
executing anything. **No recovery or production run was launched at revision5.**
No third numerical variation is authorised at this new stop boundary.

## Frozen inputs

- Existing Manifest, no package additions/updates; laptop Julia1.12.1.
- Pinned `.cache/datastore_ScottishLower.jls`, SHA256
  `c786e2fc03be0494ae3b9d447f0ad1840a787de19c171ea929b1f8cb46b423b4`,
  max_age_hours10^6. Cache and all scientific settings unchanged.
- `fullbook_seeds.toml` unchanged: generation3962, recovery chains4961–4964.
- Exact failing chain coordinate:
  `[-3.549527585137839,-4.460929121755582,-2.821549571263347,7.9567722491577495]`.
- Same full 517-fixture / 91-week schedule, original initial-state prior,
  exact log-space Dirichlet draws, cutoff10 double-Poisson grid.

## Diagnostic implementation and results

`t05_newton_regression_tests.jl` is included within t05 acceptance, not C2-pending.
Its original-termination replay follows the identical forward prefix until the
first stalled fixture12476625; it returns that book's diagnostic context and
stops before a state update. The captured gradient exactly matches revision4.
New authorised termination is compared against a tighter damped Newton path
started at its accepted point; analytic prior and exact AD likelihood derivatives
are used. Displacement stopping1e-13 is for the reference only, not production.

The new solver accepts stalled_converged at decrement3.382117013018548e-12,
but mode gap4.0159674463691175e-7 exceeds1e-8 and Laplace marginal gap
6.658597637709818e-7 exceeds1e-9. Reference gradient residual7.67031e-12.
The logdet precision change explains the marginal gap, not a changed likelihood
or grid. Inputs/outputs are in `v5_newton/newton_regression.csv` and
`newton_regression_book.csv`.

Focused `%54`: 5pass/2fail/7. Fresh `%55`: t05 282pass/2fail/284, no
T05_C5_DETERMINISTIC_DONE/T05_DONE. Same two failed checks, same numerical outputs.
No third solver variation, recovery restart or threshold change. Both panes closed.
Fresh same `%55` regression checks: C2-pending29pass/10fail/39 (excluded),
t04 92/92, t03 131/131, t02 131/131, their markers and C05_C5_REGRESSIONS_FINISHED.
All inherited270 t05 checks and six stopping/accounting units pass.

Logs: `LAPTOP_C5_NEWTON_DEVELOPMENT.txt`, `LAPTOP_C5_FINAL_T05.txt`,
`LAPTOP_C5_FINAL_TESTS.txt`. The last retains an initial REPL accounting
world-age error: include and call were in the same try/finally expression.
A separate command after include returned correctly writes both accounting CSVs.
It did not rerun the solver or change scientific code.

## Reproducing the blocked checks (review only)

This recipe reproduces **failure**, not acceptance or a recovery remedy.
Check the dedicated session name is unused and create your own pane. From the
worktree root, Julia must run through a persistent REPL, never one-shot:

```bash
PANE=$(tmux new-session -d -P -F '#{pane_id}' -s pi_julia_mm_c -c "$PWD" \
  'env JULIA_PKG_PRECOMPILE_AUTO=0 GKSwstype=100 julia --project -t 8')
printf 'owned pane: %s\n' "$PANE"
tmux send-keys -t "$PANE" -l -- \
  'include("current_development/market_model/t05_pooled_tests.jl")'
sleep 0.15
tmux send-keys -t "$PANE" Enter
tmux capture-pane -t "$PANE" -p -J -S -2000
```

Wait for the expected t05 failure, **then issue a separate REPL command**, so
Julia1.12 sees the newly defined module/method world:

```julia
PooledMarket.write_newton_accounting("current_development/market_model/results/C/v5_newton"; run="exact_regression")
PooledMarket.write_newton_accounting("current_development/market_model/results/C"; run="exact_regression")
include("current_development/market_model/t05_c2_pending_tests.jl")
include("current_development/market_model/t04_copula_grid_tests.jl")
include("current_development/market_model/t03_covariance_tests.jl")
include("current_development/market_model/t02_two_stage_tests.jl")
```

Input/capture/cleanup target **only your returned pane ID**. Tests use temporary
A outputs, preserving published A/B/B2/B3. No beast sampling is part of this recipe.

## Accounting and hashes

`newton_termination.csv` and its identical `v5_newton/` copy record actual
`exact_regression` calls: decrement0, step0, stalled_converged2, maxδ
3.382117013018548e-12, acceptance gate_pass=true. Unit mocks were reset before
these solves; the old forward-prefix diagnostic does not enter new-rule counts.
This passing accounting gate does **not** make the mode/marginal verification pass.
Recovery/real runners have finally-write hooks but were not run; their counters
or fit diagnostics do not exist for revision5.

```bash
cd current_development/market_model
sha256sum -c results/C/SOURCE_SHA256SUMS.txt
(cd results/C && sha256sum -c SCIENTIFIC_SHA256SUMS.txt)
```

The scientific manifest contains historical C diagnostics and the four new
regression/accounting CSVs. It is a blocked-evidence manifest, **not two-run
production identity**. No fresh real full Gate1 regeneration or production
reproduction was attempted after the mandatory exact-case failure.

## Remote and historical evidence unchanged

No beast pane was created or operated. Read-only status/load check found the
checkout `/root/BF_runs/market_model_c` still at **9de98548**, with preserved
untracked historical recovery output directories. No checkout/reset/cleanup,
Manifest/cache write, rsync or recovery command was sent there in this session.
Original logs and both byte-identical generated binaries remain where
`HANDOVER_V4.md` / `REPRODUCIBILITY_V4.md` records them. Their binary hash is
`de6a986b1caebc2a2ad1386308c7f4613bf69ea2f71440778783476c348da0d1`,
not a fit or production-run identity. V1/V2/V3/engine checkpoint records remain.

Before any next sampling: review the demonstrated stopping/accuracy mismatch,
explicitly authorise any further numerical remedy, pass the exact-case gate,
then finish recovery and the remaining full-book priorities. C2 stays deferred.
