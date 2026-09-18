# T013 — `MultiScaleGRW` has no dispatch arm on the builder dynamics extractors, so MatchDay cannot price any GRW model

| Field | Value |
|---|---|
| Severity | high |
| Area | `src/models/pregame/builder/engine.jl`, `current_development/match_day_inference/` |
| Status | done |
| Raised | 2026-09-12 |

## Evidence

The live Scottish Lower card of 2026-09-12 was to be executed across two
models. Model A (`m12_joint_hybrid_synergy`, TimeDecay) priced, committed and
settled normally. Model B (`m05_joint_production_wealth_grw`, `MultiScaleGRW`)
never produced a stake sheet: `r12_live_slate_grw_20260912.jl` cleared every
operational gate (fixtures, Betfair tick health, lineups, Fold-43 team-map
coverage, Option B recipe) and then died inside `price_slate`:

```
ERROR: LoadError: MethodError: no method matching _cb_extract_dynamics(
    ::MCMCChains.Chains{...}, ::BayesianFootball.Models.PreGame.MultiScaleGRW, ::String, ::Int64)

Closest candidates are:
  _cb_extract_dynamics(::MCMCChains.Chains, ::Union{StaticZeroDynamics, TimeDecayDynamics}, ::String, ::Int64)
   @ BayesianFootball ~/bet_project/BayesianFootball/src/models/pregame/builder/engine.jl:500

Stacktrace:
 [1] extract_parameters(...)  @ src/models/pregame/builder/engine.jl:530
 [2] matchday_latents(...)    @ src/MatchDay/inference.jl:620
 [3] price_slate(...)         @ src/MatchDay/slate.jl:294
 [4] top-level scope          @ current_development/match_day_inference/r12_live_slate_grw_20260912.jl:273
```

The failure is unconditional and has nothing to do with the T−25 window, the
commit gate, or the posterior's convergence state — it happens during pricing,
before any ledger code is reached.

## Root cause

`src/models/pregame/builder/engine.jl` defines both composable-builder dynamics
hooks only for the two non-GRW dynamics types:

- `_cb_extract_dynamics(::Chains, ::Union{TimeDecayDynamics,StaticZeroDynamics}, ::String, ::Int)` — line 500
- `_cb_oos_dynamics(::Union{TimeDecayDynamics,StaticZeroDynamics}, draw, lineup_map, ...)` — line 506

`extract_parameters` calls the first unconditionally at line 530
(`dyn_nt = _cb_extract_dynamics(chain, model.dynamics, "dyn", n_teams)`), so any
`ComposableCountModel` whose `dynamics` is `MultiScaleGRW` cannot have its
parameters extracted at all. Every consumer of `extract_parameters` inherits the
failure; MatchDay is simply the first caller that reaches it in production.

Working implementations of both methods already exist, but as **inline
definitions in a runner**, not in `src/`:
`current_development/match_day_inference/r13_t25_grw_backtest.jl:40-130`. That
is why the T−25 GRW backtest in `results/REPORT_T25_GRW_2627.md` produced
results while the live slate could not — the backtest monkey-patches the
methods into `Builder` at load time and the live runner never did.

The two arms are not interchangeable. The TimeDecay `draw.α` is a per-team
vector; the GRW `draw.α` is a `(team, step, sample)` trajectory whose
out-of-sample level is the final step. A GRW model routed through the TimeDecay
arm would not error — it would silently read the wrong axis.

## Reproduction

```bash
julia --project -t 8 current_development/match_day_inference/r12_live_slate_grw_20260912.jl --dry-run
```

Run it at any wall-clock time at or after the script's `R12_AS_OF_UTC`
(2026-09-12T13:35:00Z), so that the early pre-flight branch does not
`exit(0)` first. It reaches `=== G-E OPTION B RECIPE ===` and then raises the
`MethodError` above.

Note: Julia buffers stdout when it is redirected to a file, so a plain
`> log 2>&1` loses the tail of the buffer when the process dies and the run
looks like a silent death. Run it under a pty (`script -qefc "..." log`) to see
the exception.

## Blast radius

- **Every** `MultiScaleGRW` model is unpriceable by MatchDay: `price_slate`,
  `matchday_latents`, and the live and replay consoles.
- Any non-MatchDay caller of `Builder.extract_parameters` on a GRW model.
- The defect is invisible until a GRW model is actually served, because the
  convergence audit, fold selection, team-map coverage check and Option B recipe
  all pass first — the operational gates give a false sense of readiness.
- Scope creep risk: `r13_t25_grw_backtest.jl` already carries one private copy.
  Each new GRW runner that pastes a second copy makes the eventual `src/` fix
  harder and invites the two copies to drift.

## Fix applied (partial — runner layer only)

The hooks were lifted out of `r13` into a shared, documented loader and wired
into the live runner:

- **added** `current_development/match_day_inference/l12_grw_dynamics_hooks.jl`
- **changed** `r12_live_slate_grw_20260912.jl` — `using MCMCChains, Statistics`
  plus `include(joinpath(@__DIR__, "l12_grw_dynamics_hooks.jl"))`

This unblocks the live GRW slate. It is deliberately **not** the real fix.

## Resolution

The production dispatch methods graduated to
`src/models/pregame/builder/grw_dynamics.jl`, backed by the validated trajectory
reconstruction in
`src/models/pregame/components/dynamics/team_level/multiscale.jl`. The temporary
`l12_grw_dynamics_hooks.jl` shim and the private `r13` monkey patch were removed;
both runners now use the standard `BayesianFootball.Models.PreGame.Builder`
implementation from `src/`.

`test/builder_tests.jl` now covers the supported-dynamics dispatch, synthetic-chain
trajectory reconstruction and shape, the final-step OOS carry-forward property,
and end-to-end finite positive rate extraction through
`PreGame.extract_parameters`.

## Verification — 2026-09-18

- `test/builder_tests.jl`: **106 / 106 passed**, including all 10 new integration
  assertions.
- `test/test_multiscale_grw.jl`: **124 / 124 passed**.
- `test/runtests.jl`: **4,043 passed, 1 broken**. The optional PostgreSQL ledger
  testsets skipped because the inherited local `BF_DB_URL` was unreachable; the
  configured `betdb` connection was exercised successfully by both operational
  runners below.
- `r12_live_slate_grw_20260912.jl --dry-run`: reached
  `=== G-F BATCH HEADER ===`, printed the complete stake sheet, and exited without
  a dynamics `MethodError`.
- `r13_t25_grw_backtest.jl`: completed all 14 tracks, rewrote the three ledgers and
  report, and performed no database write.

## Proposed fix (the real one)

Move both methods into `src/` beside the dynamics they dispatch on, so no runner
needs a shim:

- Preferred: `src/models/pregame/builder/grw_dynamics.jl`, which
  `AGENTS.md` already names as the file that registers `MultiScaleGRW` with the
  builder. **That file does not exist on `feat/scottish-lower-goal-decomposition`** —
  it belongs to Task 013 (`feat/grw-player-lineup-hybrid`). Creating it here
  would duplicate that branch's work and conflict on merge, which is why this
  ticket stops at the runner layer.
- Alternative if Task 013 is abandoned: widen the two `Union`s in
  `engine.jl:500` and `:506` and add the GRW arms directly there.

Trade-off: the shim is fast and reversible but leaves three files that must
agree (`engine.jl`, `l12`, `r13`). The `src/` fix is correct but must be
sequenced against Task 013 to avoid a merge conflict in the same file.

## Acceptance criteria

1. `r12_live_slate_grw_20260912.jl --dry-run` reaches `=== G-F BATCH HEADER ===`
   and prints a stake sheet, with no `MethodError`.
2. `r13_t25_grw_backtest.jl` reproduces its recorded
   `results/REPORT_T25_GRW_2627.md` numbers after its private copy is deleted in
   favour of the shared definition.
3. A GRW model and a TimeDecay model priced through `price_slate` on the same
   fixture both return finite `p_model` for every market in
   `MD.canonical_markets()`.
4. `_cb_oos_dynamics` for GRW is asserted to read the **final** trajectory step:
   a fit whose last step differs from its first must price differently from one
   that reads step 1.
5. When the `src/` fix lands, `l12_grw_dynamics_hooks.jl` is deleted and both
   runners still pass (1) and (2).

## Scope guard

Do **not** fix the unrelated convergence defect in the same change: the
Fold-43 posterior for `scottish_lower_multiscale_grw_2426`
(UUID `f870dbb7-9df0-4dae-a84a-cf570cf8113e`) loads as
`converged: false <- failed: R-hat`, and `r12` sets
`require_converged = false` to bypass the gate. That is a modelling problem
needing a refit, is independent of this dispatch defect, and fixing the
dispatch does **not** make that posterior fit to size money.

Do **not** touch `src/MatchDay/`. The MatchDay layer is correct here; it is
calling a builder API that is missing an arm.

Do **not** rewrite the GRW reconstruction maths while moving it. The
non-centred parameterisation, the cross-team centring, and the
`(team, step, sample)` permutation are load-bearing and already validated by the
T−25 backtest.
