# Work package — Zero-allocation compiled gradients in the builder engine (blocks TODO 031 W2)

You are the implementer, running as a Claude CLI agent (Opus 5.5) in tmux. Claude (the manager
session) reviews your result and reruns your evidence before anything samples. **Do this package
only, then stop.** The TODO 031 wave-2 grids are stopped and wait on you. The beast is free for
your tests and benchmarks.

## 0. Rules

- **Worktree** `/home/james/bet_project/.worktrees/BayesianFootball-zero-alloc`, **branch
  `fix/zero-alloc-engine`** (cut from `8effca72` on `feat/w2-tier-components`). Work only here. A
  separate pi agent is editing harness scoring files in another worktree
  (`…/BayesianFootball-experiment-harness`); don't touch that worktree, and keep out of
  `src/harness/scoring.jl`, `src/harness/store.jl`, `scripts/score_runs.jl` and
  `scripts/leaderboard.jl`. Commit in small logical steps; push the branch at the end.
- **Julia runs on mcmc-beast, not the laptop.** The user rejects one-shot `julia -e` on the laptop.
  Use `ssh root@mcmc-beast` and your own tmux session there, from a clean checkout of a pushed SHA:
  `git -C /root/BayesianFootball fetch origin fix/zero-alloc-engine && bash /root/BF_runs/8effca72/scripts/beast_checkout.sh <sha>`
  gives `/root/BF_runs/<sha>`. Logs go in `/root/BF_runs/logs/<sha>/`, never inside a checkout.
  - `set -a; source .env; set +a` before Julia, and use `/root/.juliaup/bin/julia --project`.
  - `-t 16`, `pinthreads(:cores)` and `BLAS.set_num_threads(1)` for anything multithreaded.
  - One heavy job at a time; check `pgrep -af run_candidates.jl` first. The Kaimon process in
    tmux `julia` is expected; leave it alone.
- **Never print or commit credentials.**
- **Read first:**
  - `AGENTS.md` §7;
  - `docs/turing_ad_performance_guide.md` (all of it);
  - `docs/guides/julia_coding_context_for_agents.md`.

## 1. The defect (measured by the manager, 2026-09-27)

**Symptom.** W2 grid `td_base` (TimeDecay(180) + `JointGammaPoissonObservation` + default
`ClampGuard`, pooled SPFL + cups, fold 60: 2,390 rows, 92 parameters) ran at 16 threads with the
16 sampler threads at 39–49% CPU. ~8 GC threads each had accumulated ~⅔ of a sampler thread's CPU
time.

**Measurement** (probe scripts in `/root/BF_runs/logs/w2_gc_probe.jl`, `w2_alloc_profile.jl`,
`w2_alloc_detail.jl`; logs in `/root/BF_runs/logs/1a8f4a03/`):
- compiled tape `ReverseDiff.gradient!` = 0.244 ms and **432,496 bytes per gradient**;
- the Turing `logdensity_and_gradient` path = 435,760 bytes;
- plain `logdensity` = 62 MB (not on the NUTS hot path, but report it).

**Where** (per-instruction replay): 114 of 118 instructions allocate 0. **Four
`SpecialInstruction{tracker_∇broadcast}` reverse passes allocate everything:**

| Tape # | Broadcast | Bytes per gradient |
|---|---|---|
| 69, 71 | `clamp.(η, lo, hi)` in `apply_guard(::ClampGuard, η)` | 57,624 each |
| 107, 109 | the fused Gamma pxG-arm broadcast in `JointGammaPoissonObservation` (`(ν - 1) .* log_pxg .- (ν .* pxg) .* inv_μ .- ν .* ξ .+ log_norm`), inputs `TReal, Arr, TReal, Arr, TArr, TReal, TArr, TReal` | 153,664 each |

A tracked **scalar** inside a broadcast (and `clamp` with scalar bounds) sends ReverseDiff to its
allocating `tracker_∇broadcast` adjoint instead of the preallocated `∇broadcast` path.

**This was fixed before, but only in prototypes, and it came back.** Read:
- `experiments/scottish_lower/10_momentum_multiscale_grw/l10_momentum_grw_loader.jl`
  (commit `7bea5069`: `ArrayClampGuard`, a zero-allocation audit);
- `experiments/scottish_lower/11_decompression_pxg_covariate/l11_decompression_loader.jl` §2.
  It says the production engine's "scalar broadcasts allocate on ReverseDiff replay … these local
  adapters … do not modify src-owned methods".

They lift sampled scalars into one-element tracked arrays. Nothing graduated to `src/`, and nothing
in `test/` or the harness gates on allocation. **Your job is to fix it in `src/` and make it
impossible to reintroduce silently.**

## 2. What to build

1. **Audit first.** Write `scripts/tape_allocation_audit.jl`. For a model and a real fold, it prints
   bytes per compiled `gradient!`, taking the **minimum over ≥ 20 warmed reps behind a function
   barrier** (the `l11` pattern). It then lists every allocating tape instruction with its function
   and argument shapes. Run it on the beast for **all 15 W2 candidates**
   (`experiments/scotland/04_pyramid_tier_transition_arms/candidates.jl`, each on its scope's
   largest fold and on a season-opening fold) and on **all 12 W1 recipes**
   (`experiments/scotland/03_dynamics_scope_matrix/candidates.jl`). This is the before-table.
   Expect more offenders than the four above: tracked scalars in broadcasts appear in `PyramidTiers`
   (`step .* (…)`, `carry .* …`, `old_firm .* …`, `σ .* raw`), the league deltas
   (`scale .* (raw .- mean(raw))`), the GRW and TimeDecay scale multiplies (`z .* σ`), covariate
   weights, and the NegBin/dispersion observations.
2. **Fix every allocating instruction in `src/` builder code.**
   - Use the proven pattern: lift sampled scalars to one-element tracked arrays, or restructure
     algebraically so scalars multiply **after** a reduction (e.g. `ν * sum(x .* mask)`).
     Array-valued clamp bounds for `ClampGuard`. Choose per site and document why in a comment.
   - **Do not change any struct's field layout**: old fit artefacts deserialize these types.
   - **Do not change `string(model)`**: recipe hashes must not move.
   - **Do not change the sampled sites** (names, shapes, order).
   - **Mathematics identical:** log-density and gradient equal to the pre-fix code to ≤ 1e-12
     relative at a prior draw **and** at perturbed points at sampler scale (±0.8 per coordinate;
     see guide §2/§10.3), for every W2 candidate and W1 recipe.
3. **Guard against regression, three layers:**
   - **(a) Test suite:** `test/tape_allocation_tests.jl`, in `test/runtests.jl` and
     `test/run_parallel_tests.jl`. On a synthetic league it asserts **0 bytes per compiled
     gradient** (min over warmed reps) across a matrix: {TimeDecay, MultiScaleGRW, GRW +
     target_season_step} × {Poisson, NegBin, JointGammaPoisson (Shared/Hierarchical/Competition κ),
     JointGammaNegBin} × {ClampGuard, NoGuard} × {no predictors, PyramidTiers (NoCarry/Estimated,
     NoJump/EstimatedJump/FixedTransitionPriors), a wealth covariate, PlayerLineupPillar}. Prune
     combinations the builder rejects. The test must fail on the pre-fix code; demonstrate that.
   - **(b) Harness smoke gate:** add a **hard** smoke check `tape_allocation` in `src/harness/`
     (`checks.jl` / `stages.jl`, not the files listed in §0). It measures bytes per compiled
     gradient on the smoke's fold and fails above **1,024 bytes**, naming the offending
     instructions. Add it to the `has_passing_smoke` required set. Update the harness tests.
   - **(c) Documentation:**
     - `docs/turing_ad_performance_guide.md`: a new rule and section, "bytes per gradient is a
       metric", covering the `tracker_∇broadcast` failure mode, the tracked-scalar and clamp cases,
       the fixes, and how to measure; plus §10.1 and checklist entries. Say plainly that latency
       alone hides this: 0.24 ms per gradient looked fine, and it halved throughput at 16 threads
       through GC.
     - A one-line digest in `AGENTS.md` §7 (the size budget is enforced by
       `./scripts/todo.sh check`).
     - Note in the `l10`/`l11` loaders that their local adapters are superseded by `src`.
4. **Performance evidence on the beast:**
   - W2 `td_base` and `grw_step_a2_carry_jump_ldelta`, each on its scope's largest fold, before
     vs after:
     - bytes per gradient;
     - a single-chain NUTS 150+150 `@timed` with `discard_adapt = false` (GC %, GiB allocated,
       KB and ms per leapfrog);
     - a **16-thread** run of 16 chains (`QueuedNUTSConfig`, 100+100) reporting wall time and
       sampler-thread CPU utilisation.
   - Report the throughput gain.
5. **Tracking:** create a TODO with `./scripts/todo.sh new "Zero-allocation compiled gradients in
   the builder engine"`. Fill it in (context, acceptance = this package's §4, Work Log, and
   Verification with your numbers) and link it from TODO 031's Work Log. Run
   `./scripts/todo.sh check`.

## 3. Out of scope

- Legacy hand-written engines in `src/models/pregame/engines/`: note offenders from the audit in
  your report, but don't fix them.
- Changing Turing/DynamicPPL/ReverseDiff versions. Distributions stays pinned at 0.25.126.
- Any sampling beyond the benchmarks in §2.4.

## 4. Acceptance (the manager reruns each item)

1. The before/after audit table for 15 W2 + 12 W1 recipes: after = 0 bytes on the compiled tape
   (or each residual explained and ≤ 1,024 bytes).
2. The log-density and gradient equality table (≤ 1e-12 relative, prior + perturbed points) for
   the same 27 recipes.
3. The 12 W1 and 15 W2 `recipe_hash` values are unchanged: print them next to the pre-fix values.
   A persisted W1 fit (`decb01be-f214-47bb-bc7b-bcc309026bda`) still loads.
4. `test/tape_allocation_tests.jl` passes on the fix and **fails on `8effca72`** (show both).
5. The `tape_allocation` hard smoke check exists, is required, and is tested.
6. Suites pass on the beast from a clean checkout:
   - `test/pyramid_tier_tests.jl`, `test/test_multiscale_grw.jl`, `test/builder_tests.jl`;
   - `test/harness_runner_tests.jl`, `test/harness_scoring_tests.jl`,
     `test/harness_portfolio_tests.jl`, `test/scoped_splitter_tests.jl`;
   - the new test;
   - `test/run_parallel_tests.jl` no worse than known T007.
7. The performance table (§2.4).
8. The guide, `AGENTS.md` digest and TODO are updated; `./scripts/todo.sh check` passes.

## 5. Report and stop

- Write `docs/architecture/zero_alloc_engine_report.md`:
  - the evidence per acceptance item (commands, SHAs, numbers);
  - the before/after tables;
  - the design choice per site;
  - any residual allocation and why;
  - offenders in the legacy engines.
- Leave the beast tidy (no stray Julia processes of yours).
- Commit, push, then print exactly `ZEROALLOC_DONE` on its own line and stop. If you're blocked on a
  human decision, write it under "Blocked" in the report, print `ZEROALLOC_BLOCKED`, and stop.
