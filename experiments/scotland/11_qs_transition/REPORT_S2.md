# Stage 2 — build gates and four-arm smokes complete; F6 deferred

## Summary (10 lines)
- Four recipes are committed on W2's 60-fold / 1,070-fixture panel; F6 stays deferred for James.
- The frozen 21/22–25/26 market table retains all 1,430 legacy rows exactly (1,790 rows; 1,361 full books).
- New and existing QS unit tests pass: 46 + 373 assertions.
- Exact AD, convergent FD and exactly 0-byte linked tapes pass on synthetic and actual opening/late folds.
- Post-source legacy chains (396,000 values) and latents (160,000 values) reproduce bit-identically.
- Entrant endpoints overlap the market likelihood 0 times; earlier-season EB transport overlap is declared separately.
- Three-seed recovery passes: entrant 50/51 and summer 36/39 intervals; no parameter misses all three seeds.
- All four normal two-fold smoke hard checks pass; F6 is explicitly deferred, not silently dropped.
- S3 control is blocked by stale D11 check832; review flags and DB cleanup remain manager decisions.
- No full grid, 26/27 outcomes, pushes, or DB writes outside normal smoke machinery were used.

## 1. Per-arm status

| Arm | Recipe | Two-fold smoke hard checks | Max R̂ | Min bulk / tail ESS | Divergences |
|---|---|---|---:|---:|---:|
| Control | `qs_control` | PASS | 1.0625 | 67.1 / 75.5 | 0 |
| F3 QS+A2 | `qs_a2` | PASS | 1.0533 | 50.9 / 72.0 | 2 (0.25%) |
| F1F2 entrant | `qs_entrant` | PASS | 1.0802 | 19.4 / 62.6 | 0 |
| F4 summer | `qs_summer` | PASS | 1.1393 | 15.1 / 37.0 | 0 |
| F6 weekly | No runnable arm | Deferred explicitly on resume | — | — | — |

Normal harness smokes select source folds 1 and 60, including the 23/24 opener; each uses
**2 chains × (200 warmup + 200 retained)** per fold, not the old handover's four-chain description.
S3 recipes retain **4 × (500 + 1,000)**. The load-realistic timing probe uses 16 × (50 + 50).
Control/A2 smokes were verified before the new source; they were not repeated. Post-source legacy
identity and the existing QS unit suite separately certify default/off-option preservation.

Run UUIDs:
- Control: `b58afafc-80ee-4ee6-920b-d5df860b4483`.
- A2: `9374341b-66aa-4460-83fe-af45f8d5a01b`.
- Entrant: `8ac3ccfa-3aa8-4721-9e17-7216e394a2f1`.
- Summer: `55c33930-2e27-418f-ba7d-9e330fdd6095`.

All completed smokes have convergence REVIEW failures. Performance REVIEW flags also remain:
GC shares control32.4%, A2 22.5%, entrant29.7%, summer32.2%; in-situ/bare-gradient cost
ratios4.22/4.79/3.97/4.05.
Their tapes allocate0 B; these flags concern non-tape sampler/GC overhead. No tuning or re-smoking
was used to hide them. See `build/smoke_checks_qs_*.csv`, `smoke_summary_qs_*.csv`, and
`build/logs/smoke_perf/scottish_lower_qs_transition_2326/`.

## 2. Gate table

| Gate | Result / committed evidence |
|---|---|
| Expanded market parity / panel | PASS; `build/market_coverage.csv`, `panel.csv`, `market_rates.sha256` |
| Opt-in defaults / units | PASS: 46 new +373 existing assertions; `test/test_quality_style_transition.jl`, `build/logs/component_gates.log` |
| Synthetic exact AD ≤1e-10 | PASS: four opening/observed-target entrant/summer cases, five points each; max relative error1.554e-16; `component_gates.csv` |
| Actual-fold exact AD ≤1e-10 | PASS: both arms on folds1/2/60, five points each; max relative error1.331e-15; `real_fold_gates.csv` |
| FD convergence audit | PASS: seven step sizes per case; worst best directional error3.840e-10 synthetic /9.660e-10 real; `fd_convergence.csv`, `real_fold_fd.csv` |
| Linked tape allocation | PASS: **exactly0 B**, all ten source-gate cases; normal smoke tapes also0 B; same CSVs and smoke checks |
| Three-seed recovery | PASS: 50/51 entrant (98.04%), 36/39 summer (92.31%); pooled86/90 (95.56%); no monitored parameter misses all3; `recovery_gate.csv`, `recovery_three_seeds.csv` |
| Post-src legacy byte identity | PASS: chains396000/latents160000 identical to wave2 `81fdf817-f82d-4f66-9bfd-5f456c4fe29f`, max difference0; `legacy_reproduction_post_src.csv` |
| Entrant endpoint filtration / entry node | PASS: overlap0, pre-target source chronology and first-biweekly-period assertion on folds1/2/60; `provenance_gates.csv`, `prior_manifests/` |
| Persistent prior / incumbent invariance | PASS: fold1→2 means/SDs/source IDs identical; entrant-mean perturbations leave incumbent states bit-identical in units |
| A2 axis / single application | PASS: one tier term; attack/concession0.48/0.52; total step0.47 → q0.235 and s−0.0094; new unit suite |
| Normal two-fold harness checks | All four PASS. Gradient, allocation, filtration, latents, score tensor and save/load parity; `smoke_checks_qs_*.csv` |

Commands, job IDs and source/checkout commits: `build/gate_commands.csv`. All source gates passed
on their first resume attempt, with no source changes after `e2e956cf` and no threshold changes.
The post-src legacy CSV's `git_sha=unknown` reflects a checkout ownership warning; sync transcript
and `build/logs/resume_checkout_receipt.log` establish `ba33e752` including `e2e956cf`. Only the
owned checkout's ownership/git pointer was repaired; the verified sampling was not repeated.

Recovery uses an independent equation-based generator, not engine likelihood/design calls:
16 clubs, five historical and four target periods, all-pair synthetic fixtures, two entrants for
F1F2; fixed seeds recorded in CSV; two chains400+400 per seed, 90% marginal intervals. Monitored
truths include all six QS scales/ratios, market observation SD, interception/home advantage,
entrant q/s entry draws, and realized incumbent q/s boundary states/changes. These synthetic
checks validate implementation, not the prior's real-world calibration or a powered transition test.

## 3. Entrant semantics, provenance and gauge

Separate opt-in wrappers preserve the legacy QS type/layout. All new options default off.
D10 eligibility uses absence of pre-target-season in-scope history and persists across that season's
folds. The prior is an **in-model entry-node draw**, never a backdated historical initial state or
an OOS-only replacement; only incumbents determine the centring projection. Summer support is
independent: it mirrors the MultiScaleGRW boundary map, including deterministic, per-draw,
team-centred macro uncertainty integration in a season-opening fold.

F1F2 is labelled **exact entrant endpoint, EB transport hyperparameters** under D9. Endpoint
reconstructions use only 54/55 closes strictly before the target-season cutoff. S1's A2-centred
P2 shifts/scales use earlier movers. Fallback is the destination A2 tier reference with declared
broad q/s uncertainty. The destination gauge is added once; the transport already includes the
old/new division rebase. SDs are diagonal conservative working bounds, not an identified or
certified state-jump covariance. No A2 predictor is also added to the entrant arm.

| Source fold | Target season | Entrants | Endpoint overlaps | Sum of per-club hyper overlaps |
|---|---|---:|---:|---:|
| 1 | 23/24 | 2 | 0 | 1,158 |
| 2 | 23/24 | 2 | 0 | 1,158 |
| 60 | 25/26 | 1 | 0 | 509 |

These hyper counts are **not unique-source union counts**, and are not represented as disjoint.
`prior_manifests/` includes club, source IDs/seasons, maximum endpoint source date, cutoff, entry
node date, input hash, moments, transport parameters, fallback, gauge and hyper-source IDs.
Manifest filenames use season-local fold seeds (e.g.25/26 fold20 is source fold60); the audit CSV
maps the global source folds. All chronology and overlap hard assertions passed.

## 4. D11/D12 history

The initial shared REPL failed the unchanged allocation gate. Identical Manifests and Julia1.12.4
were used in fresh archived/current/loader REPLs, all measuring0 B on matched folds1/21/41.
No package, source or gate limit changed. The original process-state cause is unproved; see
`build/ALLOCATION_INVESTIGATION.md`. Control/A2 then passed fresh normal hard smokes.
D12 stopped work while Tailscale SSH was down. Resume stopped only owned `s2`, synced the isolated
checkout, and started a fresh REPL before gates; no verified work was redone.

## 5. F6 decision note

`DataScope.dynamics_col` drives both fold cadence and feature-state geometry. Merely selecting
`:match_week` changes both factors; it is not a weekly-refit-only arm. Per Task5 and the explicit
resume instruction, **F6 remains deferred for James**, with no src change or grid command.
Cheapest proposed separation: an opt-in fold-cadence field plus a tested OOS rule that draws no
new biweekly innovation for a weekly refit inside the same state period. Manager approval and
actual nonempty weekly-fold counting are required; do not assume120. See `build/F6_DECISION.md`.

## 6. Timing projection

| Arm | Whole smoke wall | Largest-fold 16-chain probe window | Grid-amortized full-fold wall | Projected60-fold grid |
|---|---:|---:|---:|---:|
| Control | 208.95 s | 89.92 s | 337.21 s | 5.62 h |
| A2 | 242.10 s | 91.39 s | 342.71 s | 5.71 h |
| Entrant | 241.94 s | 86.58 s | 324.69 s | 5.41 h |
| Summer | 224.85 s | 90.05 s | 337.69 s | 5.63 h |
| Weekly | Deferred | Not measured | Unknown | Unknown |

Scaling: `probe_window_seconds × (60×4×1500)/(16×100)`. The full-fold figure is
**grid-amortized**, not a measured isolated four-chain fold latency. It assumes constant full-load
throughput at the largest smoke fold; excludes extra persistence/scoring overhead and adaptation
nonlinearity. Legacy opener fit/postprocessing measured65.57 s. A defensible **five-arm** total
cannot be provided while F6 is deferred. The four measured 60-fold arms total **22.37 h** of
projected sampling (240 folds); allow additional scoring/persistence and throughput uncertainty.
`build/timing_projection.csv` records the exact unrounded calculations.

## 7. Exact S3 commands — manager-controlled, not executed in S2

Read-only history audit: `has_passing_smoke=false` for control, true for A2/entrant/summer.
The sole stale failed hard check is **`harness_checks.id=832`**, control recipe
`f75623499919dd9e2e045595bdbe3735c1b2e7748b0655602844ba68a0f46ba7`, `run_id=NULL`,
`tape_allocation`, 57,568 B from the original D11 incident. The latest control smoke passes all
required hard checks at0 B, but the harness conservatively considers its whole history.
Evidence: `build/s3_readiness.csv`, `historical_failed_hard_qs_control.csv`,
`harness_history_qs_control.csv`, `logs/s3_readiness.log`. **No row was deleted or altered.**
Before the control command can start, an explicitly authorised operator must review/archive this
specific stale row and perform the narrowly targeted cleanup described in the experiment harness
guide §3, then re-evaluate readiness. S2 does not authorise that DB write or any gate bypass.
Convergence/performance flags also require manager review, not automatic promotion. Update the owned checkout to the final
approved local commit before launch. Preserve the frozen datastore (verified matching the smoke
input), and do not regenerate it from SQL for a like-for-like comparison:

```bash
ssh root@mcmc-beast 'cd /root/BF_runs/qs_transition && printf "a1e89da1a25aaaa8bc7476307338b9e29a58d5d33beee8e25e25e6126989fe12  .cache/datastore_ScottishPyramid.jls\n" | sha256sum -c -'
jl start s3 --beast --threads 16 --dir /root/BF_runs/qs_transition
jl eval s3 'empty!(ARGS); append!(ARGS, ["experiments/scotland/11_qs_transition/candidates.jl", "--stage", "grid", "--only", "qs_control"]); include("scripts/run_candidates.jl"); main()' --bg
```

Watch the printed job ID to completion, then stop/restart **owned s3** before each subsequent arm.
One grid at a time; the exact alternative launch commands are:

```bash
jl eval s3 'empty!(ARGS); append!(ARGS, ["experiments/scotland/11_qs_transition/candidates.jl", "--stage", "grid", "--only", "qs_a2"]); include("scripts/run_candidates.jl"); main()' --bg
jl eval s3 'empty!(ARGS); append!(ARGS, ["experiments/scotland/11_qs_transition/candidates.jl", "--stage", "grid", "--only", "qs_entrant"]); include("scripts/run_candidates.jl"); main()' --bg
jl eval s3 'empty!(ARGS); append!(ARGS, ["experiments/scotland/11_qs_transition/candidates.jl", "--stage", "grid", "--only", "qs_summer"]); include("scripts/run_candidates.jl"); main()' --bg
```

The script pins threads and BLAS1, uses the same frozen scope/cache, registers via the normal
harness, and refuses recipes without passing historical smoke checks. F6 has no valid command.
No S3 command was run in Stage2. Final completion/owned REPL shutdown recorded in `build/HANDOVER.md`.
