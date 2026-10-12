# Stage 2 — BLOCKED at D12 network deadline

## Summary (9 lines)
- Stage 2 is **not complete**; Stage 3 is **not cleared**.
- The unchanged wave-2 fold reproduced bit-identical chains and latents before src changes.
- W2's panel is reproduced: 60 folds, 1,070 fixtures.
- The expanded market table has 1,790 rows and 1,361 full books; all 1,430 legacy rows are unchanged.
- Control and QS+A2 completed normal two-fold smokes with every hard check passing.
- Both smokes have convergence review failures; neither is a promotion certificate.
- Entrant-node and summer-step code is committed at `e2e956cf` but **ungated and unsmoked**.
- F6 cadence and dynamics clock are tied; no falsely labelled weekly arm was built.
- SSH remained unavailable after 03:30 BST; D12 requires a blocked handover.

## 1. Per-arm status

| Arm | Recipe | Two-fold smoke | Max R̂ | Min bulk/tail ESS | Divergences |
|---|---|---|---:|---:|---:|
| Control | `qs_control` | All hard checks PASS; convergence REVIEW fail | 1.0625 | 67.1 / 75.5 | 0 |
| F3 QS+A2 | `qs_a2` | All hard checks PASS; convergence REVIEW fail | 1.0533 | 50.9 / 72.0 | 2 (0.25%) |
| F1F2 entrant | `qs_entrant` | Committed WIP; **not run** | — | — | — |
| F4 summer | `qs_summer` | Committed WIP; **not run** | — | — | — |
| F6 weekly | No runnable arm | Deferred under brief Task 5 | — | — | — |

Smokes selected source folds 1 and 60, including the 23/24 opener. They ran before the new src
was shipped/tested; do not treat their success as validation of the later source changes.
Run UUIDs:
- Control: `b58afafc-80ee-4ee6-920b-d5df860b4483`.
- QS+A2: `9374341b-66aa-4460-83fe-af45f8d5a01b`.

Sources: `build/smoke_checks_qs_{control,a2}.csv`, `build/smoke_summary_qs_{control,a2}.csv`,
`build/panel.csv`, `build/market_coverage.csv`, and `build/logs/control_a2_fresh.log`.

## 2. Gate table

| Gate | Evidence / result | Remaining requirement |
|---|---|---|
| Pre-src legacy byte identity | PASS: 396,000 chain values and 160,000 latent values, max difference 0; `build/legacy_reproduction.csv` | Repeat after all src changes |
| Frozen market parity | PASS: all 1,430 legacy rows retained exactly; table SHA256 `1098f12aff1db116d3fc46454ccc00297402b61f5011502c0ffd86f7298c9add` | None for committed input |
| Panel | PASS: 60 folds / 1,070 fixtures | None for baseline panel |
| Control/A2 harness hard checks | PASS: gradient, zero-allocation tapes on both folds, filtration, latents, score-grid coherence, DB save/load parity | Does not validate new src |
| New-component exact AD ≤ 1e-10 | **NOT RUN** | `build/r07_component_gates.jl`, then real-fold probes |
| New-component FD convergence | **NOT RUN** | Prepared in the same gate runner |
| New-component zero allocation | **NOT RUN** | Prepared opening/observed-target cases; exactly 0 B required |
| New-component unit tests | **NOT RUN** | `test/test_quality_style_transition.jl` plus existing QS suite |
| Three-seed recovery | **NOT RUN; runner not yet written** | No parameter may miss all three seeds; pooled ≥80% |
| Entrant provenance / overlap | Feature implemented; **no executed manifests** | Verify endpoint overlap 0 and chronology on actual folds |
| Entrant/summer two-fold smokes | **NOT RUN** | All hard checks plus recorded R̂/ESS/divergences |

## 3. Source implementation and manager rulings

Separate opt-in configurations preserve the existing `QualityStyleGRW` concrete field layout.
The default constructor still returns the legacy type. Summer support uses the existing GRW
season-boundary accumulator, with opening-fold macro uncertainty integrated during extraction.

D10 entrant implementation drafts a separate in-model q/s entry draw at the first target node,
not a historical-window initial state or a fresh OOS override. Incumbent-only projection is
intended to prevent entrant mean shifts changing incumbent states. Eligibility is based on
absence of pre-target-season in-scope matches and persists throughout that target season.
These statements describe **intent and committed code**, not verified behaviour.

`l01_entrant_prior.jl` freezes S1 input hashes, retains S1 A2-centred P2 parameters, uses 54/55
endpoint reconstruction only, and records fallback/gauge/moments/source IDs. Under D9, the
zero-overlap hard check applies to endpoint IDs; earlier-season hyperparameter-source overlap
is separately recorded and must not be represented as zero. SDs use a diagonal conservative
working bound, not a claimed calibrated intrinsic state-jump covariance.

The source includes an explicit first-biweekly-period entry assertion. Its satisfaction on the
real panel has **not been tested**. Any failed provenance/reproduction gate stops the stage.

## 4. D11 allocation investigation

The initial shared REPL failed the unchanged control's allocation gate. Current and wave-2
Manifests were byte-identical, and both used Julia 1.12.4. Archived grid source `081ba5a6` and
current source both measured 0 B on matched opening folds 1/21/41 in fresh REPLs. Including the
old market loader retained 0 B. Fresh normal control/A2 smokes then passed their allocation gates.
No package, source or gate limit was changed to recover the baseline. The original process-state
cause remains unproved; no package-drift or bad-source attribution is claimed.
See `build/ALLOCATION_INVESTIGATION.md` and matched-allocation CSVs.

## 5. F6 decision note

`DataScope.dynamics_col` drives both fold cadence and feature-state geometry. Merely selecting
`:match_week` would change both factors and is not the requested arm. Per Task 5, no F6 src
change was made. Cheapest proposed separation: an opt-in fold-cadence field plus a tested OOS
rule that draws no new biweekly innovation for a weekly refit inside the same dynamics period.
Manager approval and actual weekly fold counting are still required. See `build/F6_DECISION.md`.

## 6. Timing projection — partial, not a five-arm estimate

| Arm | Measured whole smoke wall | Largest-fold 16-chain probe wall | Projected grid-amortized full-fold wall | Projected 60-fold grid |
|---|---:|---:|---:|---:|
| Control | 208.95 s | 89.92 s | 337.21 s | 5.62 h |
| QS+A2 | 242.10 s | 91.39 s | 342.71 s | 5.71 h |
| Entrant / summer / weekly | Not measured | Not measured | Not available | Not available |

Projection scales the full-concurrency 16 × (50 warmup + 50 sample) probe to
60 × 4 × (500 warmup + 1,000 sample) work. It uses the largest smoke fold, assumes constant
throughput, and excludes additional persistence/scoring overhead. The per-fold number is
**grid-amortized**, not a measured standalone four-chain fold latency. The original legacy
opening-fold reproduction measured 64.31 s through fitting/postprocessing. The two measured
arms total about **11.34 h**; a defensible five-arm estimate cannot be provided before the
remaining measurements and F6 decision.

## 7. Exact S3 launch templates — NOT AUTHORISED

**Do not execute these on the current ungated source.** First complete every source gate,
post-src legacy reproduction, real-fold provenance and both missing smokes, and obtain manager
clearance. Update the isolated beast checkout to the approved commit and restore valid git
metadata. F6 has no valid S3 command. No Stage 3 command was executed during Stage 2.

```bash
jl start s3 --beast --threads 16 --dir /root/BF_runs/qs_transition
jl eval s3 'empty!(ARGS); append!(ARGS, ["experiments/scotland/11_qs_transition/candidates.jl", "--stage", "grid", "--only", "qs_control"]); include("scripts/run_candidates.jl"); main()' --bg
jl eval s3 'empty!(ARGS); append!(ARGS, ["experiments/scotland/11_qs_transition/candidates.jl", "--stage", "grid", "--only", "qs_a2"]); include("scripts/run_candidates.jl"); main()' --bg
jl eval s3 'empty!(ARGS); append!(ARGS, ["experiments/scotland/11_qs_transition/candidates.jl", "--stage", "grid", "--only", "qs_entrant"]); include("scripts/run_candidates.jl"); main()' --bg
jl eval s3 'empty!(ARGS); append!(ARGS, ["experiments/scotland/11_qs_transition/candidates.jl", "--stage", "grid", "--only", "qs_summer"]); include("scripts/run_candidates.jl"); main()' --bg
```

Run one grid at a time; watch each printed job ID to completion before launching the next.
These are blocked templates, not evidence that Stage 2 DONE WHEN passed.

## 8. Deadline / shutdown

The D12 watch ran at 600-second cadence without intervening model calls. Its deadline wake was
followed by one SSH check at 03:38:59 BST; the connection still timed out. No Julia job was active
when SSH was lost. The last confirmed owned REPL was `s2` in `/root/BF_runs/qs_transition`;
shutdown/restart could not be confirmed while unreachable. On recovery, the manager must stop
or restart **only** `s2`. See `build/HANDOVER.md` for the exact continuation.
