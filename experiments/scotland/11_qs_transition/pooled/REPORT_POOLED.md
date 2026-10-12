# Pooled QS+A2 — S4 preparation

## Summary (9 lines)
- `qs_pooled_a2` trains Championship/League One/Two goals; scores League One/Two only.
- A2 uses `:first_in_window`; other priors/sampler and frozen lower-only market table match `qs_a2`.
- Panel PASS: 60 folds, exactly the control's 1,070 target fixtures, including the 23/24 opener.
- Championship goals are present and market-masked, including 169 training rows in 25/26 on fold60.
- Three-seed recovery PASS on two/three tiers: 83/90 intervals; known tier step covered6/6.
- Normal two-fold smoke PASS: all hard checks, exactly0 B tapes; no production source changes.
- Convergence REVIEW remains: R̂1.09424, zero divergences; not production acceptance.
- Estimated60-fold grid at16threads:3.52h ideal scaling; plan roughly4–6h, not a measured16thread run.
- Historical smoke gate now PASS after expressly authorised archival/deletion of stale check890; no grids run.

## 1. Recipe and data contract

Recipe: `../candidates.jl`, candidate `qs_pooled_a2`, namespace
`scottish_lower_qs_transition_2326`, control `qs_control`.
Train `[55,56,57]`, target `[56,57]`, monitor `[55]`, clock `[56,57]`, cups`:none`,
history2, seasons23/24–25/26, dynamics`:match_biweek`.
QS weak ratio priors, A2 `EstimatedCarry()`/`EstimatedJump()`, other model components,
and grid sampler4×(500warmup+1000samples) are unchanged from `qs_a2`.
A2 anchor changes to `:first_in_window`, exactly W2's pooled convention (§2.2), rather
than the lower-scope `:full_pyramid` anchor. No entrant prior, cups, pxG or26/27 outcomes.

The frozen market table is unchanged, SHA256
`1098f12aff1db116d3fc46454ccc00297402b61f5011502c0ffd86f7298c9add`.
Its IDs have zero intersection with Championship fixtures. Missing Championship rows receive
zero market masks through the existing extractor; goals remain in the likelihood.
W2/S2 cache reused read-only, copied from
`/root/BF_runs/a76a65df/.cache/datastore_ScottishPyramid.jls` into the owned checkout.
SHA256 `a1e89da1a25aaaa8bc7476307338b9e29a58d5d33beee8e25e25e6126989fe12`.
No SQL refresh or raw-data write. `market_masks.csv` records Championship goal rows360/370/529
on folds1/2/60, zero market rows throughout, and169 current-season25/26 rows on fold60.
The Championship quote gap therefore does not remove its goals or create a missing likelihood row.

## 2. Gates and evidence

| Gate | Result | Evidence |
|---|---|---|
| Exact target panel | PASS:60 folds,1070 unique targets; per-fold ID equality to control | `panel.csv`, `r01_panel.jl` |
| Lower-only market / Championship goals | PASS: zero Championship market IDs/masks, finite goals including25/26 | `market_masks.csv` |
| Two-tier recovery,3 seeds | PASS:42/45 marginal90% intervals; step3/3 | `recovery_gate.csv`, `recovery_three_seeds.csv` |
| Three-tier recovery,3 seeds | PASS:41/45 intervals; step3/3 | Same; raw diagnostic CSVs retained |
| QS global centring | PASS: max absolute q/s draw-wise sum1.173e-15 | `recovery_centering.csv` |
| Smoke gradient | PASS: RD/FD exact relative error3.524e-16; fresh/perturbed tape error0 | `smoke_checks_qs_pooled_a2.csv` |
| Linked compiled tapes | PASS: exactly0 B on both source folds1/60 | Same |
| Filtration / latents / score tensor | PASS: normal unmodified harness checks | Same |
| Save/load fit parity | PASS:2 folds,46 held-out fixtures including16 monitor fixtures,400 draws | Same |
| Historical hard-smoke launch gate | PASS after authorised cleanup of only890 | `readiness.csv`, current history and cleanup receipt |
| Convergence review | FAIL/REVIEW: R̂1.09424, bulk ESS20.49/tail48.03, divergences0, min BFMI0.798 | Smoke summary/checks |
| Sampling-performance review | PASS: GC6.44%, sampler utilisation1.0, efficiency ratio1.522 | Smoke checks |
| Additional source-change gates | Not applicable: no `src/` changes | `logs/source_receipt.log` |

Smoke run UUID: `5cb867cd-f6b8-43ef-a4d7-3d5da9c47ca2`.
Recipe hash: `0db99f621d0e430127f62404c530f07bdbdb22a459f8a345856193e0e09df887`.
Normal smoke budget2chains×(200warmup+200samples) per fold. Folds1/60 include the23/24
opener; their target counts20/10 total30, with16 extra Championship monitor fixtures.
Used `Harness.smoke` directly with all required checks: CLI requires≥8 threads but prep was
limited to4. Registry names are pooled-specific, avoiding changes to S3's canonical entries.

### Recovery interpretation

Independent equation-based generator:8 clubs/tier,5 history+4 target periods, same-tier league
fixtures, lower-only market observations, known step0.55, carry0.30, one boundary mover with
attack/concession jumps0.06/−0.02. Two chains400+400 per seed; fixed data/sampler seeds are
in the CSV. Monitors cover six QS scale/ratio parameters, interception, home advantage,
market SD, tier step/carry and realized q/s first-target/boundary states.
Gate:≥80% interval coverage per panel, no monitored parameter misses all3 seeds, step≥2/3,
and posterior centring<1e-10. All passed; step and carry each6/6.

**Identification is prior-assisted, not likelihood-only.** Globally centred QS random effects
can absorb static tier offsets with an interception adjustment; proper exchangeable QS/A2
priors separate them. Same-tier fixtures have especially little direct step information under
0.48/0.52. The recovered step intervals remain broad (approximately0.18–0.77 across seeds).
This checks implementation/global centring and finite posterior recovery, not a precise or
powered real-world tier-step estimate. Jump scale/raw draws are not claimed recovered from one event.

### Failed attempts and authorisation

Initial panel assertion mistakenly compared targets+monitors with control targets; corrected
only the assertion's56/57 filter. Recovery attempt1 failed before draws on empty synthetic
EstimatedJump arrays (`Vector{Real}`); attempt2 failed before sampling because the diagnostic
generator reused `delta` for two purposes. Stopped and asked; user expressly authorised fixing
and continuing. Added one real synthetic mover, separated `tier_delta`, restarted only pool;
recovery then passed. No recipe/source/gate changes to make these tests pass.

First smoke failed at2464/2560 B linked-gradient allocation after recovery in the same process.
A fresh pool process, without recovery, passed at0 B; no source, package or threshold changes.
Cause remains unproved, consistent with S2's recorded process-state issue. Smoke source/models
match commit`8915cb8d`; `-dirty` receipts reflect runner/evidence files, not model changes.
Manifest SHA256 `f86f49b72b987dd4c87ac80d16f132e5342e988fededbe27e54c6afe8133b6cc`;
Julia1.12.4/Turing0.41.4/DynamicPPL0.38.10/ReverseDiff1.17.0/Distributions0.25.126.

The initial failed null-run check890 blocked historical readiness even after the good smoke.
Full row/history CSVs were committed before cleanup. User separately authorised deletion of
**only890**; a guarded transaction deleted exactly1row, requiring exact recipe/candidate/stage/
check/severity/status/NULLrun/detail. Full backup `authorised_archive_check890.csv`, SHA256
`f094f0ef37ed50489f905a0888be7ee8e283928dddd7e74a1ded26cc4bb46ef4`.
No other checks changed. Read-only post-cleanup audit confirms `has_passing_smoke=true`.
Convergence review still requires manager consideration before promotion.

## 3. Timing projection — estimate, not a grid

Smoke wall235.27s. Final-fold full-load probe:4 concurrent chains,50+50 iterations,
window56.2956s; linked gradient0.4661ms,1481 parameters,1589 fitted rows.

`56.2956 × [60×4×(500+1000)] / [4×(50+50)] = 50,666.06s = 14.07h` at measured4thread throughput.
Assuming ideal4× throughput on16threads:12,666.52s = **3.52h**.
Plan approximately4–6h allowing nonideal scaling and scoring/persistence/setup overhead.
This is not a measured bound: the probe uses the largest selected fold, short adaptation,
and a shared beast running S3. Fold sizes/trajectory costs and oversubscription can change timing.
`timing_projection.csv` preserves the arithmetic and assumptions. No16thread prep or grid run.

## 4. Exact grid command — manager-controlled, NOT executed

After S3/F6 compute is free, convergence review is considered, the manager approves S4,
and the owned checkout is updated to the final local branch commit (preserving this frozen cache):

```bash
ssh root@mcmc-beast 'cd /root/BF_runs/qs_transition_pooled && printf "a1e89da1a25aaaa8bc7476307338b9e29a58d5d33beee8e25e25e6126989fe12  .cache/datastore_ScottishPyramid.jls\n" | sha256sum -c -'
ssh root@mcmc-beast 'cd /root/BF_runs/qs_transition_pooled && /root/.juliaup/bin/julia --project=. --threads=16 scripts/run_candidates.jl experiments/scotland/11_qs_transition/candidates.jl --stage grid --only qs_pooled_a2 --log-dir experiments/scotland/11_qs_transition/pooled/logs/grid'
```

The normal runner pins cores/BLAS1, loads the frozen ScottishPyramid cache, gates the exact recipe,
and scores against `qs_control` in the same namespace. No alternative scope/market table,
entrant-prior combination, weekly cadence or cups are introduced. All prep jobs used only pool/4threads.
