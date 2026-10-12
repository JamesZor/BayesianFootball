# F6 — weekly refits, unchanged biweekly QS dynamics

## Summary (8 lines)
- `qs_weekly` changes only fold cadence: weekly refits, identical control model/priors/sampler, biweekly states.
- Default scopes retain their original serialized layout/type/representation; opt-in scopes use the approved `FoldCadenceScope` wrapper.
- Units PASS: 505 assertions, including same-biweek forecasts without a new innovation and serialization roundtrip.
- Legacy splits PASS: 1,368 exact signatures across both panels, four existing scope policies and three clocks.
- Original wave-2 fold PASS: 396,000 chain and 160,000 latent values byte-identical, maximum difference 0.
- Frozen panel PASS: 115 non-empty weekly folds, the same 1,070 fixtures; all folds pass strict day filtration.
- Normal two-fold smoke PASS: all six hard checks, exactly 0 B tapes; convergence/performance REVIEW flags remain.
- No grid ran. The 115-fold /16-thread sampling estimate is 5.65 h, not a measured or guaranteed runtime.

## Recipe and forecast rule

`DataScope(; fold_col=:match_week, dynamics_col=:match_biweek, ...)` returns an opt-in wrapper. Unset (or equal-clock) scopes remain the original nine-field `DataScope`; adding a physical field broke old Julia-serialized fits on the first attempt. James approved the wrapper fix. The original positional constructor is unchanged. Existing scope dispatch and harness candidates accept either type.

Boundaries and held-out selection use `fold_col`; feature state indices still use `dynamics_col`. If a held-out fixture's biweek is already fitted, its forecast reads that posterior state exactly. If every held-out fixture is in a fitted biweek, extraction draws no new innovation. Unseen next-biweek fixtures retain the original shared, centred QS innovation. Default/off-option feature metadata and forecasts remain unchanged.

`qs_weekly` uses the **identical** `qs_control` model (including its internal model name), market table, priors and sampler. Only scope fold cadence and the candidate's identity/hypothesis differ. Source positions 1 and 115 are selected by the normal harness smoke; each uses 2 chains × (200 warmup + 200 retained). The performance probe uses 4 chains × (50 + 50). Grid budget remains 4 chains × (500 + 1,000).

## Gates and evidence

| Gate | Result | Evidence under this directory |
|---|---|---|
| Defaults / units / fitted-state forecast / serialization | PASS, 505 assertions | `unit_gates.csv`, `logs/units.log`; tests in `test/scoped_splitter_tests.jl`, `test/test_quality_style_grw.jl` |
| Legacy boundary IDs and held-out IDs | PASS, 1,368 before/after signatures; W2 + wave-2 panels; lower, SPFL, cups with/without monitors; week/biweek/month | `legacy_split_baseline.csv`, `legacy_split_identity.csv`, `logs/split_baseline.log`, `logs/split_identity.log` |
| Original wave-2 fold byte identity | PASS, chains396000 / latents160000, identical SHA and max difference0 | `legacy_reproduction.csv`, `legacy_checkpoint_manifest.txt`, `logs/legacy.log` |
| Weekly count / unchanged fixture panel / biweekly state clock | PASS, 115 folds /1070 fixtures vs control60 /1070 | `weekly_panel.csv`, `weekly_folds.csv`, `logs/panel.log` |
| Weekly day filtration | PASS, every training date strictly before first held-out day, all115 folds | `weekly_folds.csv`, `logs/panel.log` |
| Normal two-fold hard smoke | PASS: gradient, tape_allocation, filtration, latents, score_grid_coherence, fit_parity | `smoke_checks_qs_weekly.csv`, `smoke_summary_qs_weekly.csv`, `logs/smoke.log` |
| Brief's exactly0 B / canonical DB readiness | PASS, both tapes0 B; `has_passing_smoke=true` | `smoke_gate.csv`, `logs/smoke_receipt.log` |

55 weekly folds (531 fixtures) forecast an already observed biweek. No fixture is dropped or duplicated. The smoke endpoints forecast unseen next periods; the dedicated unit tests and all-panel metadata audit cover the within-biweek rule separately.

Smoke run UUID: **`9346d3b3-874c-49eb-9bc3-0db9f07de477`**.
Recipe hash: `aaed32f47a4c97e032bfb067b6debd2e4ccc14d934df58301955f4bccffae952`.
Legacy reference: `81fdf817-f82d-4f66-9bfd-5f456c4fe29f`, original chain seeds203100101–104 / postprocessing203109000.

**Review, not promotion:** R̂1.06250 >1.05; bulk/tail ESS67.14/75.45, zero divergences, BFMI0.8071, tree-depth rate0. Performance efficiency ratio2.08675 >2.0 (GC9.72%). Both REVIEW rows fail; no thresholds were weakened or retry used to hide them. See `logs/smoke_perf/scottish_lower_qs_transition_2326/qs_weekly.md`.

Source implementation: `fbfb86a1` (after initial `bb12dee7`); legacy/filtration evidence and runner introspection fix: `91bcd7d8`. Legacy metadata has `-dirty` from generated tracked gate CSVs; the smoke checkout's untracked legacy checkpoint directory likewise makes its source status dirty. No uncommitted source delta was sampled. Failed first legacy load and the first wrapper-panel runner's `getfield` error are preserved in logs; corrected attempts pass, and no gate failed twice. Job receipts: `gate_commands.csv`.

## Timing projection — estimate only

Measured smoke wall: **317.77 s** (5.30 min). Largest-fold probe sampling window: **47.15384 s**, four chains /four threads under shared S3 load. Linear iteration-work projection with ideal four-to-sixteen-thread scaling:

`seconds16 = 47.153839282 × [folds × 4 × (500+1000)] / [4 × (50+50)] × (4/16)`.

| Folds | Chain iterations | Same-load four-thread projection | Ideal sixteen-thread estimate |
|---:|---:|---:|---:|
| 60 (reference budget) | 360,000 | 11.79 h | **2.95 h** |
| 115 (actual weekly panel) | 690,000 | 22.59 h | **5.65 h** |

`timing_projection.csv` records the arithmetic. These are **sampling-only estimates**, not a forecast of observed grid wall time: adaptation/leapfrog counts can change, sixteen-thread GC/memory contention need not scale ideally, and load differs. Scoring, persistence and setup are excluded. A lower four-thread GC share does not prove a grid speedup. Manager should budget headroom; no sixteen-thread F6 run was made.

## Exact grid command — manager only, after S3 scheduling/review

From the final synced F6 checkout, using the verified frozen cache (not a regenerated SQL panel):

```bash
cd /root/BF_runs/qs_transition_f6
echo 'a1e89da1a25aaaa8bc7476307338b9e29a58d5d33beee8e25e25e6126989fe12  .cache/datastore_ScottishPyramid.jls' | sha256sum -c - && \
julia --project -t 16 scripts/run_candidates.jl \
  experiments/scotland/11_qs_transition/candidates.jl \
  --stage grid --only qs_weekly \
  --log-dir /root/BF_runs/logs/qs_transition_f6_weekly_grid
```

This command was **not run**. The CLI pins threads and sets BLAS to1. Normal harness grid gating must remain enabled. F6's owned four-thread REPL is stopped; no S3 session/checkout was touched. Records are here because `~/bet_project/hub/experiments/` is absent.
