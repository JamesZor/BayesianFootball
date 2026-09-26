# Experiment harness Phase 2 report

Date: 2026-09-26  
Task: [TODO 030](../../todos/030_shared_experiment_harness_recipes_gates_runner_scorecard_leaderboard.md)  
Scope: Phase 2 only; no MCMC was run

## Result

Phase 2 is complete. The Data layer now owns a validated, copy-only `DataScope`, the seven-
tournament `ScottishPyramid` segment, rule-based SPFL cup filtering, and a scoped
walk-forward splitter. Features use the canonical lower-league clock for every widened row.
Scorecard v1.1 adds `entered_spfl` and monitor subsets while keeping monitor performance out
of candidate-versus-control deltas.

Implementation commits:

- `fe1a8a0c` — scoped pyramid store, folds, feature clock, neutral/exclusion input and tests.
- `96c43452` — entered-SPFL and monitor scoring under scorecard v1.1.
- `dca65a1d` — deterministic Phase-2 acceptance verifier.
- `c2ed4780` — metric-specific cohort sample sizes in leaderboard views.
- `c73bad99` — enforce `cups = :none` even when a splitter receives an unscoped pooled store.
- `a2ad2675` — post-review hardening: fail-closed cup input, union native/CSV
  exclusions, positional fold preservation, v1.1 store defaults, monitor-only regression test,
  and root-anchored rsync excludes.

## Acceptance evidence

### 1. Lower scope is canonical

Command, locally and on `mcmc-beast` (detached `harness_p2`, immutable checkout
`dca65a1dce2db679b4d76da3abb7b570d427ba95`):

```bash
set -a; source /root/BayesianFootball-scotland-cross-tier/.env; set +a
julia --project -t 16 scripts/verify_harness_phase2.jl
```

Result:

```text
LOWER_OK folds=40 unique_heldout=710
```

`DataScope(name = "lower")` after `apply_scope` was compared fold by fold with
`gph_splitter(["24/25", "25/26"])`. The ordered `history_match_ids`, ordered
`target_match_ids`, and each next-bin held-out ID set are identical: 40/40 folds, 710 unique
held-out fixtures. The source cache is not mutated; the synthetic test separately asserts
that the returned `DataStore` and its frames are detached copies.

### 2. `spfl_cups` reproduces g2 training rows

The verifier compares every column of its 40-row widening frame with
`r02_widening_g2_grw_all_spfl_cups.csv`. Result:

```text
PYRAMID_OK folds=40 widening=exact features=exact cups=363
```

All 40 folds agree exactly on `n_train`, `n_lower`, `n_upper`, `n_cup`, `n_oos`,
`n_target_steps`, and `n_teams`. The scoped store retains exactly the same 363 cup fixtures
that the prototype datastore can resolve from its 409-row allowlist (46 allowlist IDs are
not in the current finished-match store).

The rule-only cup set initially retained 404 rows, 41 more than the allowlist-resolved set.
The SofaScore schema and sampled/full raw event JSON have no neutral-ground field. The one
hand-maintained input is therefore
`src/Data/data/scottish_neutral_venue_ties.csv`, derived from that exact diff. Thirty-seven
rows are neutral venues. Four additional rows exposed pre-existing scoreability exclusions
in the allowlist derivation: two awarded fixtures and two whose event JSON has no normal-time
score (one after penalties). They are recorded explicitly rather than silently described as
neutral.

| Match | Tournament | Date | Tie | Reason |
|---:|---:|---|---|---|
| 9724815 | 1520 | 2021-09-04 | `elgin-city`–`dunfermline-athletic` | awarded |
| 9834448 | 982 | 2021-11-21 | `rangers`–`hibernian` | neutral venue |
| 9834449 | 982 | 2021-11-20 | `celtic`–`st-johnstone` | neutral venue |
| 9871600 | 1520 | 2021-11-30 | `hamilton-academical`–`kilmarnock` | missing normal-time score |
| 9970862 | 982 | 2021-12-19 | `hibernian`–`celtic` | neutral venue |
| 10171949 | 1520 | 2022-04-03 | `raith-rovers`–`queen-of-the-south` | neutral venue |
| 10194955 | 73 | 2022-04-16 | `heart-of-midlothian`–`hibernian` | neutral venue |
| 10194956 | 73 | 2022-04-17 | `celtic`–`rangers` | neutral venue |
| 10260336 | 73 | 2022-05-21 | `rangers`–`heart-of-midlothian` | neutral venue |
| 10738313 | 1520 | 2022-12-10 | `queens-park-fc`–`montrose` | neutral venue |
| 10828946 | 982 | 2023-01-15 | `rangers`–`aberdeen` | neutral venue |
| 10828954 | 982 | 2023-01-14 | `celtic`–`kilmarnock` | neutral venue |
| 10996537 | 982 | 2023-02-26 | `rangers`–`celtic` | neutral venue |
| 11006445 | 73 | 2023-01-31 | `inverness-caledonian-thistle`–`queens-park-fc` | awarded; missing normal-time score |
| 11049677 | 1520 | 2023-03-26 | `raith-rovers`–`hamilton-academical` | neutral venue |
| 11129490 | 73 | 2023-04-29 | `falkirk-fc`–`inverness-caledonian-thistle` | neutral venue |
| 11129491 | 73 | 2023-04-30 | `rangers`–`celtic` | neutral venue |
| 11252080 | 73 | 2023-06-03 | `celtic`–`inverness-caledonian-thistle` | neutral venue |
| 11694132 | 982 | 2023-11-04 | `hibernian`–`aberdeen` | neutral venue |
| 11694133 | 982 | 2023-11-05 | `heart-of-midlothian`–`rangers` | neutral venue |
| 11807775 | 982 | 2023-12-17 | `rangers`–`aberdeen` | neutral venue |
| 11813990 | 1520 | 2023-11-18 | `greenock-morton`–`airdrieonians` | missing normal-time score after penalties |
| 12153111 | 73 | 2024-04-20 | `aberdeen`–`celtic` | neutral venue |
| 12153124 | 73 | 2024-04-21 | `rangers`–`heart-of-midlothian` | neutral venue |
| 12272123 | 73 | 2024-05-25 | `celtic`–`rangers` | neutral venue |
| 12617718 | 982 | 2024-08-17 | `rangers`–`st-johnstone` | neutral venue |
| 12885753 | 982 | 2024-11-03 | `motherwell`–`rangers` | neutral venue |
| 12885754 | 982 | 2024-11-02 | `celtic`–`aberdeen` | neutral venue |
| 13047121 | 73 | 2024-11-30 | `queens-park-fc`–`partick-thistle` | neutral venue |
| 13078823 | 982 | 2024-12-15 | `celtic`–`rangers` | neutral venue |
| 13247407 | 73 | 2025-01-18 | `queens-park-fc`–`montrose` | neutral venue |
| 13466308 | 1520 | 2025-03-30 | `queens-park-fc`–`livingston` | neutral venue |
| 13591643 | 73 | 2025-04-20 | `st-johnstone`–`celtic` | neutral venue |
| 13591645 | 73 | 2025-04-19 | `heart-of-midlothian`–`aberdeen` | neutral venue |
| 13797101 | 73 | 2025-05-24 | `aberdeen`–`celtic` | neutral venue |
| 14741025 | 982 | 2025-11-01 | `motherwell`–`st-mirren` | neutral venue |
| 14741026 | 982 | 2025-11-02 | `celtic`–`rangers` | neutral venue |
| 14999871 | 982 | 2025-12-14 | `st-mirren`–`celtic` | neutral venue |
| 15692822 | 73 | 2026-04-19 | `celtic`–`st-mirren` | neutral venue |
| 15692823 | 73 | 2026-04-18 | `dunfermline-athletic`–`falkirk-fc` | neutral venue |
| 16039341 | 73 | 2026-05-23 | `celtic`–`dunfermline-athletic` | neutral venue |

### 3. Features are identical to the pyramid path

For `pcx_model("g2_grw_all_spfl_cups")`, the verifier independently builds the prototype
`PyramidGRWCV` + `pcx_align_time!` path and the new `ScopedWalkForwardCV` path. Every one of
40 folds agrees exactly on:

- `ordered_match_ids`
- `time_indices`
- `n_history_steps`
- `n_target_steps`
- `n_teams`
- `team_map`

This is deterministic feature equivalence over the same model recipe and rows; no posterior
sampling was used.

### 4. Monitor fixtures and data coverage

Adding `monitor_tournaments = [54, 55]` leaves the 710 target IDs unchanged. `get_next_matches`
returns the target and monitor rows in the next lower-league clock bin, with the original
`tournament_id`. Across 40 folds it exposes 396 Premiership and 355 Championship monitor
fixtures. Betfair counts below mean a usable de-vigged TWA(−20,0] score reference, not merely
a processed SofaScore price.

| Fold | Season | T54 held / Betfair | T55 held / Betfair |
|---:|---|---:|---:|
| 1 | 24/25 | 12 / 12 | 10 / 0 |
| 2 | 24/25 | 6 / 6 | 6 / 0 |
| 3 | 24/25 | 6 / 6 | 5 / 0 |
| 4 | 24/25 | 8 / 8 | 9 / 2 |
| 5 | 24/25 | 12 / 12 | 10 / 7 |
| 6 | 24/25 | 6 / 6 | 7 / 7 |
| 7 | 24/25 | 16 / 16 | 13 / 13 |
| 8 | 24/25 | 6 / 6 | 10 / 8 |
| 9 | 24/25 | 13 / 13 | 2 / 2 |
| 10 | 24/25 | 13 / 13 | 10 / 10 |
| 11 | 24/25 | 18 / 18 | 11 / 11 |
| 12 | 24/25 | 20 / 20 | 6 / 6 |
| 13 | 24/25 | 7 / 7 | 7 / 7 |
| 14 | 24/25 | 7 / 7 | 7 / 7 |
| 15 | 24/25 | 12 / 12 | 11 / 11 |
| 16 | 24/25 | 12 / 12 | 14 / 14 |
| 17 | 24/25 | 6 / 6 | 10 / 10 |
| 18 | 24/25 | 12 / 12 | 12 / 12 |
| 19 | 24/25 | 6 / 6 | 10 / 10 |
| 20 | 24/25 | 0 / 0 | 10 / 10 |
| 21 | 25/26 | 12 / 12 | 10 / 5 |
| 22 | 25/26 | 4 / 4 | 5 / 0 |
| 23 | 25/26 | 6 / 6 | 7 / 0 |
| 24 | 25/26 | 7 / 7 | 9 / 1 |
| 25 | 25/26 | 14 / 14 | 13 / 0 |
| 26 | 25/26 | 6 / 6 | 9 / 0 |
| 27 | 25/26 | 14 / 14 | 12 / 0 |
| 28 | 25/26 | 6 / 6 | 9 / 0 |
| 29 | 25/26 | 13 / 13 | 5 / 0 |
| 30 | 25/26 | 15 / 15 | 10 / 0 |
| 31 | 25/26 | 14 / 14 | 10 / 0 |
| 32 | 25/26 | 18 / 18 | 6 / 0 |
| 33 | 25/26 | 8 / 8 | 5 / 0 |
| 34 | 25/26 | 10 / 10 | 6 / 0 |
| 35 | 25/26 | 17 / 17 | 9 / 0 |
| 36 | 25/26 | 10 / 10 | 12 / 0 |
| 37 | 25/26 | 12 / 12 | 11 / 0 |
| 38 | 25/26 | 6 / 6 | 9 / 0 |
| 39 | 25/26 | 6 / 6 | 12 / 0 |
| 40 | 25/26 | 0 / 0 | 6 / 0 |

Whole-segment Fetch/Process coverage from `ScottishPyramid`:

| Tournament | Matches | processed odds fixtures | raw Betfair fixtures | usable Betfair TWA fixtures |
|---:|---:|---:|---:|---:|
| 54 | 1,230 | 1,226 (99.7%) | 990 (80.5%) | 989 (80.4%) |
| 55 | 1,065 | 1,064 (99.9%) | 518 (48.6%) | 517 (48.5%) |
| 56 | 1,024 | 1,023 (99.9%) | 891 (87.0%) | 880 (85.9%) |
| 57 | 1,025 | 1,025 (100%) | 750 (73.2%) | 747 (72.9%) |
| 73 | 520 | 365 (70.2%) | 289 (55.6%) | 273 (52.5%) |
| 982 | 87 | 84 (96.6%) | 75 (86.2%) | 75 (86.2%) |
| 1520 | 372 | 368 (98.9%) | 258 (69.4%) | 251 (67.5%) |

This corrects the RFC preflight interpretation: processed odds are near-complete for 54/55,
but the archived Betfair close needed by scorecard v1.1 is not. In particular, only 153/355
held-out T55 monitor fixtures have the scorecard reference; 25/26 coverage is almost absent.
The scorer emits `monitor_t54`/`monitor_t55` only when latents contain them and never emits
`delta_logloss_vs_control` for a monitor subset.

### 5. W0 scorecard v1.1

On `mcmc-beast`, `scripts/score_runs.jl experiments/scotland/W0_runs.csv` loaded and rescored
all 19 W0 runs under v1.1 and wrote 12,426 rows. The additive table holds the new
`entered_spfl` first-10/first-20 rows alongside all 10,640 v1 rows; no v1 record was
replaced (`v1.1`: 12,426 rows / 19 runs; `v1`: 10,640 rows / 19 runs). `experiments/LEADERBOARD.csv` and
`experiments/LEADERBOARD.md` were regenerated from v1.1. Every transition metric in the
Markdown view carries its own `n_fixtures`, including the new entered-SPFL cohort.

### 6. Tests

Standalone:

```text
scoped_splitter_tests.jl: 48/48 pass
harness_scoring_tests.jl: 29/29 pass
```

Parallel runner:

```text
20 / 21 suites passed
```

The sole failure is the known T007 signature in `features_tests.jl`:
`UndefVarError: SplitClockProbe not defined in Main`. Both new/changed suites pass. The suite
count increased from 20 to 21, so 20/21 is the same single known failure as Phase 1's 19/20.

Repository checks:

```text
./scripts/todo.sh check
OK: 31 task(s); metadata, template and registry agree; AGENTS.md 19673 bytes (< 22000).

git diff --check
(no output)
```

## Neutral-venue finding

Neither `sofascore.events` nor `sofascore.matches` has an explicit neutral column, and the
raw event JSON has no neutral-ground key at the top level or under tournament/venue objects.
TODO 029's BBC-venue/round derivation remains the independent source. Consequently no
`neutral_venue` fetcher column was invented. The checked-in CSV is the only hand-maintained
Data input and records its provenance and four non-venue scoreability anomalies row by row.
Its absence is a hard error, and its exclusions are unioned with any future native
`neutral_venue` column so those four scoreability rows cannot silently return. Repository
rsync guidance now uses root-anchored `/.cache/` and `/data/` excludes, preserving tracked
nested inputs such as `src/Data/data/`.

## Open questions for Phase 3

1. Treat T55 monitor scores as an explicitly partial diagnostic until the Betfair archive is
   backfilled; do not compare their aggregate directly with T54's near-complete 24/25–25/26
   panel.
2. Phase 3's coverage gate should report target and monitor coverage separately. A candidate
   must still have all 710 target latents, while absent Betfair monitor references are a
   scoring-coverage diagnostic, not a fit failure.
3. The four non-venue cup exclusions should eventually become typed Data QA flags if raw event
   status/normal-time score is graduated into `MatchesData`; until then the single audited CSV
   keeps the g2 training contract exact.
4. Keep monitor subsets out of `delta_vs_control` selection. They are cross-tier transfer
   diagnostics, not promotion criteria under the approved scorecard.

## Blocked

None.
