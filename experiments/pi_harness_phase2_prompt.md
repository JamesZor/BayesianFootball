# Work package — Experiment harness, Phase 2: data scope (TODO 030)

You are the implementer; Claude is the manager and reviews against §3 before Phase 3 is
released. **Do Phase 2 only, then stop.** Same worktree, branch, git, DB, beast and
credential rules as Phase 1 (`experiments/pi_harness_phase1_prompt.md` §1). Re-read them.
**No MCMC in Phase 2.**

## 0. Read first

1. `docs/architecture/rfc_experiment_harness.md` §2, §3 (`DataScope`, `ScopedWalkForwardCV`),
   §6 Phase 2, §7 (risks: neutral venue, Features dispatch placement, monitor Betfair coverage).
2. `docs/architecture/harness_phase1_report.md`: what exists now.
3. Code you are graduating:
   - `current_development/grw_pyramid_cups/l01_loader.jl`: `ScottishPyramidCups`, `pcx_load_data`, `pcx_fs_label`, `PyramidGRWCV`, `create_id_boundaries`, `get_next_matches`, `pcx_align_time!`, the `create_features` override, `pcx_widening_report`
   - `current_development/grw_pyramid_cups/data/cup_bridge_allowlist.csv` and how it was derived (TODO 029: `experiments/scotland/02_cross_tier_cups_and_pyramid_eda/r01_extract_scottish_pyramid_dataset.jl`, `data/r01_club_season_tiers.csv`)
   - `current_development/grw_player_hybrid/l01_loader.jl`: `gph_splitter`, `gph_fold_inputs`, `gph_filtration_report`
   - `src/Data/splitting/` (`GroupedCVConfig`, `create_id_boundaries`, `get_next_matches`, `_effective_step_map`), `src/Data/fetchers/segments.jl`, and the existing `Features.create_features` splitter methods in `src/features/`
   - `src/harness/cohorts.jl` (`club_season_tiers`; reuse it for the cup rule)

## 1. Phase 1 follow-ups (do first, small)

- **New cohort direction `:entered_spfl`:** a club whose current season is in 54–57 but whose
  previous season is absent from 54–57 while the data covers that previous season (i.e.
  promoted from the Highland/Lowland pyramid, e.g. Kelty Hearts, Bonnyrigg Rose, The
  Spartans). Today these clubs fall into no cohort. Add `transition_entered_spfl_{first10,first20}`
  subsets, bump the scorecard version to `v1.1`, and re-score W0 on the beast (scores only; no sampling).
- **Cohort sizes are first-class:** every cohort row already carries `n_fixtures`. Add it to
  `LEADERBOARD.md` next to each cohort metric (e.g. `+1.8 pp (n=57)`), because
  relegated-into-L1 is only 28/57 fixtures.

## 2. What to build

| Item | Work |
|---|---|
| `Data.ScottishPyramid()` | segment for 54, 55, 56, 57, 73, 982, 1520 in `src/Data/fetchers/segments.jl`; `load_datastore_cached` works for it; record per-tournament match/odds/betfair coverage in the report |
| `Data.DataScope` | exactly the RFC §3 fields (`name, train_tournaments, cups, target_tournaments, monitor_tournaments, clock_tournaments, target_seasons, history_seasons, dynamics_col`), with validation (target ⊆ train; clock ⊆ target; cups ∈ (:none, :senior_spfl_ties)) |
| `Data.apply_scope(ds, scope) -> DataStore` | a copy (never mutate the cache): cup season relabel to football season (port `pcx_fs_label`); `:senior_spfl_ties` = keep a cup tie only if **both** clubs are in an SPFL league (54–57) that football season per `club_season_tiers`; B-teams/U21/guest sides are excluded by that rule. Neutral venues: see below. Drops tournaments not in train ∪ target ∪ monitor |
| `Data.ScopedWalkForwardCV <: AbstractSplitter` | wraps a `DataScope`. `create_id_boundaries`: canonical `GroupedCVConfig` folds on `clock_tournaments`, widened with train/cup rows strictly before each fold's first held-out kickoff (generalise `PyramidGRWCV`; keep its assertions). `get_next_matches`: target **and** monitor rows of the next clock bin, carrying `tournament_id` so the scorer can split them |
| Features clock | a `create_features` method for `ScopedWalkForwardCV` in `src/features/` next to the existing splitter methods: port `pcx_align_time!` including its cross-check against `Data._effective_step_map` for clock-tournament rows |
| Scorer | `monitor_t54`, `monitor_t55` subsets when a fit's latents include monitor fixtures (scored vs Betfair like target; never used in `delta_vs_control` selection). Test with a synthetic latents frame |
| Neutral venues | search the raw SofaScore event JSON in betdb (`sofascore.events.raw_data`, read-only) for a neutral-ground flag. If one exists, add a `neutral_venue` column in the Data layer and exclude neutral cup ties under `:senior_spfl_ties`. If not, create `src/Data/data/scottish_neutral_venue_ties.csv` from the allowlist diff (ties the rule keeps but the allowlist dropped because of venue) and document it as the one hand-maintained input |
| Tests `test/scoped_splitter_tests.jl` | on a synthetic multi-tier league: strict cutoff, no train/held-out overlap, monitor rows only held out, clock cross-check, cup rule (B-team/guest/non-SPFL excluded), season relabel, `DataScope` validation errors. Add to `run_parallel_tests.jl` |

## 3. Acceptance (Claude will check each)

1. **Lower scope is canonical:** `DataScope(name = "lower")` on `ScottishLower` data gives
   boundaries whose `history_match_ids`, `target_match_ids` and next-bin held-out IDs are
   **identical**, fold by fold, to `gph_splitter(["24/25","25/26"])` (40 folds, 710 unique held-out).
2. **spfl_cups reproduces g2's training rows:** `DataScope(name = "spfl_cups", train_tournaments = [54,55,56,57], cups = :senior_spfl_ties)`
   reproduces `current_development/grw_pyramid_cups/results/r02_widening_g2_grw_all_spfl_cups.csv`
   per fold (`n_train, n_lower, n_upper, n_cup, n_oos, n_teams`). Where the rule-based cup set
   differs from the 363 allowlisted ties, list **every** differing tie with its reason in the
   report. Exact equality is expected once neutral venues are handled.
3. **Features identical to the pyramid path:** for the g2 model recipe (`pcx_model("g2_grw_all_spfl_cups")`),
   the feature sets built through `ScopedWalkForwardCV` equal those built through
   `PyramidGRWCV` + `pcx_align_time!` on **all 40 folds**: `ordered_match_ids`,
   `time_indices`, `n_history_steps`, `n_target_steps`, `n_teams` and the team index maps.
   Run on the beast; deterministic, no sampling.
4. **Monitor fixtures:** with `monitor_tournaments = [54, 55]`, report per-fold monitor
   held-out counts and Betfair coverage for 54/55; the 710 target IDs are unchanged.
5. **W0 re-scored at v1.1** with the `entered_spfl` cohort; `LEADERBOARD.md` shows cohort n.
6. **Tests:** `test/scoped_splitter_tests.jl` and `test/harness_scoring_tests.jl` pass
   standalone; `run_parallel_tests.jl` is no worse than 19/20 (T007).
7. `./scripts/todo.sh check` passes; TODO 030 Work Log has dated `@pi` lines.

## 4. Reporting and stopping

- Write `docs/architecture/harness_phase2_report.md`: evidence per acceptance item, the cup
  diff table, neutral-venue finding, coverage table, open questions for Phase 3.
- Commit, push `feat/experiment-harness`, print exactly `PHASE2_DONE` and stop. If blocked on
  a human decision: write it under "Blocked", print `PHASE2_BLOCKED`, and stop.
