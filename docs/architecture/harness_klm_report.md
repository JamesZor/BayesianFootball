# Harness KLM — build phase report (2026-09-28)

## Resolved block — MAP-screen source

The original §2 Item 5 / §5 procedure required re-scoring 12 MAP-screen runs at v1.2 with `scripts/score_runs.jl`, without re-fitting. That was impossible under the original contract:

- `src/harness/stages.jl:screen` fits MAP in memory, persists only `harness_scores` and `harness_checks`, and never calls `save_fit`. The v1.1 leaderboard itself warns: screen recipe UUIDs are **not** rows in `mcmc_experiments.runs`.
- Read-only production query (on mcmc-beast, `docker exec mcmc_experiments_postgres psql -U postgres -d mcmc_experiments`): `SELECT count(DISTINCT h.run_id), count(DISTINCT r.run_id) FROM harness_scores h LEFT JOIN runs r ON r.run_id=h.run_id WHERE h.scorecard_version='v1.1' AND h.stage='screen';` → **12 | 0**.
- `scripts/score_runs.jl:main` calls `Harness.score_runs`; that function calls `Training.load_fit(PostgresStorage(ref.experiment), ref.run_id)` for each run. No screen fit exists to load, so a CSV cannot re-score these rows. Re-running `screen` would fit MAP again (sampling/fits disallowed by §0); copying old v1.1 scores into v1.2 is not re-scoring and was expressly rejected as a version fallback.

Under Amendment 1 the human authorised re-running the W1 MAP screen only. Its test-database rehearsal and completed-register protection are part of this build; production remains read-only until the manager's explicit go.

## Build implementation and iterations (production deferred)

- Implemented diagnostic metric-carrying failures, harness-only 1.05 / 0.1% gate, per-fold PostgreSQL shell/blob layout and legacy load, append-on-extension (legacy stays single-blob), no thinning fallback, smoke nonce and duplicate policy, and per-panel/per-control leaderboard rows. These remain **build-phase only and not approved for production**.
- Added unit/integration checks for failures, gate boundaries, multi-control leaderboard, duplicate policy, legacy-row load, cascade, and fold-blob hash stability.
- Beast development sync: `rsync -a --exclude '/.cache/' --exclude '/data/' --exclude '/.git/' ./ root@mcmc-beast:/root/BF_runs/klm_dev/`; test DB `mcmc_experiments_test` created via Docker Postgres; `.env` and `Manifest.toml` linked, Scottish caches copied. Credentials were derived at runtime and not logged.
- `test/harness_scoring_tests.jl` and `test/harness_runner_tests.jl`: runner **159/159 in 1m15.8s**, scoring tests to this point passed (see `/root/BF_runs/logs/klm/focused.log`). Initial synthetic scoring test errored due to a typed `Vector` of missing-only control IDs; corrected and second run passed the scoring section. The second combined invocation could not connect to the test DB because the production-only pgpass entry did not match the new database; resolved using the same Docker credential in process environment (never printed).
- `test/test_extension.jl` with test DB: **43/43, 27.0 s** (`/root/BF_runs/logs/klm/extension.log`). `test/test_db_storage.jl`: **133/133, 33.5 s** (`/root/BF_runs/logs/klm/db_storage.log`). Test-created run rows removed from `mcmc_experiments_test`; database retained.
- The 14 W0 UUIDs missing from v1.2 were identified read-only by comparing `git show 090db11e:experiments/LEADERBOARD.md` against the production v1.2 grid UUIDs. `docs/architecture/harness_klm_W0_rescore_v12.csv` records their production experiments and original controls; `scripts/score_runs.jl` already includes the three required prototype loaders. No MAP CSV is needed under Amendment 1.
- First MAP rehearsal in `/root/BF_runs/klm_dev/` produced 12 v1.2 synthetic screen UUIDs / 8,640 score rows but **no checks**. Screen silently swallowed `write_checks!`'s transaction error: `_latent_audit(require_variance=false)` emits `NaN` in `min_sd`, and JSON3 correctly refuses to serialize NaN. Reproduced `JSON3.write((; min_sd=NaN))` on the beast. Fixed by recursively mapping nonfinite diagnostic values to JSON null at the persistence boundary and by no longer swallowing the screen check write failure. The clean-SHA rerun passed (below). No production write.

### Build-phase validation (clean pushed SHAs)

| SHA / checkout | Check | Result / wall | Evidence log |
|---|---|---|---|
| `af5645b4` | Separate test DB, `test/test_db_storage.jl` + `test/test_extension.jl` | **150/150** (35.8 s) + **47/47** (13.7 s); combined 82 s including compile. Tests cover expected/unexpected diagnostic failures and JSON null for NaN, duplicate run UUID and nonce parity, binary bytea, legacy single-blob compatibility, per-fold parity and cascade, per-fold append with unchanged `md5(fold_blob)` and legacy-extension rewrite, completed W1 register preservation. | `/root/BF_runs/logs/klm/af_focus.log` |
| `af5645b4` | W1 12-cell MAP screen, `scripts/run_candidates.jl ... --stage screen --test-db` | **12/12** hard summaries, 12 UUIDs / 8,640 `v1.2` screen score rows, 12 each of `screen_validity`, `filtration`, `latents` checks; all score panels `56+57\|24/25,25/26\|n=710`; **568 s**. Production unchanged. | `/root/BF_runs/logs/klm/af_screen.log` |
| `5d1103f7` | Read-only production legacy `load_fit` after verifying **324 legacy artefacts, no layout column** | **4/4**, 24 s total: `ce7ea22f` stride 2, 60 folds, first/last chain (500,194,4)/(500,1830,4); `c4a0fa94` stride 4, 60, (250,238,4)/(250,1960,4); W0 `132df5c2`, 45, (800,69,4)/(800,71,4); pre-PR #42 `aad544b3` (2026-08-30), 42, (800,65,4)/(800,67,4). No production schema change. | `/root/BF_runs/logs/klm/5d_legacy.log` |
| `5d1103f7` | Synthetic >1.2 GiB round trip in test DB from real W2 `grw_base`, 120 renumbered deep-copied folds, no sampling | **1,993,184,800 raw serialized fold bytes** (>1.2 GiB); compressed fold bytes **1,896,612,795**; largest blob **28,621,776 B**; save **11.38 s**, load **5.54 s**, total process **44 s**, peak VmHWM **6,501,460 KiB**. All chains, fold metadata, config, diagnostics, metadata, latents and save path structurally/bit-exact equal. Run deleted from test DB after verification. | `/root/BF_runs/logs/klm/5d_large.log` |

Required existing-suite matrix from the clean `5d1103f7` checkout, each suite run separately on mcmc-beast (`/root/BF_runs/logs/klm/matrix_<suite>.log`; `matrix.tsv`, **9/9 green**): harness scoring **46/46** (previous recorded 43), harness runner **165/165** (previous partial-build 159), harness portfolio **27/27** (previous 27), portfolio **92/92**, unified portfolio **792/792**, pyramid tiers **94/94** (previous 94), scoped splitter **48/48** (previous 48), database storage **150/150** (previous partial-build 133), extension **47/47** (previous partial-build 43). No skipped suite was treated as a pass. Clean `5d1103f7` `test/run_parallel_tests.jl`: **24/25 in 389.0 s** (`/root/BF_runs/logs/klm/5d_parallel2.log`); the sole failure is exactly known T007: `features_tests.jl` references `SplitClockProbe` absent in its isolated process. Every other suite passed, including tape allocation (365.8 s). A first parallel launcher attempt found zero child suites because `/root/.juliaup/bin` was missing from the shell PATH; the corrected environment supplied it and ran all 25 suites.

Read-only production postcheck: `fit_artifacts` still has 324 legacy rows, `fit_fold_artifacts` table and `fit_artifacts.layout` column are absent, and production has **zero** v1.2 screen UUIDs. Independent W1 register comparison against the saved production JSON: all completed-row metadata and `run_commits` match exactly; original `run_ids` are retained. Only the separate test database received rehearsal writes.

Screen-render preview (test DB, no board files written): `scripts/validate_klm_screen_rehearsal.jl` confirms a distinct warning-labelled MAP section, 12 displayed rows, 12 validity UUIDs, one panel and the completed W1 register metadata unchanged against a read-only production snapshot. Its initial invocation omitted the screen-validity column (the production leaderboard script supplies it); the corrected validator passed on clean pushed SHA **`37429d59`** in **39 s** (`/root/BF_runs/logs/klm/374_screen_check.log`, `KLM_SCREEN_PASS ids=12 checks=12 score_rows=8640 ... register=preserved markdown_map_rows=12`).

After capturing the final rehearsal counts (8,640 scores, 36 checks, one W1 register row), test-created objects were removed with `DROP SCHEMA public CASCADE; CREATE SCHEMA public` **only in `mcmc_experiments_test`**; zero public tables remain and the test database still exists. Production was not changed.

The runbook and 14-run W0 CSV are `docs/architecture/harness_klm_prod_runbook.md` and `harness_klm_W0_rescore_v12.csv`. Dry run: CSV parsed with 14 unique UUIDs, both original controls (`m12_td`, `g2_harness_repro`) present, and a read-only production `runs` lookup found all **14/14** UUIDs; the referenced experiment/control grouping is also covered by the harness runner suite. The W0 rescoring/production board are deliberately deferred until §5. `experiments/LEADERBOARD.md`, `REGISTER.md` and production `mcmc_experiments` were not changed. No NUTS/smoke/grid sampling was run.

## Build disposition and reproduction

- **(k) / gate:** `src/harness/checks.jl` retains metric-carrying `DiagnosticFailure.value`; `src/harness/store.jl` persists it as valid JSON, with unavailable nonfinite telemetry as null. Boundary, ESS-note-only and unexpected-exception checks: `test/harness_runner_tests.jl` 165/165 and `test/test_db_storage.jl` 150/150 (test DB). Training's own `fold_results.converged` rule is unchanged.
- **(l):** `src/training/inference/db/schema.sql`, `db_storage.jl`, `extension.jl` and `src/harness/stages.jl` implement binary per-fold artefacts, legacy loads before and after migration, append-on-extension and stride 1 without the thinning fallback. `test/test_extension.jl` 47/47, `test/test_db_storage.jl` 150/150, 4/4 legacy production read-only loads, and the 1.993 GB raw-fold synthetic test DB round trip above verify this. No production schema migration occurred.
- **(j):** `src/harness/stages.jl` smoke nonce and grid `on_duplicate=:error`, `db_storage.jl` duplicate policy; the 165/165 runner and 150/150 database tests include smoke A/B unique UUID + parity and default `:return` compatibility.
- **(m):** `src/harness/scoring.jl` and `scripts/leaderboard.jl` compare controls within each panel, keeping W2 pairings distinct. The 46/46 scoring tests cover panel isolation and multiple controls. W0 CSV dry-run finds 14/14 production run UUIDs; W1 12-cell screen is verified only in the separate test DB. The actual W0 production rescore, W1 production re-screen and board/register generation are **held for an explicit go prod** under `harness_klm_prod_runbook.md`.

Reproduction on mcmc-beast, using a clean pushed-SHA checkout, Julia 1.12.4, the existing `.env`, Scottish caches and credentials sourced at runtime (never printed): `julia --project -t 16 scripts/validate_klm_legacy_loads.jl` (production SELECT only); with `BF_EXPERIMENTS_TEST_DB_URL` naming **exactly** `mcmc_experiments_test`, `julia --project -t 16 scripts/validate_klm_large_roundtrip.jl`, `julia --project -t 16 scripts/run_candidates.jl experiments/scotland/03_dynamics_scope_matrix/candidates.jl --stage screen --test-db`, and `julia --project -t 16 scripts/validate_klm_screen_rehearsal.jl`. Suite logs and exact SHA paths are in the table above. Do **not** execute the production runbook until the manager's explicit go; no merge, PR review or production rescore was performed in this build phase.

## Other R̂ 1.01 / ESS 400 threshold sites (not changed)

The harness review is deliberately separate from these gates/monitors. Targeted audit: `rg -n '1\.01|([<>]=?\s*400(\.0)?)|min_ess\s*=\s*400' src scripts test current_development experiments -g '*.jl'` (filter out odds/HTTP/tick prices and bootstrap counts). Non-harness sites:

- Core Training: `src/training/inference/types.jl:224-247` documents 1.01 and ESS 400; its **actual** default `max_rhat` is already 1.05, `min_ess=400`. `src/training/inference/convergence.jl:summarise_convergence` applies the defaults; `engine.jl:fit_model`, `io.jl:load_fit`, `compat.jl:upgrade_to_fit`, `db_storage.jl:save_fit`, and `extension.jl:_extension_insert_fold!` use the Training summary (including `fold_results.converged`). MatchDay consumes it. Not changed.
- Harness: `src/harness/checks.jl` records ESS <400 as a **note**; no longer a review-failure gate.
- Active/experimental strict R̂ or ESS: `current_development/multiscale_grw/l01_loader.jl` (R̂ 1.01, ESS 400), `current_development/scottish_lower/_protocol/sampling.jl` (1.01/400), `current_development/scottish_lower/00_team_poisson/l04_sampling_gates.jl` (1.01), `current_development/scottish_lower/archive/open_play_rebuild/r06_remote_nuts_smoke.jl` (1.01/400), `current_development/scottish_lower/05_composable_count_builder/l05_parity.jl` and `r01_demo.jl` (1.01); `current_development/grw_joint_negbin/l01_loader.jl`, `grw_player_hybrid/l01_loader.jl`, `grw_market_smile/l01_loader.jl` (strict advisory 1.01); `experiments/scottish_lower/11_decompression_pxg_covariate/l12_workflow.jl` and `12_decoupled_generative_xg/l13_workflow.jl` (strict 1.01); `experiments/scottish_lower/06_joint_player_lineup_fusion/r64_smoke_hierarchical_kappa.jl` (ESS 400), `r69_verify_matchday_2627.jl` (tail ESS 400 monitor); `experiments/scottish_lower/08_goal_decomposition/r08_sampling_budget_benchmark.jl` (ESS 400).
- Other standalone diagnostics at 1.01: `current_development/team_wealth/r04_train_wealth_ireland.jl`, `r07_train_wealth_anchored.jl`; `current_development/smile_negbin/r02_train_ireland.jl`, `r03_pipeline_smoke.jl`; `current_development/orderbook_layer2/r02_train_ireland.jl`, `r08_train_ireland_noanchor.jl`; `current_development/scottish_upper/r01_smoke.jl`, `r02_grid_family.jl`, `r03_eval_family.jl`, `r04_grid_pillar.jl`; `current_development/scottish_proxy_xg/r02_smoke.jl`, `r03_grid.jl`; `current_development/bbc_xg_proxy/r03_funnel_smoke.jl`, `r04_funnel_hier_smoke.jl`, `r06_funnel_full_spec.jl`, `r07_funnel_iso_grid.jl`, `r07b_funnel_iso_pilot.jl`; archived `current_development/scottish_lower/archive/proxy_xg/r02_smoke.jl`, `r03_grid.jl`, `archive/wealth/r03_grid_wealth.jl`, `archive/neg_bin/r01_smoke_negbin.jl`; `current_development/match_day_inference/r17_extend_to_card.jl` (comment about a strict legacy fold).
- Tests with threshold assertions: `test/inference_tests.jl:192-213` (1.01/400); the unrelated 1.01 odds in portfolio/MatchDay tests and match-odds tick bands are **not** convergence rules. No non-harness thresholds were weakened.
## Fix round 1 (Claude CLI builder, 2026-09-28)

Responds to the independent review `docs/architecture/harness_klm_review.md` (CHANGES_REQUIRED,
F1–F11). pi started this round and ran out of quota. Claude CLI (Opus 5.5) took over the
uncommitted draft, reviewed it finding by finding, and fixed what was missing or wrong. That
covered one test-breaking `include`, the F5 test, the runbook's F2 SQL gate and the indentation.
The brief is `experiments/claude_klm_fix1_brief.md`.

| ID | Disposition | Change | Test / evidence |
|---|---|---|---|
| F1 (major) | **Fixed** | `scripts/run_candidates.jl:parse_args` rejects `--test-db` unless `--stage screen`, naming the production-bound stages. Harness guide and runbook say the flag is screen-only. | Runner suite: `parse_args` refuses smoke, grid and portfolio, and `main()` refuses `--stage smoke --test-db`. |
| F2 (major) | **Fixed** | `harness_klm_W0_rescore_v12.csv` adds the 12 W1 grid UUIDs to the `m12_td` (`132df5c2`) group under their v1.1 labels, including the six aliases (26 distinct UUIDs). The runbook step-2 count becomes 26. Step 4 requires every v1.1 (UUID, label), and it now includes a read-only SQL gate: after the re-score, 12 W1 UUIDs have a v1.2 delta against `132df5c2`, and the `97c7a3d9` count is unchanged. | Runner suite §8: 26 refs, 26 distinct UUIDs, alias→UUID map, 25 refs in the `m12_td` group. Production, read-only: the CSV's 25 (label, UUID) candidate pairs **equal** the v1.1 `delta_logloss_vs_control` pairs against `132df5c2`, and all 26 `runs.experiment_name` match. Gate SQL baseline: `97c7a3d9` → 12, `132df5c2` → none. |
| F3 | **Fixed** | New `scripts/klm_test_db_guard.jl:assert_klm_test_database!` keeps the parsed-name check. It then connects and requires `SELECT current_database() = 'mcmc_experiments_test'` before any schema write. It is used by `run_candidates.jl` and both `validate_klm_*` test-DB scripts. | Runner suite: an unset URL and a production-named URL are refused by `main()`. `<test url>?dbname=postgres` parses as `mcmc_experiments_test` but is refused after connecting. |
| F4 | **Fixed** | `write_experiment!` SQL and `_merge_experiment_run_ids` split on `[,;]` and trim. A `preserve_completed` upsert that contributes no run IDs keeps the completed row's `run_ids` **byte-identical**. Without that, the new split would rewrite W1's seeded `;` list as `,` on re-screen. | DB suite (test DB): a seeded `;` completed row survives an empty re-screen unchanged, and a later merge yields the comma union. Offline mirror asserts the same. |
| F5 | **Fixed** (runbook + fallback) | Runbook §1 now runs "immediately after merge", before any other `save_fit`/`extend_fit`. `extend_fit` reads `layout` only when the column exists, so a pre-migration row is single-blob. `save_fit` **refuses** with a clear error when the per-fold schema is absent. It does not fall back, because a whole-Fit blob reinstates the 1 GiB wall. It checks before any write. | New `test_extension` testset: rename `fit_artifacts.layout` and `fit_fold_artifacts` in the test DB (renames, so no row is lost), then `save_fit` throws "per-fold schema is not installed" and `extend_fit` extends a legacy run 2→3 folds. The schema is restored in `finally`, and the run reloads with 3 folds. |
| F6 | **Fixed** | `Harness._chain_parity` compares `parent(chain.value)` (draws **and** internals), `names`, `name_map` and `logevidence` with `isequal`. `_fit_parity` and `validate_klm_large_roundtrip.jl` use it. | Runner suite: an identical chain passes. A changed internal value fails, and so does a renamed internal. Strict round trip below. |
| F7 | **Fixed** | (a) `main()` refusal tests (see F1/F3). (b) The pairing test uses different deltas per panel and asserts each (panel, control) value. (c) The rehearsal validator requires exact `run_ids` equality. | Scoring suite, runner suite; rehearsal below. |
| F8 | **Not changed; documented** | `harness_checks` has no invocation identifier. A failed smoke writes its rows with `run_id` NULL, so "latest invocation" can only be guessed from timestamps. A reliable rule needs a new column, which is beyond this additive-schema round. The harness guide now gives the manual procedure: confirm the latest smoke passes all hard checks, archive, then delete only the specific stale failed hard-check IDs, and only with authorisation. | Production has zero failed hard smoke rows (review), so nothing is blocked today. |
| F9 | **Fixed** | Harness guide states the rule, R̂ > 1.05 or divergence > 0.1% of post-warm-up draws, cites `HARNESS_REVIEW_MAX_RHAT` and `HARNESS_REVIEW_MAX_DIVERGENCE_RATE`, and lists ESS/BFMI/tree depth as notes. | Docs. |
| F10 | **Fixed** (2 of 3); **annotated** (1) | The BFMI note reads `<=`, matching the test. `DualStorage.save_fit` forwards `on_duplicate` to PostgreSQL only, and the DB guide documents it. `g2_harness_repro` stays self-controlled. `leaderboard` deliberately includes self-controls (as does `m12_td`), and the runbook says to annotate its Δ 0 as not a v1.1 pairing. | DB suite: `DualStorage` `:return` returns the run, and `:error` throws. |
| F11 | **Fixed** | `test_db_storage` and `test_extension` delete their experiment's `runs`, `config_registry`, `harness_experiments` and `harness_checks` rows in `finally`. | `runs`, `config_registry` and `harness_experiments` counts in `mcmc_experiments_test` are 0 after each suite (below). |

Also fixed: pi's last dev run (`/root/BF_runs/logs/klm/fix1_dev.log`, status 1) errored in "Screen CLI
test-database opt-in". The test includes `run_candidates.jl` into a bare `Module(...)`, which
has no `include`. The script now uses `Base.include(@__MODULE__, …)`.

### Fix round 1 evidence — clean pushed SHA `5e70f2a2`

These were run on mcmc-beast from `beast_checkout.sh 5e70f2a2cca00acf4285ed38b8f452ad30505635`, with the Scottish caches
copied from `/root/BF_runs/a76a65df/.cache/`. Julia 1.12.4 was run with `-t 16` through the `klm_env.sh` credential wrapper, and
no credential was printed. `BF_EXPERIMENTS_TEST_DB_URL` named `mcmc_experiments_test`. Each item ran in its own process in tmux
`claude_klm_fix1`, one after another. The runner is `/root/BF_runs/logs/klm/fix1_suite.sh`, logs are
`/root/BF_runs/logs/klm/fix1_clean_<name>.log`, and the summary is `fix1_clean_summary.tsv`. No sampling was run.

| Check | Result | Wall (process) |
|---|---|---|
| `test/test_db_storage.jl` (test DB) | **154/154** (was 150; +4 F4/F10) | 68 s (tests 36.1 s) |
| `test/test_extension.jl` (test DB) | **51/51** (was 47; +4 F5 pre-migration) | 44 s (tests 28.0 s) |
| `test/harness_scoring_tests.jl` | **46/46** (9 top-level sets) | 30 s |
| `test/harness_runner_tests.jl` | **182/182** (was 165) | 110 s (tests 1m35.3 s) |
| `scripts/validate_klm_large_roundtrip.jl` (strict `_chain_parity`) | **PASS**: 1,993,184,800 raw fold bytes, 1,896,612,795 compressed, 120 folds, largest blob 28,621,776 B, save 10.62 s, load 5.49 s, peak RSS 6,500,580 KiB | 41 s |
| `scripts/validate_klm_legacy_loads.jl` (production, read-only) | **4/4**: `ce7ea22f` 60 folds (500,194,4)…(500,1830,4); `c4a0fa94` 60 (250,238,4)…(250,1960,4); `132df5c2` 45 (800,69,4)…(800,71,4); `aad544b3` 42 (800,65,4)…(800,67,4) | 25 s |
| Test-DB debris after each item (F11) | `runs`/`config_registry`/`harness_experiments` = **0/0/0** after every item | — |

**W1 MAP screen rehearsal (Amendment 1).** It was re-run because this round touches the screen path (the
`--test-db` guard and the register upsert). The runner is `fix1_rehearsal.sh`, with logs `fix1_clean_{seed,screen,screen_check}.log`.

- The test DB was seeded with the W1 register row from `w1_register_before.json`. Before seeding, read-only, that snapshot
  matched production field for field.
- `run_candidates.jl … --stage screen --test-db` scored **12/12 hard PASS** in **563 s**.
- `validate_klm_screen_rehearsal.jl` (21 s) printed `KLM_SCREEN_PASS ids=12 checks=12 score_rows=8640
  panel=56+57|24/25,25/26|n=710 register=preserved markdown_map_rows=12`. That check now requires **exact** `run_ids`
  equality, so the seeded semicolon list survived byte for byte.

The same set also passed on the rsynced dev tree before the push (`fix1_dev2_*`), with the same counts.

**Production.** Access was read-only: SELECTs, `load_fit`, and a `SET default_transaction_read_only` session for the
runbook gate SQL. A final post-run production recheck was not executed, because the session's permission layer
blocked it. Nothing in this round writes to production. The test-DB guard asserts `current_database()` before any write.

**Test DB state.** The rehearsal left its rows in `mcmc_experiments_test`: 8,640 screen scores, its `harness_checks` rows and the W1
register row. The reset (`DROP SCHEMA public CASCADE`) was not run from this session. It is left for the operator.

## Fix round 2 (Claude CLI builder, 2026-09-28)

Responds to "Re-review 1" in `docs/architecture/harness_klm_review.md` (CHANGES_REQUIRED: F12 major, F13 minor).
F12 uses the reviewer's preferred code-only approach, so the six W1 alias labels that F2 required stay on the board.

| ID | Disposition | Change | Test / evidence |
|---|---|---|---|
| F12 (major) | **Fixed** | `Harness.leaderboard` (`src/harness/scoring.jl`) now groups by `(panel, run_id)`, not `(panel, run_id, model)`. The headline (Target LL, 1X2 LL, ECE, compression) and transition-cohort extras are label-independent, so they come from the whole group, whichever label `write_scores!` left the NULL-control rows under. It emits one row per **(label, control)** pairing present in that run's delta rows. A run with no delta rows keeps one row per label, and legacy NULL-control deltas still pair. Runbook §2 now states that the re-score **rewrites the 12 W1 runs' v1.2 non-delta rows, relabelled**, while their `97c7a3d9` delta rows keep the native label. §4 gains a gate: every grid board row must have a finite Target LL, 1X2 LL, ECE and compression (no `—`). It also requires `KLM_BOARD_DRYRUN_PASS` from the new read-only `scripts/validate_klm_board_dryrun.jl` before §2. | New scoring testset "Leaderboard keeps metrics for a UUID scored under two labels" (14 assertions). A control and an arm each carry a native label and an alias; the alias owns the NULL-control metric rows and the m12 delta, and the native label owns only its delta. Across both row orders it asserts 5 rows, all headline and cohort values finite, and each (label, control) → (LL, Δ) pair. Board dry run below. |
| F13 | **Fixed** | A control is named by the label of its own self-pair row (`run_id == control_run_id` on the target/all delta), falling back to `first(model)`. | The same testset puts the control's alias rows first (the reviewer's failing order) and asserts that its native pairings name `td_lower_joint`. The dry run names the control of all 12 W1 `97c7a3d9` rows `td_lower_joint`. |

`scripts/score_runs.jl`: `main` is split into `include_run_loaders` and `score_csv_groups`. Behaviour is unchanged
(`main` still loads, scores, then calls `write_scores!`), so the dry run exercises exactly the production scoring path.

### Fix round 2 evidence — clean pushed SHA `c33d942e`

These were run on mcmc-beast from `beast_checkout.sh c33d942edccd32ce7e495bd83c63d248db772ede`, with Scottish caches from `a76a65df`. Julia was
run with `-t 16` through `klm_env.sh`, in tmux `claude_klm_fix2`. Logs are `/root/BF_runs/logs/klm/fix1_fix2{final,dry}_*`. No sampling was run.

| Check | Result | Wall (process) |
|---|---|---|
| `test/harness_scoring_tests.jl` | **60/60** (was 46; +14 F12/F13) | 40 s |
| `test/harness_runner_tests.jl` | **182/182** | 131 s (tests 1m52.2 s) |
| `scripts/validate_klm_board_dryrun.jl` (production **read-only**: `read_scores`, `read_experiments`, `load_fit`; no `write_scores!`, no schema call) | **`KLM_BOARD_DRYRUN_PASS`**: 26/26 CSV runs scored in memory (287 s, 0 failures). The replace-by-key removed **7,596** production v1.2 rows from the in-memory copy, the reviewer's count. **68** grid board rows, **0** with a NaN headline, **0** `—` Target LL in the rendered Markdown. **32/32** v1.1 first-table (label, UUID) pairs present, and all 32 reproduce their v1.1 Target LL within 5e-6. 12 W1 UUIDs carry a `132df5c2` delta and 12 a `97c7a3d9` delta, whose control is named `td_lower_joint`. | 338 s |

The scoring and runner suites also passed on `d227f1f7` (60/60, 182/182). That run differs only in the dry-run script's
`invokelatest` fix: the first clean dry run stopped with a world-age `MethodError` when it called the included
`leaderboard_markdown`.

The six W1 rows that the reviewer saw render as "—" now read as follows (rendered board
`/root/BF_runs/logs/klm/fix2_clean_board.md`, posterior grid, panel `56+57|24/25,25/26|n=710`):

| Model | Target LL | 1X2 LL | ECE | Compression | Control | Δ LL vs control |
|---|---:|---:|---:|---:|---|---:|
| `td_lower_joint` | 0.64375 | 0.61746 | 0.01427 | 1.928 | `td_lower_joint` | 0.00000 |
| `s12_m02_td_joint` (alias, same UUID) | 0.64375 | 0.61746 | 0.01427 | 1.928 | `m12_td` | 0.00038 |
| `td_lower_poisson` | 0.64679 | 0.62040 | 0.01249 | 2.781 | `td_lower_joint` | 0.00304 |
| `grw_lower_poisson` | 0.64460 | 0.61689 | 0.01617 | 1.190 | `td_lower_joint` | 0.00085 |
| `grw_spfl_poisson` | 0.64644 | 0.61939 | 0.01221 | 1.099 | `td_lower_joint` | 0.00270 |
| `grw_spfl_cups_poisson` | 0.64617 | 0.61634 | 0.01806 | 1.252 | `td_lower_joint` | 0.00242 |
| `grw_spfl_cups_joint` | 0.64385 | 0.61718 | 0.00953 | 1.105 | `td_lower_joint` | 0.00010 |

Production was only read. The runner suite's guard test connects to `mcmc_experiments_test` but creates nothing: before and after every item it still had no `runs` table (the empty schema left by the reviewer).

## Production (manager, 2026-09-28)

Merged SHA `01e9936a`, run on mcmc-beast (`/root/BF_runs/01e9936a`, `-t 16`; logs `/root/BF_runs/logs/klm/`). Timings: §1 16:22:45–16:23:19, §2 16:23:19–16:28:57, §4 16:28:57–16:29:19 (CEST). **§3 (the MAP re-screen) was not run; the human deferred it.**

| Check | Before | After |
|---|---|---|
| `fit_artifacts` rows / layouts | 324 / (no column) | 324 / 1 (`single`) |
| `fit_fold_artifacts` | absent | present, 0 rows |
| v1.1 score rows | 26,256 | 26,256 |
| v1.2 score rows | 20,262 | 30,350 (+17,684 written, −7,596 replaced) |
| W1 grid deltas vs `132df5c2` (m12) | 0 | 12 |
| W1 grid deltas vs `97c7a3d9` | 12 | 12 |
| `harness_experiments` row md5s | — | all unchanged |
| Board rows / rows missing a headline | — | 68 / 0 |
| v1.1 UUIDs and labels missing from the board | — | 0 / 0; all six aliases present |
| MAP-screen section | present (v1.1) | **absent (§3 deferred)** |
