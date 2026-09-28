# Harness KLM: production runbook (manager **go prod** only)

**Not executed during build.** Manager supplies merged `feat/w2-tier-components` SHA and worktree. Run from `bash /root/BF_runs/8effca72/scripts/beast_checkout.sh <merged-sha>` on `mcmc-beast`, with linked `.env`/Manifest, Scottish caches, and `-t 16`; logs under `/root/BF_runs/logs/klm/`, never in the checkout. Check `pgrep -af julia` first. No NUTS/smoke/grid or other model fits; the W1 MAP screen below is the sole permitted re-fit (Amendment 1).

## 1. Schema preflight

```bash
cd /root/BF_runs/<merged-sha>
set -a; source .env; set +a
/root/.juliaup/bin/julia --project -t 16 -e 'using BayesianFootball; Training.ensure_schema!(PostgresStorage("harness")); Harness.ensure_harness_schema!(PostgresStorage("harness"))'
docker exec mcmc_experiments_postgres psql -U postgres -d mcmc_experiments -c '\d fit_fold_artifacts'
docker exec mcmc_experiments_postgres psql -U postgres -d mcmc_experiments -c '\d fit_artifacts'
```

The sole schema change is additive: `fit_artifacts.layout` defaults to `single` on legacy rows; new saves use `per_fold` and the `(run_id, fold_idx)`-keyed table. Do **not** migrate/rewrite the existing 324 fits.

## 2. W0 reference re-score, v1.2

`docs/architecture/harness_klm_W0_rescore_v12.csv` lists exactly the 14 unique v1.1 W0 UUIDs absent from the v1.2 grid plus their reference control. Thirteen retain their recorded `m12_td` control (`132df5c2`); `g2_harness_repro` had no recorded v1.1 delta and is self-controlled to preserve that distinction. Its self-comparison will now have a zero delta; annotate rather than presenting it as a pre-existing pairing. The file uses the frozen W1 panel **56+57 | 24/25,25/26 | n=710**. Verify UUIDs and experiments read-only in `runs` before executing. `scripts/score_runs.jl` includes the `scottish_lower_momentum_grw`, `scottish_lower_decompression`, and `scottish_lower_decoupled_xg` prototype loaders when those experiments occur; the remaining run types live in `src/`.

```bash
/root/.juliaup/bin/julia --project -t 16 scripts/score_runs.jl \
  docs/architecture/harness_klm_W0_rescore_v12.csv \
  --target-seasons 24/25,25/26 --expected-fixtures 710 \
  > /root/BF_runs/logs/klm/prod_w0_rescore.log 2>&1
```

Require 14 distinct scored runs (the 14 missing include `m12_td`), no failures, and new `v1.2` grid rows for all 14. The procedure writes only `harness_scores` v1.2; existing v1.1 rows stay untouched.

## 3. W1 MAP screen re-run (Amendment 1)

The original 12 screen UUIDs are synthetic and have no fit artifacts. Do **not** invoke `score_runs.jl` on them. Re-fit MAP, in memory, with the original W1 12-cell recipe:

```bash
/root/.juliaup/bin/julia --project -t 16 scripts/run_candidates.jl \
  experiments/scotland/03_dynamics_scope_matrix/candidates.jl --stage screen \
  > /root/BF_runs/logs/klm/prod_w1_screen.log 2>&1
```

Check **12 distinct** `stage='screen', scorecard_version='v1.2'` UUIDs and the panel `56+57|24/25,25/26|n=710`, plus 12 `screen_validity` rows in `harness_checks`. Before and after, read `harness_experiments` row `scottish_dynamics_scope_matrix_w1` and compare date, TODO, question, dimension, status (`completed`), decision, readme, and original run list. The `preserve_completed=true` re-screen upsert must leave all metadata intact; only a merged run list/commits may grow. Stop if it changes any decision field. Screen can take substantial time; this is **not** NUTS sampling.

## 4. Generate and compare the board

```bash
/root/.juliaup/bin/julia --project -t 16 scripts/leaderboard.jl \
  > /root/BF_runs/logs/klm/prod_board.log 2>&1
git show 090db11e:experiments/LEADERBOARD.md > /root/BF_runs/logs/klm/board_v11.md
```

Compare all unique grid UUIDs extracted from the first table of `board_v11.md` with the newly generated Markdown: **every v1.1 grid UUID must appear**. Confirm the v1.2 MAP section is present, warning-labelled and separate, with 12 W1 cells. For W2 read the `grw_*` / `td_*` deltas against each row's `control_name` and UUID; compare numerically with `experiments/scotland/04_pyramid_tier_transition_arms/README.md`. Panels must remain separate; no W0-vs-W2 delta. Compare the W1 register row to its pre-screen state, and confirm no v1.1 scores were deleted. Require zero unexplained omissions or changed controls.

Copy `experiments/LEADERBOARD.{csv,md}` and `experiments/REGISTER.md` to the worktree named by the manager, review `git diff`, then commit/push **only** `feat/w2-tier-components` as instructed. Append production evidence (merged SHA, counts, queries, wall times, diff) to `docs/architecture/harness_klm_report.md`. Print `KLM_PROD_DONE` and stop.

## Build rehearsal only (test database)

On a dev checkout with `BF_EXPERIMENTS_TEST_DB_URL` pointing **exactly** to database `mcmc_experiments_test` (other names are rejected), run `scripts/run_candidates.jl ... --stage screen --test-db`. The flag never falls back to production if the variable is absent or mispointed. The screen script and completed-register preservation are tested separately as well. Neither `scripts/score_runs.jl` nor `scripts/leaderboard.jl` targets the test DB via this flag; render a test-board preview with `Harness.read_scores(test_db)` and `scripts/leaderboard.jl:leaderboard_markdown` without invoking its production `main()`.
