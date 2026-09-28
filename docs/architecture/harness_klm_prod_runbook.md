# Harness KLM: production runbook (manager **go prod** only)

**Not executed during build.** Manager supplies merged `feat/w2-tier-components` SHA and worktree. Run from `bash /root/BF_runs/8effca72/scripts/beast_checkout.sh <merged-sha>` on `mcmc-beast`, with linked `.env`/Manifest, Scottish caches, and `-t 16`; logs under `/root/BF_runs/logs/klm/`, never in the checkout. Check `pgrep -af julia` first. No NUTS/smoke/grid or other model fits; the W1 MAP screen below is the sole permitted re-fit (Amendment 1).

## 1. Schema preflight — immediately after merge

Run this **immediately after the merged checkout is deployed**, before any non-harness
`save_fit`, live `extend_fit`, MatchDay extension or sampling job starts. `save_fit` intentionally
does **not** fall back to a legacy whole-Fit blob: that would reinstate the 1 GiB data-loss
limit. `extend_fit` can still update a legacy row before migration, but that is not a
substitute for applying the additive schema first. No schema write is authorised in the
build/review phase; this section is held for the manager's explicit go prod.

```bash
cd /root/BF_runs/<merged-sha>
set -a; source .env; set +a
/root/.juliaup/bin/julia --project -t 16 -e 'using BayesianFootball; Training.ensure_schema!(PostgresStorage("harness")); Harness.ensure_harness_schema!(PostgresStorage("harness"))'
docker exec mcmc_experiments_postgres psql -U postgres -d mcmc_experiments -c '\d fit_fold_artifacts'
docker exec mcmc_experiments_postgres psql -U postgres -d mcmc_experiments -c '\d fit_artifacts'
```

The sole schema change is additive: `fit_artifacts.layout` defaults to `single` on legacy rows; new saves use `per_fold` and the `(run_id, fold_idx)`-keyed table. Do **not** migrate/rewrite the existing 324 fits.

## 2. W0 reference re-score, v1.2

`docs/architecture/harness_klm_W0_rescore_v12.csv` lists the 14 unique v1.1 W0 UUIDs
absent from the v1.2 grid **plus 12 distinct W1 grid UUIDs** whose original v1.1 comparison
against `m12_td` (`132df5c2`) must be retained. Six W1 rows use their v1.1 alias labels:
`s12_m01_td_poisson`, `s12_m02_td_joint`, `m00_baseline_grw`, `g1_grw_all_spfl`,
`g2_grw_all_spfl_cups`, `g3_grw_joint_all_spfl_cups`. The W1 UUIDs already have a
v1.2 comparison against `td_lower_joint` (`97c7a3d9`); `write_scores!` adds the
m12 pairing without erasing it. Thirteen W0-only rows retain their recorded `m12_td`
control; `g2_harness_repro` had no recorded v1.1 delta and is self-controlled to preserve
that distinction. Its new zero delta [0, 0] is **not** a historical comparison; annotate
rather than claiming an old pairing. The file uses the frozen W1 panel **56+57 | 24/25,25/26 | n=710**. Verify UUIDs and experiments read-only in `runs` before executing. `scripts/score_runs.jl` includes the `scottish_lower_momentum_grw`, `scottish_lower_decompression`, and `scottish_lower_decoupled_xg` prototype loaders when those experiments occur; the remaining run types live in `src/`.

```bash
/root/.juliaup/bin/julia --project -t 16 scripts/score_runs.jl \
  docs/architecture/harness_klm_W0_rescore_v12.csv \
  --target-seasons 24/25,25/26 --expected-fixtures 710 \
  > /root/BF_runs/logs/klm/prod_w0_rescore.log 2>&1
```

Require **26 distinct scored runs** (14 W0-only plus 12 W1 grid UUIDs), no failures,
v1.2 rows for all 14 W0-only UUIDs and 12 W1 deltas against `132df5c2`. The procedure writes only `harness_scores` v1.2; existing v1.1 rows stay untouched.

## 3. W1 MAP screen re-run (Amendment 1)

The original 12 screen UUIDs are synthetic and have no fit artifacts. Do **not** invoke `score_runs.jl` on them. Re-fit MAP, in memory, with the original W1 12-cell recipe:

```bash
/root/.juliaup/bin/julia --project -t 16 scripts/run_candidates.jl \
  experiments/scotland/03_dynamics_scope_matrix/candidates.jl --stage screen \
  > /root/BF_runs/logs/klm/prod_w1_screen.log 2>&1
```

Check **12 distinct** `stage='screen', scorecard_version='v1.2'` UUIDs and the panel `56+57|24/25,25/26|n=710`, plus 12 `screen_validity` rows in `harness_checks`. Before and after, read `harness_experiments` row `scottish_dynamics_scope_matrix_w1` and compare date, TODO, question, dimension, status (`completed`), decision, readme, and original run list. The `preserve_completed=true` re-screen upsert must leave all metadata intact, and `run_ids` must be **byte-identical** (the screen passes no run IDs, so the seeded semicolon list is not re-normalised); only `run_commits` may grow. Stop if it changes any decision field. Screen can take substantial time; this is **not** NUTS sampling.

## 4. Generate and compare the board

```bash
/root/.juliaup/bin/julia --project -t 16 scripts/leaderboard.jl \
  > /root/BF_runs/logs/klm/prod_board.log 2>&1
git show 090db11e:experiments/LEADERBOARD.md > /root/BF_runs/logs/klm/board_v11.md
```

Compare every grid **(run UUID, model label)** in the first table of `board_v11.md` with
the regenerated Markdown: **no v1.1 UUID or label may disappear**. In particular, require
all six W1 aliases named in §2, and verify that **each of the 12 W1 grid UUIDs** listed in
the CSV has a v1.2 `stage='grid', subset='target', market='all',
metric='delta_logloss_vs_control'` row with `control_run_id = 132df5c2-c742-4e95-8693-3aeb2b2cbaef`.
Verify that their existing `97c7a3d9-a05a-4029-90cb-e34279b8c791` deltas were not
overwritten. The six alias labels must be visible in the board, not merely present in SQL.
Confirm the v1.2 MAP section is present, warning-labelled and separate, with 12 W1 cells. For W2 read the `grw_*` / `td_*` deltas against each row's `control_name` and UUID; compare numerically with `experiments/scotland/04_pyramid_tier_transition_arms/README.md`. Panels must remain separate; no W0-vs-W2 delta. Compare the W1 register row to its pre-screen state, and confirm no v1.1 scores were deleted. Require zero missing v1.1 UUIDs, labels or W1-vs-m12 pairings. The only intentionally
new self-delta is the annotated `g2_harness_repro` comparison; do not call it a v1.1 control.

The W1-vs-m12 part of the gate, read-only (run it before §2 and after §4; the list is the
12 W1 grid UUIDs, CSV lines 15–26):

```sql
SELECT control_run_id, count(DISTINCT run_id) AS n
FROM harness_scores
WHERE scorecard_version = 'v1.2' AND stage = 'grid' AND subset = 'target'
  AND market = 'all' AND metric = 'delta_logloss_vs_control'
  AND run_id IN (
    '4346604b-85ca-4d5b-9f92-b4732512a0c4',
    'decb01be-f214-47bb-bc7b-bcc309026bda',
    '97c7a3d9-a05a-4029-90cb-e34279b8c791',
    '9babf9e9-0a04-43af-855c-619a4b7dac8b',
    '7d8d69e7-25c7-4279-841e-68c664a78a3d',
    '82e01a3b-d686-4e29-a269-3702b96fc5ba',
    'f64a00a2-34a0-4f31-8c58-c093c92d54b7',
    'a6f62436-ec8a-461d-8bd5-dc1861a2daaa',
    'f00ec78a-28ca-464e-91d4-dd1af384415c',
    'de7fa956-87e8-418f-afb4-61ce01cb9f7d',
    '4712735f-afd0-4b73-8d52-c54b4d096daa',
    '1440ab9c-72a0-45dd-8c8e-a3b69e7509be')
GROUP BY control_run_id;
```

After: `132df5c2-…` must show **12**, and the `97c7a3d9-…` count must equal its value
before §2.

Copy `experiments/LEADERBOARD.{csv,md}` and `experiments/REGISTER.md` to the worktree named by the manager, review `git diff`, then commit/push **only** `feat/w2-tier-components` as instructed. Append production evidence (merged SHA, counts, queries, wall times, diff) to `docs/architecture/harness_klm_report.md`. Print `KLM_PROD_DONE` and stop.

## Build rehearsal only (test database)

On a dev checkout with `BF_EXPERIMENTS_TEST_DB_URL` pointing **exactly** to database `mcmc_experiments_test` (other names are rejected), run `scripts/run_candidates.jl ... --stage screen --test-db`. The flag is **screen-only** and refuses smoke, grid and portfolio. The command queries `current_database()` before `ensure_schema!` or any other write, so a misleading URL/keyword override cannot redirect it to production. The screen script and completed-register preservation are tested separately as well. Neither `scripts/score_runs.jl` nor `scripts/leaderboard.jl` targets the test DB via this flag; render a test-board preview with `Harness.read_scores(test_db)` and `scripts/leaderboard.jl:leaderboard_markdown` without invoking its production `main()`.
