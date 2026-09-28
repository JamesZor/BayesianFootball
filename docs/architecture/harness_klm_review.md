# Harness KLM — independent review (Claude CLI, Opus 5.5, 2026-09-28)

Branch `fix/harness-klm` at `7ea6cf9f`, diff `feat/w2-tier-components...fix/harness-klm`.
Reviewed against `experiments/pi_harness_klm_prompt.md`, Amendment 1, TODO 036 and the
builder's report. No code was changed by the reviewer.

**VERDICT: CHANGES_REQUIRED**

Two majors: F1 and F2. The core engineering is sound and verified independently. That covers
per-fold storage, legacy loads, the > 1 GiB round trip, (k), the gate, the (j) nonce and the
leaderboard code. The two majors are about safety at the production boundary and the
completeness of the runbook. Both fixes are small.

## Findings

| ID | Severity | Location | Defect | Failure scenario | Suggested fix |
|---|---|---|---|---|---|
| F1 | **major** | `scripts/run_candidates.jl:110`; `src/harness/stages.jl:189`; `src/harness/scoring.jl:516,539`; `src/harness/portfolio.jl:236` | `--test-db` is accepted for every stage, but only the storage passed as `db` is redirected. The smoke stage builds its own `PostgresStorage("<exp>_smoke")`. `score_runs` and the portfolio stage build `PostgresStorage(ref.experiment)`. All of these resolve to production `mcmc_experiments`. | `run_candidates.jl <file> --stage smoke --test-db` runs `ensure_schema!` on production, which applies the new DDL before the go-prod step. It then **writes** `runs`/`configs`/`fold_results`/`fit_artifacts` rows for `<exp>_smoke` to production, while its checks go to the test DB. `--stage grid --test-db` saves to the test DB and then scores by `load_fit` from production. The run is not there, so a fit that took hours to sample fails at scoring. `--stage portfolio --test-db` reads production grid runs. The brief asks whether the flag can hit production by accident: it can. (`screen` is safe: fits stay in memory and W1's `CONTROL` UUID matches no screen fit, so no production load happens. I verified this in code and in the builder's `af_screen.log`.) | Reject `--test-db` unless `stage === :screen`, with a clear error. Alternatively, thread the storage URL through smoke, `score_runs` and portfolio, but that is a larger change. Add a test that `parse_args` + `main` refuse `--stage smoke --test-db`. |
| F2 | **major** | `docs/architecture/harness_klm_W0_rescore_v12.csv`; `docs/architecture/harness_klm_prod_runbook.md:50` | The v1.1 board's Δ column was against `m12_td` (`132df5c2`) for **all** rows, including the 12 W1 cells. Production v1.1 `harness_scores` holds a `132df5c2` delta for all 12 W1 UUIDs. Six of them sit only under W0 alias labels (`s12_m01_td_poisson`, `s12_m02_td_joint`, `m00_baseline_grw`, `g1_grw_all_spfl`, `g2_grw_all_spfl_cups`, `g3_grw_joint_all_spfl_cups`). The v1.2 rows for those UUIDs carry only the `td_lower_joint` (`97c7a3d9`) control, and the CSV re-scores only the 14 W0-only UUIDs. | After §5 every v1.1 UUID survives, but those 12 rows change their reference from m12_td to td_lower_joint, so the numbers differ for the same UUID. The six alias rows vanish from the board. The runbook's own step-4 gate ("Require zero unexplained omissions or changed controls") would then fail, stopping the production phase midway, or be waved through. The W1 cells' comparison against the live champion is also lost. This is a gap in the contract wording ("the 14 W0 runs"), not a code defect, but it breaks the stated no-loss intent of (m). | Add the 12 W1 UUIDs to the `m12_td` group of the CSV, under their v1.1 labels including the aliases. It is scoring only. `write_scores!` keys on `control_run_id`, so this adds rows without erasing the existing `97c7a3d9` pairing, and `leaderboard` then shows both. Alternatively, have the manager record explicitly that the W1-vs-m12_td pairings are dropped, and relax the step-4 gate text. |
| F3 | minor | `scripts/run_candidates.jl:113-115`, `scripts/validate_klm_*.jl` | The test-DB guard trusts a parsed `dbname`: the URL path, or the *first* `dbname=` keyword. libpq honours a `?dbname=` URI query parameter and the *last* keyword. Probe output: `…/mcmc_experiments_test?dbname=mcmc_experiments` and `dbname=mcmc_experiments_test dbname=mcmc_experiments` both parse as `mcmc_experiments_test`. | A contrived or copy-pasted URL passes the guard and connects to production. Unlikely, but this is the only guard. | After connecting, assert `SELECT current_database() = 'mcmc_experiments_test'`. Fold this into the F1 fix. |
| F4 | minor | `src/harness/store.jl:327` | The register merge splits `run_ids` on commas only. 10 of 27 production register rows (W1 included) are semicolon-separated (seeded). | The **W1 re-screen is safe**: the screen stage passes an empty `run_ids`. I evaluated the exact SQL expression read-only against the production W1 value, and it returns the value unchanged. `run_commits` merges `{}`. A smoke, grid or portfolio run on any of the 10 semicolon rows would instead produce `"<new>,a;b;…;l"`: mixed separators, and duplicate IDs if an old ID recurs. This bug predates the branch. | Use `regexp_split_to_array(…, '[,;]')` in the SQL, and the same in `_merge_experiment_run_ids`. Optional for this TODO. |
| F5 | minor | `src/training/inference/db_storage.jl:803`; `src/training/inference/extension.jl:411` | After merge, new `save_fit` (INSERT … `layout`) and `extend_fit` (`SELECT layout`) require the migrated schema. Only `load_fit` has a pre-migration fallback. | Suppose a merge lands and then a non-harness script calls `save_fit`, or MatchDay/l17-style live `extend_fit` runs against production before runbook step 1. The transaction rolls back cleanly, but a long sampling run's fit is lost. Harness `run_candidates.jl` is unaffected because it calls `ensure_schema!` first. That also means the first harness run after merge migrates production implicitly, which is additive, so acceptable. | Add to the runbook that step 1 must run immediately after merge, before any other fit or extension. Or give `extend_fit` the same `_db_has_column` fallback that `load_fit` has. |
| F6 | minor | `scripts/validate_klm_large_roundtrip.jl:73`; `src/harness/checks.jl:417` (`_fit_parity`) | The chain comparison is `Array(chain)`, which covers parameters only. It excludes sampler internals, names and `name_map`, so "bit-exact chains" is not literally what the script asserts. | A regression that dropped or reordered internals, such as `numerical_error`, would pass both the round-trip script and `fit_parity`. **I re-ran the round trip** with `isequal(parent(chain.value))` plus names, `name_map` and `logevidence`. It **passes**, so the implementation is fine and only the check is weak. | Use the stricter comparison in the script, and ideally in `_fit_parity`. |
| F7 | minor | `test/harness_runner_tests.jl:663`; `test/harness_scoring_tests.jl:228`; `scripts/validate_klm_screen_rehearsal.jl:41` | There are test gaps and one vacuous check. (a) The `--test-db` test covers only `parse_args`; the missing-URL and non-test-DB refusals are untested, though Amendment 1 asked for "documented and tested". (b) The pairing test gives identical deltas in both panels, so a cross-panel delta mix-up could not be detected. (c) The rehearsal validator's "original run IDs retained" check splits the semicolon production value on `,`, so it is one token and trivially present. | (a) A regression in the refusal code passes CI. (b) A panel-mixing bug in the `labels`/`delta_rows` logic passes. (c) A corrupted W1 run list would still print `register=preserved`. | (a) Test `main` refusal with the env var unset and with a production-named URL. (b) Use different deltas per panel and assert each panel's values. (c) Compare `run_ids` for exact equality, which is what the screen path should preserve. |
| F8 | minor | `src/harness/store.jl:470` (`has_passing_smoke`) | Any historical hard failure for a `recipe_hash` blocks the grid forever. A fresh passing re-smoke (now possible via the nonce) does not clear it. TODO 036 (j) names "the failed record blocks the grid" as a symptom. | Production currently has **zero** failed hard smoke rows, so nothing is blocked today. The next genuine smoke failure followed by a fixed re-smoke would still block the grid. The contract does not require a change here. | Optional: judge only the latest smoke invocation per recipe, e.g. by `run_id`/`at`. Otherwise note the manual cleanup procedure. |
| F9 | nit | `docs/guides/experiment_harness_guide.md:102` | The guide says convergence is a review blocker but does not state the new rule: fail only on R̂ > 1.05 or divergence > 0.1% of post-warm-up draws, with ESS, BFMI and tree depth as notes. | Readers infer the old ESS gate. | Add one sentence that cites the constants in `checks.jl`. |
| F10 | nit | `src/harness/checks.jl:443`; `src/training/inference/db_storage.jl:827`; runbook §2 | Three small items. The BFMI note says `<` while the test is `<=`. `DualStorage.save_fit` does not forward `on_duplicate`, and passing it raises a MethodError from the file backend. `g2_harness_repro` is self-controlled, so it shows Δ 0 [0, 0] where v1.1 showed "—" (the runbook says to annotate this). | Cosmetic, or unused today. | Align the message. Document or forward the kwarg. Consider scoring `g2_harness_repro` with no control row rather than a self-pair. |
| F11 | nit | test suites | `test_extension`/`test_db_storage` left 5 `runs` rows in `mcmc_experiments_test` after my run. | The test DB accumulates debris between runs (harmless, and the builder cleaned it by hand). | Clean up in a `finally`, or accept this. |

## Verified (no finding)

- **(l) `save_fit`:** a single `BEGIN`…`COMMIT` with `ROLLBACK` on error. It writes the shell plus one
  `_db_exec_binary` insert per fold, and serializes and compresses per iteration, so peak extra
  memory is about one fold. In the round trip the largest fold blob is **28.6 MB** against a 1 GiB
  limit.
- **Layout marker:** an explicit `layout` column (`single` by default, `per_fold` for new runs), with
  unknown values rejected. When the column is absent, `load_fit` treats the row as legacy, and
  production has had no column so far.
- **`ensure_schema!`:** I restored production's `pg_dump --schema-only` into the test DB, seeded a
  legacy `fit_artifacts` row, then applied the new `schema.sql` **twice**. Both runs exited 0, the
  legacy row read `layout = single`, and `fit_fold_artifacts` was created. There is no destructive
  DDL; the only statements added are `ADD COLUMN IF NOT EXISTS` and `CREATE TABLE IF NOT EXISTS`.
  `DELETE FROM runs` cascaded to both artefact tables (0/0 left).
- **`load_fit`:** reads both layouts, rebuilds folds in shell order, and checks each fold's index.
  The latents path is unchanged.
- **`extend_fit`:** for a per-fold run it inserts only the new fold blobs, and existing blob `md5`
  values are unchanged (tested). A legacy run keeps the single-blob rewrite, which is documented in
  the extension header, the docstring and the DB guide §4.8a, and tested.
- **Stride:** the ladder is gone. Grids save at stride 1 with `on_duplicate = :error`, and
  `persistence_stride` records 1, as the contract asks. `thin_for_persistence` remains as an
  unused utility, which the contract allows. FileStorage is untouched, and DualStorage calls the
  PostgreSQL path with its defaults.
- **(j):** the `smoke_nonce:<uuid4>` tag is kept by `_db_recipe_tags`, so it is inside
  `config_hash`. `recipe_hash` is built only from the model and scope. All **15/15** W2 candidate
  recipe hashes at this SHA equal production `harness_checks.recipe_hash` (SELECT). The default
  stays `:return`, and `:error` names the existing run. The test saves two nonce smokes, and both
  pass parity.
- **(k) and the gate:** `DiagnosticFailure` carries `value`, and a plain exception still records
  `{}` plus the error text (both tested against the DB). The thresholds are constants that cite
  the 2026-09-27 decision. A run fails only on `max_rhat > 1.05` and on `divergence_rate > 0.001`.
  The rate is pooled `n_div / (n_draws × n_chains)` from post-warm-up chains, which matches the W2
  README's "70 / 240,000". Both boundaries are tested (1.05 and 0.001 pass). ESS, BFMI and tree
  depth only produce `notes`. `Training` defaults and `fold_results.converged` are untouched.
- **NaN → null:** `_harness_json_safe` is applied only in `write_checks!` (probe:
  `{"a":null,"b":[1.0,null],…}`). Screen check-write failures now propagate instead of being
  swallowed.
- **(m) code:** one row per (panel, run, model, control), with the control named. The panels
  always come from `groupby(:panel)`. The MAP section is kept, in its own warning-labelled block.
- **Runbook:** the steps run schema, then W0 re-score, then W1 re-screen, then board, and are
  additive (`write_scores!` replaces only same-key v1.2 rows). The 14 UUIDs in the CSV equal
  exactly the production set of v1.1 grid UUIDs minus v1.2 grid UUIDs (SELECT), and all 14 exist
  in `runs`. Their controls match v1.1: 13 against `132df5c2`, and `g2` had none.
- **Production is untouched:** after all runs, `fit_artifacts` = 324 rows, with no `layout` column,
  no `fit_fold_artifacts` table and 0 v1.2 screen rows.

## Commands run (mcmc-beast, clean `beast_checkout.sh 7ea6cf9f`, tmux `claude_klm_review`, logs `/root/BF_runs/logs/klm_review/`)

`BF_EXPERIMENTS_TEST_DB_URL` pointed at `mcmc_experiments_test` through the existing
runtime-credential wrapper, and no credential was printed. All production access was
`SELECT`/`pg_dump --schema-only`/`load_fit`.

| Check | Result | Builder's claim |
|---|---|---|
| `test/test_db_storage.jl` (test DB) | **150/150**, 35.8 s | 150/150 |
| `test/test_extension.jl` (test DB) | **47/47**, 27.4 s | 47/47 |
| `test/harness_scoring_tests.jl` | **46/46** (sum of 9 top-level sets) | 46/46 |
| `test/harness_runner_tests.jl` | **165/165**, 1m27 | 165/165 |
| Synthetic round trip (stricter reviewer variant of `validate_klm_large_roundtrip.jl`) | **PASS**: 1,993,184,800 raw bytes, 120 folds, largest blob 28,621,776 B, save 11.41 s, load 5.58 s, peak RSS 6.45 GB | same figures |
| `scripts/validate_klm_legacy_loads.jl` (production, read-only) | **4/4**: `ce7ea22f` (500,194,4)…(500,1830,4); `c4a0fa94` (250,238,4)…; `132df5c2` (800,69,4)…; `aad544b3` (800,65,4)… | same |
| Schema replay (production schema → new `schema.sql` ×2, cascade) | pass | not tested by the builder against the production schema |
| Probe (`probe.jl`): `--test-db` parser, recipe hashes, JSON mapping | see F3; 15/15 hashes equal | — |

**Not verified:**
- `test/run_parallel_tests.jl` and the other matrix suites (portfolio, pyramid, splitter). The
  brief did not ask for them; the builder reports 24/25, with only the known T007 failing.
- The actual 12-cell screen rehearsal. I did not re-run it (568 s of MAP fitting); I reviewed the
  builder's log instead.
- `./scripts/todo.sh check`.

Housekeeping: afterwards the test DB was reset to an empty `public` schema, and the tmux
session was closed.
