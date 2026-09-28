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

## Re-review 1 (Claude CLI, Opus 5.5, 2026-09-28) — head `b04abd02`

**VERDICT: CHANGES_REQUIRED**

F1 and F3–F11 are resolved, and F8's documentation-only disposition is acceptable. The F2 CSV
fix restores every v1.1 (label, UUID) pairing and its m12_td delta. However, the same fix
introduces a new **major (F12)**: runbook §2 would overwrite the existing v1.2 metric rows of
the 12 W1 runs. As a result, six W1 board rows, including the W1 control `td_lower_joint`,
would lose every headline metric. I found this with a read-only dry run of §2 and §4. The
runbook's own gate would not catch it.

### Disposition check

| ID | Status | Evidence |
|---|---|---|
| F1 | **Resolved** | `parse_args` refuses `--test-db` for smoke, grid and portfolio before any I/O (`scripts/run_candidates.jl:69`). The runner suite tests both `parse_args` and `main()`. Screen was already safe. |
| F2 | **Resolved as specified, but see F12** | CSV: 26 refs, with the 12 W1 UUIDs under their v1.1 labels in the `m12_td` group. Dry run: **26/26 scored, 0 failures, 250 s**. All **32/32** v1.1 first-table (label, UUID) pairs appear on the in-memory v1.2 board. All 12 W1 UUIDs carry both a `132df5c2` and a `97c7a3d9` delta, and the m12 Δs reproduce v1.1 exactly (e.g. `td_spfl_cups_joint` −0.00046, `s12_m02_td_joint` 0.00038, `g1_grw_all_spfl` 0.00307). |
| F3 | **Resolved** | `assert_klm_test_database!` checks `current_database()` after connecting and before `ensure_schema!`. The runner suite's `?dbname=postgres` override test really ran: 182 = 165 + 17 new assertions, which includes that branch. |
| F4 | **Resolved** | The merge splits on `[,;]` and trims. A `preserve_completed` upsert with an empty run list keeps `run_ids` byte-identical (DB test with a seeded `;` list). Production's W1 row is unchanged (`md5(run_ids)` = `4a164c17…`). |
| F5 | **Resolved** | `save_fit` refuses with a clear error when the `layout` column is absent. `extend_fit` falls back to single-blob. Runbook §1 is marked "immediately after merge". The test simulates the old schema by renaming objects and restores it in `finally`. |
| F6 | **Resolved** | `_chain_parity` compares values plus internals, names, `name_map` and `logevidence`. The round trip and `_fit_parity` both use it. Unit tests cover a changed internal and a renamed internal. |
| F7 | **Resolved** | `main()` refusal tests, different per-panel deltas asserted per (panel, control), and exact `run_ids` equality in the rehearsal validator. |
| F8 | **Accepted as documented** | The rationale holds. A failed smoke writes its rows with `run_id` NULL, and `harness_checks` has no invocation key, so "latest invocation" would be a timestamp heuristic. Doing it properly needs a new column. Production has zero failed hard smoke rows, so nothing is blocked. The guide's manual procedure is scoped, authorised and never automatic. Track it as a follow-up if smoke failures recur. |
| F9 | **Resolved** | The harness guide states the R̂ > 1.05 / 0.1% rule and cites the constants. |
| F10 | **Resolved** | The BFMI message says `<=`. `DualStorage` forwards `on_duplicate`, and both `:return` and `:error` are tested. The `g2` self-pair is annotated. |
| F11 | **Resolved** | After each suite the test DB held `runs`/`config_registry`/`harness_experiments` = 0/0/0. |

### New findings

| ID | Severity | Location | Defect | Failure scenario | Suggested fix |
|---|---|---|---|---|---|
| F12 | **major** | `src/harness/store.jl:157` (`write_scores!` replace key); `src/harness/scoring.jl:585` (`leaderboard` groups by `model`); runbook §2 (lines 31, 45) | `write_scores!` replaces all rows sharing `(run_id, scorecard_version, control_run_id, stage, subset)`. The non-delta metrics (LL, ECE, compression, transition cohorts, …) have a **NULL** `control_run_id`. Re-scoring the 12 W1 UUIDs therefore **deletes their existing v1.2 metric rows (7,596 rows) and re-inserts them under the CSV label**. Only the `97c7a3d9` delta rows keep the native W1 label. `leaderboard` builds headline metrics per `(panel, run_id, model)`. | Read-only dry run on `b04abd02`: score the CSV in memory, apply `write_scores!`'s replace-by-key to a copy of production v1.2, then run `leaderboard`. `td_lower_joint`, `td_lower_poisson`, `grw_lower_poisson`, `grw_spfl_poisson`, `grw_spfl_cups_poisson` and `grw_spfl_cups_joint` render with Target LL, 1X2 LL, ECE, Compression and all transition-cohort columns as **"—"** (NaN), next to their Δ vs `td_lower_joint`. So the W1 control's own row loses its numbers. The six same-label W1 rows are fine. Runbook §2 says `write_scores!` "adds the m12 pairing without erasing it" and that only v1.1 rows stay untouched; in fact it rewrites existing production v1.2 rows. The §4 gate (UUID/label presence and delta counts) passes regardless, so the regression would ship to `LEADERBOARD.md`. | Preferred, code only: in `leaderboard`, take the headline and extras from the whole `(panel, run_id)` group, because these metrics do not depend on the label. Then emit one row per (label, control) pairing present, where a label is paired with the controls of its own delta rows. Add a scoring test with a run carrying two labels, one of which owns the NULL-control metrics. Also correct runbook §2: it rewrites the 12 W1 runs' v1.2 non-delta rows, relabelled. Add a §4 gate that every grid board row has a finite Target LL. Alternative: re-score the W1 UUIDs under their native labels. That leaves the metrics intact but drops the six alias labels, which F2 required. |
| F13 | minor | `src/harness/scoring.jl:583` | `control_name` is `first(g.model)` over the control run's `(panel, run_id)` rows. Once a UUID carries two labels (e.g. `97c7a3d9` = `td_lower_joint` / `s12_m02_td_joint` after §2), the name shown depends on row order. My first dry run, where the row order differed, printed `s12_m02_td_joint` as the control of all 12 W1 rows. The type-normalised run printed `td_lower_joint`. | The board can name the W1 control by its W0 alias. The numbers stay correct. | Prefer the label of the control's self-pair row (`run_id == control_run_id`), and fall back to `first(model)`. Or show the control UUID in a column. |

### Commands run (mcmc-beast, clean `beast_checkout.sh b04abd02`, tmux `claude_klm_review`, logs `/root/BF_runs/logs/klm_review/rr1/`)

The test DB was reset to an empty schema before the runs. Integration suites went through the
existing runtime-credential wrapper. Production access was SELECT and `load_fit` only; the dry
run calls no `write_scores!` and no `ensure_harness_schema!`.

| Check | Result | Builder's claim |
|---|---|---|
| `test/test_db_storage.jl` | **154/154** | 154/154 |
| `test/test_extension.jl` | **51/51** | 51/51 |
| `test/harness_scoring_tests.jl` | **46/46** | 46/46 |
| `test/harness_runner_tests.jl` | **182/182** | 182/182 |
| Test-DB debris after each suite | 0/0/0 | 0/0/0 |
| `scripts/validate_klm_large_roundtrip.jl` (strict `_chain_parity`) | **PASS**: 1,993,184,800 B, 120 folds, largest blob 28,621,776 B, save 11.31 s, load 5.55 s, peak RSS 6.42 GB | same |
| F2/§2+§4 dry run (`f2_dryrun.jl`, then `f2_dryrun2.jl` with run-ID types normalised) | 26/26 scored in 250 s; 32/32 v1.1 pairs present; W1 deltas 12 × `132df5c2` + 12 × `97c7a3d9`; **6 W1 rows with NaN headline metrics (F12)** | not tested |
| Production recheck (read-only) | 324 `fit_artifacts`, no `layout` column, no `fit_fold_artifacts` table, 20,262 v1.2 score rows, W1 register `completed` and unchanged | not executed by the builder |

Not re-run in this round: the 12-cell screen rehearsal (563 s of MAP fitting; this round's screen
path changes are covered by the runner/DB tests and the builder's log), the legacy loads (no
change to `load_fit`) and the parallel suite. The test DB was reset to an empty schema afterwards.

## Re-review 2 (Claude CLI, Opus 5.5, 2026-09-28) — head `f44cda6b`

**VERDICT: ACCEPT**

F12 and F13 are fixed. The `score_runs.jl` split does not change behaviour, and F1–F11 have not
regressed. The runbook's §2 + §4 rehearsal passes read-only against current production. No
blocker or major findings remain open.

### F12 / F13

- **Code** (`src/harness/scoring.jl:590–642`). `leaderboard` now groups by `(panel, run_id)`
  and takes the label-independent headline and cohort metrics from the whole group. It emits one
  row per `(label, control)` pair found in that run's target/all delta rows. A run with no deltas
  keeps one row per label. The sort now includes `:model` and `:control_name` as tie-breaks, so
  the order is deterministic. `control_name` comes from the control's self-pair row
  (`run_id == control_run_id`), falling back to `first(model)`. This fixes F13 independently of
  row order.
- **Test.** The new scoring testset reproduces the §2 shape: the alias owns the NULL-control
  metrics and the m12 delta, while the native label owns only its delta. It runs both row orders
  and asserts finite headline and cohort values plus exact `(label, control) → (LL, Δ)` pairs.
  Before this fix, the old `(panel, run_id, model)` grouping would fail its NaN assertions.
- **Builder's `scripts/validate_klm_board_dryrun.jl`** (clean `f44cda6b`, production read-only)
  prints `KLM_BOARD_DRYRUN_PASS`, with scored = 26 in 248 s, `replaced_rows` = 7,596 and
  `board_rows` = 68. There are 0 NaN headline rows and 0 "—" Target LL cells. All 32/32 v1.1
  pairs are present with Target LL matching v1.1 within 5e-6. W1 has 12 m12 deltas and 12
  td_lower_joint deltas, and every W1 row names its control `td_lower_joint`. I read the script
  before running it. It calls only `read_scores`, `read_experiments` and `load_fit` (through
  `score_runs`); there is no `write_scores!` and no schema call.
- **My own dry run** (`f2_dryrun3.jl`, run-ID types normalised; this is my code, not the
  builder's). It applies `write_scores!`'s replace-by-key to a copy of production v1.2 and
  checks invariants across **all panels**:
  - the set of board `(panel, run, label, control)` rows equals the set of grid delta pairings
    plus the label rows of runs without deltas (68 = 68; missing 0, extra 0, no duplicates);
  - 0 of the 47 grid `(panel, run, label)` triples are dropped from the board, and 0 rows have a
    NaN Target LL;
  - all six W1 aliases and their native labels show identical finite metrics, for example
    `td_lower_joint` and `s12_m02_td_joint` both at 0.64375 / slope 1.928;
  - the W2 panel has all 30 production pairings. Spot checks against the W2 README match:
    `td_lower_a2full_carry_jump` vs `td_lower_base` −0.0014, `grw_step_ldelta` vs `grw_base`
    −0.0025, `grw_step_a2_carry_jump_ldelta` vs `grw_base` −0.0029.
- **Runbook.** §2 now states correctly that the W1 v1.2 non-delta rows are rewritten and
  relabelled. §4 adds a finite-headline gate and requires `KLM_BOARD_DRYRUN_PASS` before §2.

### `score_runs.jl` split

`main` still loads the groups, `unique!`s the refs, includes the loaders, loads the datastore,
opens the harness DB, calls `ensure_harness_schema!` and handles the portfolio branch, in the
same order. It then calls `score_csv_groups`, which contains the loop moved verbatim (same
`invokelatest`, kwargs and failures vector), followed by `write_scores!` and the same printout
and exit code. The only differences are that `tiers` is computed inside the helper, and that
`include_run_loaders` `include`s into the same (script) module as before. The dry run over this
path produced **17,684** score rows for 26/26 runs, identical to my re-review 1 run on
`b04abd02`, which used the pre-split code path. The runner suite §8 still covers CSV parsing and
grouping.

### F1–F11 regression check

The four suites were re-run on clean `f44cda6b`. Counts match or exceed round 1, and the only
additions are the 14 F12/F13 scoring assertions. The test DB held 0/0/0
`runs`/`config_registry`/`harness_experiments` rows after each suite (F11). None of the F1–F11
code sites changed in this round; the only source diff is `scoring.jl:leaderboard`.

### Remaining non-blocking notes (no action required for merge)

- **Runbook §2 rewrites 7,596 existing production v1.2 rows.** They are the W1 non-delta rows,
  relabelled to the CSV alias. The step is therefore not strictly additive, but the runbook now
  says so, the values are recomputed from the same fits and the same panel, and the new
  `leaderboard` makes the relabelling invisible on the board. Optional hardening: have §4 also
  confirm that the rewritten Target LL of each W1 UUID equals its pre-§2 value.
- **F8 is still documented rather than changed.** This remains acceptable, as judged in
  re-review 1.
- **Nit** (`scoring.jl:603–606`). Consider a run that has deltas, plus a label that owns only
  NULL-control metric rows and no delta. That label gets no board row. The rows' numbers stay
  visible under the run's other labels, so no metric is lost. The invariant check found no such
  label in the current production data (0 of 47 dropped), so this is a note, not a defect.

### Commands run (mcmc-beast, clean `beast_checkout.sh f44cda6b`, tmux `claude_klm_review`, logs `/root/BF_runs/logs/klm_review/rr2/`)

| Check | Result | Builder's claim |
|---|---|---|
| `test/test_db_storage.jl` | **154/154** | — (unchanged since round 1: 154) |
| `test/test_extension.jl` | **51/51** | — (51) |
| `test/harness_scoring_tests.jl` | **60/60** | 60/60 |
| `test/harness_runner_tests.jl` | **182/182** | 182/182 |
| Test-DB debris after each suite | 0/0/0 | — |
| `scripts/validate_klm_board_dryrun.jl` (production read-only) | **`KLM_BOARD_DRYRUN_PASS`** (figures above) | PASS, same figures |
| Reviewer invariant dry run `f2_dryrun3.jl` | 26/26, 17,684 rows, 0 failures; 68/68 pairings; 0 dropped labels; 0 NaN | — |

Before launching, I found another session's Julia REPL (`claude_clv_napkin`, `-t 8`). It was
idle (2.7% CPU, load 0.29), so I did not wait for it. Final read-only production recheck after all runs: 324
`fit_artifacts`, no `fit_fold_artifacts` table, no `layout` column, 20,262 v1.2 score rows (as
in re-review 1), and the W1 register `completed` with `md5(run_ids)` `4a164c17…` unchanged.
Production is untouched. The test DB was then reset to an empty schema and the tmux session
closed.
