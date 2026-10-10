# Phase 0 static read-path review (manager ruling 1)

The actual runner/loader pass `check_read_paths.py` before any DB connection. The
checker rejects write/schema SQL, persistence/sampling calls, unknown qualified
Training calls, harness execution, and datastore SQL/cached-loader fallback paths.
All direct LibPQ executions go through `read_query`, which accepts only one
SELECT/SHOW statement. This is a static review of these entry points, not a proof
that every function in the repository is read-only.

## Reachable database call tree

| Called path | Reviewed source | Operations |
|---|---|---|
| Training.PostgresStorage(namespace) | `src/training/inference/io.jl:68–120` | String/endpoint construction only; no migration or SQL |
| Inference._db_connect | `db_storage.jl:11` | Opens existing libpq connection; no schema/write helper |
| Training.load_fit(db, UUID) | `db_storage.jl:947,864–907` | UUID dispatch; SELECT artefact layout/blob and optional per-fold blobs |
| _db_has_column / _db_rows | `db_storage.jl:27–46` | SELECT information_schema; reads result and closes it |
| _db_query_blob | `db_storage.jl:109–144` | PQexecParams with caller's SELECT; fetches binary result, clears result |
| _db_artifact_value | `db_storage.jl:270–271` | Decompresses/deserialises memory value; no SQL |
| _db_load_count_latents | `db_storage.jl:836–861` | SELECT match_latents JOIN fold_results; reconstructs CountLatents |
| Experiment read_query / audit_snapshot | `l00_phase0_inventory.jl` | SHOW default once; SELECT table metadata, counts, timestamp maxima |

`ensure_schema!`, registry persistence, save_fit, harness schema/score recording,
portfolio persistence and calibration persistence are **not called**. Portfolio
construction/simulation returns in-memory objects and writes experiment CSVs only.
The datastore uses `Serialization.deserialize` on an existing cache: failure stops
rather than falling back to betdb SQL. No betdb table is touched by this runner.

## Runtime audit contract

Before the fit load and again in `finally` after every run/failure, save table-wide
counts and maximum timestamp values for runs, configs, config_registry, fold_results,
match_latents, fit_artifacts, fit_fold_artifacts and all installed public harness_*
tables. Explicit rows mark absent tables and absent created_at/updated_at columns;
existing other timestamp columns (e.g. finished_at, harness_checks.at) are included.
Compare both snapshots exactly, log per-table `unchanged`, and stop on any change.
No namespace filter or exemption for unrelated writers is applied.

Limits: counts/maxima do not detect every same-count in-place edit on untimestamped
tables. The static call-path review and before/after evidence are complementary,
as the manager requested. SHOW's actual value is a record, not a passing condition.
