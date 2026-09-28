# Hand-off brief — TODO 036 fix round 1 (Claude CLI builder, taking over from pi)

You are the **builder** for fix round 1 on branch `fix/harness-klm`. Claude (the manager) reviews
your result, and a separate Claude CLI session will re-review it.
- **Do not spawn subagents.**
- Do not merge. Push only `fix/harness-klm`.

## Situation

- pi (Sol 6.0) built this branch (`docs/architecture/harness_klm_report.md`).
- An independent review (`docs/architecture/harness_klm_review.md`, verdict CHANGES_REQUIRED)
  raised findings **F1–F11**. **Do not edit the review file.**
- pi started fix round 1 and hit its model quota partway through. Its work is **uncommitted in this
  worktree**: `git status` shows 15 modified files and a new `scripts/klm_test_db_guard.jl`. Treat it
  as a draft. Read the diff (`git diff`) finding by finding; keep what is right and finish what is
  missing or wrong.
- The manager's instruction to pi for this round:
  - **F1:** reject `--test-db` unless `stage === :screen`, with a clear error.
  - **F3:** after connecting, assert `SELECT current_database() = 'mcmc_experiments_test'`.
  - **F2:** add the 12 W1 UUIDs, under their v1.1 labels including the six aliases, to the
    `m12_td` (`132df5c2`) group of `docs/architecture/harness_klm_W0_rescore_v12.csv`. Make the
    runbook's no-loss gate check them.
  - **F4:** split `run_ids` on `[,;]` in both the SQL merge and `_merge_experiment_run_ids`.
  - **F5:** in the runbook, the schema step runs immediately after merge. Also add the
    `_db_has_column` fallback to `extend_fit`/`save_fit` if that is small.
  - **F6:** strict chain comparison (`parent(chain.value)`, names, `name_map`, `logevidence`) in the
    round-trip script and in `_fit_parity`.
  - **F7:** close the three test gaps.
  - **F8:** judge only the latest smoke invocation per recipe; if that is not small, document the
    manual cleanup instead.
  - **F9–F11:** the nits.
- **pi's last dev test run failed.** `/root/BF_runs/logs/klm/fix1_dev.log` (status 1) has an
  `Error During Test` in the "Screen CLI test-database opt-in" testset
  (`test/harness_runner_tests.jl`), from the rsynced dev tree `/root/BF_runs/klm_dev/`. Start by
  reading that log.

## Rules (the original brief `experiments/pi_harness_klm_prompt.md` §0 still applies)

- **The laptop runs no Julia.** All Julia runs on mcmc-beast (`ssh root@mcmc-beast`) in your own
  tmux session there, named `claude_klm_fix1`.
  - **Iterating:** rsync to `/root/BF_runs/klm_dev/` with
    `--exclude '/.cache/' --exclude '/data/' --exclude '/.git/'`.
  - **Final evidence:** from a clean `bash /root/BF_runs/8effca72/scripts/beast_checkout.sh <sha>`
    checkout of a pushed SHA, after copying `/root/BF_runs/a76a65df/.cache/datastore_Scottish*.jls`
    into its `.cache/`.
  - Julia is `/root/.juliaup/bin/julia --project -t 16`, after `set -a; source .env; set +a`.
  - Logs go in `/root/BF_runs/logs/klm/fix1_*`.
  - Check `pgrep -af julia` first. The Kaimon process in tmux `julia` is expected. One heavy job at
    a time.
- **Test DB only:** `BF_EXPERIMENTS_TEST_DB_URL` must name `mcmc_experiments_test`.
  `/root/BF_runs/logs/klm/klm_env.sh` is pi's runtime-credential wrapper. Use it, and never print
  credentials. **Production `mcmc_experiments` is read-only** (SELECT and `load_fit`).
- **No sampling** (no smoke or grid). You may re-run the W1 screen rehearsal on the test DB
  (Amendment 1) if your changes touch that path.
- Code style: `docs/guides/julia_coding_context_for_agents.md`. Keep edits surgical and match the
  surrounding idiom.

## Deliverables

1. For every finding F1–F11, either fixed with a test, or explicitly documented as not changed,
   with the reason.
2. From a clean pushed-SHA checkout, green on the beast:
   - `test/test_db_storage.jl` and `test/test_extension.jl` (test DB);
   - `test/harness_scoring_tests.jl`;
   - `test/harness_runner_tests.jl`;
   - `scripts/validate_klm_large_roundtrip.jl` (the strict version);
   - `scripts/validate_klm_legacy_loads.jl` (read-only).

   Report the counts and wall times.
3. `./scripts/todo.sh check` passes.
4. Commit in logical steps, including this brief file. Append a **"Fix round 1"** section to
   `docs/architecture/harness_klm_report.md`: a table of finding → disposition → change → test and
   evidence. Add a dated `@claude` line to the TODO 036 Work Log. Push.
5. Print exactly `KLM_FIX1_DONE` on its own line and stop. If blocked, print `KLM_BLOCKED` with a
   one-line reason and stop.
