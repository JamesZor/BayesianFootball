# Amendment 1 to `pi_harness_klm_prompt.md`: MAP-screen re-run (human decision, 2026-09-28)

Your `KLM_BLOCKED` was correct: the manager confirmed that the 12 W1 MAP-screen UUIDs have no
`runs` or `fit_artifacts` rows, and that `screen()` never calls `save_fit`. **The human's
decision: re-run the MAP screen.** This supersedes the "re-score the MAP-screen runs" wording in
§2 Item 5 and §5.

## What changes

1. **Fitting permission, MAP screen only.** You may run the harness `screen` stage for the 12 W1
   cells in `experiments/scotland/03_dynamics_scope_matrix/candidates.jl`
   (`scripts/run_candidates.jl <file> --stage screen`, `-t 16`, on the beast). It produces fresh
   v1.2 screen rows on the W1 panel. No NUTS, smoke or grid runs; sampling stays forbidden
   otherwise.
2. **Build phase: rehearse against the test DB.**
   - Run the screen for the 12 W1 cells with storage pointed at `mcmc_experiments_test`, so no
     production writes happen. If the code has no clean way to target another DB, add one (e.g. an
     env override or flag, documented and tested).
   - Show that it writes v1.2 `stage = "screen"` rows with the W1 panel label
     (`56+57|24/25,25/26|n=710`), plus `screen_validity` checks.
   - Show that the v1.2 leaderboard renders the MAP-screen section from them, still in its own
     warning-labelled section and never mixed into grid Δs.
   - Report the wall time.
3. **Register side effect.** `run_candidates.jl` upserts `harness_experiments` after a stage, and
   would overwrite the W1 row's `date`, `status` and `decision` with whatever `candidates.jl` holds.
   - The W1 row (`scottish_dynamics_scope_matrix_w1`) must stay unchanged, apart from a merged run
     list if new IDs are added.
   - Make the screen stage's register upsert safe for a completed experiment: do not downgrade the
     status or change date/decision on a re-screen. Test it.
4. **Production phase (§5, still only after the manager says "go prod").** Replace "re-score the
   MAP-screen runs" with: run the W1 screen stage against production `mcmc_experiments`, then
   verify the rows and that the W1 register row is intact, before regenerating the leaderboard.
   Put this in the runbook.
5. **Everything else in the brief stands.** Finish the outstanding items:
   - the synthetic > 1.2 GiB round trip;
   - the four legacy loads;
   - the full suite matrix from a clean pushed-SHA checkout;
   - the runbook and CSVs for the 14 W0 re-scores;
   - the full threshold-site audit.

   Remove the "Blocked" section from your report, or mark it resolved, and set TODO 036 back to
   `ACTIVE` (file and README row).

Commit this amendment file with your next commit. Finish with `KLM_BUILD_DONE`, or `KLM_BLOCKED`
under the §6 rules.
