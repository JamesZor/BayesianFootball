# Review brief — TODO 038 market evaluation layer, branch `feat/market-eval`

You are the **independent reviewer** (a Claude CLI session separate from the builder).
- Verify; don't fix. No subagents.
- Treat the builder's report as claims.

## Inputs and rules

- **Worktree:** `/home/james/bet_project/.worktrees/BayesianFootball-market-eval`. Pull first.
  Review `git diff <base>...feat/market-eval`, where `<base>` is the TODO 036 merge commit named in
  TODO 038.
- **Contract:** `experiments/claude_market_eval_prompt.md` (the agreed design; don't re-litigate it)
  and the TODO 038 acceptance criteria.
- **Builder report:** `docs/architecture/market_eval_report.md`.
- **Environment:**
  - Julia only on mcmc-beast, in your own tmux there named `claude_market_review`, from a clean
    `beast_checkout.sh <head>` checkout with the caches copied from `/root/BF_runs/a76a65df/.cache/`.
  - Logs go in `/root/BF_runs/logs/market_review/`.
  - Production and betdb are **read-only**; use the test DB via the KLM guard. No sampling.

## Check

1. **Dispatch design.**
   - Are axes and evaluators selected purely by multiple dispatch, with no Symbol or `if`
     switches on axis values?
   - Are the types concrete and parametric, and the hot loops type-stable (spot-check with
     `@code_warntype` on `ledger` and `evaluate(::CLVEvaluator, …)`)?
   - Do the domain types live in `src/Portfolio/evaluation/`, with the harness only orchestrating?
   - Does a new axis type plug in with one method (tested)?
2. **CLV correctness.**
   - Recompute `clv_pp`, `clv_pct` and `beat_close` by hand for about 10 real ledger rows,
     covering a back, a lay and a filled order where they exist, against the de-vigged
     TWA(−20, 0] close.
   - Is the slate-clustered CI clustered by slate?
   - Are the legacy wrappers (`Calibration.bet_clv`/`clv_summary`, `MatchDay.clv_for_order`)
     bit-identical to the pre-branch outputs? Re-run their parity tests, and compare against the
     base commit's functions on the same inputs.
3. **Migration parity.** Are the old `close_option_b` / `t25_calibrated` finalist metrics
   reproduced **bit-identically** on the 632/611 panels? Compare against a read-only production
   SELECT of the `stage = 'finalist'` rows. Do the `portfolio_runs` and `--portfolio` wrappers
   still work?
4. **No leakage in the ledger.**
   - Entry prices come only from the book at the recipe's instant.
   - Close prices are used only for CLV, never for selection or staking.
   - Calibration uses only the entry book.
   - Panels are not mixed.
5. **Storage.**
   - The schema changes are additive and idempotent (replay the production schema into the test
     DB and apply twice).
   - `market_recipe_hash` is stable and complete, and `harness_market_recipes.spec` reproduces the
     recipe.
   - The `--test-db` guard holds.
   - The production and betdb backfill steps are insert-only and idempotent. Read the code; don't
     run them.
6. **Reporting honesty.**
   - Does the report show every slice with a multiplicity caveat?
   - Is `Close()` CLV reported as a zero sanity check?
   - Are significance statements supported by the CIs?
7. **Tests.**
   - Re-run `test/market_eval_tests.jl`, the portfolio, calibration and MatchDay ledger suites,
     and the harness portfolio and scoring suites. Compare the counts with the report.
   - Look for vacuous tests.

## Output

Write `docs/architecture/market_eval_review.md`:
- `VERDICT: ACCEPT` or `VERDICT: CHANGES_REQUIRED`;
- a findings table: ID, severity, `file:line`, the defect, a failure scenario, and the fix;
- the commands you ran with their counts.

**Do not commit; the manager commits it.** Print exactly `MKT_REVIEW_DONE` on its own line and
stop. For a re-review, append "Re-review N" and print `MKT_REREVIEW<N>_DONE`.
