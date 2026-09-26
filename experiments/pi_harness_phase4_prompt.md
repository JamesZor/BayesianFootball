# Work package — Experiment harness, Phase 4: finalist portfolio, cleanup, docs (TODO 030)

You are the implementer; Claude is the manager and reviews against §3. **Do Phase 4 only,
then stop.** Same worktree, branch, git, DB, beast and credential rules as Phase 1
(`experiments/pi_harness_phase1_prompt.md` §1) and the compute rules of Phase 3
(`experiments/pi_harness_phase3_prompt.md` §1). Re-read both. Phase 4 needs **no MCMC**.
Use `scripts/beast_checkout.sh <sha>` and write logs to `/root/BF_runs/logs/<sha>/`,
never inside a checkout.

## 0. Read first

1. `docs/architecture/rfc_experiment_harness.md` §5–§6 (Phase 4), §8.
2. `docs/architecture/harness_phase{1,2,3}_report.md`.
3. Code to graduate: `current_development/grw_player_hybrid/l02_evaluation.jl` §5–§6
   (`gph_option_b`, `gph_buildable_panel`, `gph_simulate`, `gph_portfolio_row`,
   `gph_partition_bets`, `gph_edge_summary`), `current_development/grw_player_hybrid/r05_portfolio_attribution.jl`
   (how r05 used them), `src/MatchDay` (`option_b_system`), `src/Portfolio`
   (`build_books_reported`, `simulate_portfolio`), `src/Calibration` (`point_in_time_book`,
   `calibrate_fit`, `GenerativeRateCalibrator`).

## 1. Manager findings to address (from the Phase 3 review)

1. **The screen stage isn't comparable across model classes.** MAP collapses `MultiScaleGRW`'s
   walk scales (W1 screen: GRW LogLoss 0.68–0.79, slope 0.17–0.36 vs ~0.6445 / ~1.19 under
   NUTS). Even TimeDecay's MAP slope (0.97) differs from NUTS (2.78).
   - Record a `screen_validity` diagnostic per candidate: `limited` for any model with a
     hierarchical/random-walk scale (`MultiScaleGRW`, any learned σ); `ranking_only` otherwise.
   - `leaderboard.jl`: screen rows go in a **separate** section of `LEADERBOARD.md` with a
     warning line, never mixed into the grid table or its Δ columns.
   - In the report, propose (don't build) one better cheap screen, e.g. MAP with walk scales
     fixed at a reference posterior median, or a short single-chain NUTS on every 4th fold.
2. **Stale screen rows.** `harness_scores` holds a pre-fix screen row set for `td_lower_poisson`
   with random `run_id = fcab115f-21f6-4506-8b32-fbd0e2440bc5` (654 rows). The correct
   deterministic one is `630d12ec-ef5c-5163-b967-78f0ece987a0`.
   - Add `scripts/purge_harness_scores.jl <run_id> [--stage screen]`, which deletes rows **only
     from `harness_scores`** for an explicit run_id and stage, prints the count before
     deleting, and refuses without an explicit run_id. Run it for `fcab115f…` and record the
     output. This is the one permitted DELETE, on a harness_* table.
3. **Throwaway runs.** List (don't delete) the improvised runs in `mcmc_experiments.runs`:
   `dd8a9005-ad02-42cf-8e7c-5f34bd2e70e6` (`g2_clean_smoke`, git `unknown`) and
   `07d182a1-fbf3-469a-8d7e-9bb730a99c94` (`g2_clean_smoke2`), namespace `harness_clean_smoke`,
   in the report under "Human decisions". The human decides on deleting them.

4. **Precompile warnings.** Loading the package prints `WARNING: Imported binding Portfolio.TurnoverCommission (IndependentKelly, StaticFamilyTrust, ShrinkToMarketTrust, MatchDrawdown, FixedFraction, PerMatchCap, WeeklySlate, MatchSlate) was undeclared at import time`. Check whether this also happens on `main`/`6c1da1cd` (pre-existing) or was introduced by the harness branch; if introduced, fix it; either way report which.

## 2. What to build

| Item | Work |
|---|---|
| `src/harness/portfolio.jl` | port the gph portfolio helpers. `portfolio_runs(refs; ds, container = :close_option_b | :t25_calibrated, panel = :common)` → per-run summary rows (return %, CAGR, growth/slate [CI], ROI, P(ROI>0), Sharpe, Calmar, max DD, win rate, mean edge, exposure, capture ratio, n_bets, n_slates, n_panel). The common buildable panel is shared across all refs (`gph_buildable_panel`). Rows go to `harness_scores` with `stage = "finalist"`, `subset = "portfolio_<container>"`, `market = "book"`, `control_run_id` set. `:t25_calibrated` uses `point_in_time_book(as_of_minutes = -25)` + `GenerativeRateCalibrator(law = InverseGaussianLaw(w_base = 0.25, sigma = 0.35), book_as_of_minutes = -25)` as in AGENTS.md §2 |
| Runner | `scripts/run_candidates.jl ... --stage portfolio [--container close_option_b\|t25_calibrated]` for candidates with a completed grid run (looked up by config hash); also usable directly on a runs CSV via `scripts/score_runs.jl <runs.csv> --portfolio <container>` |
| Tests | `test/harness_portfolio_tests.jl` on synthetic books/latents: common-panel intersection, a skipped-fixture refusal, deterministic summary for a fixed seed, and rows written with `stage = "finalist"`. Add to `run_parallel_tests.jl` |
| Guide | `docs/guides/experiment_harness_guide.md`: the three lanes (EDA / component / experiment); writing `candidates.jl`; stages and what each check means; `beast_checkout.sh` and log location; scorecard versions and `control_run_id`; the screen-validity caveat; reading `LEADERBOARD`/`REGISTER`; how to add a register row. Keep it practical |
| AGENTS.md | link the guide in §1 and mention it in §5 (experiments now = `candidates.jl` + README through the harness). **Correct §6**: replace the old-scorer numbers (0.6597 / 0.6598 / 0.6571 vs 0.6568) with a pointer to `experiments/LEADERBOARD.md` and a one-line note that 01–03 README numbers use an older scorer. `./scripts/todo.sh check` enforces the 22,000-byte budget |
| Other docs | same correction in `docs/guides/model_generations_guide.md`; add a caveat block at the top of `experiments/scottish_lower/UNIFIED_PARADIGM_REPORT.md` naming which of its eight bench rows used a `synthetic-no-mcmc` run (query `runs.git_commit` for each run it lists; Gen 1 `m00_baseline` is `013af743…`); point `experiments/scottish_lower/README.md` to `REVIEW.md`, `LEADERBOARD.md` and `REGISTER.md`; `experiments/TEMPLATE_README.md` for new experiment folders |
| Register | add rows for TODO 030 (harness, validated), `harness_validation` (g2 reproduced: `75190012`), and TODO 031 W1 (`scotland/03_dynamics_scope_matrix`, status: screened, grid pending) |

## 3. Acceptance (Claude will check each)

1. **Portfolio reproduces r05.** `portfolio_runs` with `:close_option_b` on m12_td
   `132df5c2-c742-4e95-8693-3aeb2b2cbaef` and m05_joint_td `ed541a7c-01e2-447e-a771-783517728d47`,
   on r05's common panel, reproduces
   `current_development/grw_player_hybrid/results/portfolio/r05_portfolio_summary.csv` rows
   `m12_hybrid_td_raw` / `m05_joint_td_raw`: `n_panel` (632), `n_slates` (100), `n_bets`
   (1302 / 1280), and `total_return_pct` (606.5369616689156 / 495.50427678971437), ROI,
   Sharpe, Calmar, max DD, to 1e-9. Bootstrap CI columns may differ only if the seed differs.
   If r05 used all seven of its arms to form the common panel, reproduce that panel (read
   r05) rather than a two-run panel. Report exactly what defined it.
2. **T−25 container** runs for the same two runs and writes finalist rows; report the numbers.
   No reproduction target: this is new.
3. **Screen caveat implemented:** `screen_validity` rows exist for the 12 W1 candidates;
   `LEADERBOARD.md` shows screen results only in their own section with the warning.
4. **Stale rows purged:** before/after counts for `fcab115f…` shown; `630d12ec…` untouched.
5. **Docs:** the guide exists; AGENTS.md §1/§5/§6 updated and under budget; UNIFIED caveat
   names the synthetic rows (with UUIDs); README pointers and template added.
6. **Tests:** all four harness suites pass standalone; `run_parallel_tests.jl` no worse than
   the known T007 failure.
7. `./scripts/todo.sh check` passes; TODO 030 Work Log has dated `@pi` lines. **Don't** mark
   TODO 030 COMPLETED; the manager does that after review.

## 4. Reporting and stopping

Write `docs/architecture/harness_phase4_report.md` (evidence per item, the portfolio tables,
the proposed better screen, and "Human decisions": the throwaway runs, and anything else
needing sign-off). Commit, push, print exactly `PHASE4_DONE` and stop. If blocked: write it
under "Blocked", print `PHASE4_BLOCKED`, and stop.
