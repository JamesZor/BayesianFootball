# Work package — TODO 031 Wave 2, Phase B: scoring for W2 (harness follow-ups g, i, h + panel)

You are the implementer; Claude is the manager. **Do Phase B only, then stop.** The W2 grids are
running on mcmc-beast from a pinned checkout while you work. **Never start sampling, never kill or
touch a running `run_candidates.jl`, and never run anything heavy on the beast while
`pgrep -af run_candidates.jl` shows a job.** Light scoring runs (below) are allowed. Same git, DB,
credential and AD rules as `experiments/pi_w2_phaseA_prompt.md` §0. Branch
`feat/w2-tier-components`: pull first, commit in small steps, push at the end.

## 0. Read first

1. `experiments/scotland/04_pyramid_tier_transition_arms/README.md` §4 (decision rule) and §5 (this work).
2. `src/harness/scoring.jl` (all of it), `src/harness/cohorts.jl`, `src/harness/store.jl`,
   `scripts/score_runs.jl`, `scripts/run_candidates.jl`, `scripts/leaderboard.jl`.
3. `docs/guides/experiment_harness_guide.md` (scorecard, leaderboard, register sections).

## 1. What to build

**Item 0 — Phase A correction (do this first, own commit, push immediately, and report the SHA in
the pane).** `_grw_oos_boundary_innovations` (`multiscale.jl`) draws the summer boundary innovation
per team without centring. The fitted GRW states are zero-centred over teams at every time state
(`raw .- mean(raw, dims = 1)`), so centre the OOS innovation across teams per draw as well
(`z .- mean(z, dims = 2)` before scaling). Keep the per-fold seeding. Add a test: per draw, the
innovations sum to 0 across teams, and reload parity still holds. The GRW grids will be launched
from the commit containing this fix, so print `PHASEB_ITEM0_PUSHED <sha>` on its own line when
it's pushed, then continue with the rest of Phase B.


| Item | Work |
|---|---|
| **Panel parameterisation** | `score_runs`/`score_fits` today default to `target_seasons = ["24/25", "25/26"]` and `expected_fixtures = 710`. Make both explicit inputs. `scripts/score_runs.jl` gets `--target-seasons 23/24,24/25,25/26` and `--expected-fixtures N\|none`. The defaults stay the W1 values, so old invocations reproduce. `grid()`'s in-process scoring must take target seasons from the candidate's `DataScope`, not the default |
| **Per-row controls** | The run CSV gains an optional `control` column (label or UUID of the row's paired control). `score_runs.jl` scores each control group separately: each arm against its own base. If the column is absent, keep today's single-`role=:control` behaviour |
| **Scorecard v1.2 + panel label** | `SCORECARD_VERSION = "v1.2"`. Add a `panel` column to `harness_scores` (additive `ALTER TABLE … ADD COLUMN IF NOT EXISTS`), e.g. `"56+57|23/24,24/25,25/26|n=<fixtures>"`. Backfill existing v1.1 rows with the W1 panel label. `scripts/leaderboard.jl` must **never rank rows from different panels together**: group or section by panel, and state the panel in each table header |
| **(g) Transition-bias CIs** | `transition_bias_pp` gets `lo`/`hi` from a **club-season-clustered** bootstrap: resample the cohort's (club, season) units with replacement, B = 10,000, fixed seed. Add metric `delta_abs_bias_vs_control`: paired over the same resampled club-seasons, (|bias_arm| − |bias_control|) in pp with lo/hi. Emit it for every transition subset whenever a control is present. Keep `transition_bias_pp` point estimates **bit-identical** to v1.1 |
| **(i) Monitor scoring** | `scripts/score_runs.jl` scores on a datastore that contains 54/55 (`ScottishPyramid` or `ScottishAll`, whichever the harness runner uses for the fit's scope), so `monitor_t54` / `monitor_t55` rows are produced for pooled scopes. The **target** subset rows must be unchanged vs the `ScottishLower` datastore (test it) |
| **(h) Register merge** | `run_candidates.jl --only X` must **merge** X's run_id into the `harness_experiments` row's run list, not overwrite it. Record per-candidate commit metadata without clobbering the others' |

## 2. Tests (extend `test/harness_scoring_tests.jl` and `test/harness_runner_tests.jl`)

1. **Clustered bootstrap on a synthetic cohort:**
   - the CI widens when fixtures are duplicated within a club-season (fixture-level would narrow);
   - the point estimate equals the v1.1 value;
   - deterministic under a fixed seed.
2. **`delta_abs_bias_vs_control`** is 0 with a degenerate CI when arm = control, and has the right
   sign on a constructed example.
3. **Panel:** a v1.2 row carries `panel`; the leaderboard refuses to rank or mix two panels (a
   synthetic two-panel table).
4. **Per-row controls:** two control groups in one CSV give deltas against the right control
   (synthetic).
5. **Register merge:** two sequential `--only` upserts leave both run_ids.
6. All existing harness suites still pass, with counts; `test/run_parallel_tests.jl` no worse than T007.

## 3. Validation on real runs (light; the beast is shared with a running grid)

Re-score the 12 W1 runs with the new code:
`experiments/scotland/03_dynamics_scope_matrix/W1_runs.csv`, control `td_lower_joint`, W1 panel,
**v1.2**. Run it on the beast from a `beast_checkout.sh` checkout, with `-t 4` only (a grid is
using the other cores). If a grid is running, start it with `nice -n 19`.

1. Every v1.1 metric/subset value that v1.2 also emits is **identical** (|Δ| < 1e-12) for all 12
   runs. Print the max absolute difference per metric.
2. v1.2 adds lo/hi for `transition_bias_pp` and `delta_abs_bias_vs_control` rows. Report the
   relegated/promoted-into-L1 first-20 bias with its CI for all 12 cells.
3. The pooled W1 cells now have `monitor_t54`/`monitor_t55` rows. Report their LogLoss.
4. Do **not** delete or overwrite the v1.1 rows.

## 4. Report and stop

- Write `docs/architecture/w2_phaseB_report.md`: evidence per item (commands, SHAs, numbers), the
  validation tables, design choices.
- `./scripts/todo.sh check` passes; TODO 031 Work Log gets a dated `@pi` line.
- Commit, push, print exactly `PHASEB_DONE` and stop. If blocked: write it under "Blocked", print
  `PHASEB_BLOCKED`, and stop.
