QS transition experiment, Stage 1: EDA on why clubs that change tier are mispriced
Role: pi implementer (gpt-6.1-sol) in .worktrees/BayesianFootball-qs-transition, branch exp/qs-transition. Stage 1 only.
Goal: diagnose the cause of the transition bias (cold start, tier scale, summer step, lag) and propose a market-informed prior.
Done: §5 DONE WHEN; write REPORT_S1.md in experiments/scotland/11_qs_transition/.
Stop: print PHASES1_DONE alone on its line at the end. No new model fits, no src edits, no DB writes.

## 0. Records

`~/bet_project/hub/experiments/` does not exist. Keep every record in this worktree:
- `experiments/scotland/11_qs_transition/eda/` for code and results;
- `REPORT_S1.md` one level up.

Local commits only; the manager pushes. Don't ask about this.

## 1. Context card

Read `experiments/scotland/11_qs_transition/DESIGN.md` in full (about 170 lines); it is the shared
design. Then `research/EDA_QUESTIONS.md` and `research/PRIOR_FORMS.md` (Stage 0 output). Beyond
those, read only the ranges listed here.

- **Inverted market rates** (per-match KL inversion of the de-vigged Betfair close; use
  `accepted == true` only):
  `current_development/market_model/results/QS/rates_sco_{premiership,championship,league_one,league_two}.csv`.

  | Tier | Seasons | Accepted |
  |---|---|---:|
  | Premiership | 21/22–25/26 | 956 |
  | Championship | 21/22–24/25 (no 25/26) | 477 |
  | League One | 21/22–25/26 | 835 |
  | League Two | 21/22–25/26 | 693 |

  Columns: `match_id, date, season, tournament, home, away, lambda_h, lambda_a, …`. From them, per
  club-season, q and s come from a simple team fit within each league-season (see DESIGN §3).
- **Tiers and cohorts:** `src/harness/cohorts.jl:14-60` (`club_season_tiers`, `transition_cohort`).
  The transition bias definition is in `src/harness/scoring.jl:246-330`
  (`_transition_bias_observations`, `_clustered_bias_bootstrap`, `_delta_abs_bias_vs_control`):
  mean(model − market) win probability of the transitioning club, first 20 league games,
  club-season-clustered.
- **Existing fits** to decompose (read-only, held-out predictions through the harness scoring path):
  - qs_marketobs `81fdf817-f82d-4f66-9bfd-5f456c4fe29f` (namespace `scottish_lower_qs_wave2_2426`,
    24/25–25/26);
  - W2, panel 23/24–25/26: `td_lower_base` `0dd74a52`, `td_lower_a2full_carry_jump` `6c0aa3ea`,
    `grw_base` `ce7ea22f`, `grw_step_a2_carry_jump` `c4a0fa94`, `m12_td_ref` `973baafa`.

  W2's scoring recipe is in `experiments/scotland/04_pyramid_tier_transition_arms/README.md` §Reproduction:
  `--target-seasons 23/24,24/25,25/26 --expected-fixtures 1070`, with the grids' datastore cache.
  Resolve the full UUIDs from the harness tables (read-only).
- **A2 component:** `src/models/pregame/builder/pyramid_tiers.jl:1-80`. Its tier step prior is
  TruncNormal(0.47, 0.16), with a 0.48/0.52 attack/concession split.
- **Cups:** Scottish Cup, League Cup (knockouts only) and Challenge Cup are in betdb.
  - Cups have no Betfair. SofaScore `match_odds` 'Full time' covers all three, so use de-vigged
    bookmaker 1X2 for the cup market rating.
  - `sofascore.match_statistics` has almost no cup rows.
- **pxG** starts 23/24. Premiership pxG reads about 21% high relative to goals.
- **DB access** is read-only: the repo's Data fetchers, LibPQ with `BF_DB_URL` from `.env` (never
  print it), or DuckDB `ATTACH … (TYPE postgres, READ_ONLY)`. Wrap every Julia runner in the
  10-table before/after DB audit used in `experiments/scotland/10_qs_marketobs_2627_paper/l00_p27.jl`
  (`with_audit` pattern).

**Reading rules**
- Use `read` with offset/limit, or `rg -n`, for anything not listed. Don't read whole reports.
- Print at most 40 lines of any command. Send longer output to `eda/logs/`, then `rg`/`tail` it,
  or use `context_get`.
- Keep `eda/DECISIONS_LOG.md` and `eda/PROGRESS.md` current.

**After a compaction:** re-read this brief's §5, then `PROGRESS.md` and `DECISIONS_LOG.md`. Take
paths and IDs from `recall` or `rg --files`, never from memory. Don't redo finished work.

## 2. Rules

- Julia runs only on mcmc-beast, through `jl` (`jl start s1 --beast --threads 16`; `jl run s1 <file>`;
  long jobs `--bg` plus `watch_job`). BLAS 1. Checkout at `/root/BF_runs/qs_transition`. Small
  CSV/DuckDB work on committed outputs may run on the laptop.
- Never touch other tmux sessions, locally or on the beast. On the beast, `julia` and `mbtop` belong
  to James.
- **No look-ahead.** Any quantity that will become a prior must be computed per fold, from seasons
  strictly before the fold's target season.
- **26/27 is held back.** Only coverage counts are allowed.
- **No new model fits.** S2 does those.

## 3. Questions

### 3a. Base set (DESIGN §6)

1. **Market re-rating of movers**, by direction:
   - new-tier rating against last season's old-tier rating;
   - how fast the market settles: the first quote, then games 5, 10 and 20.
2. **Bias decomposition of the existing fits** by game number 1–20 and by fold age, mapped onto
   causes a–d. Include non-movers at the season start, as the summer-step check.
3. **Tier value:**
   - from cup ties (goals, and bookmaker odds);
   - from movers (market);
   - compared with 0.47; check linearity across tiers and the Old Firm.
4. **Summer movement** of non-movers' market ratings, against QS's boundary step.
5. **Coverage:** market history per club and season across tiers, the Championship 25/26 gap, and
   clubs with no market history.

### 3b. From Stage 0 research (merged by the manager)

Stage 0 wrote 12 testable questions in `research/EDA_QUESTIONS.md`. **Use them as the EDA's question
list.** They supersede §3a where they overlap: Q1 covers base 2, Q2 the prior check in §4, Q3 base 3,
Q4 base 4, and Q6 base 5. Base 1 (market re-rating of movers) feeds Q2 and Q7.

Read EDA_QUESTIONS.md's "Common contract" and follow it; in particular, a quote identifies only
D = q_h − q_a and U = s_h + s_a.

**Order and priority (manager, D2):**
1. Q6 and Q12 first, as prerequisites: coverage, and the provenance and no-look-ahead audit.
2. ★ Q1–Q5 (the cause verdict depends on these).
3. Then Q7, Q10, Q8.
4. Q9 (cups vs bookmaker sources) and Q11 (squad discontinuity) only if time allows. If you skip
   them, say so in REPORT_S1.

Prior candidates: P1/P2/P3 in `research/PRIOR_FORMS.md`. §4's out-of-sample check is Q2.

### 3c. Free exploration

James asked to "play around with it". Anything else that sheds light on the causes is welcome. Log
each exploration in DECISIONS_LOG with one line on what it showed.

## 4. The prior proposal

Choose one of the `research/PRIOR_FORMS.md` candidates, or a better one. Justify it with the
results above and write it as equations on q/s, with:
- its per-fold, earlier-seasons-only construction;
- its parameters estimated from movers before each fold;
- its fallback for clubs without market history (A2's tier step, or a results-based rating);
- **an out-of-sample check of the prior on its own:** for movers in seasons before the panel and
  within it (estimated only from earlier seasons each time), how well the prior predicts the
  market's first-quote and first-5-games rating. Report the error and calibration, against two
  baselines: "league mean" (today's QS) and A2's tier step.

## 5. DONE WHEN

1. Each question in §3 has a CSV in `eda/results/`, and figures in `eda/figures/` where useful.
2. `REPORT_S1.md` (in `experiments/scotland/11_qs_transition/`) has:
   - a summary of at most 12 lines;
   - a **cause verdict**: which of a–d drive the bias, with numbers;
   - the prior proposal with its out-of-sample check;
   - a **recommended S2 build list and S3 arm list**;
   - open questions.

   Every number traces to a CSV.
3. DB audits are unchanged. Commit on `exp/qs-transition` (local). Stop the beast REPL. Print
   `PHASES1_DONE`.

## 6. Budget

- gpt-6.1-sol, thinking high (diagnosis).
- If you hit the usage limit, write `eda/HANDOVER.md` with the exact next step and stop. The
  manager resumes you.
- If a gate fails (an existing fit won't score, or the cohort counts don't match W2's 5 / 6 / 2),
  stop and `ask_user_question`.
