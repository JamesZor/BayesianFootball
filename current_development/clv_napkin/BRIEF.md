# Napkin brief — CLV across our models and all Betfair markets (prototype)

You are a **solo prototyper** (Claude CLI, Opus 5.5). Do not spawn subagents. Claude, the
manager in another session, reviews your result.
- This is **napkin/prototype work**: fast, honest and approximately right beats polished.
- It lives in `current_development/clv_napkin/`, not in `src/`.

## The question

**Do any of our models beat the closing line, and where?** The human's hypothesis:
- W1/W2 were hyper-focused on **compression** (model spread too narrow vs the market).
- Fixing compression brings the model closer to the market, so it may *reduce* ROI: fewer and
  smaller disagreements to bet on.
- What actually matters is whether the disagreements that remain are right, and CLV measures that.

So answer three things:
1. **CLV and ROI by model**, flat stakes, for bets placed before kick-off.
2. **CLV by market**, for *every* market we can price and Betfair archives. Up to now we have only
   bet 1X2 and O/U 2.5.
3. **Compression vs CLV/ROI across models.** Does less compression buy or cost CLV and ROI?

## Rules

- **Worktree** `/home/james/bet_project/.worktrees/BayesianFootball-clv-napkin`, branch
  `proto/clv-napkin`.
  - Commit only inside `current_development/clv_napkin/`. Push this branch; never merge.
  - No edits to `src/`, `test/`, `scripts/`, `experiments/` or `todos/`.
- **The laptop runs no Julia.** Julia runs on mcmc-beast (`ssh root@mcmc-beast`) in your own tmux
  session there, named `claude_clv_napkin`.
  - rsync the worktree to `/root/BF_runs/clv_napkin_dev/` with
    `--exclude '/.cache/' --exclude '/data/' --exclude '/.git/'`.
  - Symlink `.env` and `Manifest.toml` from `/root/BayesianFootball/`, and copy
    `/root/BF_runs/a76a65df/.cache/datastore_Scottish*.jls` into its `.cache/`.
  - Julia is `/root/.juliaup/bin/julia --project`, after `set -a; source .env; set +a`.
  - Logs go in `/root/BF_runs/logs/clv_napkin/`.
  - **Another agent may be running tests on the beast.** Check `pgrep -af julia`. If another
    non-Kaimon Julia job is running, use `nice -n 19` and `-t 8`; otherwise use `-t 16`. Load one
    fit at a time, since pooled GRW fits are large.
- **No sampling, no new fits.** Use persisted fits (`Training.load_fit`) and their held-out
  latents or score grids only.
- **Databases are read-only.** That covers `mcmc_experiments` (`PostgresStorage`) and `betdb`
  (`BF_DB_URL`, including the `betfair` archive schema). Never print credentials.
- Follow the loader/runner pair convention (`docs/prototype_runner_style_guide.md`):
  - `l01_clv_napkin.jl` holds the functions;
  - `r01_clv_napkin.jl` is the readable runner with numbered sections;
  - `REPORT.md` holds the results.

## Reuse, don't rebuild (read these first)

- Pricing a fit's held-out matches into books: `src/harness/portfolio.jl` (`_portfolio_source`,
  `_portfolio_build_books`, `_restrict`, `_betfair_closing_odds`),
  `Portfolio.build_books_reported`, `Portfolio.BookSpec`, `Data.MarketConfig` and the market
  types:
  - `MarketOverUnder(line)`;
  - `MarketBTTS`, `MarketCorrectScore`, `MarketDC`, `MarketDrawNoBet`, `MarketAsianHandicap`;
  - 1X2.
- Entry books before kick-off: `Calibration.point_in_time_book(ds; config = PointInTimeBookConfig(as_of_minutes = -25.0))`.
  If it doesn't cover every market or line, build the T−n price from `betfair.odds_history`
  yourself (last quote at or before T−n).
- CLV code that exists (reuse or mirror it; don't invent a third definition):
  - `Calibration.bet_clv`/`clv_summary` (`src/Calibration/book.jl`);
  - `MatchDay.clv_for_order` (`src/MatchDay/ledger/settle.jl`).
- The Betfair archive (betdb schema `betfair`): `markets.market_type` includes `MATCH_ODDS`,
  `OVER_UNDER_05` … `OVER_UNDER_55`, `CORRECT_SCORE` and `BOTH_TEAMS_TO_SCORE` (about 17.5k matches
  each), with sparse `DOUBLE_CHANCE`, `ASIAN_HANDICAP` and `DRAW_NO_BET`. `odds_history.odds_data`
  holds the price series; `match_meta` maps events to kick-off times.

## Definitions (use exactly these)

- **Entry:** the back price at **T−25** (primary). If cheap, also T−60 and the earliest
  available, to show how CLV changes with lead time.
- **Close:** the de-vigged Betfair price, time-weighted over (−20, 0] minutes, as the harness
  scorecard does. De-vig multiplicatively within each market (all runners of that market).
- `p_entry = 1/odds_entry` (the taken price, not de-vigged).
- `clv_pp = 100·(p_close − p_entry)`.
- `clv_pct = 100·(odds_entry·p_close − 1)`.
- `beat_close = clv_pp > 0`.
- **Model edge:** `100·(p_model − p_entry_fair)`, where `p_entry_fair` is the de-vigged entry book.
- **Bets:** flat 1-unit **back** bets on every selection with model edge ≥ `e`, for
  `e ∈ {0, 2, 4, 6}` pp. Back only; no lays.
- **ROI:** realised return per unit staked. Apply Betfair commission on net winnings per market at
  **2% and 5%** (sensitivity).
- **Uncertainty:** a slate-clustered bootstrap (a slate = one match day; B = 2,000, fixed seed).
  Report 95% CIs for mean `clv_pp`, the beat-close share and ROI.
- **No leakage:** entry prices only from quotes at or before T−n. The close is used only for CLV,
  never for selection. Model probabilities come from held-out, walk-forward latents only.

## Models to cover

Report each run on its own panel, and never pool across panels. Record each run's compression
slope from `harness_scores` (`subset = 'target'`, `market = '1X2'`, `metric = 'compression_slope'`,
`scorecard_version = 'v1.2'` or `v1.1`) so you can relate compression to CLV and ROI.

- **W2**, 1,070-fixture panel (23/24–25/26), experiment `scottish_pyramid_tier_transition_w2`:
  all 15 grid runs. Get the IDs from `experiments/scotland/04_pyramid_tier_transition_arms/W2_runs.csv`.
- **W1**, 710 panel: the 12 runs in `experiments/scotland/03_dynamics_scope_matrix/W1_runs.csv`.
- **W0**, 710 panel: `experiments/scotland/W0_runs.csv`, including `m12_td` `132df5c2`
  (the live model) and `m05_joint_td` `ed541a7c`.
- **Floor control:** a pure team-strength Poisson (`s12_m01_td_poisson` `de7fa956`) in the same
  table, to show what "no modelling edge" looks like.
- **A market-only null:** CLV of betting *every* selection (e = −∞) at T−25. That is roughly the
  cost of the spread, the baseline any model must beat.

If time is short, do W2's 5 key runs, W0 `m12`/`m05` and the floor first:
`td_base`, `td_lower_base`, `td_lower_a2full_carry_jump`, `grw_base`, `grw_step_a2_carry_jump`.

## Tournaments

- **Primary:** Scottish League One and Two (56/57) held-out target fixtures.
- **Secondary, if the odds exist:** the pooled W1/W2 runs also hold 54/55 (Premiership and
  Championship) held-out *monitor* latents. Report CLV there too, where Betfair archive prices
  exist (the Championship is patchy; see TODO 033). Keep those tables separate from 56/57.

## Output: `current_development/clv_napkin/REPORT.md`

Keep it short and readable:
1. **The one-paragraph answer.** Is there positive CLV anywhere after the spread? Where, and how
   sure are we?
2. **Table A, by model:** n_bets, mean `clv_pp` [CI], beat-close %, ROI at 2% and 5% commission
   [CI], and compression slope. Show `e = 2` as the headline, with the other thresholds in an
   appendix.
3. **Table B, by market** (1X2, each O/U line, BTTS, correct score, DC, DNB, AH where there are
   enough bets), for the best 3 models plus the floor.
4. **Table C:** CLV by edge bucket (monotonic or not?) and by league 56 vs 57. Add lead time if
   done.
5. **Compression vs CLV/ROI** across all runs, one line per panel. Say plainly whether the data
   supports the human's hypothesis.
6. **54/55 monitor** results, if done.
7. **Caveats:**
   - napkin method;
   - multiple comparisons: many slices, so say explicitly which results would survive a rough
     correction;
   - data gaps.

Add a small ASCII chart or two if it helps; no plotting dependencies needed.

When done: commit, push `proto/clv-napkin`, print exactly `CLV_NAPKIN_DONE` on its own line, and
stop. If blocked, print `CLV_NAPKIN_BLOCKED` with the reason.
