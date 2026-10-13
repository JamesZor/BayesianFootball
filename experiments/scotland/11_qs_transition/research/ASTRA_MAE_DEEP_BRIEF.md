Deep research + EDA: what else could cut the model-vs-market error on promoted and relegated clubs? (gpt-6-astra)
Role: research agent in .worktrees/BayesianFootball-qs-transition. Read-only on the DB and repo; you write only under research/mae_deep/.
Goal: find what the market knows about tier movers that QS does not, test it on our data, and rank concrete ways to reduce MAE_club.
Done: write research/mae_deep/REPORT.md; print PHASEMAE_DONE alone on its line.
Stop: no Julia, no model fits, no DB writes, no git writes, no 26/27 outcomes (coverage counts only).

## 1. The question and what we have learned

**Original question (James):** QualityStyleGRW with market-rate observations (qs_marketobs) misprices
clubs that change tier. In the 26/27 paper trade its 37 bets on transition-club fixtures returned
−39%, against +22% on the rest.

**What we have tested** (all on 23/24–25/26 Scottish League One/Two, 60-fold walk-forward, each fix
against the control). Read `experiments/scotland/11_qs_transition/REPORT_S3B.md` §1–2,
`MANAGER_DECISIONS.md` D21–D24, and `research/LOGLOSS_THEORY_REPORT.md`'s summary.
- **The fixes:**
  - A2 tier term;
  - a market-derived entrant prior;
  - a summer state step;
  - weekly refits;
  - pooling with Championship goals (+A2).
- **The result:**
  - none moves log loss (all |ΔLL| ≤ 0.0004);
  - none moves **MAE_club**, the per-club mean |P_model(win) − P_market(win)| over the first 20
    games (DESIGN.md §9), by more than 0.33 pp, against about 5–6 pp in every arm;
  - A2 and pooling cut the worst-club error (promoted 2.7 → 1.1 pp) but not the mean.
- **Theory report:** the fixes act only before a club's first matches, and they move most fixtures
  by under 1 pp. A learned, partially pooled boundary transition is recommended.
- **S1 EDA** (`REPORT_S1.md` summary; `eda/results/`): the market's opening re-rating is mostly
  quality; cup-based tier gaps are unequal (0.40 / 0.28 / 0.19); summer dispersion is larger than
  within-season.
- **Earlier finding** (CLV napkin): the market's early rating of a club is roughly last season's
  goal difference plus the tier change (R² 0.70).

**So the open question:** the 5–6 pp error is mostly per-match, not a club-level offset these fixes
reach. **What information does the market use for movers that QS lacks, and how could QS use it
without leaking the target match's own price?**

## 2. What to investigate (EDA plus literature)

Candidate information the market may use, to test against our data:
1. **Squad strength and turnover:** players' market values (SofaScore
   `match_player_lineups.proposed_market_value`), lineup continuity from last season, players arriving
   from higher tiers, and the age profile.
2. **Manager changes** (`sofascore.matches.manager_home/away`).
3. **Last-season underlying performance** beyond goals: xG or shots (BBC pxG from 23/24; shots), goal
   difference, points, and end-of-season form.
4. **Club resources:** parachute/relegation effects, stadium or venue proxies, and whether the club is
   full-time or part-time (if derivable, e.g. from venue or history; say if not).
5. **Opponent-specific effects:** are the errors concentrated in particular matchups (vs top or bottom
   sides, home or away)?
6. **Market dynamics:** how the market's rating of a mover evolves over its first 20 games against
   QS's. Is the error a lag, an anchoring, or noise?
7. **Anything the literature suggests** (promoted-team prediction, squad-value models, Transfermarkt-
   value studies, market-efficiency studies), with web and browser.

For each candidate, measure how much of the **per-match model − market residual** for transition
fixtures it explains, against a non-mover baseline. Use only information available **before** the
match. Never use the match's own price as a feature.

## 3. Data access (read-only)

- **Model and market per fixture:** `experiments/scotland/11_qs_transition/grid/observations_qs_*.csv`
  (all six arms; 4,406 selection rows, 938 fixtures; the model and market probabilities are both
  there). Per-club and bin tables are in `grid/abs_error_table.csv` and `grid/bias_game_bins.csv`.
  `eda/results/` has S1's mover tables.
- **Inverted market rates:** `current_development/market_model/results/QS/rates_sco_*.csv`.
- **betdb, read-only, via DuckDB:**
  `set -a; . /home/james/bet_project/BayesianFootball/.env; set +a` and then
  `duckdb -c "LOAD postgres; ATTACH '$BF_DB_URL' AS db (TYPE postgres, READ_ONLY); …"`.
  - **Never print the URL or `.env`.** Pipe output through `sed -E 's#postgres(ql)?://[^ ]*#<url>#g'`.
  - Useful schemas: `sofascore.matches`, `sofascore.match_player_lineups`, `sofascore.match_statistics`,
    `sofascore.events`, `betfair.*`, `bbc.*`. List columns with `db.information_schema.columns`.
  - `start_timestamp` is timestamptz: compare with `left(start_timestamp::varchar,10)`.
  - Keep queries targeted (tournaments 54–57 plus cups). No full-table dumps.
- **Library:** `~/.pi/agent/skills/research-library/bin/lib-grep PATTERN [-i]`, `lib-show`, `lib-sql`.
- **Web:** `web_search`, `web_fetch`, and browser tools (enabled).

## 4. Deliverables in `research/mae_deep/`

1. `REPORT.md`:
   - a summary of at most 15 lines;
   - one section per candidate (what, data, effect on the residual, with numbers and the
     query/CSV);
   - the literature;
   - **a ranked list of concrete ways to reduce MAE_club**, each with: the mechanism, the data
     needed, a model change (equation or feature), the expected size, the leakage risk, and how to
     test it as a QS arm.
2. Small CSVs of your EDA results, plus the SQL you used in `queries.sql`.

## 5. Budget and rules

- gpt-6-astra (costly): no context or fetch cap (James, 2026-10-12: pi-blackhole handles compaction). Write
  the report as you go.
- Print at most 40 lines per command. Write longer results to CSV, then summarise.
- **No `git add` or `git commit`** (another agent may be in this worktree); the manager commits.
- `~/bet_project/hub/experiments` doesn't exist; keep records here and don't ask about it.
- If a scope question blocks you, use `ask_user_question`.

**After a compaction:** re-read §4, then your REPORT.md so far; continue from where it stops.
