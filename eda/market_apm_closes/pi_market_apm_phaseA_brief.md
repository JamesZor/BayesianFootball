Market-APM on closes, Phase A: data join and headroom EDA
Role: pi implementer in .worktrees/BayesianFootball-market-apm, branch eda/market-apm-closes. Phase A only.
Goal: measure whether the starting XI explains inverted closing rates beyond team strength, per league.
Done: DONE WHEN (§4); progress in eda/market_apm_closes/results/A/PROGRESS.md.
Stop: after Phase A print PHASEMAPM_HANDOVER and stop. Codex weekly is nearly spent; no Phase B work.

## 0. Objective and the idea

James's idea is a "market-APM": regress the market's inverted closing log-rates on the
starting XI, so the response is the market's goal expectation and not a noisy goal count. The
analogy is RAPM. Phase A builds the joined data and answers one cheap question before any
player-rating model:

> After team strength, do inverted closes move with who starts?

Phase B (player ridge, reliability, comparison with goals/PxG RAPM, forecast value) is a later
brief, after the Codex weekly reset. **Scottish League One/Two (56/57) is the focus.** The other
Scottish and English leagues are context and give sample size.

## 1. Context card

Read this instead of whole files. The manager wrote it from the sources; trust it, and open a
source only for the line ranges listed.

- **Inverted closes already exist for all leagues.** They are in
  `current_development/market_model/results/QS/rates_<league>.csv`, with these columns:
  `match_id, date, season, tournament, home, away, lambda_h, lambda_a, kl, n_markets, n_selections,
  lines_quoted, has_1x2, accepted, reason, …`. Use `accepted == true` rows only.
  - **Close:** Betfair time-weighted average over (−20, 0] min.
  - **Inversion:** per-match KL fit of a double Poisson to every quoted market (`l01_market_model.jl:53-63`
    for the close, `:146-182` for `invert_panel`).
  - **Accepted rows per league:**

    | League | Accepted | Seasons |
    |---|---:|---|
    | sco_league_one (56) | 835 | 21/22–25/26 |
    | sco_league_two (57) | 693 | 21/22–25/26 |
    | sco_premiership (54) | 956 | 21/22–25/26 |
    | sco_championship (55) | 477 | 21/22–24/25 (25/26 excluded) |
    | eng_premier (1) | 1507 | 22/23–25/26 |
    | eng_championship (2) | 2194 | 22/23–25/26 |
    | eng_league_one (3) | 2183 | 22/23–25/26 |
    | eng_league_two (84) | 2173 | 22/23–25/26 |

  - Ireland and Finland are out of scope.
  - **Don't re-invert.** If a row looks wrong, log it and move on.
- **Lineups:** `ds.lineups` in the pinned datastores. The SQL is in `src/Data/fetchers/sql/lineups.jl:1-30`
  (SofaScore `match_player_lineups`, with a BBC fallback).
  - Columns: `match_id, team_side, player_id, player_name, position, is_substitute, minutes_played, …`.
  - **Starting XI = `is_substitute == false`.**
  - On 56/57, `minutes_played` is 0 before 23/24 and mostly NULL in 25/26, so don't use minutes.
- **Pinned datastores** (no DB queries needed) are on mcmc-beast in `/root/BF_runs/market_model_qs/.cache/`:
  - `datastore_MarketModelEnglish.jls`, SHA256 `2efe200c…5168bf`;
  - `datastore_ScottishUpper.jls`, `56207e63…3630a`;
  - `datastore_ScottishLower.jls`, `e894ac6d…1006a`.

  Verify each SHA first. Load with `BayesianFootball.Data.load_datastore_cached(segment; max_age_hours=10^6)`.
  The English segment is stream-local: `MarketModelEnglish` in
  `current_development/market_model/l06_qs_eda.jl:40-41` (tournaments 1, 2, 3, 84). The beast
  checkout `/root/BF_runs/market_model_qs` already defines it; reuse that pattern
  (`DATA_SNAPSHOTS.md` in `results/QS/`).
- **The starting XI is not a leak for a close-price regression.** Lineups are published about
  60 minutes before kick-off, before the T−20 close window, so the close has priced the announced XI.
  (A forecasting use before announcement would leak; that's out of scope here, see
  `LastHistorical leaks the played XI` in the repo history.)
- **Prior work** that frames the result:
  - Goals/xG RAPM as a covariate **hurt** log loss once the model had explicit team strength:
    `rapm_xg` +0.0108 nats [+0.0026, +0.0193] (`current_development/scottish_lower/FEATURE_DISCOVERY_REPORT.md` §4).
    It was carrying team strength, not player skill.
  - In the plus-minus stream, the market-price target was scoped away and never built
    (`current_development/plus_minus_ratings/NOTES.md:722-765`). xG-PM was the best target there.
  - Scotland 54–57 has 3,192 players, of whom 576 moved between upper and lower tiers.

**Reading rules**
- Use `read` with offset/limit, or `rg -n`, for anything not listed. Don't read a whole README,
  report or design doc.
- Don't re-read a file you've already read unless it changed or a compaction dropped it.
- Print at most 40 lines of any command. Send longer output to `results/A/logs/<name>.log`, then
  `rg`/`tail` it, or use `context_get` for stored output.
- Write decisions and their reasons to `results/A/DECISIONS_LOG.md` as you go.
- Keep `results/A/PROGRESS.md` current: phase, done, next action, REPL names and job IDs.

**After a compaction** (your context starts with `[Session Goal]` and a compaction note):
1. Re-read this brief's DONE WHEN and stop rules, then `PROGRESS.md` and `DECISIONS_LOG.md`.
2. Never guess a path, command or ID. Take it from the summary's `[Files And Changes]`, from
   `recall`, or from `rg --files`. Re-read only the lines you are about to edit.
3. Don't redo work the summary or `PROGRESS.md` marks done; check that its output exists.

**Context:** compaction is automatic at about 200K. At the phase end, commit, write `HANDOVER.md`
and stop with the sentinel.

## 2. Rules

- **Folder:** everything goes in `eda/market_apm_closes/`, with code as `l01_*.jl`/`r01_*.jl`
  and results in `results/A/`. No `src/` changes. Commit small steps on `eda/market-apm-closes`
  and push that branch only. Never merge or rebase.
- **Julia:** on mcmc-beast through `jl` (`jl start mapm --beast --threads 16`, `jl run mapm <file>`,
  long jobs `--bg` plus `watch_job`). Set BLAS to 1 thread. Give the beast a checkout of
  this branch at `/root/BF_runs/market_apm`.
  - The laptop is allowed for small DuckDB/CSV work on committed outputs.
  - Never touch other tmux sessions, locally or on the beast. On the beast that includes
    `claude_w5_repl` (the wave 5 agent), `julia`, `claude_clv_napkin`, `pi_klm`, `w2_watch` and
    `mbtop`. Keep `/root/BF_runs/*`.
- **Read-only:** no betdb writes, no `mcmc_experiments` writes.
- **No polling:** use one blocking `jl run` or `watch_job` per job. No sleep loops.

## 3. Phase A tasks

1. **Join and coverage**, per league × season:
   - fixtures;
   - accepted inverted closes;
   - closes with a complete starting XI (11 + 11) on both sides;
   - the final joined n.

   Also list the failures by reason.
2. **Design diagnostics** per league (pooled seasons):
   - players with at least 1, 5 and 20 starts;
   - median starts per player;
   - per team-season, the share of starts by the 11 most-used players;
   - mean overlap of the XI with the previous match's XI (rotation);
   - within-league movers between clubs.

   Also give the rank of the starter design **after** absorbing team-season attack and
   defence effects. That is the dimension player deviations can be identified on.
3. **Headroom: how much is left after team strength.** Response `y = (log λ_h, log λ_a)`, and the
   derived supremacy `log λ_h − log λ_a` and total `log λ_h + log λ_a`.
   - Fit team-season fixed effects (attack and defence per team-season, one home term per
     league) by least squares.
   - Report the residual SD per league for supremacy and total.

   That residual is the most a player term could explain. It also includes in-season form,
   which a season-constant team effect misses.
4. **The cheap "who starts" test.** Define a **regular** causally: a player among the team's 11
   most-started players in that season **before** this match. Use no future matches, and drop
   each team's first 5 matches of the season.
   - Per match and side: `absent = 11 − (number of regulars in the XI)`.
   - Regress the step-3 residual supremacy on `absent_home − absent_away`, and the residual total
     on `absent_home + absent_away`, per league.
   - Report the slopes with match-clustered (or league-season block) standard errors.
   - Express the slope as Δ supremacy per missing regular, and as a 1X2 home-probability change at
     a typical line.
   - **Placebo:** do the same with the **next** match's absences (they must not predict this
     close).
   - **Form control:** add a rolling or within-season time trend per team, or use the previous
     close's residual, so that "absent" doesn't pick up a run of bad form.
5. **One figure per league**, plus a pooled one: binned residual supremacy against
   `absent_home − absent_away`.

## 4. DONE WHEN

1. `results/A/coverage.csv`, `design.csv`, `headroom.csv` and `absent_slopes.csv` (with the
   placebo and form-control rows) are committed, and the figures are in `results/A/figures/`.
2. `eda/market_apm_closes/REPORT_A.md` has a summary of at most 10 lines first. It must answer:
   - Is the who-starts effect visible in closes after team strength, and how big is it per
     missing regular?
   - Is it bigger in 56/57 or in the English leagues?
   - Does the placebo come out null?
   - Is a player-level market-APM (Phase B) worth the build, with what identifiable dimension
     and sample?

   Every number must trace to a CSV.
3. `results/A/HANDOVER.md` (what's done, Phase B recommendations, open questions) is committed and
   pushed, the beast REPL is stopped, and `PHASEMAPM_HANDOVER` is printed alone on its line.

## 5. Out of scope

- A player ridge or a Bayesian market-APM, which is Phase B.
- Lineup-announcement timing and order-book data (the data doesn't exist before 26/27).
- Any forecasting or betting backtest.
- Re-inverting closes or changing the inversion.

## 6. Budget and blocks

- `openai-codex/gpt-6.1-sol`, thinking medium (data plumbing). Codex weekly is about 91% used.
  If you hit the usage limit, write `HANDOVER.md` with the exact next step and stop. The manager
  resumes after the reset.
- Stop and ask (`ask_user_question`) if a lineup or league join loses more than 15% of the accepted
  closes, or if a datastore SHA doesn't match.
