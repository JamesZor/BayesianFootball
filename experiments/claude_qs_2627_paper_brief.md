# 26/27 paper trade (Claude CLI): replay the frozen qs_marketobs + scenario Kelly config on this season's matches
Role: Claude Code implementer on `exp/qs-goal-vs-market`. The launching Claude session (the wave-5 session, local pane %114) is your manager. James set the scope.
Goal: predict every 26/27 League One/Two match played so far with qs_marketobs (walk-forward, no look-ahead), then paper-trade the FROZEN wave-5 config on them at the order-book back price, next to m12 canonical.
DONE WHEN is §5. Stop at the end of each phase with P27_HANDOVER. Stop at any failed gate with P27_BLOCKED.
No DB writes of any kind. src is frozen. The config is frozen: no tuning, no new baskets or λ.

## 1. Context card

Read this instead of whole files. The manager wrote it from the sources, so trust it, and open a source only at the line ranges listed.

### What is being paper-traded (wave 5, TODO 044, `experiments/scotland/09_qs_marketobs_wave5_config/`)
- **Frozen config** (`FREEZE.md`, commit `04fdba71`):
  - model qs_marketobs = `QualityStyleGRW` (weak r) + `MarketRateObservation`, trained on KL-inverted market close rates only (no goals);
  - backtest fit: run `81fdf817-f82d-4f66-9bfd-5f456c4fe29f`, namespace `scottish_lower_qs_wave2_2426`, 40 biweekly folds over 24/25 + 25/26;
  - basket B1 = 1X2 H/D/A + Under 2.5, with legs outside the basket dropped BEFORE the joint solve;
  - TieredTrust H 0.35, U2.5 0.35, D 0.25, A 0.25;
  - `ScenarioKelly()` (1000 scenarios, seed 20261010, budget 0.99, per-leg cap 0.50) jointly per day;
  - `SlateDrawdown(12.0)`, `FixedCap(0.25)`, canonical BookSpec (all canonical markets, `DeArb`), 2% commission.
  - `PAPER_TRADING_CONFIG.md` restates it.
- **Fallback:** the same config at λ 23. **Benchmark:** m12 × `MD.canonical_scottish_lower_policy()` (per-match allocator, λ 23), run `132df5c2-c742-4e95-8693-3aeb2b2cbaef`.
- **Wave-5 25/26 result** (T−25m last-traded prices, `REPORT.md`):
  - frozen: wealth 1.6241209548752367, 494 bets, g 9.70×10⁻³/day, MDD −19.26%;
  - m12: 1.327330130248303, 557 bets;
  - fallback: 1.3208208815934512, 496 bets.
  - Ruling-8 drawdown watch: **−20.39%**.
- **Reusable code, all in `l00_wave5.jl` (module QSWave5):**
  - `day_problems` (:159-183) builds one joint scenario problem per day;
  - `solve_days(days, keep)` (:211-249) solves the column subset for a basket;
  - `basket`, `in_basket`, `CANON_TIERS` (:139-154);
  - `basket_policy`, `simulate_slates`, `simulate_books` (:114-133);
  - `curve_row` (:269-280), `with_audit` (:283-320).
  - In `l03_phase2_eval.jl`: `half_cells` (:67-) scores cells with `P3.cell_tables`, and `block_bootstrap_circular` sits beside wave 4's `P3.block_bootstrap`.
  - These assume the 710-fixture research universe and the pinned datastore (`load_inputs` :72-108), so you will need a 26/27 loader. Do not edit wave-5 files; include them, or copy a function and log the copy.
- **Wave 4** (`08_qs_marketobs_backtest/`):
  - `l01_scenario_kelly.jl` holds the scenario adapter `QSBacktestScenario.scenario_problem(...; native = true)`;
  - `l02_phase3_grid.jl` has `goal_model` (:77-96, loads a fit's latents by UUID), `close_index` (:137-159), `bet_ledger`/`clv_stats` (:164-195);
  - `l00_phase0_inventory.jl` has `snapshot`, `read_query`, `audit_snapshot`;
  - `check_read_paths.py` is the static write-verb checker. Run it on your files too, and widen its allowlist only for the read-only SQL loader, logging that.

### 26/27 data (manager's read-only checks on 2026-10-11)
- **Results:** betdb `sofascore.matches` has **79** finished L1/L2 (tournaments 56/57) 26/27 matches, from 2026-08-01 to **2026-09-19**. Nothing later is ingested and there are no upcoming fixtures, even though the order book has rows up to 2026-10-08.
  - Do not ingest anything. Report the gap; James decides.
- **Historical odds:** `betfair.odds_history` has **no** 26/27 L1/L2 rows. So every 26/27 price, both entry and close, comes from `betfair_live.order_book_1m`.
- **Order book:**
  - 70 of the 79 fixtures map through `betfair.match_meta` → `betfair_live.market_metadata` (`md.event_id = mm.betfair_event_id`).
  - 60 have books in the 30 min before the off for MATCH_ODDS, OVER_UNDER_05/15/25/35 (and 45), and CORRECT_SCORE; 59 also have BOTH_TEAMS_TO_SCORE.
  - The 08-22 round (book starts after the off) and the 08-29 round (no market) have nothing.
- **The CLV napkin did this before** for another model (branch `proto/clv-napkin` @ `1c215df4`, worktree `/home/james/bet_project/.worktrees/BayesianFootball-clv-napkin`; read only, never commit there). See REPORT.md §13 (lines 1349-1642).
  - Coverage by match day (T−25m: 60 fixtures with a complete two-sided book ≤ 15 min old).
  - Snapshots every ~3 min.
  - Prices and sizes ×10⁴; `bid` = back side, `ask` = lay side.
  - LTP is populated from 08-08 on.
  - The median back size is £9 at T−25m.
  - At the back price, CLV was −0.6 pp; the LTP-era edge fits inside a ~4-tick spread.
  - Transition clubs (Ross Co, Airdrie, East Kilbride, Kelty) drive the largest model–market gaps.
- **Code from that study:** `current_development/clv_napkin/l04_live_orderbook.jl`:
  - `c4_extension_plan` :52, `c4_filtration` :77, `c4_extend_inmemory` :110, `c4_convergence_table` :134;
  - `c4_markets` :157 (MATCH_ODDS only, so extend it to the OU/BTTS market types), `c4_snapshots` :185;
  - `c4_entry_book` :238 (fresh two-sided book at an instant, ≤ 15 min stale), `c4_close` :289 (TWA close).
  - Also `r04_extend_grw_spfl_joint.jl`, the in-memory fold extension: the run's own splitter with "26/27" appended to `target_seasons`, positions 1–40 checked against the persisted held-out sets, only new positions sampled, nothing persisted to the DB, output serialised to a `.jls` on the beast.
  - Read it with `git show origin/proto/clv-napkin:<path>`. Copy what you need into your folder and log each copy.
- **src/MatchDay:**
  - `matchday-module.jl:1-75` (docstring: order-book scaling, back/lay, collector caveats);
  - `db.jl:60` (market_type ↔ symbol mapping), `:141` (match_meta join), `:169` (order-book query).
  - The MatchDay paper ledger (`ledger/`) writes to the DB. **Do not use it.**
- **m12's 26/27 predictions:** the live m12 run `132df5c2` has been extended weekly in the DB by `r17_extend_to_card.jl` (branch `feat/scottish-cross-tier-cups-eda`, `current_development/match_day_inference/`). Phase 0 must establish:
  - whether folds covering 08-01 → 09-19 exist;
  - that each fold trained only on days strictly before its held-out block.
  - If they don't, plan an in-memory extension for m12 too, as for qs.

### Prior conversations (reference only)
These are Claude Code transcripts (jsonl). Search them with `rg -o -m5 '<term>.{0,200}' <file>`; never read or cat one whole. Use them only when a report or DECISIONS_LOG leaves something unclear.
- `~/.claude/projects/-home-james-bet-project-BayesianFootball/3fa8246d-377e-4d80-92ca-c88d9982c384.jsonl` (26 MB): the old manager. It covers the market-model stream (TODO 039), the design of waves 1–5, and James's grill answers.
- `~/.claude/projects/-home-james-bet-project-BayesianFootball/d4671e26-0926-46fd-9b6e-b9746b2a02d4.jsonl`: the current manager, from 2026-10-11 (market-APM, wave-5 review).
- `~/.claude/projects/-home-james-bet-project--worktrees-BayesianFootball-qs-experiment/1e5b2135-93d8-4e28-89b8-9e21e0727d04.jsonl`: wave 3 (fusion with bias, TODO 042).
- `…-qs-experiment/706ce404-b7fd-4edd-818e-5c63cf42e4cf.jsonl`: wave 4 Phases 3–4 (backtest grid, TODO 043).
- `…-qs-experiment/43feb3fa-104e-43ac-8bdd-ecd152da9023.jsonl`: wave 5 (TODO 044) and the launch of this brief.
- `~/.claude/projects/-home-james-bet-project--worktrees-BayesianFootball-clv-napkin/802992ce-0394-4a12-ae88-e8c156316275.jsonl`: the CLV napkin, including brief 04 (the 26/27 order book).
- Memory notes: `~/.claude/projects/-home-james-bet-project-BayesianFootball/memory/`. The relevant ones are `qs-goal-vs-market-experiment-todo-040.md`, `clv-napkin-2026-09-28-results.md`, `live-fold-staleness-card-injection.md`, `julia-runs-via-jl-repl.md` and `mcmc-beast-use-16-threads.md`.
- Reports to read first, at the section level: wave-5 `REPORT.md` and `PAPER_TRADING_CONFIG.md`, wave-4 `REPORT.md` §summary, and CLV napkin §13.

### Manager rulings (binding; copy into `DECISIONS_LOG.md` at Phase 0)
1. **Scope:** the 79 finished 26/27 L1/L2 matches (2026-08-01 → 2026-09-19). Bets only where a T−25m book exists (about 60 fixtures). Fixtures with no book are predicted and scored for log loss but never bet. No price imputation.
2. **Predictions without look-ahead:** extend qs_marketobs `81fdf817` in memory with the run's own splitter and "26/27" appended, biweekly folds as in the CLV napkin r04.
   - Positions 1–40 must reproduce the persisted held-out sets exactly.
   - Each new fold trains on every in-scope fixture on a calendar day strictly before its first held-out day.
   - Use the fit's own model, sampler and seeds rule.
   - A fold with R̂ > 1.05 gets one seeded rerun. Report divergences and tail-ESS.
   - **No persistence in the DB:** serialise the extension to `/root/BF_runs/qs_2627/out/`, and commit only small CSVs (per-fixture probabilities, convergence).
3. **Market-rate observations for 26/27 training matches:**
   - Compute the KL-inverted rates from the order-book close with the same inversion code and full-book rule (≥ 5 selections) as wave 1/2 (`05_.../market_arm.jl`, `06_...` rates table).
   - Close = TWA over (−20, 0] min of the de-vigged **mid** (`c4_close`).
   - **Gate:** the inversion code, run on the pinned 24/25–25/26 closes, reproduces the committed wave-1/2 rates table for those matches (state the tolerance before you run).
   - The 24/25–25/26 rates stay exactly as fitted. Log the source change (odds_history LTP before, order-book mid in 26/27).
   - Rounds with no book contribute no observation (masking, as for incomplete books).
4. **Benchmark predictions:** use m12's DB folds if Phase 0 shows they cover the period with strictly-earlier training. Otherwise extend m12 in memory the same way; that is a second sampling job and needs the manager's go.
5. **Prices:**
   - The primary cell is the **best back price** at T−25m, from a complete two-sided book ≤ 15 min old. These are executable.
   - Secondary cells, the same bets' world at **mid** and at **LTP**, are comparable with the backtests.
   - 2% commission on net winnings in every cell. CLV is measured against the order-book close (mid TWA, as in ruling 3).
6. **Cells:** frozen (B1, λ 12), fallback (B1, λ 23) and m12 canonical (λ 23), × back / mid / LTP. That is 9 cells, all on one continuous 26/27 bankroll from 1.0 starting on 2026-08-01, with MDD measured on that path. Report the drawdown watch against −20.39%. Nothing else is scored, and every cell computed is reported.
7. **Fill check:** at a notional bankroll of £1,000 (and £200), report:
   - the share of the frozen config's back-price bets whose stake exceeds the best-level back size;
   - the share exceeding all 3 levels;
   - the wealth if each bet is capped at the available size.
8. **Statistics:** with about 60 bets per cell and 8 match days, report point values with slate-clustered intervals as in the CLV napkin. Draw no ranking conclusion. Add 1X2 per-selection log loss of qs and m12 against the mid close on the same fixtures, split into transition clubs vs the rest.
9. **Engine gate before any 26/27 number:** the 26/27 replay code, fed wave 5's 25/26 T−25m inputs, must reproduce the frozen cell bit-identically: 1.6241209548752367, 494 bets.

### Reading rules
- Use `Read` with offset/limit, or `rg -n`. Never read whole long files or transcripts.
- At most 40 lines of command output. Send longer output to `logs/` and grep it.
- Keep `PROGRESS.md` and `DECISIONS_LOG.md` current in the new folder.

**After a compaction:**
1. Re-read this brief's §2, §4 and §5, then `PROGRESS.md` and `DECISIONS_LOG.md`.
2. Never guess a path, run ID or pane ID; take it from those files or `rg --files`.
3. Don't redo work marked done; check that its output exists.

## 2. Rules
- **Folder:** `experiments/scotland/10_qs_marketobs_2627_paper/`. Reuse wave 4 and wave 5 code by `include`. Copy a function only if you must change it, and say so in DECISIONS_LOG.
- **TODO 045:** create it from `todos/template.md` (assignee claude, P2), add it to `todos/README.md`, keep its work log current, and close it at the end.
- **Git:**
  - Commit small steps and push only `exp/qs-goal-vs-market`. Never merge, rebase or force-push.
  - src is frozen. If you think src must change, stop with P27_BLOCKED.
  - `logs/` is gitignored, so commit logs with `git add -f`, after checking that they contain no connection strings.
- **Julia only on mcmc-beast:**
  - Connect with `ssh root@mcmc-beast`, and make your own checkout at `/root/BF_runs/qs_2627`, at your pushed commit.
  - Drive one persistent REPL with `jl` (`~/.pi/agent/bin/jl`): `jl start claude_p27 --beast --dir /root/BF_runs/qs_2627 --threads 16`. Then `jl run claude_p27 <file.jl> --timeout <s>` blocks until the job ends; for long jobs use `--bg` plus `jl wait`.
  - Set BLAS 1 and `pinthreads(:cores)`. Logs go to `/root/BF_runs/logs/qs_2627/`. No laptop Julia.
  - Before the sampling job, check that the beast is idle (`uptime`, `jl ls`).
- **Never touch other tmux panes or sessions**, locally or on the beast.
  - Locally: `%109` (pi), `%110` (James's Claude) and `%114` (your manager).
  - On the beast, leave `julia`, `claude_clv_napkin`, `pi_klm`, `w2_watch`, `mbtop`, and any `jl-*` session that isn't yours, alone.
  - Keep `/root/BF_runs/*`.
- **Read-only DB:**
  - Connect with `default_transaction_read_only=on`.
  - Run wave 4's before/after audit of the fit, config and harness tables around every runner, and stop on any change.
  - The DataStore is rebuilt from betdb with the read-only SQL loader and saved to a file on the beast; record its sha256. **Never print connection strings or credential-bearing exceptions** (`.env` holds BF_DB_URL).
- **Waiting on beast jobs:** one blocking command (`jl run`/`jl wait`, or one background shell command that blocks until an end marker or `ERROR:`) with an explicit long timeout. No polling loops.
- **Honesty:** report every cell you computed, losers included, and every gate result. Say plainly that 8 match days cannot rank anything.

## 3. Phases

### Phase 0: setup, data audit, gates, plan (no sampling)
1. Set up: the folder, TODO 045, DECISIONS_LOG (rulings 1–9), PROGRESS, the beast checkout and the `jl` REPL.
2. Rebuild the DataStore from betdb (read-only) and record its sha256. Confirm the 79 fixtures and their results.
3. Build a coverage table by match day, as CLV napkin §13.1. It must reproduce 79 / 70 / 60 at T−25m, or explain any difference.
4. Build the odds frame from the order book at T−25m for all canonical markets (1X2, BTTS, OU0.5–3.5), in back, mid and LTP versions, plus the mid TWA close. Use the column layout of wave 4's snapshot frame (`match_id`, `market_name`, `market_line`, `selection`, odds), so `PF.build_books_reported` accepts it.
5. Run the gates:
   - the inversion reproduction (ruling 3);
   - the engine reproduction (ruling 9);
   - the m12 fold inventory and filtration (ruling 4);
   - a dry run of the qs extension plan: positions 1–40 match, the new positions are listed with their held-out fixtures and training cut-off days, and nothing is sampled.
6. Plan the sampling job: the number of new folds, the ETA from one timed warm fold if cheap, and the memory needed.
7. Stop with P27_HANDOVER. The manager approves the sampling job(s).

### Phase 1: walk-forward predictions
1. Run the approved in-memory extension(s). Report the convergence table (R̂, ESS, divergences) and any reruns.
2. Re-run the filtration check on the sampled folds.
3. Write per-fixture held-out probabilities for qs (and m12): 1X2, OU lines and BTTS, with the fold id and training cut-off.
4. Report 1X2 log loss vs the mid close (ruling 8).
5. Stop with P27_HANDOVER.

### Phase 2: paper trade, report
1. Replay the 9 cells (ruling 6) on the 26/27 days with books. Build the per-day ledger, which feeds the bankroll path, and record per bet: stake, price used, back size available, P&L and CLV.
2. Run the fill check (ruling 7) and the drawdown watch.
3. Write `REPORT.md`:
   - a summary of at most 10 lines;
   - then §§ data and coverage, predictions, paper trade (all 9 cells), fills, comparison with the wave-5 backtest, and caveats.
   - The caveats must cover: 8 match days; about 60 bets; the source change for market rates; transition clubs; the ingestion gap after 09-19; LTP missing on 08-01; and that the order-book close is not the odds_history close.
4. Write `PAPER_LEDGER.csv`: one row per bet of the frozen config at the back price.
5. Close out: close TODO 045 with follow-ups (the first is the ingestion gap after 09-19 and the src/MatchDay integration), push everything, `jl stop claude_p27`, and print P27_DONE.

## 4. Sentinels
Print each alone on its own line:
- `P27_HANDOVER` at the end of Phases 0 and 1;
- `P27_BLOCKED` with a one-line reason;
- `P27_DONE` at the end.

## 5. DONE WHEN
1. Every gate passed (rulings 3, 4, 9 and the 1–40 position check), and the DB audit was unchanged throughout.
2. No sampled fold saw a fixture on or after its first held-out day, and the filtration CSV is committed.
3. The 9 cells, the fill check and the ledger are committed, and REPORT.md answers §3 Phase 2 with traceable numbers.
4. TODO 045 is closed with follow-ups, everything is pushed, the REPL is stopped, and `P27_DONE` is printed.
