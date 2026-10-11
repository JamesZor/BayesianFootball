# Decisions log — 26/27 paper trade (TODO 045, Claude CLI)

Brief: `experiments/claude_qs_2627_paper_brief.md` (`57cf058e`). Manager: the wave-5 Claude session
(local pane %114). Sentinels: P27_HANDOVER / P27_BLOCKED / P27_DONE.

## Manager rulings (binding; copied from the brief at Phase 0)

1. **Scope:** the 79 finished 26/27 L1/L2 matches (2026-08-01 → 2026-09-19). Bets only where a
   T−25m book exists (about 60 fixtures). Fixtures with no book are predicted and scored for log
   loss but never bet. No price imputation.
2. **Predictions without look-ahead:** extend qs_marketobs `81fdf817` in memory with the run's own
   splitter and "26/27" appended, biweekly folds as in the CLV napkin r04.
   - Positions 1–40 must reproduce the persisted held-out sets exactly.
   - Each new fold trains on every in-scope fixture on a calendar day strictly before its first
     held-out day.
   - Use the fit's own model, sampler and seeds rule.
   - A fold with R̂ > 1.05 gets one seeded rerun. Report divergences and tail-ESS.
   - **No persistence in the DB:** serialise the extension to `/root/BF_runs/qs_2627/out/`, and
     commit only small CSVs (per-fixture probabilities, convergence).
3. **Market-rate observations for 26/27 training matches:**
   - Compute the KL-inverted rates from the order-book close with the same inversion code and
     full-book rule (≥ 5 selections) as wave 1/2 (`05_.../market_arm.jl`, `06_...` rates table).
   - Close = TWA over (−20, 0] min of the de-vigged **mid** (`c4_close`).
   - **Gate:** the inversion code, run on the pinned 24/25–25/26 closes, reproduces the committed
     wave-1/2 rates table for those matches (state the tolerance before you run).
   - The 24/25–25/26 rates stay exactly as fitted. Log the source change (odds_history LTP before,
     order-book mid in 26/27).
   - Rounds with no book contribute no observation (masking, as for incomplete books).
4. **Benchmark predictions:** use m12's DB folds if Phase 0 shows they cover the period with
   strictly-earlier training. Otherwise extend m12 in memory the same way; that is a second
   sampling job and needs the manager's go.
5. **Prices:**
   - The primary cell is the **best back price** at T−25m, from a complete two-sided book ≤ 15 min
     old. These are executable.
   - Secondary cells, the same bets' world at **mid** and at **LTP**, are comparable with the
     backtests.
   - 2% commission on net winnings in every cell. CLV is measured against the order-book close
     (mid TWA, as in ruling 3).
6. **Cells:** frozen (B1, λ 12), fallback (B1, λ 23) and m12 canonical (λ 23), × back / mid / LTP.
   That is 9 cells, all on one continuous 26/27 bankroll from 1.0 starting on 2026-08-01, with MDD
   measured on that path. Report the drawdown watch against −20.39%. Nothing else is scored, and
   every cell computed is reported.
7. **Fill check:** at a notional bankroll of £1,000 (and £200), report the share of the frozen
   config's back-price bets whose stake exceeds the best-level back size; the share exceeding all
   3 levels; the wealth if each bet is capped at the available size.
8. **Statistics:** with about 60 bets per cell and 8 match days, report point values with
   slate-clustered intervals as in the CLV napkin. Draw no ranking conclusion. Add 1X2
   per-selection log loss of qs and m12 against the mid close on the same fixtures, split into
   transition clubs vs the rest.
9. **Engine gate before any 26/27 number:** the 26/27 replay code, fed wave 5's 25/26 T−25m
   inputs, must reproduce the frozen cell bit-identically: 1.6241209548752367, 494 bets.

## Implementation decisions (implementer; the manager may overrule)

- **D1 (2026-10-11) Folder and reuse.** Code lives in `experiments/scotland/10_qs_marketobs_2627_paper/`
  as loader `l00_p27.jl` (module `QSPaper2627`) plus runners `rNN_*.jl`. Wave-4 modules
  (`QSBacktestPhase0`, `QSBacktestScenario`, `QSBacktestPhase3`) and wave 5 (`QSWave5`) are
  `include`d unchanged. Copies from `proto/clv-napkin` @ `1c215df4` are listed in D-entries below.
