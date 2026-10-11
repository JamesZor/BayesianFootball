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
- **D2 (2026-10-11) Read-only betdb.** `betdb_connect` passes `default_transaction_read_only=on` through
  LibPQ's `options` connection parameter (LibPQ.jl sends its own `options`, which is presumably why wave 4's
  PGOPTIONS had no effect). The session SHOW is checked and must be `on` (logged: `on`). The
  `mcmc_experiments` connection is unchanged (SHOW `off`, logged once); there the before/after audit is the guard.
- **D3 (2026-10-11) 26/27 DataStore.** `Data.get_datastore(conn, ScottishLower())` over the read-only session:
  the src pipeline `load_datastore_sql` runs, minus its own connection. The checker allowlist was **not**
  widened, because no forbidden name is called. Pinned to `/root/BF_runs/qs_2627/out/datastore_ScottishLower_2627.jls`,
  sha256 `f407a1b4c3188472b803804245dd7dbb0011aafc6471390e67038bb8c426c2b6`, 2,049 matches, last 2026-09-19.
  The 79 L1/L2 26/27 fixtures (08-01 → 09-19) all have results.
- **D4 (2026-10-11) Order book, copies from `proto/clv-napkin` @ `1c215df4` (`l04_live_orderbook.jl`).**
  The order book was pulled once and pinned to `out/orderbook_2627.jls`, sha256
  `2adb12715f9bfd73b06ae0a71bda5724fdd4b298776789670b5e4bd9f8a5443a`: 420 markets, 35,234 pre-off snapshots,
  70 fixtures. Copies, each changed:
  - `c4_markets` → `ob_markets`: all canonical market types, not MATCH_ODDS only, and one market per (fixture, type) asserted;
  - `c4_snapshots` → `ob_snapshots`: runners mapped by `MD.betfair_to_key`;
  - `c4_entry_book`/`_c4_book_at` → `entry_book`: per market rather than per 1X2. Every runner's last snapshot
    at or before T−25m must be two-sided and ≤ 15 min old, the market complete, and the mid overround in
    [0.90, 1.10]. Refusals are kept;
  - `c4_scope_extended` → `scope_extended`: verbatim;
  - `c4_extension_plan` → `walkforward_plan`: adds a position-by-position equality check (against the
    persisted folds' held-out sets on the pinned store) to the union check, and works for both splitters;
  - `c4_filtration` → `filtration`: the pxG column is replaced by the count of full-book market rates.
  - `c4_extend_inmemory` and `c4_convergence_table` will be copied into `l01_sampling.jl` in Phase 1.
- **D5 (2026-10-11) Close.** TWA over (−20, 0] min of the order-book mid, through the TODO 039 closing-book
  call (`Data.summarize_odds`, default overround filter), then `MM.gate_book`: complete markets, overround in
  [0.90, 1.10], multiplicative de-vig. For complete markets this equals `c4_close` + de-vig. One gated book
  (`results/phase0/phase0_close_mid.csv`) feeds the inversion (ruling 3), CLV (ruling 5) and log loss (ruling 8).
- **D6 (2026-10-11) Inversion gate (ruling 3).** The tolerance was stated before the run: IDs and full-book flags
  exact, zero dummies exact, max |Δ log λ| ≤ 1e-12 on full-book rows. **Result: PASS**, with max |Δ| = 0.0 on all
  1,430 committed rows (22/23–25/26). The code is `MM.gated_close` + `MM.invert_panel` with
  `QSMarketArm.panel_config()` on the pinned cache (sha `c786e2fc…`), and the wave-2 table rule.
- **D7 (2026-10-11) 26/27 market rates.** The same `invert_panel` with `seasons = ["26/27"]` on the order-book
  mid close covers 79 fixtures: 60 have a close book, 56 are accepted (all with ≥ 5 selections, so all full
  book), 4 are refused for "summed KL > 0.01" (16362588, 16362589, 16362607, 16362792), and 19 have no book.
  **Source change:** the 24/25–25/26 rows stay the odds_history LTP closes as fitted, while the 26/27 rows are
  order-book mid closes. Fixtures without a full book are masked (zero dummy, mask 0).
- **D8 (2026-10-11) The 26/27 qs model.** It is the fit's own `PoissonCountModel`, with the identical
  interception, dynamics, home-advantage, covariate and guard objects. Only `MarketRateObservation.feature` is
  replaced, by the committed 1,430 rows plus the 79 rows for 26/27; the σ prior is unchanged. The fitted
  rows were verified element-wise, and the fit's feature digest equals the committed table's
  (`1c8f4749…`). The extended feature digest is `17b3ceb3…`.
- **D9 (2026-10-11) Transition clubs** (napkin rule, `club_season_tiers`, tier in 25/26 ≠ tier in 26/27):
  airdrieonians, east-kilbride, kelty-hearts-fc, ross-county. They appear in 28 fixtures, the same as the napkin.
- **D10 (2026-10-11) Engine gate (ruling 9): PASS, bit-identical** on all three cells. `replay_cells` on
  wave 5's 25/26 T−25m inputs gives frozen 1.6241209548752367 / 494, fallback 1.3208208815934512 / 496
  and m12 1.327330130248303 / 557. B1 is spelt `W5.basket("B1_canonical", [H, D, A, U25])`.
- **D11 (2026-10-11) Splitter semantics.** The run's `ScopedWalkForwardCV`, with 26/27 appended, gives 4 new
  biweekly positions (41–44), which hold out 79/79. With `history_seasons = 2` counted back from the
  target season, a 26/27 fold trains on 24/25 + 25/26 + the 26/27 days before its step: fold 41 trains on
  710 fixtures, while folds 1–40 also had 22/23–23/24. This is the run's own rule. As in the napkin, a step's
  later days are predicted with 1–2 weeks less data; for example, 09-05 is predicted from data through 08-22.
- **D12 (2026-10-11) Price cells (ruling 5): implementer's reading, for the manager to confirm.** Each price
  cell is a full replay that uses that price both to decide and to settle, as the backtests used LTP for both.
  So back, mid and LTP can bet different sets. The LTP frame drops any market with a missing runner LTP,
  which removes all of 08-01 (50 fixtures remain, against 60 for back and mid).
- **D13 (2026-10-11) m12 inventory (ruling 4): the DB folds do NOT cover the period.**
  - DB folds 41–44 hold 69 of the 79 fixtures. **The 09-05 round (10 fixtures, all with T−25m books) has no
    m12 latent**: fold 43 held out only 08-28/08-29.
  - Folds 41–43 share one runtime (260.94 s), so they were sampled in one batch. Fold 45 holds the
    unplayed 10-03 card.
  - Reconstructed training is strictly earlier for every fold.
  - Under ruling 4 this means an in-memory m12 extension (positions 41–44, the run's own
    `GroupedCVConfig` with 26/27 and its sampler 4 × (800 + 800)) is needed. That is a second sampling job,
    so it needs the manager's go. Its dry run passes: positions 1–40 equal 40/40 with union 710, the
    filtration passes, and it covers 79/79.

## Manager rulings after Phase 0 (2026-10-11)

- Phase 0 accepted. Sampling job 1 (qs, in-memory positions 41–44) and job 2 (m12, in-memory positions 41–44
  for all 79 fixtures) approved; the m12 DB folds are kept as a check on the 69 shared fixtures only.
- D12 approved: each price cell decides and settles at its own price; back is primary; report the differing
  bet sets and the LTP loss of 08-01.
- m12 condition: held-out 26/27 fixtures must take their XI by exactly the rule of positions 1–40 (no
  LastHistorical-style shortcut); state which XI they use and list it as a caveat if it is the played XI.
- Run qs first, then m12, one blocking wait each.

## Phase 1 implementation decisions

- **D14 (2026-10-11) Sampling path.** `l01_sampling.jl` copies `c4_extend_inmemory` and
  `c4_convergence_table`, with one change: the chain fan-out is wave 2's `sample_pending!` loop (copied, no
  checkpoints), so each chain runs in its own task after `Random.seed!(seed)`. The rest is unchanged:
  `Features.create_features` → `Training.sample_fold` per chain → `FoldFit` → `Inference.audit_convergence`
  (the fit's thresholds) → `Inference._extension_extract_latents`, after `Random.seed!(post_seed)`.
  Training internals are called through the aliases `TR = BF.Training` and `TI = TR.Inference`. The
  checker's `BF.Training.` allowlist does not see these alias calls, so they are listed here: `sample_fold`,
  `FoldFit`, `_inf_narrow`, `audit_convergence`, `sampler_max_depth`, `_extension_extract_latents` and
  `merge_latents`. None of them persists anything. The before/after DB audit is the guard, and it was
  unchanged around both jobs.
- **D15 (2026-10-11) Seed rules.**
  - **qs** uses the fit's own wave-2 schedule (`06_.../l05_grid.jl`, REPRODUCIBILITY.md), arm 4: chain seed
    202700000 + 400000 + 100·fold + 10·attempt + chain (fold 41 → 203104101…), post seed 203109000 + attempt.
  - **m12** has no recorded seed rule; its DB extensions ran `extend_fit` on the task RNG. So a schedule was
    frozen before sampling: 2026101100 + 100·fold + 10·attempt + chain, post 2026101100 + 9000 + attempt.
  - The sampler is each fit's own: qs 4 × (500 + 1000); m12 4 × (800 + 800), matching its 800-draw folds as
    `extend_fit` requires.
- **D16 (2026-10-11) Results of the jobs.**
  - **qs** sampled in 1.5 min: max R̂ 1.0056 / 1.0072 / 1.0061 / 1.0115, min tail ESS 1522 / 1097 / 1174 / 768,
    0 divergences out of 4000 per fold.
  - **m12** sampled in 0.5 min: max R̂ 1.0061 / 1.0084 / 1.0044 / 1.0053, min tail ESS 1085 / 873 / 1387 / 1504,
    0 divergences out of 3200 per fold.
  - No rerun was due. The extensions are at `out/ext_qs_marketobs_2627.jls` (sha `5f6a7e28…`) and
    `out/ext_m12_2627.jls` (sha `f7328646…`).
  - The sampled-fold filtration re-checks what actually went into each feature set (`ordered_match_ids`):
    the training set equals the plan, there is no overlap, the last training day is strictly before the
    first held-out day, and the latents cover exactly the held-out set. It passes for all 8 folds.
  - qs market observations per fold: 517 / 534 / 543 / 554. The 26/27 ones among them: 0 / 17 / 26 / 37.
- **D17 (2026-10-11) m12 held-out XI (manager condition).** The 26/27 held-out fixtures take the **played XI**:
  the starting rows of `ds.lineups` for that fixture.
  - The rule lives in the src feature builder (`features/extractors/plus_minus_extractors.jl`). The RAPM
    ratings are fitted only on the fold's history (`fit_on = :history`). The starting-XI map is then built
    over every match in the store, and the held-out fixture is looked up in it.
  - Positions 1–40 used the same builder, so the rule is identical. No MatchDay `LastHistorical` source is
    involved.
  - All 79 held-out fixtures have 11 + 11 starters and an entry in the lineup map.
  - **Caveat for the report:** m12's 26/27 predictions use the confirmed starting XI. That XI is normally
    published about 60 min before the off, so it is known at T−25m. It still is not a forecast XI, and any
    change after T−25m would leak.
- **D18 (2026-10-11) m12 DB-fold check (69 shared fixtures).** The chain shapes are identical (67 / 71 / 71 / 71
  params × 3200 draws). In-memory vs DB 1X2 probabilities differ by a mean |Δp| of 0.00043 (max 0.0027), with
  corr(p_home) 0.99992 and mean |Δ log λ| 0.0015. The difference is Monte Carlo only.
- **D19 (2026-10-11) Log loss (ruling 8).**
  - The 1X2 per-selection binary log loss and the 3-way log loss come from posterior-mean probabilities
    through the production double-Poisson grid. The close is the de-vigged mid TWA (D5).
  - Intervals are slate-clustered 95%, resampling match days (napkin `cn_summary`: B 2000, seed 20260928).
  - Model minus close is reported only on the 60 fixtures that have a close.
  - In `phase1_logloss.csv`, the `mid_close` rows under panel `all_79` cover only the 60 fixtures with a close.

## Phase 2 implementation decisions

- **D20 (2026-10-11) The 26/27 model inputs to the replay.** `l03_paper.jl::model_2627` builds the NamedTuple
  that `W5.day_problems` / `Q.replay_cells` expect (as `P3.goal_model` returns) from the Phase 1 extension files,
  after checking their sha256 against the Phase 1 logs:
  - `source = latents = ext.latents` (79 fixtures, merged folds 41–44);
  - the witness groups draws by fold: `"<label>|<run>+mem2627|fold=<k>|<layout>"`, with layouts 4chains_x_1000
    (qs) and 4chains_x_800 (m12);
  - `converged = true`, because every fold's used attempt has R̂ ≤ 1.05. The flag is only carried into the
    report; it does not change stakes.
  - The engine gate (ruling 9) was run again at the Phase 2 head before any 26/27 number: bit-identical on all
    three cells. The odds frames were rebuilt from the pinned order book and checked equal to the committed
    Phase 0 CSVs.
- **D21 (2026-10-11) Bankroll path and MDD.** Each cell starts at 1.0 on its first day with a book (2026-08-01
  for back and mid, 2026-08-08 for LTP). MDD is measured on the path including the opening point
  (`W5.drawdown_pct(trajectory.bankroll)`). It equals the PF summary's MDD in every cell.
- **D22 (2026-10-11) Settlement price.** Bets settle at `odds_used` (the `DeArb` price), as in every backtest.
  - At the back price, no bet was de-arbed (back overround > 1 on every book).
  - At the mid, 62 of the frozen cell's 118 bets were settled below the quoted mid, because the mid book's
    overround can fall below 1. LTP had 45 of 100.
  - The ledger keeps both the quoted and the settled price.
- **D23 (2026-10-11) Fill check (ruling 7).** It covers the frozen back-price bets only.
  - The money stake is the stake fraction × the opening bankroll on the paper path × the notional (£1,000 or £200).
  - The shares compare this stake with the best-level back size and with the 3-level depth (`back_depth3`) of
    the entry snapshot.
  - The capped wealth re-runs the bankroll path day by day, with each bet's stake capped at the available size
    and its P&L scaled by the filled share.
  - **Approximation:** the policy's stake fractions are kept from the uncapped run, so the SlateDrawdown state
    is not re-solved on the capped path.
  - The depth-3 cap settles the whole fill at the best price, so it is optimistic about price.
  - The uncapped re-run reproduces the cell's final wealth bit-identically (`FILL_REPRODUCES`).
- **D24 (2026-10-11) Intervals (ruling 8).** All intervals resample match days with replacement (B 2000, seed
  20260928, as in Phase 1). They cover:
  - g, the mean daily log growth;
  - wealth, as exp of the summed resampled daily log growth;
  - money ROI, as a ratio of sums over the resampled days' bets;
  - mean CLV in pp against the mid close.
  The paired frozen − other daily log growth uses zero for a day missing from one path. It is reported only to
  show the width; no ranking is drawn from it.
- **D25 (2026-10-11) Bet sets.** Frozen and fallback share one joint solve and the same basket, so their bet sets
  are identical at every price; λ only scales the stakes. The back, mid and LTP sets differ (D12), as shown in
  `phase2_bet_sets.csv`.
